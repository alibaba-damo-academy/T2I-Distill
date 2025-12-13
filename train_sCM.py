#!/usr/bin/env python
# coding=utf-8

import gc
import os
import math
import logging
import argparse
import numpy as np
from tqdm.auto import tqdm
from functools import partial
from datetime import timedelta

import torch
import torch.nn as nn
from torchvision import transforms

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs, set_seed
from accelerate.utils import DataLoaderConfiguration

import transformers
from transformers import T5TokenizerFast
from transformers import CLIPTokenizer

from datasets import load_dataset

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from data.custom_dataset import Text2ImageDataset
from model.mmdit_flux import FluxTransformer2DModelTimestepX1GuidanceX1
from model.utils import pack_latents, prepare_latent_image_ids, load_text_encoders, import_model_class_from_model_name_or_path, unwrap_model
from model.pipeline import compute_text_embeddings
from model.attn_processor import FluxAttnProcessor2_0_vanilla, FluxSingleAttnProcessor2_0_vanilla

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # ----- loaded flux model -----
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--rescaled_checkpoint_path",
        type=str,
        default=None,
    )
    # ----- text encoder  -----
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum sequence length to use with with the T5 text encoder",
    )

    # ----- dataset  -----
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    
    # ----- directory  -----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dyn",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # ----- validation  -----
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )

    # ----- optimizer  -----
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )

    # ----- optimization hyper-paramter  -----
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    # ----- save & resume  -----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # ----- diffusion sample timestep  -----
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )

    # ----- logging -----
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # distributed training / deepspeed config / fsdp config
    parser.add_argument(
        "--ds_file",
        type=str,
        default='deepspeed_config/deepspeedconfig_bf16_batch8.json',
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    # sCM
    parser.add_argument(
        "--sigma_data", type=float, default=0.5, help="Standard deviation of data distribution is supposed to be 0.5, from sana-sprint"
    )
    parser.add_argument(
        "--scm_cfg_scale", type=float, nargs="+", default=[4, 4.5, 5], help="Range for classifier-free guidance scale"
    )
    parser.add_argument(
        "--tangent_warmup_steps", type=int, default=4000, help="Number of warmup steps for tangent vectors"
    )
    parser.add_argument("--scm_lambda", type=float, default=1.0, help="Weighting coefficient for SCM loss")
    # parser.add_argument(
    #     "--guidance_embeds_scale", type=float, default=0.1, help="Scaling factor for guidance embeddings"
    # )


    args = parser.parse_args()

    return args


def save_model_hook_partial(models, weights, output_dir, accelerator):
    if accelerator.is_main_process:
        for i, model in enumerate(models):
            if isinstance(unwrap_model(model.student, accelerator), FluxTransformer2DModelTimestepX1GuidanceX1):
                unwrap_model(model.student, accelerator).save_pretrained(os.path.join(output_dir, "transformer"))
            else:
                raise ValueError(f"Wrong model supplied: {type(model)=}.")
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()


def load_model_hook_partial(models, input_dir, accelerator):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model.student, accelerator), FluxTransformer2DModelTimestepX1GuidanceX1):
            load_model = FluxTransformer2DModelTimestepX1GuidanceX1.from_pretrained(input_dir, subfolder="transformer")
            model.student.register_to_config(**load_model.config)

            model.student.load_state_dict(load_model.state_dict())
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

        del load_model


class FluxTrigFlow(FluxTransformer2DModelTimestepX1GuidanceX1):
    def __init__(self, original_model, guidance=False):
        self.__dict__ = original_model.__dict__
        self.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.guidance = guidance
        if self.guidance:
            hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
            self.logvar_linear = nn.Linear(hidden_size, 1)
            nn.init.xavier_uniform_(self.logvar_linear.weight)
            nn.init.constant_(self.logvar_linear.bias, 0)

    def forward(
        self, hidden_states, encoder_hidden_states, pooled_projections, timestep, txt_ids, img_ids,
        guidance=None, jvp=False, return_logvar=False, **kwargs
    ):
        batch_size = hidden_states.shape[0]    # Teacher G Part:  bsz, int        .
        latents = hidden_states                # Teacher G Part: [bsz, 1024, 64]  . for 512 image
        prompt_embeds = encoder_hidden_states  # Teacher G Part: [bsz, 256, 4096] .
        t = timestep                           # Teacher G Part: [bsz]            .

        # TrigFlow --> Flow Transformation
        timestep = t.expand(latents.shape[0]).to(prompt_embeds.dtype)  # Teacher G Part: [bsz]
        latents_model_input = latents                                  # Teacher G Part: [bsz, 1024, 64]

        flow_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))  # [bsz]

        flow_timestep_expanded = flow_timestep.view(-1, 1, 1)  # [bsz, 1, 1]
        latent_model_input = latents_model_input * torch.sqrt(
            flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2
        )  # [bsz, 1024, 64], FP32
        latent_model_input = latent_model_input.to(prompt_embeds.dtype)

        # forward in original flow

        if jvp and self.gradient_checkpointing:  # Student \theta^{-}
            self.gradient_checkpointing = False
            model_out = super().forward(
                hidden_states=latent_model_input,
                timestep=flow_timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
            )[0]
            self.gradient_checkpointing = True
        else:  # Teacher G Part
            model_out = super().forward(
                hidden_states=latent_model_input,
                timestep=flow_timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
            )[0]  # model_out.shape=[bsz, 1024, 64], BF16

        # Flow --> TrigFlow Transformation
        trigflow_model_out = (
            (1 - 2 * flow_timestep_expanded) * latent_model_input
            + (1 - 2 * flow_timestep_expanded + 2 * flow_timestep_expanded**2) * model_out
        ) / torch.sqrt(flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2)

        temb = self.time_text_embed(
            timestep, guidance, pooled_projections
        )

        if return_logvar:
            logvar = self.logvar_linear(temb)
            return trigflow_model_out, logvar

        return trigflow_model_out


class TeacherStudentModelPair(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.student.train()
        self.student.requires_grad_(True)


def initialize_models(args, accelerator):
    # Load the tokenizers
    logger.info(f"[INFO] loading tokenizers")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    
    # import correct text encoder classes
    logger.info(f"[INFO] loading text encoders")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)

    gc.collect()

    logger.info(f"[INFO] loading vae")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    vae.enable_slicing()
    gc.collect()
    
    logger.info("[INFO] init teacher and student Flux model: START")
    # init teacher and student model
    ori_transformer_student = FluxTransformer2DModelTimestepX1GuidanceX1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,
        state_dict={},
    )
    # as teacher, ori_transformer_no_guide in sana-sprint-diffusers
    ori_transformer_teacher = FluxTransformer2DModelTimestepX1GuidanceX1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,
        
    )

    paras = torch.load(args.rescaled_checkpoint_path, map_location='cpu', weights_only=False)
    missing_keys, unexpected_keys = ori_transformer_teacher.load_state_dict(
        paras["module"], strict=False
    )
    logger.info(f"[INFO] successfully load params, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
    missing_keys, unexpected_keys = ori_transformer_student.load_state_dict(
        paras["module"], strict=False
    )
    logger.info(f"[INFO] successfully load params, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

    # set ori_transformer_student
    for block in ori_transformer_student.transformer_blocks:
        block.attn.set_processor(FluxAttnProcessor2_0_vanilla())
    for block in ori_transformer_student.single_transformer_blocks:
        block.attn.set_processor(FluxSingleAttnProcessor2_0_vanilla())    

    # 这里两个模型都有guidence了，所以load的是一样的
    transformer = FluxTrigFlow(ori_transformer_student, guidance=True).train()
    pretrained_model = FluxTrigFlow(ori_transformer_teacher, guidance=True).eval()
    logger.info("[INFO] init teacher and student model: DONE")
    gc.collect()
    
    transformer.requires_grad_(True)
    pretrained_model.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)


    args.weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        args.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        args.weight_dtype = torch.bfloat16

    logger.info(f"[INFO] moving models to cuda")
    transformer.to(accelerator.device, dtype=args.weight_dtype)
    pretrained_model.to(accelerator.device, dtype=args.weight_dtype)
    vae.to(accelerator.device, dtype=args.weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=args.weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=args.weight_dtype)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    return vae, transformer, pretrained_model, tokenizers, text_encoders


def initialize_optimizer(args, params_to_optimize):

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
        
    return optimizer


class sCMLoss:

    def sample_time_steps(self, batch_size, logit_mean, logit_std):
        """Compute the density for sampling the timesteps from log-normal distribution."""
        sigma = torch.randn(batch_size, device="cpu")   # shape = [bsz]
        sigma = (sigma * logit_std + logit_mean).exp()  # shape = [bsz]
        u = torch.atan(sigma / 0.5)  # TODO: 0.5 should be a hyper-parameter  # shape = [bsz]
        return u

    def __call__(
        self,
        modelpair, model_input, sigma_data, latent_image_ids,
        prompt_embeds, pooled_prompt_embeds, text_ids, device,
        accelerator, global_step
    ):
        scm_cfg_scale = torch.tensor(
            np.random.choice(args.scm_cfg_scale, size=model_input.shape[0], replace=True),
            device=device,
        )  # args.scm_cfg_scale=[4, 4.5, 5], shape=[bsz], bsz个不一定是一样的
        transformer_kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "guidance": scm_cfg_scale,
        }
        pack_kwargs = {
            "batch_size": model_input.shape[0],
            "num_channels_latents": model_input.shape[1],
            "height": model_input.shape[2],
            "width": model_input.shape[3],
        }

        # Sample noise that we'll add to the latents
        bsz = model_input.shape[0]
        noise = torch.randn_like(model_input) * sigma_data                         # noise.shape = [bsz, 16, 128, 128]

        u = self.sample_time_steps(bsz, args.logit_mean, args.logit_std).to(device)   # shape = [bsz], continues

        # Add noise according to TrigFlow. zt = cos(t) * x + sin(t) * noise
        t = u.view(-1, 1, 1, 1).to(args.weight_dtype)                              # shape = [bsz, 1, 1, 1]
        noisy_model_input = torch.cos(t) * model_input + torch.sin(t) * noise      # [bsz, 16, 512/8, 512/8], FP32

        def model_wrapper(scaled_x_t, t):                        
            pred, logvar = accelerator.unwrap_model(modelpair.student)(
                hidden_states=scaled_x_t,
                timestep=t.reshape([bsz]),
                jvp=True,                   # 这两个是新的
                return_logvar=True,         # 这两个是新的
                **transformer_kwargs,
            )
            return pred, logvar

        packed_noisy_model_input = pack_latents(noisy_model_input, **pack_kwargs)  # [bsz, 1024, 64, 64], 512input

        with torch.no_grad():
            cfg_pretrain_pred = modelpair.teacher(
                hidden_states=(packed_noisy_model_input / sigma_data),      # packed_noisy_model_input.shape = [bsz, 1024, 64]
                timestep=t.reshape([bsz]),                                  # t.shape = [bsz, 1, 1, 1] --> [bsz]
                **transformer_kwargs,
            )  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
            cfg_dxt_dt = sigma_data * cfg_pretrain_pred  # cfg_dxt_dt.shape=[bsz, 1024, 64]
            dxt_dt = cfg_dxt_dt

        t = t.to(args.weight_dtype)
        v_x = torch.cos(t.reshape([bsz, 1, 1])) * torch.sin(t.reshape([bsz, 1, 1])) * dxt_dt / sigma_data  # [bs,1,1] * [bs,1,1] * [bsz, 1024, 64] / int
        v_t = torch.cos(t.reshape([bsz, 1, 1])) * torch.sin(t.reshape([bsz, 1, 1]))                        # [bs,1,1] * [bs,1,1]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.weight_dtype):
            F_theta, F_theta_grad, logvar = torch.func.jvp(
                model_wrapper, (packed_noisy_model_input / sigma_data, t.reshape([bsz, 1, 1])), (v_x, v_t), has_aux=True
            )

        F_theta, logvar = modelpair.student(
            hidden_states=(packed_noisy_model_input / sigma_data),
            timestep=t.reshape([bsz]),
            return_logvar=True,
            **transformer_kwargs,
        )

        logvar = logvar.view(-1, 1, 1)  # logvar.shape = [1, 1, 1]
        F_theta_grad = F_theta_grad.detach()
        F_theta_minus = F_theta.detach()

        # Warmup steps
        r = min(1, global_step / args.tangent_warmup_steps)  # args.tangent_warmup_step = 4000, r is float

        # Calculate gradient g using JVP rearrangement
        g = -torch.cos(t.reshape([bsz, 1, 1])) * torch.cos(t.reshape([bsz, 1, 1])) * (sigma_data * F_theta_minus - dxt_dt)  # [bs,1,1,1] * [bs,1,1,1] * (float * [bs,32,32,32] - [bs,32,32,32])
        second_term = -r * (torch.cos(t.reshape([bsz, 1, 1])) * torch.sin(t.reshape([bsz, 1, 1])) * packed_noisy_model_input + sigma_data * F_theta_grad)  # int * ([bs,1,1,1] * [bs,1,1,1] * [bs,32,32,32] + float * [bs,32,32,32])
        g = g + second_term  # [bs,32,32,32] + [bs,32,32,32]

        # Tangent normalization
        g_norm = torch.linalg.vector_norm(g, dim=(1, 2), keepdim=True)  # g_norm.shape = [1, 1, 1 ,1]
        g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper

        sigma = torch.tan(t.reshape([bsz, 1, 1])) * sigma_data  # [bs,1,1,1] * float
        weight = 1 / sigma                 # [bs,1,1,1]

        l2_loss = torch.square(F_theta - F_theta_minus - g)  # all [bs,32,32,32] for 4 variables

        # Calculate loss with normalization factor
        loss_sCM = (weight / torch.exp(logvar)) * l2_loss + logvar  # weight & logvar: [bs, 1, 1, 1]. l2_loss: [bs,32,32,32]

        loss_sCM = loss_sCM.mean()  # torch float scalar
    
        loss_no_logvar = weight * torch.square(F_theta - F_theta_minus - g)  # weight: [bs, 1, 1, 1]. others: [bs,32,32,32].
        loss_no_logvar = loss_no_logvar.mean()  # torch float scalar
        g_norm = g_norm.mean()  # torch float scalar

        total_loss = args.scm_lambda * loss_sCM

        loss_logs = {
            "loss_sCM": loss_sCM.detach().item(),
            "l2_loss": l2_loss.mean().detach().item(),
            "g_norm": g_norm.mean().detach().item(),
            "mean_weight": weight.mean().detach().item(),
        }

        return total_loss, loss_logs


def train(
        accelerator, train_dataloader, modelpair, text_encoders, tokenizers, vae,
        optimizer, params_to_optimize, lr_scheduler, first_epoch, progress_bar, global_step, loss_func
    ):
    sigma_data = args.sigma_data
    for epoch in range(first_epoch, args.num_train_epochs):
        modelpair.student.train()
        models_to_accumulate=[modelpair.student]
        modelpair.teacher.eval()
            
        for step, (pixel_values, prompts) in enumerate(train_dataloader):
            with accelerator.accumulate(models_to_accumulate):
                
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers, args.max_sequence_length, accelerator.device
                )  # prompt_embeds.shape = [bsz, 256, 4096], pooled_prompt_embeds.shape = [bsz, 768], text_ids.shape = [bsz, 256, 3]
      
                pixel_values = pixel_values.to(dtype=vae.dtype)              # [bsz, 3, 1024, 1024], FP32-->vae.dtype(BF16)
                model_input = vae.encode(pixel_values).latent_dist.sample()  # model_input.shape=[bsz, 16, 128, 128], BF16
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor * sigma_data
                model_input = model_input.to(dtype=args.weight_dtype)

                latent_image_ids = prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    args.weight_dtype,
                )  # latent_image_ids.shape = [bsz, 4096, 3]

                total_loss, loss_logs = loss_func(
                    modelpair, model_input, sigma_data, latent_image_ids,
                    prompt_embeds, pooled_prompt_embeds, text_ids, accelerator.device,
                    accelerator, global_step
                )

                total_loss = total_loss / args.gradient_accumulation_steps
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    
                    if accelerator.is_main_process:
                        # # official save method, but not work in alibaba
                        # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # logger.info(f"Saved state to {save_path}")

                        transformer_unwrap = unwrap_model(modelpair.student, accelerator)
                        save_dict = {
                                'module': transformer_unwrap.state_dict(),
                            }
                        
                        ckpt_path = f"{args.output_dir}/ckpts"
                        if accelerator.is_main_process:
                            os.makedirs(ckpt_path, exist_ok=True)
                        ckpt_name = f"{ckpt_path}/transformer_{global_step}.pt"
                        # 异步保存（可选）
                        def save_async():
                            torch.save(save_dict, ckpt_name)
                            logger.info(f"Saved checkpoint {ckpt_name} for step {global_step}")
                        import threading
                        threading.Thread(target=save_async).start()

            logs = loss_logs
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if f"{args.output_dir}/norm_log" is not None:
                os.makedirs(f"{args.output_dir}/norm_log", exist_ok=True)
            with open(f"{args.output_dir}/norm_log/step_gnorm.txt", "a", encoding="utf-8") as f:
                line = f"{step},{logs['g_norm']:.8f}\n"
                f.write(line)

            if global_step >= args.max_train_steps:
                break


def main(args):

    # =================================================================================================
    # 1. PATHS & DIRECTORY MANAGEMENT
    # =================================================================================================

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Handle the repository creation
    args.output_dir = f"{args.output_dir}/resolution{args.resolution}_bs{args.train_batch_size}_lr{args.learning_rate}"
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    run_name = f"sCM_optimizer{args.optimizer}_lr{args.learning_rate}_2"
    logging_dir_for_run = os.path.join(args.output_dir, "tensorboard_logs", run_name) # 更清晰的路径
    

    # =================================================================================================
    # 2. ACCELERATOR & SEED & LOGGING INITIALIZATION
    # =================================================================================================
    # create Accelerator
    kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(hours=24))
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, # 这个是 Accelerate 存放模型检查点等的顶层目录
        logging_dir=logging_dir_for_run # 确保这里包含了你希望的 run_name
    )
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.ds_file, zero3_init_flag=True)
    dl_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        dataloader_config=dl_config
    )

    # set seed
    if args.seed is not None:
        set_seed(args.seed)

    # create logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # create tracker
    cleaned_args = {}
    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool)):
            cleaned_args[k] = v
        elif v is None:
            cleaned_args[k] = "None"
        else:
            cleaned_args[k] = str(v)
    accelerator.init_trackers("flux_sCM", config=cleaned_args)

    # =================================================================================================
    # 3. DATASET & DATALOADER SETUP
    # =================================================================================================

    hf_dataset = load_dataset(
        "brivangl/midjourney-v6-llava",
        data_files=["data/train_000.parquet", "data/train_001.parquet", "data/train_002.parquet"],
        split="train",
    )
    train_dataset = Text2ImageDataset(
        hf_dataset=hf_dataset,
        resolution=args.resolution,
    )
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty! Please check the image directory path and ensure images are accessible.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    logger.info("[INFO] build dataset!")


    # =================================================================================================
    # 4. MODEL & LOSS FUNCTION & OPTIMIZER INITIALIZATION
    # =================================================================================================


    # Initialize all models
    vae, transformer, pretrained_model, tokenizers, text_encoders = initialize_models(args, accelerator) 
    modelpair = TeacherStudentModelPair(teacher_model=pretrained_model, student_model=transformer)

    save_model_hook = partial(save_model_hook_partial, accelerator=accelerator)
    load_model_hook = partial(load_model_hook_partial, accelerator=accelerator)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    params_to_optimize = modelpair.student.parameters()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Loss function
    loss_func = sCMLoss()
    
    # Optimizer creation
    logger.info("[INFO] prepare params for optimizer!")
    optimizer = initialize_optimizer(args, params_to_optimize)
    logger.info("[INFO] prepare optimizer done!")


    # =================================================================================================
    # 5. FINAL SETUP
    # =================================================================================================

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    modelpair, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        modelpair, optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info(vars(args))
    
    # =================================================================================================
    # 6. Train!
    # =================================================================================================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    train(
        accelerator, train_dataloader, modelpair, text_encoders, tokenizers, vae,
        optimizer, params_to_optimize, lr_scheduler, first_epoch, progress_bar, global_step, loss_func
    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
