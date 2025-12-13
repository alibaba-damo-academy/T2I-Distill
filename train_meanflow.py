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
from model.mmdit_flux import FluxTransformer2DModelTimestepX1GuidanceX1, FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1
from model.utils import pack_latents, unpack_latents, prepare_latent_image_ids, load_text_encoders, import_model_class_from_model_name_or_path, unwrap_model
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
    
    # distilled guidance input
    parser.add_argument(
        "--cfg_scale", type=float, nargs="+", default=[4, 4.5, 5], help="Range for classifier-free guidance scale"
    )

    # meanflow specific parameters
    parser.add_argument("--time_sampler", type=str, default="logit_normal", choices=["uniform", "logit_normal"], 
                       help="Time sampling strategy")
    parser.add_argument("--time_mu", type=float, default=-0.4, help="Mean parameter for logit_normal distribution")
    parser.add_argument("--time_sigma", type=float, default=1.0, help="Std parameter for logit_normal distribution")
    parser.add_argument("--ratio_r_not_equal_t", type=float, default=0.75, help="Ratio of samples where r≠t")
    parser.add_argument("--adaptive_p", type=float, default=1.0, help="Power param for adaptive weighting")
    parser.add_argument("--cfg_omega", type=float, default=1.0, help="CFG omega param, default 1.0 means no CFG")
    parser.add_argument("--cfg_kappa", type=float, default=0.0, help="CFG kappa param for mixing")
    parser.add_argument("--cfg_min_t", type=float, default=0.0, help="Minum time for cfg trigger")
    parser.add_argument("--cfg_max_t", type=float, default=1.0, help="Maxium time for cfg trigger")
    parser.add_argument(
        "--use_cfg_meanflow",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="whether use cfg version of meanflow"
    )

    parser.add_argument("--path_type", type=str, default="linear")
    parser.add_argument("--weighting", type=str, default="adaptive")
    parser.add_argument("--cfg_prob", type=float, default=0.1)

    args = parser.parse_args()

    return args


def save_model_hook_partial(models, weights, output_dir, accelerator):
    if accelerator.is_main_process:
        for i, model in enumerate(models):
            if isinstance(unwrap_model(model.student, accelerator), FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1):
                unwrap_model(model.student, accelerator).save_pretrained(os.path.join(output_dir, "transformer"))
            else:
                raise ValueError(f"Wrong model supplied: {type(model)=}.")
            # # make sure to pop weight so that corresponding model is not saved again
            # weights.pop()


def load_model_hook_partial(models, input_dir, accelerator):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model.student, accelerator), FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1):
            load_model = FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1.from_pretrained(input_dir, subfolder="transformer")
            model.student.register_to_config(**load_model.config)

            model.student.load_state_dict(load_model.state_dict())
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

        del load_model


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
    ori_transformer_student = FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,
        state_dict={},
    )
    # as teacher, ori_transformer_no_guide in sana-sprint-diffusers
    ori_transformer_teacher = FluxTransformer2DModelTimestepX1GuidanceX1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,
        state_dict={},
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

    source_state_dict = ori_transformer_student.time_text_embed.timestep_embedder.state_dict()
    ori_transformer_student.timestep_embedder2.load_state_dict(source_state_dict, assign=True, strict=False)

    # set ori_transformer_student
    for block in ori_transformer_student.transformer_blocks:
        block.attn.set_processor(FluxAttnProcessor2_0_vanilla())
    for block in ori_transformer_student.single_transformer_blocks:
        block.attn.set_processor(FluxSingleAttnProcessor2_0_vanilla())    

    # 这里两个模型都有guidence了，所以load的是一样的
    transformer = ori_transformer_student.train()
    pretrained_model = ori_transformer_teacher.eval()
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
    # transformer.to(accelerator.device, dtype=args.weight_dtype)
    # pretrained_model.to(accelerator.device, dtype=args.weight_dtype)
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


class MeanFlowLoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-0.4,                 # Mean parameter for logit_normal distribution
            time_sigma=1.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            # CFG related params
            cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
            cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
            cfg_min_t=0.0,                # Minium CFG trigger time 
            cfg_max_t=0.8,                # Maximum CFG trigger time
            ):
        self.weighting = weighting
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t

    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device, weight_dtype):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device, dtype=weight_dtype)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device, dtype=weight_dtype)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t 
    
    def __call__(
        self,
        modelpair, model_input, latent_image_ids,
        prompt_embeds, pooled_prompt_embeds, text_ids, 
        neg_prompt_embeds, neg_pooled_prompt_embeds, neg_text_ids,
        device, vae_scale_factor
    ): 
    # def __call__(self, model, images, model_kwargs=None):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """

        batch_size = model_input.shape[0]
        cfg_scale = torch.tensor(
            np.random.choice(args.cfg_scale, size=batch_size, replace=True),
            device=device,
        )  # args.cfg_scale=[4, 4.5, 5], shape=[bsz], bsz个不一定是一样的

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)  # puyifan: shape=[bsz]
        if self.label_dropout_prob > 0:
            dropout_mask = torch.rand(batch_size, device=device) < self.label_dropout_prob
            prompt_embeds[dropout_mask] = neg_prompt_embeds
            pooled_prompt_embeds[dropout_mask] = neg_pooled_prompt_embeds
            text_ids[dropout_mask] = neg_text_ids
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        transformer_kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "guidance": cfg_scale,
        }
        pack_kwargs = {
            "batch_size": model_input.shape[0],
            "num_channels_latents": model_input.shape[1],
            "height": model_input.shape[2],
            "width": model_input.shape[3],
        }
        unpack_kwargs = {
            "height": model_input.shape[2] * (vae_scale_factor // 2) ,
            "width": model_input.shape[3] * (vae_scale_factor // 2),
            "vae_scale_factor": vae_scale_factor
        }
        

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device, weight_dtype=model_input.dtype)

        noises = torch.randn_like(model_input)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * model_input + sigma_t * noises  # (1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * model_input + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)
                
        u_target = torch.zeros_like(v_t)

        packed_z_t = pack_latents(z_t, **pack_kwargs)
        packed_u = modelpair.student(
            hidden_states=packed_z_t,      # packed_noisy_model_input.shape = [bsz, 1024, 64]
            timestep=t.reshape([batch_size]),                   # t.shape = [bsz, 1, 1, 1] --> [bsz]
            timestep2=time_diff.reshape([batch_size]),                  # t.shape = [bsz, 1, 1, 1] --> [bsz]
            **transformer_kwargs,
        )[0]  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
        u = unpack_latents(packed_u, **unpack_kwargs)
        
        # get teacher's v_t
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.weight_dtype):
            packed_v_t_teacher = modelpair.teacher(
                hidden_states=packed_z_t,      # packed_noisy_model_input.shape = [bsz, 1024, 64]
                timestep=t.reshape([batch_size]),                   # t.shape = [bsz, 1, 1, 1] --> [bsz]
                **transformer_kwargs,
            )[0]  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
            v_t_teacher = unpack_latents(packed_v_t_teacher, **unpack_kwargs)
        
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)

        if args.use_cfg_meanflow and cfg_time_mask.any():
            # Split samples into CFG and non-CFG
            cfg_indices = torch.where(cfg_time_mask)[0]  # puyifan: indices
            no_cfg_indices = torch.where(~cfg_time_mask)[0]
            
            u_target = torch.zeros_like(v_t)
            
            # Process CFG samples
            if len(cfg_indices) > 0:
                cfg_z_t = z_t[cfg_indices]
                cfg_v_t = v_t[cfg_indices]
                cfg_r = r[cfg_indices]
                cfg_t = t[cfg_indices]
                cfg_time_diff = time_diff[cfg_indices]
                
                cfg_transformer_kwargs = {
                    "encoder_hidden_states": prompt_embeds[cfg_indices],
                    "pooled_projections": pooled_prompt_embeds[cfg_indices],
                    "txt_ids": text_ids[cfg_indices],
                    "img_ids": latent_image_ids[cfg_indices],
                    "guidance": cfg_scale[cfg_indices],
                }

                half_cfg_bsz = cfg_z_t.shape[0]
                
                cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
                cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_time_diff_batch = torch.cat([cfg_time_diff, cfg_time_diff], dim=0)
                
                cfg_transformer_kwargs_batch = {
                    "encoder_hidden_states": torch.cat([prompt_embeds[cfg_indices], neg_prompt_embeds.repeat_interleave(half_cfg_bsz, dim=0)], dim=0),
                    "pooled_projections": torch.cat([pooled_prompt_embeds[cfg_indices], neg_pooled_prompt_embeds.repeat_interleave(half_cfg_bsz, dim=0)], dim=0),
                    "txt_ids": torch.cat([text_ids[cfg_indices], neg_text_ids.repeat_interleave(half_cfg_bsz, dim=0)], dim=0),
                    "img_ids": torch.cat([latent_image_ids[cfg_indices], latent_image_ids[cfg_indices]], dim=0),
                    "guidance": torch.cat([cfg_scale[cfg_indices], cfg_scale[cfg_indices]], dim=0),
                }

                with torch.no_grad():
                    cfg_pack_kwargs = {
                        "batch_size": 2 * half_cfg_bsz,
                        "num_channels_latents": model_input.shape[1],
                        "height": model_input.shape[2],
                        "width": model_input.shape[3],
                    }
                    packed_cfg_z_t_batch = pack_latents(cfg_z_t_batch, **cfg_pack_kwargs)
                    packed_cfg_combined_u_at_t = modelpair.student(
                        hidden_states=packed_cfg_z_t_batch,                         # packed_noisy_model_input.shape = [bsz, 1024, 64]
                        timestep=cfg_t_batch.reshape([2 * half_cfg_bsz]),           # t.shape = [bsz, 1, 1, 1] --> [bsz]
                        timestep2=cfg_time_diff_batch.reshape([2 * half_cfg_bsz]),  # t.shape = [bsz, 1, 1, 1] --> [bsz]
                        **cfg_transformer_kwargs_batch,
                    )[0]  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
                    cfg_combined_u_at_t = unpack_latents(packed_cfg_combined_u_at_t, **unpack_kwargs)
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    # cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
                    #         self.cfg_kappa * cfg_u_cond_at_t + 
                    #         (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)
                    cfg_v_tilde = (self.cfg_omega * v_t_teacher[cfg_indices] + 
                            self.cfg_kappa * cfg_u_cond_at_t + 
                            (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)

                # Compute JVP with CFG velocity
                def fn_current_cfg(z, cur_r, cur_t):
                    half_cfg_pack_kwargs = {
                        "batch_size": half_cfg_bsz,
                        "num_channels_latents": model_input.shape[1],
                        "height": model_input.shape[2],
                        "width": model_input.shape[3],
                    }
                    packed_z = pack_latents(z, **half_cfg_pack_kwargs)
                    modelpair.student.gradient_checkpointing = False
                    packed_out = modelpair.student(
                        hidden_states=packed_z,              # packed_noisy_model_input.shape = [bsz, 1024, 64]
                        timestep=cur_t.flatten(),            # t.shape = [bsz, 1, 1, 1] --> [bsz]
                        timestep2=(cur_t - cur_r).flatten(), # t.shape = [bsz, 1, 1, 1] --> [bsz]
                        **cfg_transformer_kwargs,
                    )[0]  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
                    modelpair.student.gradient_checkpointing = True
                    out = unpack_latents(packed_out, **unpack_kwargs)
                    return out
                
                primals = (cfg_z_t, cfg_r, cfg_t)
                tangents = (cfg_v_tilde, torch.zeros_like(cfg_r), torch.ones_like(cfg_t))
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.weight_dtype):
                    _, cfg_dudt = torch.func.jvp(fn_current_cfg, primals, tangents)

                cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt
                u_target[cfg_indices] = cfg_u_target
            
            # Process non-CFG samples (including unconditional ones)
            if len(no_cfg_indices) > 0:
                no_cfg_z_t = z_t[no_cfg_indices]
                no_cfg_v_t = v_t[no_cfg_indices]
                no_cfg_r = r[no_cfg_indices]
                no_cfg_t = t[no_cfg_indices]
                no_cfg_time_diff = time_diff[no_cfg_indices]
                
                no_cfg_transformer_kwargs = {
                    "encoder_hidden_states": prompt_embeds[no_cfg_indices],
                    "pooled_projections": pooled_prompt_embeds[no_cfg_indices],
                    "txt_ids": text_ids[no_cfg_indices],
                    "img_ids": latent_image_ids[no_cfg_indices],
                    "guidance": cfg_scale[no_cfg_indices],
                }

                def fn_current_no_cfg(z, cur_r, cur_t):
                    no_cfg_pack_kwargs = {
                        "batch_size": no_cfg_z_t.shape[0],
                        "num_channels_latents": model_input.shape[1],
                        "height": model_input.shape[2],
                        "width": model_input.shape[3],
                    }
                    packed_z = pack_latents(z, **no_cfg_pack_kwargs)
                    modelpair.student.gradient_checkpointing = False
                    packed_out = modelpair.student(
                        hidden_states=packed_z,              # packed_noisy_model_input.shape = [bsz, 1024, 64]
                        timestep=cur_t.flatten(),            # t.shape = [bsz, 1, 1, 1] --> [bsz]
                        timestep2=(cur_t - cur_r).flatten(), # t.shape = [bsz, 1, 1, 1] --> [bsz]
                        **no_cfg_transformer_kwargs,
                    )[0]  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
                    modelpair.student.gradient_checkpointing = True
                    out = unpack_latents(packed_out, **unpack_kwargs)
                    return out
                
                primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
                tangents = (no_cfg_v_t, torch.zeros_like(no_cfg_r), torch.ones_like(no_cfg_t))
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.weight_dtype):
                    _, no_cfg_dudt = torch.func.jvp(fn_current_no_cfg, primals, tangents)
                
                # no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt
                no_cfg_u_target = v_t_teacher[no_cfg_indices] - no_cfg_time_diff * no_cfg_dudt
                u_target[no_cfg_indices] = no_cfg_u_target

        else:
            # No labels or no CFG applicable samples, use standard JVP
            primals = (z_t, r, t)
            tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
            
            def fn_current(z, cur_r, cur_t):
                packed_z = pack_latents(z, **pack_kwargs)
                modelpair.student.gradient_checkpointing = False
                packed_out = modelpair.student(
                    hidden_states=packed_z,      # packed_noisy_model_input.shape = [bsz, 1024, 64]
                    timestep=cur_t.reshape([batch_size]),                   # t.shape = [bsz, 1, 1, 1] --> [bsz]
                    timestep2=(cur_t - cur_r).reshape([batch_size]),                  # t.shape = [bsz, 1, 1, 1] --> [bsz]
                    **transformer_kwargs,
                )[0]  # cfg_pretrain_pred.shape=[bsz, 1024, 64], others TBD
                modelpair.student.gradient_checkpointing = True
                out = unpack_latents(packed_out, **unpack_kwargs)
                return out

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.weight_dtype):
                _, dudt = torch.func.jvp(fn_current, primals, tangents)

            # u_target = v_t - time_diff * dudt
            u_target = v_t_teacher - time_diff * dudt
                
        # Detach the target to prevent gradient flow        
        error = u - u_target.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid          
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error**2))
        loss = loss.mean()

        loss_logs = {
            "loss": loss.detach().item(),
            "loss_mean_ref": loss_mean_ref.detach().item(),
        }

        return loss, loss_logs


def train(
        accelerator, train_dataloader, modelpair, text_encoders, tokenizers, vae,
        optimizer, params_to_optimize, lr_scheduler, first_epoch, progress_bar, global_step, loss_func
    ):

    neg_prompt_embeds, neg_pooled_prompt_embeds, neg_text_ids = compute_text_embeddings(
        " ", text_encoders, tokenizers, args.max_sequence_length, accelerator.device
    )  # prompt_embeds.shape = [bsz, 256, 4096], pooled_prompt_embeds.shape = [bsz, 768], text_ids.shape = [bsz, 256, 3]
      

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
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=args.weight_dtype)

                latent_image_ids = prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    args.weight_dtype,
                )  # latent_image_ids.shape = [bsz, 4096, 3]

                total_loss, loss_logs = loss_func(
                    modelpair, model_input, latent_image_ids,
                    prompt_embeds, pooled_prompt_embeds, text_ids,
                    neg_prompt_embeds, neg_pooled_prompt_embeds, neg_text_ids,
                    accelerator.device, vae_scale_factor=2 ** (len(vae.config.block_out_channels))
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

    run_name = f"meanflow_optimizer{args.optimizer}_lr{args.learning_rate}"
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
    accelerator.init_trackers("flux_meanflow", config=cleaned_args)

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
    loss_func = MeanFlowLoss(
        path_type=args.path_type, 
        time_sampler=args.time_sampler,
        time_mu=args.time_mu,
        time_sigma=args.time_sigma,
        ratio_r_not_equal_t=args.ratio_r_not_equal_t,
        adaptive_p=args.adaptive_p,
        weighting=args.weighting,
        label_dropout_prob=args.cfg_prob,
        cfg_omega=args.cfg_omega,
        cfg_kappa=args.cfg_kappa,
        cfg_min_t=args.cfg_min_t,
        cfg_max_t=args.cfg_max_t
    )

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
    
    if args.resume_from_checkpoint:
        paras = torch.load(args.resume_from_checkpoint, map_location='cpu', weights_only=False)
        missing_keys, unexpected_keys = modelpair.student.load_state_dict(
            paras["module"], strict=False
        )
        logger.info(f"[INFO] successfully resume student model, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
        initial_global_step = int(args.resume_from_checkpoint.split(".")[-2].split("_")[-1])
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and False:  # need revise before release
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 'True', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'False', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a boolean value.')


if __name__ == "__main__":
    args = parse_args()
    main(args)
