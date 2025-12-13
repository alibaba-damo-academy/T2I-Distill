import os
import json
import logging
import argparse
from tqdm.auto import tqdm
from datetime import timedelta
from contextlib import nullcontext

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration, set_seed, InitProcessGroupKwargs
import transformers
from transformers import T5TokenizerFast
from transformers import CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils import check_min_version

from model.mmdit_flux import FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1
from model.pipeline import FluxPipeline_MeanFlow
from model.utils import load_text_encoders, import_model_class_from_model_name_or_path, read_jsonl_file


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.1")

logger = get_logger(__name__)


def parse_args(input_args=None):
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
        "--resume_path",
        type=str,
        default=None,
        help="The model checkpoint path.",
    )

    # ----- text encoder  -----
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
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

    # ----- directory  -----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dyn",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # distributed training / deepspeed config / fsdp config
    parser.add_argument(
        "--ds_file",
        type=str,
        default='deepspeed_config/deepspeedconfig_bf16_batch8.json',
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # ----- inference hyper-paramter  -----
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def initialize_all_models(args, accelerator):

    logger.info("[INFO] building transformer model")
    transformer = FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=None,
            variant=None,
            low_cpu_mem_usage=False,
    )
    logger.info("[INFO] building transformer model done")

    # load ckpt
    logger.info(f"[INFO] loading pretrained model from {args.resume_path}")
    paras = torch.load(args.resume_path, map_location='cpu', weights_only=False)
    missing_keys, unexpected_keys = transformer.load_state_dict(
        paras["module"], strict=False
    )
    logger.info(f"[INFO] successfully resumed, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
  
    transformer.requires_grad_(False)
    transformer.eval()

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
    
    logger.info(f"[INFO] loading text encoders")
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )   
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    
    logger.info(f"[INFO] loading vae")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)


    logger.info(f"[INFO] moving models to cuda")
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    args.weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        args.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        args.weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=args.weight_dtype)
    transformer.to(accelerator.device, dtype=args.weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=args.weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=args.weight_dtype)
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    return vae, transformer, tokenizers, text_encoders


def generate_images(
    pipeline,
    args,
    accelerator,
    global_step=0,
):
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    autocast_ctx = nullcontext()
    with open('evaluation/generation_prompts.txt', 'r', encoding='utf-8') as file:
        prompts = file.readlines()
    prompts = [p.strip() for p in prompts]
    meta_data = read_jsonl_file('evaluation/evaluation_metadata.jsonl')
    
    logger.info(f"Running validation... \n Generating {len(prompts)} images with prompt: {prompts}")
    
    with autocast_ctx:
        images =[]
        for i in tqdm(range(len(prompts))):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                res = pipeline(
                    prompt=prompts[i],
                    height=args.resolution,
                    width=args.resolution,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=4,
                    max_sequence_length=256,
                    generator=generator,
                )
            images.append(res[0])
            
            save_path = f"{args.save_dir}/global_steps{global_step}"
            save_path = f"{save_path}/{i:05d}"
            os.makedirs(save_path, exist_ok=True)
            
            meta_path = f"{save_path}/metadata.jsonl"
            with open(meta_path, 'w', encoding='utf-8') as out_f:
                json_line = json.dumps(meta_data[i])
                out_f.write(json_line + '\n')
            
            img_path = f"{save_path}/samples"
            os.makedirs(img_path, exist_ok=True)
            
            for j in range(len(res[0])):
                img_path_j = f"{img_path}/{j:04d}.png"
                res[0][j].save(img_path_j)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def main(args):

    # =================================================================================================
    # 1. PATHS & DIRECTORY MANAGEMENT & Accelerator
    # =================================================================================================

    logging_dir_for_run = os.path.join(args.output_dir, "tensorboard_logs")
    kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(hours=24))
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir_for_run)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.ds_file, zero3_init_flag=True)
    dl_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed_plugin,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        dataloader_config=dl_config
    )

    # Handle the repository creation
    if accelerator.is_main_process:

        args.output_dir = f"{args.output_dir}/GenEval_RESOLUTION{args.resolution}_NFE{args.num_inference_steps}_cfg{args.guidance_scale}"
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        args.save_dir = f"{args.output_dir}/images/"
        os.makedirs(args.save_dir, exist_ok=True)  

    # =================================================================================================
    # 2. SEED & LOGGING INITIALIZATION
    # =================================================================================================

    if args.seed is not None:
        set_seed(args.seed)

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

    # =================================================================================================
    # 3. MODEL & SCHEDULER & PIPELINE INITIALIZATION
    # =================================================================================================

    # Load scheduler
    logger.info("[INFO] Loading scheduler...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    logger.info("[INFO] Loaded scheduler!!!")

    # Initialize all models
    vae, transformer, tokenizers, text_encoders  = initialize_all_models(args, accelerator)
    
    text_encoder_one, text_encoder_two = text_encoders
        
    pipeline = FluxPipeline_MeanFlow.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizers[0],
        tokenizer_2=tokenizers[1],
        revision=args.revision,
        variant=args.variant,
        torch_dtype=args.weight_dtype,
    )

    # =================================================================================================
    # 4. GENERATE!
    # =================================================================================================

    ckpt_global_step = args.resume_path.split("_")[-1].split(".")[0]

    generate_images(
        pipeline=pipeline,
        args=args,
        accelerator=accelerator,
        global_step=ckpt_global_step
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
