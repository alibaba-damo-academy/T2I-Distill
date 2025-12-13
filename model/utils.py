import os
import sys
import json
from collections import OrderedDict

from PIL import Image
from transformers import PretrainedConfig
from diffusers.utils.torch_utils import is_compiled_module

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader


def ck(tensor):  # for debugging
    print(tensor.shape, tensor.dtype, tensor.device)


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if "batch_sampler" in kwargs:
        dataloader = DataLoader(
            dataset, batch_sampler=kwargs["batch_sampler"], num_workers=num_workers, pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, **kwargs
        )
    return dataloader


class Logger:
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name=None, file_mode="w", should_flush=True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, text):
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self):
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self):
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def adapt_lora_weights(merged_state_dict):
    new_state_dict = OrderedDict()
    for key, value in merged_state_dict.items():
        # 处理含base_layer的权重（LoRA合并后的核心参数）
        if '.base_layer.' in key:
            new_key = key.replace('.base_layer', '')  # 去除.base_layer后缀
            new_state_dict[new_key] = value
        # 过滤所有LoRA特定参数
        elif 'lora_A' in key or 'lora_B' in key:
            continue
        # 保留其他非LoRA参数
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def async_save(
    save_dir: str,
    model: nn.Module = None,
    global_step: int = None,
    device_mesh=None,
    is_latest=False
):
    if not is_latest:
        save_dir = os.path.join(save_dir, f"global_step{global_step}")
    else:
        save_dir = os.path.join(save_dir, f"latest")
    if dist.get_rank() == 0:
        os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
    dist.barrier()
    # this line automatically manages DDP/FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
    # model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    state_dict = {
        "module": model.state_dict(),
        # "optimizer": optimizer_state_dict,
    }
    # if ema is not None:
    #     ema_state_dict = get_model_state_dict(ema)
    #     state_dict['ema'] = ema_state_dict
    
    if device_mesh['rep'].get_group().rank() == 0:
        dcp_handle = dcp.async_save(state_dict, checkpoint_id=os.path.join(save_dir, "model"), process_group=device_mesh['shard'].get_group())
    else:
        dcp_handle = None
    
    if dist.get_rank() == 0:
        running_states = {
            # "epoch": epoch,
            # "step": step,
            "global_step": global_step,
            # "batch_size": batch_size,
        }
        save_json(running_states, os.path.join(save_dir, "running_states.json"))

        # if sampler is not None:
        #     # only for VariableVideoBatchSampler
        #     torch.save(sampler.state_dict(step), os.path.join(save_dir, "sampler"))
            
        # if lr_scheduler is not None:
        #     torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "lr_scheduler"))
    dist.barrier()
    return dcp_handle


def combine_images2x2(image_paths):
    """将4张图片组合成2x2排列的大图"""
    images = [Image.open(path) for path in image_paths]
    
    # 验证所有图片尺寸一致
    widths, heights = zip(*(img.size for img in images))
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise ValueError("图片尺寸不一致")
    
    # 计算新图片尺寸
    w, h = images[0].size
    new_img = Image.new('RGB', (2*w, 2*h))
    
    # 排列顺序：左上、右上、左下、右下
    positions = [
        (0, 0),    # 0000.png
        (w, 0),    # 0001.png
        (0, h),    # 0002.png
        (w, h)     # 0003.png
    ]
    
    for img, pos in zip(images, positions):
        new_img.paste(img, pos)
    
    return new_img


def resize_image(image_path, output_size=(256, 256)):
    with Image.open(image_path) as img:
        resized_img = img.resize(output_size, Image.LANCZOS)
        return resized_img


def save_downsample_images(image_paths, rows=8, output_file='output.png'):
    images = [resize_image(img_path) for img_path in image_paths]
    
    # Calculate the dimensions of the final image
    max_height = max(img.height for img in images)
    max_width = max(img.width for img in images)
    
    # rows = 8
    cols = (len(images) + rows - 1) // rows  # Calculate the number of columns needed
    
    large_image = Image.new('RGB', (cols * max_width, rows * max_height))
    
    for index, img in enumerate(images):
        x = (index // rows) * max_width
        y = (index % rows) * max_height
        large_image.paste(img, (x, y))
    
    large_image.save(output_file)


def combine_images_horizontally(image_paths):
    """将给定路径中的图片横向拼接"""
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    
    # 计算新图像的大小
    total_width = sum(widths)
    max_height = max(heights)
    
    # 创建空白图像
    combined_image = Image.new('RGB', (total_width, max_height))
    
    # 拼接图片
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined_image


def combine_images_grid(image_paths, images_per_row=4):
    """将给定路径中的图片每n个一排垂直拼接"""
    # 打开所有图像
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    
    # 获取单个图像的宽度和高度，并用于单行的尺寸
    max_width = max(widths)
    max_height = max(heights)
    
    # 计算整合图像的尺寸
    total_width = max_width * min(len(images), images_per_row)
    total_height = max_height * ((len(images) + images_per_row - 1) // images_per_row)
    
    # 创建空白组合图像
    combined_image = Image.new('RGB', (total_width, total_height))
    
    # 逐个拼接图像
    x_offset = 0
    y_offset = 0
    for idx, img in enumerate(images):
        combined_image.paste(img, (x_offset, y_offset))
        x_offset += max_width
        
        # 每images_per_row个图片换行
        if (idx + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += max_height
    
    return combined_image


def combine_images_vertically(image_paths, images_per_column=4):
    # Load all images
    images = [Image.open(path) for path in image_paths]

    # Determine the dimensions for each column    
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    
    # Calculate the number of rows and columns
    num_columns = (len(images) + images_per_column - 1) // images_per_column
    num_rows = images_per_column
    
    # Calculate the final image dimensions
    large_image_width = num_columns * max_width
    large_image_height = num_rows * max_height

    # Create a new blank image with calculated size
    large_image = Image.new("RGB", (large_image_width, large_image_height), (255, 255, 255))

    # Paste images into large image
    for index, image in enumerate(images):
        column = index // images_per_column
        row = index % images_per_column
        x_offset = column * max_width
        y_offset = row * max_height
        large_image.paste(image, (x_offset, y_offset))

    return large_image


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = int(height // vae_scale_factor)
    width = int(width // vae_scale_factor)
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    # print(model_class)
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def read_jsonl_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析jsonl文件中的这一行文本
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError as error:
                print(f"Error decoding JSON: {error}")
    return data


def get_image_paths(root_folder):
    all_image_paths = []
    
    # 遍历根目录下的每一个子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # 检查是否是目录
        if os.path.isdir(folder_path):
            image_paths = []
            
            # 遍历子文件夹中的每一个文件
            for file_name in os.listdir(folder_path):
                # 检查是否是四位数格式命名的图片
                if file_name.endswith('.png') and file_name[:4].isdigit() and len(file_name.split('.')[0]) == 4:
                    image_path = os.path.join(folder_path, file_name)
                    image_paths.append(image_path)
                    
            all_image_paths.extend(image_paths)
    
    return all_image_paths


def merge_json_files(directory, output_file):
    merged_data = []

    # 遍历目录下的所有json文件
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # 读取每个json文件并加载它们的内容
            print(f"processing {file_path}")
            data = read_jsonl_file(f"{file_path}")
            if len(merged_data) == 0:
                merged_data = data
            else:
                merged_data.extend(data)
    # 将合并的数据写入到一个新的json文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


