# T2I-Distill

**A Systematic Study of Diffusion Distillation for Text-to-Image Synthesis**

[![arXiv](https://img.shields.io/badge/arXiv-2512.13006-b31b1b.svg)](https://arxiv.org/abs/2512.13006v1)

## Highlights

- 🔬 **Unified Framework**: Casting existing distillation methods (sCM, MeanFlow) into a unified framework for fair comparison
- 🚀 **Fast Generation**: 1-4 step image generation with high fidelity
- 🛠️ **Production Ready**: Practical guidelines on input scaling, network architecture, and hyperparameters
- 📦 **Open Source**: Fully reproducible codebase with pretrained student models


## 💻 Released Checkpoints

You can download the models directly from huggingface:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="Alibaba-DAMO-Academy/T2I-Distill")
print(f"Model downloaded to: {model_path}")
```

## Project Structure

```
T2I-Distill/
├── data/
│   ├── __init__.py
│   └── custom_dataset.py       # Text2Image dataset for HuggingFace datasets
├── model/
│   ├── __init__.py
│   ├── attn_processor.py       # Custom attention processors
│   ├── mmdit_flux.py           # Modified FLUX transformer models
│   ├── pipeline.py             # Inference pipelines for MeanFlow and sCM
│   └── utils.py                # Utility functions
├── parallel_config/
│   └── deepspeed_bf16_*.json   # DeepSpeed configurations
├── evaluation/
│   ├── evaluation_metadata.jsonl
│   └── generation_prompts.txt  # GenEval prompts
├── script/
│   ├── train_meanflow.sh       # MeanFlow training script
│   ├── train_sCM.sh            # sCM training script
│   └── geneval_*.sh            # Evaluation scripts
├── train_meanflow.py           # MeanFlow training entry
├── train_sCM.py                # sCM training entry
├── test_meanflow.py            # MeanFlow inference
├── test_sCM.py                 # sCM inference
└── test_flux8b.py              # Base FLUX inference (teacher)
```

## Installation

### Dependencies

```bash
pip install torch torchvision
pip install accelerate>=0.25.0
pip install transformers>=4.36.0
pip install diffusers>=0.33.1
pip install datasets
pip install deepspeed
pip install wandb  # optional, for logging
```
## Quick Start

### 1. Teacher Model

We use [Freepik/flux.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B) as the teacher model. Ensure you have access to this model on HuggingFace.

### 2. Prepare Rescaled Checkpoint

Before training, prepare a rescaled checkpoint of the base model (`.pt` file with state dict under `"module"` key).

### 3. Training

#### MeanFlow Distillation

```bash
bash script/train_meanflow.sh \
    1e-6 \                    # Learning rate
    512 \                     # Resolution
    8 \                       # Batch size per GPU
    True \                    # Use CFG MeanFlow
    1.0 \                     # CFG omega
    0.5 \                     # CFG kappa
    0.0 \                     # CFG min time
    1.0 \                     # CFG max time
    /path/to/output \         # Output directory
    /path/to/rescaled.pt      # Rescaled checkpoint path
```

#### sCM Distillation

```bash
bash script/train_sCM.sh \
    1e-6 \                    # Learning rate
    512 \                     # Resolution
    8 \                       # Batch size per GPU
    /path/to/output \         # Output directory
    /path/to/rescaled.pt      # Rescaled checkpoint path
```

### 4. Inference

#### MeanFlow (2-4 steps)

```bash
python test_meanflow.py \
    --pretrained_model_name_or_path Freepik/flux.1-lite-8B \
    --resume_path /path/to/checkpoint.pt \
    --resolution 512 \
    --num_inference_steps 4 \
    --guidance_scale 3.5 \
    --output_dir ./outputs
```

#### sCM (1-2 steps)

```bash
python test_sCM.py \
    --pretrained_model_name_or_path Freepik/flux.1-lite-8B \
    --resume_path /path/to/checkpoint.pt \
    --resolution 512 \
    --num_inference_steps 2 \
    --guidance_scale 4.5 \
    --output_dir ./outputs
```
<!-- 
## Citation

```bibtex
@article{t2i-distill,
  title={T2I-Distill: A Systematic Study of Diffusion Distillation for Text-to-Image Synthesis},
  author={...},
  year={2025}
}
``` -->
