export MODEL_NAME="Freepik/flux.1-lite-8B"
export NCCL_DEBUG=INFO

LR=${1:-1e-5}
RESOLUTION=${2:-512}
BATCH_SIZE=${3:-8}
CFG_MEANFLOW=${4:-True}
CFG_OMEGA=${5:-1}
CFG_KAPPA=${6:-0.5}
CFG_MIN_T=${7:-0.0}
CFG_MAX_T=${8:-1.0}
OUT_DIR=${9:-1}
RESCALED_PATH=${10:-1}

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv-conf=timeout=36000 \
  train_meanflow.py \
  --pretrained_model_name_or_path $MODEL_NAME  \
  --rescaled_checkpoint_path=$RESCALED_PATH \
  --resolution=$RESOLUTION --dataloader_num_workers=4 \
  --output_dir=$OUT_DIR \
  --train_batch_size=$BATCH_SIZE --gradient_accumulation_steps=1 \
  --ds_file=parallel_config/deepspeed_bf16_batch$BATCH_SIZE.json \
  --max_train_steps=10_000_000 --max_grad_norm=0.1 \
  --gradient_checkpointing \
  --learning_rate=$LR \
  --checkpointing_steps=1000 \
  --allow_tf32 \
  --report_to=tensorboard \
  --use_cfg_meanflow=$CFG_MEANFLOW \
  --cfg_omega=$CFG_OMEGA \
  --cfg_kappa=$CFG_KAPPA \
  --cfg_min_t=$CFG_MIN_T \
  --cfg_max_t=$CFG_MAX_T


# clean-code-meanflow-4m32g_0924_cfg_omg1x0_kpa0x5_mint0x3_max0x8
# bash script/train_meanflow.sh 1e-6 512 8 /mnt/eff_nas/puyifan/work_dirs/clean_code/MeanFlow_0924_cfg_omg1x0_kpa0x5_mint0x3_max0x8/ RESCALED_PATH
