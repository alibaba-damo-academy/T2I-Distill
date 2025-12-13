MODEL_NAME="Freepik/flux.1-lite-8B"
DEVICE=0
RESOLUTION=512
NFE=4
CFG=3.5
CKPT_PATH=YOUR/CKPT/PATH
CKPT_ITER=YOUR_CKPT_ITER



CUDA_VISIBLE_DEVICES=$DEVICE torchrun --nnodes=1 --nproc_per_node=1 --master_port="3${DEVICE}331" test_sCM.py \
--pretrained_model_name_or_path $MODEL_NAME \
--max_sequence_length 256 --seed 42 --ds_file parallel_config/deepspeed_bf16_batch1.json \
--resolution $RESOLUTION --num_inference_steps $NFE --guidance_scale $CFG \
--output_dir ${CKPT_PATH}/GenEval/ \
--resume_path ${CKPT_PATH}/ckpts/transformer_${CKPT_ITER}.pt
