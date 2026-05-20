# cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="qwen25vl-videorefer-refined-detailed-125k-detail-125k-insit-21k-miou-multilayer-test-single-turn-prod"
export LOG_PATH="./new_log/debug_log_$RUN_NAME.txt"



export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x32"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=448
export VIDEO_MINRES=288
export PAD2STRIDE=1
export FORCE_NO_DOWNSAMPLE=1
export LOAD_VISION_EARLY=1

export HF_HOME=./cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=./Q-R1/src/open-r1-multimodal/src/


WANDB_MODE=offline torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="23555" \
    src/open_r1/sft_videorefer_qwen25vl.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path <MODEL_PATH> \
    --dataset_name data_config/videorefer.yaml \
    --image_root ./data/coco/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4  \
    --logging_steps 1 \
    --bf16 true\
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 300 \
    --save_only_model false
