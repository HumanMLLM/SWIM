export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="./data/huggingface_hub"
huggingface-cli login --token 

# MODEL_NAME="SWIM"
# MODEL_PATH='<MODEL_PATH>/'

MODEL_NAME="SWIM"
MODEL_PATH='<MODEL_PATH>'

accelerate launch --num_processes 8 --main_process_port 23553 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=$MODEL_PATH,use_flash_attention_2=true \
    --tasks mvbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix eval \
    --output_path ./logs/$MODEL_NAME

echo $MODEL_NAME 
echo $MODEL_PATH
