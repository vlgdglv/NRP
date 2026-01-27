test_lora_name=$1
save_name=${2:-$1}

CUDA_VISIBLE_DEVICES=0  python inference/infer_lumina.py \
    --row_parallel \
    --lora_path training_outputs/lumina/$test_lora_name \
    --save_name $save_name