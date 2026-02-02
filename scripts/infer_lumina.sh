test_lora_name=$1
save_name=${2:-$1}
block_size=${3:-48}
infer_count=${4:-1}

conda activate gsd310

CUDA_VISIBLE_DEVICES=7 python inference/infer_lumina.py \
    --row_parallel \
    --lora_path training_outputs/lumina/$test_lora_name \
    --save_name $save_name \
    --block_size $block_size \
    --infer_count $infer_count # --draft_use_causal_mask