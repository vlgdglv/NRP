test_lora_name=$1
save_name=${2:-$1}
infer_count=${3:-1}
ar_rows=$4

# conda activate gsd310

CUDA_VISIBLE_DEVICES=7 python inference/infer_lumina.py \
    --row_parallel \
    --lora_path training_outputs/lumina/$test_lora_name \
    --save_name $save_name \
    --infer_count $infer_count \
    --ar_rows $ar_rows # --draft_use_causal_mask