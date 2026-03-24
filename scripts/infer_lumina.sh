test_lora_name=$1
save_name=${2:-$1}
infer_count=${3:-1}
ar_rows=$4

# conda activate gsd310

CUDA_VISIBLE_DEVICES=7 python -m inference.infer_lumina \
    --row_parallel \
    --lora_path training_outputs/lumina/$test_lora_name \
    --save_name $save_name \
    --infer_count $infer_count \
    --ar_rows $ar_rows --return_anything_dict # --draft_use_causal_mask
