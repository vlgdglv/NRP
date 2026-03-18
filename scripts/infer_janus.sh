test_lora_name=$1
# save_name=${2:-$1}
# infer_count=${3:-1}
# ar_rows=$4

# conda activate gsd310

CUDA_VISIBLE_DEVICES=5 python inference/infer_janus.py \
    --row_parallel \
    --do_decode --ar_rows 4 \
    --save_name $2 \
    --lora_path training_outputs/janus/$test_lora_name \

    # --lora_path training_outputs/lumina/$test_lora_name \
    # --infer_count $infer_count \
    # --ar_rows $ar_rows # --draft_use_causal_mask

# CUDA_VISIBLE_DEVICES=0 python -m inference.infer_janus.py --do_decode --save_name baseline_fmt_prompt