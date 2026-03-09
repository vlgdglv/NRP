#!/usr/bin/env bash
set -e

lora_name=$1
save_name=$2

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py \
    --save_name rk64_lm_ce_e5_ar1 --batched_cfg  --ar_rows 12 \
    --row_parallel --lora_path training_outputs/emu3/rk64_lm_ce_e5

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py --save_name baseline --batched_cfg

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py     --save_name baseline --batched_cfg 

CUDA_VISIBLE_DEVICES=0 python data_gen/generate_token_for_emu3.py  --begin 5000 --end 5625
CUDA_VISIBLE_DEVICES=1 python data_gen/generate_token_for_emu3.py  --begin 5625 --end 6250
CUDA_VISIBLE_DEVICES=2 python data_gen/generate_token_for_emu3.py  --begin 6250 --end 6875
CUDA_VISIBLE_DEVICES=3 python data_gen/generate_token_for_emu3.py  --begin 6875 --end 7500
CUDA_VISIBLE_DEVICES=4 python data_gen/generate_token_for_emu3.py  --begin 7500 --end 8125
CUDA_VISIBLE_DEVICES=5 python data_gen/generate_token_for_emu3.py  --begin 8125 --end 8750
CUDA_VISIBLE_DEVICES=6 python data_gen/generate_token_for_emu3.py  --begin 8750 --end 9375
CUDA_VISIBLE_DEVICES=7 python data_gen/generate_token_for_emu3.py  --begin 9375 --end 10000