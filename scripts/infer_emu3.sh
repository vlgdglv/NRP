#!/usr/bin/env bash
set -e

lora_name=$1
save_name=$2

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py \
    --save_name rk64_lm_ce_e5_ar12 --batched_cfg  --ar_rows 12 \
    --row_parallel --lora_path training_outputs/emu3/rk64_lm_ce_e5

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py --save_name baseline --batched_cfg

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py --save_name baseline_360 --batched_cfg  --image_area 129600

CUDA_VISIBLE_DEVICES=0 python data_gen/generate_token_for_emu3.py  --begin 10000 --end 11000
CUDA_VISIBLE_DEVICES=1 python data_gen/generate_token_for_emu3.py  --begin 11000 --end 12000
CUDA_VISIBLE_DEVICES=2 python data_gen/generate_token_for_emu3.py  --begin 12000 --end 13000
CUDA_VISIBLE_DEVICES=3 python data_gen/generate_token_for_emu3.py  --begin 13000 --end 14000

CUDA_VISIBLE_DEVICES=0 python data_gen/generate_token_for_emu3.py  --begin 14000 --end 15000
CUDA_VISIBLE_DEVICES=1 python data_gen/generate_token_for_emu3.py  --begin 15000 --end 16000
CUDA_VISIBLE_DEVICES=2 python data_gen/generate_token_for_emu3.py  --begin 16000 --end 17000
CUDA_VISIBLE_DEVICES=3 python data_gen/generate_token_for_emu3.py  --begin 17000 --end 18000

CUDA_VISIBLE_DEVICES=0 python data_gen/generate_token_for_emu3.py  --begin 18000 --end 19000
CUDA_VISIBLE_DEVICES=1 python data_gen/generate_token_for_emu3.py  --begin 19000 --end 20000
CUDA_VISIBLE_DEVICES=2 python data_gen/generate_token_for_emu3.py  --begin 20000 --end 21000
CUDA_VISIBLE_DEVICES=3 python data_gen/generate_token_for_emu3.py  --begin 21000 --end 22000