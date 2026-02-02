wandb login

export WANDB_PROJECT="Lumina-Row-Draft"
export WANDB_WATCH="false"

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  \
    --deepspeed config/ds_config_zero3.json --batch_size 1 --epochs 10 --enable_wandb --run_name bm_kl_test

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 12345  train.py  --deepspeed config/ds_config_zero3.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 32 --run_name rk32_lm_ce_b48_e5_bs1_causal --use_standard_causal --enable_wandb; 
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 12345  train.py  --deepspeed config/ds_config_zero3.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 32 --run_name rk32_lm_ce_b48_e5_bs1  --enable_wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zero3.json --batch_size 1 --epochs 5 --losses ce kd --kd_temp 2.0 --use_teacher --block_size 48 --run_name rk32_lm_cekd_b48_e5_bs1_causal --use_standard_causal --lora_rank 32  --enable_wandb
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zero3.json --batch_size 1 --epochs 5 --losses ce kd --kd_weight 4.0 --kd_temp 2.0 --use_teacher --block_size 48 --run_name rk32_lm_ce1kd4_b48_e5_bs1 --lora_rank 32  --enable_wandb