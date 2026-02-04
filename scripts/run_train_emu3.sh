wandb login

export WANDB_PROJECT="Emu3-Row-Draft"
export WANDB_WATCH="false"

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  \
    --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 10 --enable_wandb --run_name bm_kl_test

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 12345  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 32 --run_name rk32_lm_ce_b48_e5_bs1_causal --use_standard_causal --enable_wandb; 
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 12345  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 32 --run_name rk32_lm_ce_b48_e5_bs1  --enable_wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --model_name emu3 --batch_size 1 --epochs 5 --losses ce kd --kd_temp 2.0 --use_teacher --block_size 48 --run_name rk32_lm_cekd_b48_e5_bs1_causal --use_standard_causal --lora_rank 32  --enable_wandb
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --model_name emu3 --batch_size 1 --epochs 5 --losses ce kd --kd_weight 4.0 --kd_temp 2.0 --use_teacher --block_size 48 --run_name rk32_lm_ce1kd4_b48_e5_bs1 --lora_rank 32  --enable_wandb



CUDA_VISIBLE_DEVICES=4,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --model_name emu3 --model_path /home/ffc3/bht/model_home/Emu3-Gen/ --data_path /home/ffc3/bht/NRP/datasets/COCO_Emu3_tokens_for_train --output_dir training_outputs/emu3 --batch_size 1 --epochs 5 --losses ce --block_size 48 --run_name emu3_test --lora_rank 64
