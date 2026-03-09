wandb login

export WANDB_PROJECT="Lumina-Row-Draft"
export WANDB_WATCH="false"

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  \
    --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 10 --enable_wandb --run_name bm_kl_test

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 12345  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 32 --run_name rk32_lm_ce_b48_e5_bs1_causal --use_standard_causal --enable_wandb; 
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 12345  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 32 --run_name rk32_lm_ce_b48_e5_bs1  --enable_wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce kd --kd_temp 2.0 --use_teacher --block_size 48 --run_name rk32_lm_cekd_b48_e5_bs1_causal --use_standard_causal --lora_rank 32  --enable_wandb
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce kd --kd_weight 4.0 --kd_temp 2.0 --use_teacher --block_size 48 --run_name rk32_lm_ce1kd4_b48_e5_bs1 --lora_rank 32  --enable_wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce  --block_size 48 --lora_rank 8 --lora_alpha 16 --run_name rk8_lm_ce_b48_e5_bs1

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce --block_size 48 --lora_rank 8 --lora_alpha 16 --run_name rk8_a16_lm_ce_b48_e5_bs1 --enable_wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --losses ce --block_size 48 --lora_rank 8 --lora_alpha 16 --run_name rk8_a16_lm_ce_b48_e5_bs1 --enable_wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --gradient_accumulation_steps 8 --losses ce --block_size 48 --lora_rank 64 --lora_alpha 128 --run_name rk64_lm_ce_e5_3data_40k --dataset_name COCO laion midjourney --data_path /home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train /home/ffc3/bht/NRP/datasets/laion_Lumina7B_tokens_for_train /home/ffc3/bht/NRP/datasets/midjourney_Lumina7B_tokens_for_train

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --batch_size 1 --epochs 5 --gradient_accumulation_steps 8 --losses ce kd --kd_temp 2.0 --use_teacher --block_size 48 --lora_rank 64 --lora_alpha 128 --run_name rk64_lm_cekd_e5_3data_40k --dataset_name COCO laion midjourney --data_path /home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train /home/ffc3/bht/NRP/datasets/laion_Lumina7B_tokens_for_train /home/ffc3/bht/NRP/datasets/midjourney_Lumina7B_tokens_for_train --teacher_data_dir datasets/COCO_Lumina7B_training_teacher datasets/laion_Lumina7B_tokens_for_train_teacher datasets/midjourney_Lumina7B_tokens_for_train_teacher
 