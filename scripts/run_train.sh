wandb login

export WANDB_PROJECT="Lumina-Row-Draft"
export WANDB_WATCH="false"

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  \
    --deepspeed config/ds_config_zero3.json --batch_size 1 --epochs 10 --enable_wandb --run_name bm_kl_test