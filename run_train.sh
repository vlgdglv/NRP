wandb login

export WANDB_PROJECT="Lumina-Row-Draft"
export WANDB_WATCH="false"

deepspeed --num_gpus=4 train.py \
    --deepspeed config/ds_config_zero3.json \
    --run_name test_run