wandb login

export WANDB_PROJECT="Janus-Row-Draft"
export WANDB_WATCH="false"




CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros_janus.json --model_name janus --model_path /home/ffc3/bht/model_home/Janus-Pro-7B/ --data_path /home/ffc3/bht/NRP/datasets/COCO_Janus_tokens_for_train --output_dir training_outputs/janus --image_width 24 --image_height 24 --batch_size 8 --epochs 20 --losses ce --run_name rk64_lm_ce_e20 --lora_rank 64 --enable_wandb
