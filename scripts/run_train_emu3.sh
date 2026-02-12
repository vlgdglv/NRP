wandb login

export WANDB_PROJECT="Emu3-Row-Draft"
export WANDB_WATCH="false"




CUDA_VISIBLE_DEVICES=4,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --model_name emu3 --model_path /home/ffc3/bht/model_home/Emu3-Gen/ --data_path /home/ffc3/bht/NRP/datasets/COCO_Emu3_tokens_for_train --output_dir training_outputs/emu3 --batch_size 1 --epochs 5 --losses ce --block_size 48 --run_name emu3_test --lora_rank 64
