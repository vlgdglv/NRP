wandb login

export WANDB_PROJECT="Emu3-Row-Draft"
export WANDB_WATCH="false"




CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  train.py  --deepspeed config/ds_config_zeros.json --model_name emu3 --model_path /jizhicfs/pkuhetu/bht/model_home/Emu3-Gen/ --data_path /jizhicfs/pkuhetu/bht/NRP/datasets/emu3-mscoco-train2017-tokens/COCO_Emu3_tokens_for_train --output_dir training_outputs/emu3 --batch_size 1 --gradient_accumulation_steps 8 --epochs 3 --losses ce --image_width 91 --image_height 90 --block_size 90 --use_standard_causal --run_name emu3_test --lora_rank 64 --lora_rank 128
