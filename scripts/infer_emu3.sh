save_name=$1

conda init
conda activate emu3

CUDA_VISIBLE_DEVICES=7 python inference/infer_emu3.py --save_name $save_name  --batched_cfg