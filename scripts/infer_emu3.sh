save_name=$1

conda init
conda activate emu3

CUDA_VISIBLE_DEVICES=6 python inference/infer_emu3.py \
    --save_name $save_name 