# !

set -euo

python evaluation/geneval/generate_images.py \
    --metadata_path /jizhicfs/pkuhetu/bht/geneval/prompts/evaluation_metadata.jsonl \
    --model_name janus --model_path /jizhicfs/pkuhetu/bht/model_home/Janus-Pro-7B/ \
    --save_dir inference_outputs/geneval/janus/smoke \
    --lora_path training_outputs/janus/rk64_lm_ce_e3 \
    --row_parallel --ar_rows 4 --max_per_tag 5 --n_samples 1


cd ~/geneval && python -m evaluation.evaluate_images \
    /jizhicfs/pkuhetu/bht/NRP/inference_outputs/geneval/janus/smoke \
    --outfile results_smoke.jsonl --model-path /jizhicfs/pkuhetu/bht/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth

    
 cd /path/to/NRP && python -m evaluation.geneval.summarize ~/geneval/results_smoke.jsonl