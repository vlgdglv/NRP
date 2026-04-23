# GenEval Evaluation

Light wrapper around the [official GenEval](https://github.com/djghosh13/geneval)
evaluator. We only own image generation + result aggregation; scoring is done by
the upstream evaluator using a Mask2Former object detector.

## What GenEval scores

553 prompts across 6 compositional tasks: `single_object`, `two_object`,
`counting`, `colors`, `position`, `color_attr`. 4 images per prompt.
Each generated image is scored 0/1 by Mask2Former + rule checks.

## Files here

- `generate_images.py` — generates images in GenEval's expected layout for
  Janus or Lumina. Resumable (skips existing samples).
- `summarize.py` — reads GenEval's `results.jsonl` and prints per-tag accuracy.

## One-time setup on the remote box

```bash
# 1. Clone GenEval
cd ~ && git clone https://github.com/djghosh13/geneval.git
cd geneval && pip install -r requirements.txt

# 2. Install mmdetection 2.x + mmcv-full (matching your torch version)
pip install openmim
mim install mmcv-full==1.7.0
git clone -b 2.x https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -v -e .

# 3. Download the Mask2Former checkpoint GenEval uses
cd ~/geneval && bash evaluation/download_models.sh ./model_weights

# 4. Verify
ls prompts/evaluation_metadata.jsonl   # 553 prompt records
```

## Per-run workflow on remote

### Step 1 — generate images (uses our infra)

```bash
cd /path/to/NRP

# Janus row-parallel example
python -m evaluation.geneval.generate_images \
    --metadata_path ~/geneval/prompts/evaluation_metadata.jsonl \
    --save_dir inference_outputs/janus/geneval_rk64_ar4 \
    --model_name janus \
    --model_path /path/to/Janus-Pro-7B \
    --lora_path /path/to/lora_ckpt \
    --row_parallel --ar_rows 4 \
    --cfg_guidance_scale 5.0 \
    --n_samples 4

# Lumina row-parallel example
python -m evaluation.geneval.generate_images \
    --metadata_path ~/geneval/prompts/evaluation_metadata.jsonl \
    --save_dir inference_outputs/lumina/geneval_rk64_ar4 \
    --model_name lumina \
    --model_path /path/to/Lumina-mGPT-7B-768 \
    --vae_tokenizer_path /path/to/chameleon_tokenizer/tokenizer \
    --lora_path /path/to/lora_ckpt \
    --row_parallel --ar_rows 4 \
    --target_size 768 \
    --cfg_guidance_scale 3.0 \
    --n_samples 4
```

The output layout will be:
```
inference_outputs/.../geneval_.../00000/samples/0000.png
inference_outputs/.../geneval_.../00000/samples/0001.png
inference_outputs/.../geneval_.../00000/samples/0002.png
inference_outputs/.../geneval_.../00000/samples/0003.png
inference_outputs/.../geneval_.../00000/metadata.jsonl
inference_outputs/.../geneval_.../00001/...
```

553 prompts × 4 samples = 2212 images per eval. Re-running the script skips
any sample whose PNG already exists (use `--overwrite` to redo).

### Step 2 — score with GenEval's evaluator

```bash
cd ~/geneval
python evaluation/evaluate_images.py \
    /path/to/NRP/inference_outputs/janus/geneval_rk64_ar4 \
    --outfile results_janus_rk64_ar4.jsonl \
    --model-path ./model_weights
```

This writes `results_<...>.jsonl` with per-image `correct` + `tag` fields.

### Step 3 — aggregate into the paper-ready table

```bash
cd /path/to/NRP
python -m evaluation.geneval.summarize ~/geneval/results_janus_rk64_ar4.jsonl
```

Produces:
```
Tag                     N    Acc (%)
--------------------------------------
single_object         320      ...
two_object            396      ...
counting              320      ...
colors                256      ...
position              400      ...
color_attr            320      ...
--------------------------------------
OVERALL              2012      ...
```

## Notes

- Sampling is per-seed: sample `k` uses `seed_base + k`, so two runs with the
  same `seed_base` produce identical images (good for reproducibility, bad if
  you want sample diversity for the same eval — bump `seed_base` to re-roll).
- Resumable: if generation gets interrupted, just re-run the same command;
  finished images are skipped.
- For ablations (multiple `ar_rows`, multiple LoRA checkpoints), pick a
  distinct `--save_dir` per config. Scoring is separate per dir.
