"""Generate images for GenEval scoring.

GenEval expects this folder layout per prompt:
    <out>/00000/samples/0000.png   (sample 0)
    <out>/00000/samples/0001.png   (sample 1)
    ...
    <out>/00000/metadata.jsonl     (one-line copy of the prompt's metadata record)
    <out>/00001/...

Then run GenEval's official evaluator on remote:
    python <geneval_repo>/evaluation/evaluate_images.py <out> \
        --outfile results.jsonl --model-path <mask2former_ckpt>

Then summarize with `summarize.py`.

This script supports both `--model_name janus` and `--model_name lumina` so we don't
duplicate model-loading code in two places.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def iter_geneval_prompts(metadata_path: str):
    with open(metadata_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield idx, rec


def write_metadata(prompt_dir: Path, record: dict):
    """Write the per-prompt metadata.jsonl that the GenEval evaluator reads."""
    with open(prompt_dir / "metadata.jsonl", "w") as f:
        f.write(json.dumps(record) + "\n")


def sample_path(out_dir: Path, idx: int, sample_idx: int) -> Path:
    return out_dir / f"{idx:05d}" / "samples" / f"{sample_idx:04d}.png"


def run_janus(args, prompts):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
    from inference.janus.generation import (
        build_prompt, generate, gererate_row_parallel, decode_image,
    )

    proc = VLChatProcessor.from_pretrained(args.vl_processor_path or args.model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    if args.lora_path:
        vl_gpt = PeftModel.from_pretrained(vl_gpt, args.lora_path)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in tqdm(prompts, desc="janus geneval"):
        prompt_dir = out_dir / f"{idx:05d}"
        (prompt_dir / "samples").mkdir(parents=True, exist_ok=True)
        write_metadata(prompt_dir, record)

        full_prompt = build_prompt(record["prompt"], proc)

        for k in range(args.n_samples):
            target = sample_path(out_dir, idx, k)
            if target.exists() and not args.overwrite:
                continue
            seed = args.seed_base + k
            if args.row_parallel:
                tokens = gererate_row_parallel(
                    vl_gpt, proc, full_prompt,
                    cfg_weight=args.cfg_guidance_scale,
                    ar_rows=args.ar_rows, seed=seed,
                    row_attention_mode=args.row_attention_mode,
                    row_attention_window=args.row_attention_window,
                )
            else:
                tokens = generate(
                    vl_gpt, proc, full_prompt,
                    image_token_num_per_image=args.image_token_num_per_image,
                    cfg_weight=args.cfg_guidance_scale, seed=seed,
                )
            decode_image(
                vl_gpt, tokens,
                img_size=args.target_size, patch_size=args.patch_size,
                save_dir=str(prompt_dir / "samples"),
                save_name_base=f"{k:04d}",
            )
            # decode_image saves as .jpg; rename to .png expected by evaluator
            jpg = prompt_dir / "samples" / f"{k:04d}.jpg"
            if jpg.exists():
                Image.open(jpg).convert("RGB").save(target)
                jpg.unlink()


def run_lumina(args, prompts):
    from inference.lumina.inference_solver import FlexARInferenceSolver

    solver = FlexARInferenceSolver(
        model_path=args.model_path,
        vae_tokenizer_path=args.vae_tokenizer_path,
        precision=args.dtype,
        target_size=args.target_size,
        device="cuda:0",
        row_parallel=args.row_parallel,
        lora_path=args.lora_path,
    )
    image_logits_processor = solver.create_logits_processor(
        cfg=args.cfg_guidance_scale, image_top_k=args.image_top_k,
    )
    template = f"Generate an image of {args.target_size}x{args.target_size} according to the following prompt:\n"

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in tqdm(prompts, desc="lumina geneval"):
        prompt_dir = out_dir / f"{idx:05d}"
        (prompt_dir / "samples").mkdir(parents=True, exist_ok=True)
        write_metadata(prompt_dir, record)

        prompt_text = template + record["prompt"]

        for k in range(args.n_samples):
            target = sample_path(out_dir, idx, k)
            if target.exists() and not args.overwrite:
                continue
            seed = args.seed_base + k
            with torch.no_grad():
                gen = solver.generate(
                    images=[], qas=[[prompt_text, None]],
                    max_gen_len=8192, temperature=1.0,
                    cfg_guidance_scale=args.cfg_guidance_scale,
                    logits_processor=image_logits_processor,
                    seed=seed, block_size=args.block_size,
                    ar_rows=args.ar_rows,
                    row_attention_mode=args.row_attention_mode,
                    row_attention_window=args.row_attention_window,
                )
            new_image = gen[1][0]
            new_image.save(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_path", required=True,
                   help="Path to GenEval evaluation_metadata.jsonl")
    p.add_argument("--save_dir", required=True,
                   help="Output dir; images will be in <save_dir>/{idx:05d}/samples/")
    p.add_argument("--model_name", required=True, choices=["janus", "lumina"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--lora_path", default=None)
    p.add_argument("--n_samples", type=int, default=4,
                   help="Images per prompt (GenEval default = 4)")
    p.add_argument("--seed_base", type=int, default=0,
                   help="seed for sample k = seed_base + k")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    p.add_argument("--max_per_tag", type=int, default=-1,
                   help="Stratified subset: keep at most N prompts per tag (e.g. 10 -> ~60 prompts total)")
    p.add_argument("--overwrite", action="store_true")
    # generation knobs (shared)
    p.add_argument("--cfg_guidance_scale", type=float, default=5.0)
    p.add_argument("--target_size", type=int, default=384)
    p.add_argument("--row_parallel", action="store_true")
    p.add_argument("--ar_rows", type=int, default=1)
    p.add_argument("--row_attention_mode", default="full",
                   choices=["full", "bidirectional_window", "causal_window", "no_intrarow"])
    p.add_argument("--row_attention_window", type=int, default=4)
    # janus-specific
    p.add_argument("--vl_processor_path", default=None)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--image_token_num_per_image", type=int, default=576)
    # lumina-specific
    p.add_argument("--vae_tokenizer_path", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--image_top_k", type=int, default=2000)
    p.add_argument("--block_size", type=int, default=48)
    args = p.parse_args()

    prompts = list(iter_geneval_prompts(args.metadata_path))
    if args.end_idx > 0:
        prompts = prompts[args.start_idx:args.end_idx]
    elif args.start_idx > 0:
        prompts = prompts[args.start_idx:]

    if args.max_per_tag > 0:
        from collections import defaultdict
        bucket = defaultdict(list)
        for idx, rec in prompts:
            bucket[rec.get("tag", "unknown")].append((idx, rec))
        subset = []
        for tag in sorted(bucket):
            subset.extend(bucket[tag][:args.max_per_tag])
        subset.sort(key=lambda x: x[0])  # keep original idx order so folder numbers match metadata
        print(f"[geneval] stratified: kept {len(subset)}/{len(prompts)} prompts "
              f"(<= {args.max_per_tag} per tag across {len(bucket)} tags)")
        prompts = subset

    print(f"[geneval] {len(prompts)} prompts × {args.n_samples} samples = "
          f"{len(prompts)*args.n_samples} images -> {args.save_dir}")

    t0 = time.time()
    if args.model_name == "janus":
        run_janus(args, prompts)
    else:
        run_lumina(args, prompts)
    print(f"[geneval] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
