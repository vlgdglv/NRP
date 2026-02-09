import os, sys
import argparse
import json
import gc
from PIL import Image
import PIL.Image
import torch
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np

from utils.logger import get_logger
from inference.lumina.inference_solver import FlexARInferenceSolver


logger = get_logger(__name__)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0, help="Start index for sampling")
    parser.add_argument("--end", type=int, default=1000, help="End index for sampling (None means to the end)")
    parser.add_argument("--split", type=str, default="train", help="train or val")
    # parser.add_argument("--save_dir", type=str, default="./COCO_Lumina7B_tokens_for_train/", help="Save directory")
    # parser.add_argument("--prompt_path", type=str, default="/home/ffc3/bht/GSD/eval_coco/coco_data/coco2017_train_prompts.json") # datasets/midjourney_20k.json
    parser.add_argument("--dataset_name", type=str, default="COCO")
    
    args = parser.parse_args()
    
    model_path = "/home/ffc3/bht/model_home/Lumina-mGPT-7B-768"
    vae_tokenizer_path = "/home/ffc3/bht/GSD/ckpts/chameleon/tokenizer"
    
    dataset_home_dir = "/home/ffc3/bht/NRP/datasets"
    if args.dataset_name == "COCO":
        prompt_path = "/home/ffc3/bht/GSD/eval_coco/coco_data/coco2017_train_prompts.json"
        save_dir = os.path.join(dataset_home_dir, "COCO_Lumina7B_tokens_for_train")
    elif args.dataset_name == "laion":
        prompt_path = "/home/ffc3/bht/NRP/datasets/laion_20k.json"
        save_dir = os.path.join(dataset_home_dir, "laion_Lumina7B_tokens_for_train")
    elif args.dataset_name == "midjourney":
        prompt_path = "/home/ffc3/bht/NRP/datasets/midjourney_20k.json"
        save_dir = os.path.join(dataset_home_dir, "midjourney_Lumina7B_tokens_for_train")
    else:
        raise NotImplementedError
    
    target_size = 768
    target_size_h, target_size_w = 768, 768
    device = "cuda:0"
    cache_dir = ".cache"
    dtype = "bf16"
    cfg_guidance_scale = 3.0
    image_top_k = 2000
    template_condition_sentences = f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n"

    save_stats_dir = Path(save_dir)
    save_stats_dir.mkdir(parents=True, exist_ok=True)

    inference_solver = FlexARInferenceSolver(
        model_path=model_path,
        vae_tokenizer_path=vae_tokenizer_path,
        precision=dtype,
        target_size=target_size,
        device = device,
        row_parallel=False,
        
    )

    # N_SAMPLE = 800
    # N_SAMPLE = len(all_prompts)
    do_decode_image = False
    seed = 42
    template_prefix = f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n"
    total_valid_samples = 0
    with open(prompt_path, "r") as f:
        all_prompts = json.load(f)
    all_prompts = all_prompts[args.begin : args.end]
    
    for offset, item in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="Collecting Stats"):
        idx = offset + args.begin
        prompt = item["prompt"]
        
        full_prompt_text = template_prefix + prompt
        print(f"full_prompt_text: {full_prompt_text}")
        time_start = time.time()
        torch.cuda.synchronize()

        with torch.no_grad():
            returns = inference_solver.collect_tokens(
                images=[],
                qas=[[full_prompt_text, None]],
                max_gen_len=8192,
                temperature=1.0,
                cfg_guidance_scale=cfg_guidance_scale,
                logits_processor=inference_solver.create_logits_processor(cfg=cfg_guidance_scale, image_top_k=image_top_k),
                seed=seed,
                do_decode_image=do_decode_image
            )
        torch.cuda.synchronize()

        time_end = time.time()
        time_uesd = time_end - time_start

        if do_decode_image:
            logger.info(f"Image {offset} generation time elapsed: {time_uesd:.2f} s)")
            tokens_sequence, generated = returns
            a1, new_image = generated[0], generated[1][0]
            result_image = inference_solver.create_image_grid([new_image], 1, 1)
            result_image.save(save_stats_dir / f"image_sample_{idx}.png")
        else:
            tokens_sequence = returns
            
        print(f"tokens_sequence: {tokens_sequence.shape}")
        data_load = {
            # "hidden_states": hidden_states,
            # "logits": logits,
            "tokens": tokens_sequence[0] # shape [seq_len]
        }
        torch.save(data_load, save_stats_dir / f"token_sample_{idx}.pt")
        total_valid_samples += 1

    print(f"\nProcessing complete. Valid samples: {total_valid_samples}/{len(all_prompts)}")
