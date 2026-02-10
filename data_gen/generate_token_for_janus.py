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
from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
from inference.janus.generation import build_prompt, generate, gererate_row_parallel, decode_image


logger = get_logger(__name__)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0, help="Start index for sampling")
    parser.add_argument("--end", type=int, default=1000, help="End index for sampling (None means to the end)")
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument("--json_key", type=str, default="prompt")
    # parser.add_argument("--prompt_path", type=str, default="/home/ffc3/bht/GSD/eval_coco/coco_data/coco2017_train_prompts.json") # datasets/midjourney_20k.json
    parser.add_argument("--dataset_name", type=str, default="COCO")
    
    args = parser.parse_args()
    
    
    dataset_home_dir = "/home/ffc3/bht/NRP/datasets"
    if args.dataset_name == "COCO":
        prompt_path = "/home/ffc3/bht/GSD/eval_coco/coco_data/coco2017_train_prompts.json"
        save_dir = os.path.join(dataset_home_dir, "COCO_Janus_tokens_for_train")
    elif args.dataset_name == "laion":
        prompt_path = "/home/ffc3/bht/NRP/datasets/laion_20k.json"
        save_dir = os.path.join(dataset_home_dir, "laion_Janus_tokens_for_train")
    elif args.dataset_name == "midjourney":
        prompt_path = "/home/ffc3/bht/NRP/datasets/midjourney_20k.json"
        save_dir = os.path.join(dataset_home_dir, "midjourney_Janus_tokens_for_train")
    else:
        raise NotImplementedError
    
    model_path = "/home/ffc3/bht/model_home/Janus-Pro-7B/"
    device = "cuda:0"
    cache_dir = ".cache"
    dtype = "bf16"
    cfg_guidance_scale = 5.0
    target_size = 384
    image_token_num_per_image = 576
    patch_size = 16

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    save_stats_dir = Path(save_dir)
    save_stats_dir.mkdir(parents=True, exist_ok=True)

    include_prefill = True
    do_decode_image = False
    seed = 42
    
    total_valid_samples = 0
    with open(prompt_path, "r") as f:
        all_prompts = json.load(f)
    all_prompts = all_prompts[args.begin : args.end]
    
    for offset, item in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="Collecting Stats"):
        idx = offset + args.begin
        prompt = item[args.json_key]
        
        full_prompt_text = prompt
        print(f"full_prompt_text: {full_prompt_text}")
        time_start = time.time()
        torch.cuda.synchronize()

        with torch.no_grad():
            returns = generate(
                vl_gpt, vl_chat_processor, prompt, 
                image_token_num_per_image=image_token_num_per_image,
                cfg_weight=cfg_guidance_scale,
                seed=42,
                include_prefill=include_prefill
            )
        torch.cuda.synchronize()

        time_end = time.time()
        time_uesd = time_end - time_start

        if include_prefill:
            image_tokens, tokens_sequence = returns
        else:
            image_tokens = returns
            
        if do_decode_image:
            paths = decode_image(
                vl_gpt, image_tokens, 
                img_size=target_size, patch_size=patch_size, 
                save_dir=save_dir, save_name_base=f"image_sample_{idx}" 
            )
        if tokens_sequence.dim() == 1:
            tokens_sequence = tokens_sequence.unsqueeze(0)
        data_load = {
            "tokens": tokens_sequence[0]
        }
        
        torch.save(data_load, save_stats_dir / f"token_sample_{idx}.pt")
        total_valid_samples += 1

    print(f"\nProcessing complete. Valid samples: {total_valid_samples}/{len(all_prompts)}")
