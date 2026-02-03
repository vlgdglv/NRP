import os, sys
import argparse
from pathlib import Path
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, TextStreamer
from typing import List
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import os, sys
import argparse
import gc
import math
from PIL import Image
import PIL.Image
import torch
import time

from datetime import datetime
import random

from inference.emu3.emu3_generation import Emu3Sampler, batched_cfg_sample, unbatched_cfg_sample
from utils.logger import get_logger

from model.emu3_arch.mllm.processing_emu3 import Emu3Processor
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0, help="Start index for sampling")
    parser.add_argument("--end", type=int, default=1000, help="End index for sampling (None means to the end)")
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument("--save_dir", type=str, default="datasets/COCO_Emu3_tokens_for_train/", help="Save directory")
    args = parser.parse_args()
    
    save_stats_dir = Path(args.save_dir)
    save_stats_dir.mkdir(parents=True, exist_ok=True)
    
    with open(f"/home/ffc3/bht/GSD/eval_coco/coco_data/coco2017_{args.split}_prompts.json", "r") as f:
        all_prompts = json.load(f)
    # N_SAMPLE = 800
    # N_SAMPLE = len(all_prompts)
    all_prompts = all_prompts[args.begin : args.end]
    
    emu_model_path = "/home/ffc3/bht/model_home/Emu3-Gen/"
    emu_vq_model_path = "/home/ffc3/bht/model_home/Emu3-VisionTokenizer/"
    device = "cuda:0"
    max_new_tokens = 8192
    image_top_k = args.image_top_k
    cfg_guidance_scale = 3.0
    
    model = AutoModelForCausalLM.from_pretrained(
        emu_model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(emu_model_path, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(emu_vq_model_path, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(emu_vq_model_path, device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    image_area = model.config.image_area
    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=image_top_k,
    )

    sampler = Emu3Sampler(model, tokenizer)
    
    for offset, item in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="Collecting Stats"):
        idx = offset + args.begin
        caption = item["caption"]
        
        mm_list, time_uesd, tokens = batched_cfg_sample(
            caption + POSITIVE_PROMPT,
            NEGATIVE_PROMPT,
            model,
            processor,
            sampler,
            device,
            cfg_guidance_scale,
            image_area,
            pad_token_id=model.config.pad_token_id,
            max_new_tokens=max_new_tokens,
            seed=SEED,
            return_tokens=True
        )
        
        for img in mm_list:
            if isinstance(img, Image.Image):
                result_image = img
                break
        
        result_image.save(os.path.join(save_stats_dir, "image_sample_{}.png".format(idx)))
        print("Sample", idx, " token shape: ", tokens.shape)
        data_load = {
            # "hidden_states": hidden_states,
            # "logits": logits,
            "tokens": tokens
        }
        torch.save(data_load, save_stats_dir / f"token_sample_{idx}.pt")
        total_valid_samples += 1

    print(f"\nProcessing complete. Valid samples: {total_valid_samples}/{len(all_prompts)}")
