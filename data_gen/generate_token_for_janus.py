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
from peft import PeftModel
from utils.logger import get_logger
from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
from inference.janus.generation import build_prompt, generate, gererate_row_parallel, decode_image, gererate_row_parallel_tinyar


logger = get_logger(__name__)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0, help="Start index for sampling")
    parser.add_argument("--end", type=int, default=1000, help="End index for sampling (None means to the end)")
    parser.add_argument("--name_offset", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument("--json_key", type=str, default="prompt")
    # parser.add_argument("--prompt_path", type=str, default="/jizhicfs/pkuhetu/bht/GSD/eval_coco/coco_data/coco2017_train_prompts.json") # datasets/midjourney_20k.json
    parser.add_argument("--dataset_name", type=str, default="COCO")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    parser.add_argument("--prompt_path", type=str, default=None)

    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--cfg_guidance_scale", type=float, default=1)
    parser.add_argument("--row_parallel", action="store_true")
    parser.add_argument("--ar_rows", type=int, default=1)
    parser.add_argument("--row_attention_mode", type=str, default=None,
                    choices=["full", "bidirectional_window", "causal_window", "no_intrarow"],
                    help="Row attention mode for inference (overrides draft_use_causal_mask)")
    parser.add_argument("--row_attention_window", type=int, default=4,
                        help="Window size for windowed row attention modes")
    parser.add_argument("--include_prefill", action="store_true")
    parser.add_argument("--do_decode_image", action="store_true")
    parser.add_argument("--do_save_token", action="store_true")
    parser.add_argument("--tinyar_path", type=str, default=None,
                        help="Path to tinyar_head.pt (enables TinyAR inference)")
    parser.add_argument("--tinyar_layers", type=int, default=1)
    parser.add_argument("--tinyar_heads", type=int, default=8)
    parser.add_argument("--tinyar_ffn_mult", type=int, default=4)

    args = parser.parse_args()
    
    dataset_home_dir = "/jizhicfs/pkuhetu/bht/NRP/datasets/image_token_train_Janus"
    if args.prompt_path is None:
        if args.dataset_name == "COCO":
            prompt_path = "/jizhicfs/pkuhetu/bht/NRP/datasets/coco2017_train_prompts.json"
            save_dir = os.path.join(dataset_home_dir, "COCO_Janus_tokens_for_train")
        elif args.dataset_name == "laion":
            prompt_path = "/jizhicfs/pkuhetu/bht/NRP/datasets/laion_20k.json"
            save_dir = os.path.join(dataset_home_dir, "laion_Janus_tokens_for_train")
        elif args.dataset_name == "midjourney":
            prompt_path = "/jizhicfs/pkuhetu/bht/NRP/datasets/midjourney_20k.json"
            save_dir = os.path.join(dataset_home_dir, "midjourney_Janus_tokens_for_train")
        else:
            raise NotImplementedError
    else:
        prompt_path = args.prompt_path
        save_dir = args.save_dir
    
    model_path = "/jizhicfs/pkuhetu/bht/model_home/Janus-Pro-7B/"
    device = "cuda:0"
    cache_dir = ".cache"
    dtype = "bf16"
    cfg_guidance_scale = args.cfg_guidance_scale
    target_size = 384
    image_token_num_per_image = 576
    patch_size = 16
    name_offset = args.name_offset
    
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    if args.lora_path is not None:
        vl_gpt = PeftModel.from_pretrained(
            vl_gpt,
            args.lora_path
        )
        print("model type:", type(vl_gpt))
        print("has peft_config:", hasattr(vl_gpt, "peft_config"))
        print("peft_config:", getattr(vl_gpt, "peft_config", None))
    
    tinyar_head = None
    if args.tinyar_path is not None:
        from model.tinyar import TinyARHead, get_backbone_hidden_dim
        hidden_dim = get_backbone_hidden_dim(vl_gpt)
        tinyar_head = TinyARHead(
            hidden_dim=hidden_dim,
            num_layers=args.tinyar_layers,
            num_heads=args.tinyar_heads,
            ffn_mult=args.tinyar_ffn_mult,
        )
        sd = torch.load(args.tinyar_path, map_location="cpu")
        tinyar_head.load_state_dict(sd)
        tinyar_head = tinyar_head.to(torch.bfloat16).cuda().eval()
        logger.info(f"Loaded TinyAR head from {args.tinyar_path}")


    save_stats_dir = Path(save_dir)
    save_stats_dir.mkdir(parents=True, exist_ok=True)

    include_prefill = args.include_prefill
    do_decode_image = args.do_decode_image
    do_save_token = args.do_save_token
    seed = 42
    
    total_valid_samples = 0
    with open(prompt_path, "r") as f:
        all_prompts = json.load(f)
    all_prompts = all_prompts[args.begin : args.end]
    
    logger.info(f"Source file: {prompt_path}, on this run will generate {len(all_prompts)} samples.")
    logger.info(f"Save to: {save_stats_dir}")
    
    for offset, item in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="Collecting Stats"):
        idx = offset + args.begin
        prompt = item[args.json_key]
        img_id = item["image_id" if args.dataset_name == "COCO" else "id"]
        full_prompt = build_prompt(prompt, vl_chat_processor)
        # full_prompt_text = prompt

        time_start = time.time()
        torch.cuda.synchronize()

        with torch.no_grad():
            if args.row_parallel:
                if tinyar_head is not None:
                    returns = gererate_row_parallel_tinyar(
                        vl_gpt, vl_chat_processor, prompt,
                        tinyar_head=tinyar_head,
                        cfg_weight=cfg_guidance_scale,
                        ar_rows=args.ar_rows,
                        seed=42,
                        row_attention_mode=args.row_attention_mode,
                        row_attention_window=args.row_attention_window,
                    )
                else:
                    returns = gererate_row_parallel(
                        vl_gpt, vl_chat_processor, full_prompt, 
                        cfg_weight=cfg_guidance_scale,
                        ar_rows=args.ar_rows,
                        seed=42,
                        do_sample=not args.argmax,
                        row_attention_mode=args.row_attention_mode,
                        row_attention_window=args.row_attention_window,
                    )
            else:
                returns = generate(
                    vl_gpt, vl_chat_processor, full_prompt, 
                    image_token_num_per_image=image_token_num_per_image,
                    cfg_weight=cfg_guidance_scale,
                    seed=42,
                    include_prefill=include_prefill,
                    do_sample=not args.argmax
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
                save_dir=save_dir, save_name_base=f"generated_{img_id}" 
            )
        
        if do_save_token:
            if include_prefill and tokens_sequence.dim() == 1:
                tokens_sequence = tokens_sequence.unsqueeze(0)
            data_load = {
                "tokens": tokens_sequence[0]
            }
            torch.save(data_load, save_stats_dir / f"token_sample_{idx+name_offset}.pt")
        total_valid_samples += 1

    print(f"\nProcessing complete. Valid samples: {total_valid_samples}/{len(all_prompts)}")
