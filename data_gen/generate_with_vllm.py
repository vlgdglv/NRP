# Use vllm environment

import argparse
import json
import time
import os
from safetensors.torch import safe_open
import os, glob
import numpy as np
from pathlib import Path
from collections import Counter
from inference.lumina.data.item_processor import FlexARItemProcessor

# TODO
# Fast inference token samples

from vllm import LLM, SamplingParams

STOP_TOKEN_ID = 8710

def build_prompt_token_ids(item_processor, templeta_prefix: str, caption: str):
    full_prompt_text = templeta_prefix + caption

    conversations = [
        {"from": "human", "value": full_prompt_text},
        {"from": "gpt", "value": None},
    ]
    item = {"image": [], "conversations": conversations}

    _prompt = item_processor.process_item(item)

    prompt = []
    for value in _prompt:
        if isinstance(value, int):
            prompt.append(value)
        else:
            prompt += value["input_ids"]

    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0, help="Start index for sampling")
    parser.add_argument("--end", type=int, default=1000, help="End index for sampling (None means to the end)")
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument("--prompt_path", type=str, default="/home/ffc3/bht/GSD/eval_coco/coco_data/coco2017_train_prompts.json", help="Prompt path")
    parser.add_argument("--save_dir", type=str, default="datasets/COCO_Lumina7B_tokens_for_train/", help="Save directory")
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Lumina-mGPT-7B-768")
    parser.add_argument("--vae_tokenizer_path", type=str, default="/home/ffc3/bht/GSD/ckpts/chameleon/tokenizer")
    parser.add_argument("--target_size", type=int, default=768)
    args = parser.parse_args()
    
    save_stats_dir = Path(args.save_dir)
    save_stats_dir.mkdir(parents=True, exist_ok=True)
    with open(args.prompt_path, "r") as f:
        all_prompts = json.load(f)

    item_processor = FlexARItemProcessor(target_size=args.target_size, vae_tokenizer_path=args.vae_tokenizer_path)

    all_prompts = all_prompts[args.begin : args.end]
    total_valid = 0

    # model_dir = "/home/ffc3/bht/model_home/Lumina-mGPT-7B-768"
    # files = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    # print(files)
    # for f in files:
    #     with safe_open(f, framework="pt", device="cpu") as sf:
    #         bad = []
    #         for k in sf.keys():
    #             shape = sf.get_tensor(k).shape
    #             if 1 in shape and len(shape) <= 2:
    #                 bad.append((k, shape))
    #         print(f, "num_suspicious", len(bad))
    #         print("examples:", bad[:20])
    # shapes = Counter()
    # qn = []
    # kn = []
    # for f in files:
    #     with safe_open(f, framework="pt", device="cpu") as sf:
    #         for k in sf.keys():
    #             t = sf.get_tensor(k)
    #             shapes[tuple(t.shape)] += 1
    #             if ".self_attn.q_norm." in k:
    #                 qn.append((k, tuple(t.shape)))
    #             if ".self_attn.k_norm." in k:
    #                 kn.append((k, tuple(t.shape)))
    # print("Top shapes:", shapes.most_common(20))
    # print("q_norm unique shapes:", sorted(set(s for _, s in qn)))
    # print("k_norm unique shapes:", sorted(set(s for _, s in kn)))
    # print("q_norm examples:", qn[:5])
    # print("k_norm examples:", kn[:5])
    
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2600,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    
    target_size_h, target_size_w = args.target_size, args.target_size
    sample_text = "A black Honda motorcycle parked in front of a garage."
    template_condition_sentences = f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n"

    token_ids = build_prompt_token_ids(item_processor, sample_text, template_condition_sentences)
    print(token_ids)
    tokenizer = llm.get_tokenizer()
    
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(prompt)

    t1 = time.time()
    for _ in range(2):
        outs = llm.generate(prompt, sampling_params)
    t2 = time.time()
    print("Time:", t2 - t1)
    print(outs[0])

