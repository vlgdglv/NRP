import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import AutoModelForCausalLM, AutoConfig

from data.dataset import TokenDataset, ImageRowCollator
from model.lumina_arch.chameleon import ChameleonForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def is_rank0():
#     return True # 单机脚本默认 True，如果是多卡 DDP 需要自行修改


def run_offline_inference(
    teacher_model,
    data_dir, 
    save_dir,
    image_width=48+1,
    image_height=48,
    batch_size=4,
    top_k=20,
    start_idx=-1,
    end_idx=-1
):
    os.makedirs(save_dir, exist_ok=True)

    dataset = TokenDataset(data_dir=data_dir, file_ext=".pt")

    collator = ImageRowCollator(
        image_width=image_width,
        image_height=image_height,
        use_standard_causal=True 
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    teacher_model.eval()
    
    global_sample_idx = start_idx

    logger.info(f"Starting inference on {len(dataset)} samples. Saving Top-{top_k} logits/tokens.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferencing")):
            input_ids = batch["input_ids"].cuda() 
            attention_mask = batch["attention_mask"].cuda()
            
            outputs = teacher_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=False, 
                return_dict=True
            )
            full_logits = outputs.logits # [B, L, Vocab]
            
            topk_values, topk_indices = torch.topk(full_logits, k=top_k, dim=-1)

            B = input_ids.shape[0]
            
            topk_values_cpu = topk_values.detach().cpu().to(torch.float16)
            topk_indices_cpu = topk_indices.detach().cpu().to(torch.int32)

            for i in range(B):
                token_fname = f"token_{global_sample_idx}_top{top_k}.pt"
                logits_fname = f"logits_{global_sample_idx}_top{top_k}s.pt"
                
                torch.save(topk_indices_cpu[i], os.path.join(save_dir, token_fname))
                torch.save(topk_indices_cpu[i], os.path.join(save_dir, logits_fname))
                global_sample_idx += 1
            # return 
    logger.info(f"Done! Saved {global_sample_idx} pairs of files to {save_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Lumina-mGPT-7B-768")
    parser.add_argument("--data_path", type=str, default="/home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train")
    parser.add_argument("--save_dir", type=str, default="/home/ffc3/bht/NRP/datasets/COCO_Lumina7B_training")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    
    args = parser.parse_args()
    
    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True
    )
    
    data_path = args.data_path
    run_offline_inference(
        teacher_model=model, 
        data_dir=data_path, 
        save_dir=args.save_dir,
        top_k=args.topk,
        start_idx=args.start,
        end_idx=args.end
    )
   