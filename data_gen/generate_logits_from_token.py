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

    dataset = TokenDataset(data_dir=data_dir, file_ext=".pt", start_idx=start_idx, end_idx=end_idx)

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
        num_workers=1,
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
            topk_indices_cpu = topk_indices.detach().cpu().to(torch.int16)
            for i in range(B):
                end_pos = (input_ids[i] == collator.eos_token_id).nonzero(as_tuple=False)
                end_pos = int(end_pos[-1][0].item()) + 1
                # print("sample_idx", global_sample_idx, "end_pos", end_pos)
                save_fname = f"teacher_token_logits_{global_sample_idx}_top{top_k}.pt"
                torch.save((topk_indices_cpu[i][:end_pos, :], topk_values_cpu[i][:end_pos, :]), os.path.join(save_dir, save_fname))
                global_sample_idx += 1

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
   