# 
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.logger import get_logger
from utils import is_rank0

from typing import Callable
from data.dataset import TokenDataset

logger = get_logger(__name__)


class JanusImageRowCollator:
    """
        No eol token in janus series, things are simpler
    """
    def __init__(
        self,
        image_width: int,
        image_height: int,
        pad_token_id: int = 100015,
        use_standard_causal: bool = False,
        use_teacher: bool = False,
        token_check_func: Callable = None
    ):
        
        self.W = image_width
        self.H = image_height
        self.pad_token_id = pad_token_id
        self.image_len = self.W * self.H
        self.use_teacher = use_teacher

        self.image_ending_length = 0
        self.image_start_length = 0
        
        self.invalid_label = -100
        self.use_standard_causal = use_standard_causal
        
        self.token_check_func = token_check_func
        
    def __call__(self, batch):
        # batch: List[Dict[str, torch.Tensor]]
        
        if self.token_check_func is not None:
            self.token_check_func(batch, self.W, self.H)
        valid_len = [sample["input_ids"].shape[0] for sample in batch]
        image_starts = [valid_len[i] - self.image_len for i in range(len(valid_len))]
        token_ids = [sample["input_ids"] for sample in batch]

        if self.use_teacher:
            raise NotImplementedError
        

        input_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_token_id)
        # print("input_ids.shape", input_ids.shape)
        B, L = input_ids.shape
        device = input_ids.device
        labels = torch.full_like(input_ids, self.invalid_label)

        causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        final_mask_bool = causal_mask.unsqueeze(0).expand(B, L, L).clone() # (B, L, L)
        
        for i in range(B):
            seq = input_ids[i]
            
            pad_pos = (seq == self.pad_token_id).nonzero(as_tuple=False)
            Li = int(pad_pos[0].item()) if pad_pos.numel() > 0 else L
            
            img_begin = Li - self.image_len
            img_end = Li
            base = img_begin
            # assert image_starts[i] == base, f"image_starts[i]: {image_starts[i]}, base: {base}"
            # assert valid_len[i] == img_begin + self.image_len, f"valid_len[i]: {valid_len[i]}, img_begin: {img_begin}, self.image_len: {self.image_len}"

            for r in range(self.H):
                row_start = base + r * self.W
                row_end = row_start + self.W  #
                
                src = torch.arange(row_start, row_end, device=device)
                tgt = torch.arange(row_start + self.W, row_end + self.W, device=device)
                if r == self.H - 1:
                    continue
                labels[i, src] = seq[tgt]
           
            labels[i, :img_begin] = self.invalid_label

            if Li < L:
                labels[i, Li:] = self.invalid_label
           
            if self.use_standard_causal:
                pass
            else:
                seg_len = self.image_len
                seg_pos = torch.arange(seg_len, device=device)
                
                row_id = seg_pos // self.W
                
                r_i = row_id.unsqueeze(1)
                r_j = row_id.unsqueeze(0)
                # b_i = col_id.unsqueeze(1)
                # b_j = col_id.unsqueeze(0)

                row_visible = (r_i == r_j)
                abs_idx = torch.arange(img_begin, img_end, device=device)
                final_mask_bool[i, abs_idx.unsqueeze(1), abs_idx.unsqueeze(0)] |= row_visible
            pad_pos = (seq == self.pad_token_id).nonzero(as_tuple=False)
            if pad_pos.numel() > 0:
                pad_pos = pad_pos.squeeze(-1)
                final_mask_bool[i, :, pad_pos] = False  

        attention_mask_bool = final_mask_bool.unsqueeze(1)

        min_value = torch.finfo(torch.bfloat16).min 
        attention_mask = torch.full((B, 1, L, L), min_value, dtype=torch.bfloat16)
        attention_mask.masked_fill_(attention_mask_bool, 0.0)
        attention_mask = attention_mask.contiguous()
    
        if self.use_teacher:
            raise NotImplementedError
            # return {
            #     "input_ids": input_ids,
            #     "labels": labels,
            #     "attention_mask": attention_mask,
            #     "teacher_token": teacher_token,
            #     "teacher_logits": teacher_logits,
            # }
        else:
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
        

if __name__ == "__main__":
    import time
    import psutil
    proc = psutil.Process(os.getpid())
    def rss_mb() -> float:
        return proc.memory_info().rss / (1024 ** 2)

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    base_rss = rss_mb()
    base_tensor_bytes = 0

    Janus_COCO17_path = "/home/ffc3/bht/NRP/datasets/COCO_Janus_tokens_for_train"
    # Janus_laion_path = "/home/ffc3/bht/NRP/datasets/laion_Lumina7B_tokens_for_train"
    # Janus_midjourney_path = "/home/ffc3/bht/NRP/datasets/midjourney_Lumina7B_tokens_for_train"

    data_dir = [
        Janus_COCO17_path, 
        # Lumina_laion_path, 
        # Lumina_midjourney_path
    ]
    dataset_name = [
        "COCO",
    ]
    
    B, N, NB = 32, 1, True
    W, H = 24, 24
    cnt = 0
    use_teacher = False

    dataset = TokenDataset(
        data_dir=data_dir, 
        dataset_name=dataset_name,
        use_teacher=False,
    )

    collator = JanusImageRowCollator(
        image_width=24,
        image_height=24,
        pad_token_id=2366222,
        use_teacher=use_teacher,
        use_standard_causal=False
    )

    num_workers = min(os.cpu_count(), 8)

    loader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=collator,

        num_workers=1,
        pin_memory=True,
        persistent_workers=False, # False for testing
        prefetch_factor=2,
        drop_last=True,
    )

    Lmax = 0

    t0 = time.time()
    for i, batch in enumerate(loader):
        inputs = batch["input_ids"].cuda(non_blocking=NB)
        mask = batch["attention_mask"].cuda(non_blocking=NB)
        labels = batch["labels"].cuda(non_blocking=NB)
        # print(inputs[0][:50])
        # print(labels[0][:50])
 
        # print(mask[0][0][30])
        if use_teacher:
            teacher_token = batch["teacher_token"]
            teacher_logits = batch["teacher_logits"]
            print("inputs.shape", inputs.shape, "mask.shape", mask.shape, "labels.shape", labels.shape)
            print("teacher_token.shape", teacher_token.shape, "teacher_logits.shape", teacher_logits.shape)
        base_tensor_bytes += tensor_bytes(inputs)
        base_tensor_bytes += tensor_bytes(mask)
        base_tensor_bytes += tensor_bytes(labels)
        Lmax = max(Lmax, inputs.shape[1])    
        cnt += 1
        if i == N:
            break

    t1 = time.time()
    print(f"Throughput: {cnt * B / (t1 - t0):.2f} samples/sec")

    final_rss = rss_mb()
    print(f"RSS increase: {final_rss - base_rss:.2f} MB (process-level, includes overhead)")
    print(f"Tokens tensors total: {base_tensor_bytes / (1024**2):.2f} MB (sum of tensor buffers only)")
    print(f"Final RSS: {final_rss:.2f} MB")
    print(f"Max sequence length: {Lmax}")
