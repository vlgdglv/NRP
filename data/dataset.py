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
from model import lumina_img_token_config

logger = get_logger(__name__)


class TokenDataset(Dataset):
    def __init__(
        self, 
        data_dir: str | list[str], 
        dataset_name: str | list[str] | None = None,
        file_ext: str = ".pt",
        start_idx: int = -1,
        end_idx: int = -1,
        use_teacher: bool = False,
        teacher_data_dir: str = None,
        teacher_file_name: str = "teacher_token_logits_{}_top16.pt",
    ):
        super().__init__()

        if dataset_name is None or isinstance(dataset_name, str):
            dataset_names_list = None
        else:
            dataset_names_list = list(dataset_name)

        dirs = []

        if dataset_names_list is None:
            if not isinstance(data_dir, str):
                raise TypeError("data_dir must be a string if dataset_names is None")
            dirs = [(dataset_name, data_dir)]
        else:
            if isinstance(data_dir, str):
                dirs = [(name, os.path.join(data_dir, name)) for name in dataset_names_list]
            else:
                if len(data_dir) != len(dataset_names_list):
                    raise ValueError(f"data_dir list length ({len(data_dir)}) must match dataset_names length ({len(dataset_names_list)}).")
                dirs = list(zip(dataset_names_list, data_dir))

        teacher_dirs: list[str | None] = []
        if use_teacher:
            if teacher_data_dir is None:
                teacher_dirs = [None] * len(dirs)
            elif isinstance(teacher_data_dir, str):
                teacher_dirs = [teacher_data_dir] * len(dirs)
            else:
                if len(teacher_data_dir) != len(dirs):
                    raise ValueError(
                        f"teacher_data_dir list length ({len(teacher_data_dir)}) must match number of datasets ({len(dirs)})."
                    )
                teacher_dirs = list(teacher_data_dir)

        self.samples = []
        self.use_teacher = use_teacher
        if use_teacher:
            self.teacher_samples = []
        # files = sorted(glob.glob(os.path.join(data_dir, f"*{file_ext}")), key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        if start_idx != -1 and end_idx != -1:
            files = files[start_idx:end_idx]

        def _idx_key(path: str) -> int:
            base = os.path.basename(path)
            return int(base.split("_")[-1].split(".")[0])

        total_loaded = 0
        for ds_i, (ds_name, ds_dir) in enumerate(dirs):
            files = sorted(glob.glob(os.path.join(ds_dir, f"*{file_ext}")), key=_idx_key)

            if start_idx != -1 and end_idx != -1:
                files = files[start_idx:end_idx]
            
            loaded_this = 0

            iterator = tqdm(files, desc=f"Loading {ds_name}") if is_rank0() else files
            for f_path in iterator:
                try:
                    payload =  torch.load(f_path, map_location="cpu", weights_only=True)
                    token_seq = payload["tokens"]

                    if not isinstance(token_seq, torch.Tensor):
                        token_seq = torch.tensor(token_seq, dtype=torch.int32)
                    if token_seq.shape[0] == 1:
                        token_seq = token_seq.squeeze(0)

                    if use_teacher and teacher_data_dir and teacher_file_name:
                        tdir = teacher_dirs[ds_i]
                        if tdir and teacher_file_name:
                            sample_idx = os.path.basename(f_path).split(".")[0].split("_")[-1]
                            teacher_token, teacher_logits = torch.load(
                                os.path.join(tdir, teacher_file_name.format(sample_idx)),
                                map_location="cpu",
                                weights_only=True,
                            )
                            self.teacher_samples.append((teacher_token, teacher_logits))
                        else:
                            raise ValueError(f"use_teacher=True but teacher_data_dir not provided for dataset '{ds_name}'.")

                    self.samples.append(token_seq)
                    loaded_this += 1
                    total_loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load {f_path}: {e}")
            
            if is_rank0():
                logger.info(f"[{ds_name}] Loaded {loaded_this} samples from {ds_dir}.")
        
        if is_rank0():
            if dataset_names_list is None:
                logger.info(f"Found and loaded {total_loaded} samples into RAM from {dirs[0][1]}.")
            else:
                logger.info(f"Found and loaded {total_loaded} samples into RAM from {len(dirs)} datasets: {[n for n, _ in dirs]}.")
        
        if self.use_teacher and len(self.teacher_samples) != len(self.samples):
            raise RuntimeError(f"Teacher/sample length mismatch: teacher={len(self.teacher_samples)}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_teacher:
            return {
                "input_ids": self.samples[idx], 
                "teacher_token": self.teacher_samples[idx][0], 
                "teacher_logits": self.teacher_samples[idx][1]
            }
        else:
            return {
                "input_ids": self.samples[idx], 
            }
    
"""
    Lumina Image Token Sequence:

    | ---------- Prompt token ------ | --Image Ctrl--| -- Image token -- |----Ending----|
    | 0 ------------------ <imgspan> | <boi><sz><sz> | {-{W}-<eol>} x {H}|<eoi><imgspan>|
                            pl-1       pl-0
    <imgspan>: 8710
    <boi>: 8197
    <sz>: variable, 8828
    <eol>: 8803
    <eoi>: 8196
"""    

class ImageRowCollator:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        pad_token_id: int = 0,
        use_standard_causal: bool = False,
        block_size: int = 48,
        use_teacher: bool = False,
        eoi_token_id: int = 8196,
        boi_token_id: int = 8197,
        eol_token_id: int = 8803,
        eos_token_id: int = 8710,
        img_token_id: int = -1,
        token_check_func: Callable = None
    ):
        self.W = image_width
        self.H = image_height
        self.pad_token_id = pad_token_id
        self.image_len = self.W * self.H
        self.use_teacher = use_teacher
        
        self.block_size = block_size
        self.img_W = self.W - 1
        self.num_blocks = (self.img_W + self.block_size - 1) // self.block_size
        self.last_block_id = (self.img_W - 1) // self.block_size
        self.first_block_len = min(self.block_size, self.img_W)

        self.image_ending_length = 2
        self.image_start_length = 3
        
        self.invalid_label = -100
        self.use_standard_causal = use_standard_causal
        
        self.eoi_token_id = eoi_token_id
        self.boi_token_id = boi_token_id
        self.eol_token_id = eol_token_id
        self.eos_token_id = eos_token_id
        self.img_token_id = img_token_id

        self.token_check_func = token_check_func
        
    def __call__(self, batch):
        # batch: List[Dict[str, torch.Tensor]]
        if self.token_check_func is not None:
            self.token_check_func(batch, self.W, self.H)
        token_ids = [sample["input_ids"] for sample in batch]

        if self.use_teacher:
            # check length
            teacher_token = [sample["teacher_token"] for sample in batch]
            teacher_logits = [sample["teacher_logits"]for sample in batch]  
            
            for input_ids, teacher_token, teacher_logits in zip(token_ids, teacher_token, teacher_logits):
                assert input_ids.shape[0] == teacher_token.shape[0] == teacher_logits.shape[0]
            teacher_token = pad_sequence(
                teacher_token,
                batch_first=True,
                padding_value=self.pad_token_id
            ) 
            teacher_logits = pad_sequence(
                teacher_logits,
                batch_first=True,
                padding_value=0.0
            ) 
            
        input_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_token_id)
        
        B, L = input_ids.shape
        device = input_ids.device

        labels = torch.full_like(input_ids, self.invalid_label, dtype=torch.long)
        causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        final_mask_bool = causal_mask.unsqueeze(0).expand(B, L, L).clone() # (B, L, L)
        
        for i in range(B):
            seq = input_ids[i]
            
            img_end_pos = (seq == self.eoi_token_id).nonzero(as_tuple=False)
            if img_end_pos.numel() == 0: continue
            
            img_end_pos = int(img_end_pos[0].item())
            img_tokens_begin = img_end_pos - self.image_len
            img_tokens_end = img_end_pos

            # print(seq[img_tokens_begin-3], seq[img_tokens_begin-2], seq[img_tokens_begin-1], seq[img_tokens_begin])
            for r in range(self.H):
                row_base = img_tokens_begin + r * self.W
                
                for b in range(self.num_blocks):
                    c0 = b * self.block_size
                    c1 = min((b + 1) * self.block_size, self.img_W)
                    
                    src_idx = torch.arange(row_base + c0, row_base + c1, device=device)
                    if b < self.num_blocks - 1:
                        tgt_c0 = (b + 1) * self.block_size
                        tgt_c1 = min((b + 2) * self.block_size, self.img_W)
                        tgt_idx = torch.arange(row_base + tgt_c0, row_base + tgt_c1, device=device)
                    else:
                        if r == self.H-1: continue
                        # assert seq[src_idx[-1]+1] == self.eol_token_id
                        next_row_base = img_tokens_begin + (r + 1) * self.W
                        tgt_idx = torch.arange(next_row_base + 0, next_row_base + self.first_block_len, device=device)
                        
                    n = min(src_idx.numel(), tgt_idx.numel())
                    src_idx, tgt_idx = src_idx[:n], tgt_idx[:n]
                    labels[i, src_idx] = input_ids[i, tgt_idx]

                # deal with eol
                if r == self.H-1: continue
                if self.num_blocks == 1:
                    labels[i, row_base + self.W - 1] = self.eol_token_id
                else:
                    labels[i, row_base + self.W - 1] = input_ids[i, row_base + self.W]
                
            labels[i, seq == self.pad_token_id] = self.invalid_label
            labels[i, seq == self.eos_token_id] = self.invalid_label
            labels[i, :img_tokens_begin] = self.invalid_label

            if self.use_standard_causal:
                pass
            else:
                seg_len = self.image_len
                seg_pos = torch.arange(seg_len, device=device)
                
                row_id = seg_pos // self.W
                col_id = seg_pos % self.W
                is_eol = (col_id == self.W - 1)
            
                block_id = torch.empty_like(col_id)
                block_id[~is_eol] = (col_id[~is_eol] // self.block_size)
                block_id[is_eol] = self.last_block_id
                
                r_i = row_id.unsqueeze(1)
                r_j = row_id.unsqueeze(0)
                b_i = block_id.unsqueeze(1)
                b_j = block_id.unsqueeze(0)
                
                block_visible = (r_i == r_j) & (b_j <= b_i)
                
                abs_idx = torch.arange(img_tokens_begin, img_tokens_end, device=device)
                final_mask_bool[i, abs_idx.unsqueeze(1), abs_idx.unsqueeze(0)] |= block_visible
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
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "teacher_token": teacher_token,
                "teacher_logits": teacher_logits,
            }
        else:
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }


def create_dataloader(
    data_dir, dataset_name, batch_size=32, image_width=49, image_height=48, block_size=48,
    use_teacher=False, teacher_data_dir="/home/ffc3/bht/NRP/datasets/COCO_Lumina7B_training",
):
    dataset = TokenDataset(
        data_dir=data_dir, 
        dataset_name=dataset_name,
        use_teacher=use_teacher, 
        teacher_data_dir=teacher_data_dir
    )

    collator = ImageRowCollator(
        image_width=image_width,
        image_height=image_height,
        block_size=block_size,
        use_teacher=use_teacher,
        **lumina_img_token_config,
    )

    num_workers = min(os.cpu_count(), 8)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,

        num_workers=1,
        pin_memory=True,
        persistent_workers=False, # False for testing
        prefetch_factor=2,
        drop_last=True,
    )
    return loader


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

    Lumina_COCO17_path = "/home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train"
    Lumina_laion_path = "/home/ffc3/bht/NRP/datasets/laion_Lumina7B_tokens_for_train"
    Lumina_midjourney_path = "/home/ffc3/bht/NRP/datasets/midjourney_Lumina7B_tokens_for_train"

    data_dir = [
        Lumina_COCO17_path, 
        Lumina_laion_path, 
        Lumina_midjourney_path
    ]
    dataset_name = [
        "COCO",
        "laion", 
        "midjourney"
    ]
    
    # Emu3_COCO_path = "/home/ffc3/bht/NRP/datasets/COCO_Emu3_tokens_for_train"
    B, N, NB = 32, 1, True
    W, H = 49, 48
    cnt = 0
    use_teacher = False
    loader = create_dataloader(data_dir=data_dir, dataset_name=dataset_name, batch_size=B, use_teacher=use_teacher, image_width=W, image_height=H)
    Lmax = 0

    t0 = time.time()
    for i, batch in enumerate(loader):
        inputs = batch["input_ids"].cuda(non_blocking=NB)
        mask = batch["attention_mask"].cuda(non_blocking=NB)
        labels = batch["labels"].cuda(non_blocking=NB)
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
