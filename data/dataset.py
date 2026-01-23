# 
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.logger import get_logger
from utils import is_rank0

logger = get_logger(__name__)


class TokenDataset(Dataset):
    def __init__(self, data_dir, file_ext=".pt"):
        super().__init__()
        self.samples = []
        files = sorted(glob.glob(os.path.join(data_dir, f"*{file_ext}")))

        for f_path in tqdm(files):
            try:
                payload =  torch.load(f_path, map_location="cpu", weights_only=True)
                token_seq = payload["tokens"]

                if not isinstance(token_seq, torch.Tensor):
                    token_seq = torch.tensor(token_seq, dtype=torch.int32)

                self.samples.append(token_seq)
            except Exception as e:
                logger.error(f"Failed to load {f_path}: {e}")
        
        if is_rank0():
            logger.info(f"Found and loaded {len(self.samples)} samples into RAM from {data_dir}.")


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx]}
    
"""
    Image Token Sequence:

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
    ):
        self.W = image_width
        self.H = image_height
        self.pad_token_id = pad_token_id
        self.image_len = self.W * self.H

        self.image_ending_length = 2
        self.image_start_length = 3

        self.invalid_label = -100

    def __call__(self, batch):
        # batch: List[Dict[str, torch.Tensor]]
        check_image(batch, self.W, self.H)
        token_ids = [sample["input_ids"][:-self.image_ending_length] for sample in batch]

        input_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_token_id)

        B, L = input_ids.shape

        labels = input_ids.clone()
        labels = torch.roll(labels, shifts=-self.W, dims=1)

        labels[:, -self.W] = self.invalid_label
        labels[input_ids == self.pad_token_id] = self.invalid_label

        before_image_length = L - self.image_len
        labels[:, :before_image_length] = self.invalid_label

        casual_mask = torch.tril(torch.ones((L, L), dtype=torch.bool))
        final_mask = casual_mask.clone()

        img_range = torch.arange(self.image_len)

        row_ids = img_range // self.W

        r_i = row_ids.unsqueeze(1)
        r_j = row_ids.unsqueeze(0)

        row_visible = (r_i == r_j)

        current_block = final_mask[before_image_length:, before_image_length:]
        final_mask[before_image_length:, before_image_length:] = current_block | row_visible
        attention_mask_bool = final_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

        min_value = torch.finfo(torch.bfloat16).min 
        attention_mask = torch.full((B, 1, L, L), min_value, dtype=torch.bfloat16)
        attention_mask.masked_fill_(attention_mask_bool, 0.0)
        attention_mask = attention_mask.contiguous()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        } 

def check_image(batch, W=49, H=48):
    img_length = W * H
    for sample in batch:
        token_ids = sample["input_ids"]
        assert token_ids[-1] == 8710
        assert token_ids[-2] == 8196
        prompt_len = len(token_ids) - 2 - 3 - img_length
        assert token_ids[prompt_len-1] == 8710
        assert token_ids[prompt_len] == 8197
        for idx in range(prompt_len+2, prompt_len+2+img_length, W):
            assert token_ids[idx+W] == 8803


def create_dataloader(data_dir, batch_size=32, image_width=49, image_height=48):
    dataset = TokenDataset(data_dir)

    collator = ImageRowCollator(
        image_width=image_width,
        image_height=image_height
    )

    num_workers = min(os.cpu_count(), 8)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,

        num_workers=num_workers,
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

    COCO17_path = "/home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train"
    B, N, NB = 32, 100, True
    cnt = 0
    loader = create_dataloader(COCO17_path, batch_size=B)
    Lmax = 0

    t0 = time.time()
    for i, batch in enumerate(loader):
        inputs = batch["input_ids"].cuda(non_blocking=NB)
        mask = batch["attention_mask"].cuda(non_blocking=NB)
        labels = batch["labels"].cuda(non_blocking=NB)
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
