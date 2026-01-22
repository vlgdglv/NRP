import os
import time
import torch
import random
import numpy as np

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache

from utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SamplerEngine:
    def __init__(
            self, 
            model, 
            tokenizer = None,
        ):
        self.model = model
        self.tokenizer = tokenizer
    
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        temperature: float = 1.0,
        max_length: int = 4096,
        eos_token_id: int = 8710,
        do_sample: bool = True,
        sample_mode: str = "baseline",
        do_cfg: bool = True,
        cfg_scale: float = 3.0,
        seed: int = None,
        use_cache: bool = True,
        **kwargs
    ):
        is_prefill = True
        device = input_ids.device
        prefill_length = input_ids.shape[-1]

        if seed is not None:
            set_seed(seed)
            self.generator = torch.Generator(device).manual_seed(seed)

        # Preparse everything
        input_ids = input_ids.contiguous()
        token_sequence = input_ids

        # Use dynamic cache as default
        past_key_values = DynamicCache() if use_cache else None
        # past_key_values = None
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)

        # Prepare cfgs
        if do_cfg:
            input_ids = input_ids.repeat(2, 1)
            attention_mask = attention_mask.repeat(2, 1, 1)
            attention_mask[1::2, :, :prefill_length - 1] = 0

        while True:
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_hidden_states=False,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]
            

            if do_cfg: 
                cond, uncond = last_logits.chunk(2, dim=0)
                if is_prefill:
                    last_logits = cond
                else:
                    last_logits = uncond + cfg_scale * (cond - uncond)

            scores = logits_processor(token_sequence, last_logits)
      
            if do_sample:
                probs = torch.nn.functional.softmax(scores / temperature, dim=-1)
                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                    generator=self.generator,
                )
            else:
                next_token = torch.argmax(scores, dim=-1, keepdim=True)

            token_sequence = torch.cat([token_sequence, next_token], dim=-1)
            
            input_ids = next_token.repeat(2, 1) if do_cfg else next_token
            
            ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, ones], dim=-1)

            is_prefill = False if is_prefill else False
            
            if (next_token.item() == eos_token_id) or (token_sequence.shape[-1] == max_length):
                break

        return token_sequence