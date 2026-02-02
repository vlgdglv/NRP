import os
import time
import torch
import random
import numpy as np

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache

from utils.logger import get_logger
from utils import rollback_kv_cache

logger = get_logger(__name__)


DO_EVAL_DRAFT = False
RT_SAVE_NAME = "full_ar"
# RT_SAVE_NAME = "lora_64_128_10_2_0.0001_bm"

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
            next_token, outputs = self._forward_and_sample(
                input_ids,
                attention_mask,
                past_key_values,
                temperature,
                logits_processor,
                token_sequence,
                do_sample,
                do_cfg,
                cfg_scale,
                is_prefill
            )
            past_key_values = outputs.past_key_values

            token_sequence = torch.cat([token_sequence, next_token], dim=-1)
            
            input_ids = next_token.repeat(2, 1) if do_cfg else next_token
            
            ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, ones], dim=-1)

            is_prefill = False if is_prefill else False
            
            if (next_token.item() == eos_token_id) or (token_sequence.shape[-1] == max_length):
                break

        return token_sequence

    def _forward_and_sample(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        temperature,
        logits_processor,
        token_sequence,
        do_sample: bool = True,
        do_cfg: bool = True,
        cfg_scale: float = 3.0,
        is_prefill: bool = True,
        is_multi_token: bool = False
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_hidden_states=False,
        )
        if is_multi_token:
            last_logits = outputs.logits
        else:
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
            probs = probs.view(-1, probs.shape[-1]) if is_multi_token else probs
            next_token = torch.multinomial(
                probs,
                num_samples=1,
                generator=self.generator,
            )
            next_token = next_token.view(1, -1) if is_multi_token else next_token
        else:
            next_token = torch.argmax(scores, dim=-1, keepdim=True)

        return next_token, outputs
        

class RowParallelSampler(SamplerEngine):
    """
        Only for single batch sampling (for now)
    """
    def __init__(
        self,
        model,
        lora_model=None,
        tokenizer=None,
        *,
        image_start_token,
        image_end_token,
        image_end_line_token,
        **kwargs
    ):
        super().__init__(model=model, tokenizer=tokenizer)
        self.lora_model = lora_model
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.image_end_line_token = image_end_line_token
        self.inner_cnt = 0
        self.image_end_line_token_tensor = torch.tensor([self.image_end_line_token,], dtype=torch.long, device=model.device).unsqueeze(0)

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
        ar_rows: int = 12,
        parallel_as_draft: bool = False,
        draft_use_bi_mask: bool = True,
        block_size: int = 48,
        **kwargs
    ):
        is_prefill = True
        device = input_ids.device
        prefill_length = input_ids.shape[-1]
        self._init_image_position_info()

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

        ar_row_done = False
        generated_hidden_states, generated_tokens = [], []
        
        with self.model.disable_adapter():
            while True:
                next_token, outputs = self._forward_and_sample(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    temperature,
                    logits_processor,
                    token_sequence,
                    do_sample,
                    do_cfg,
                    cfg_scale,
                    is_prefill
                )
                past_key_values = outputs.past_key_values
                generated_hidden_states.append(outputs.logits[:, -1:, :])
                generated_tokens.append(next_token)
                
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token
                
                ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                is_prefill = False if is_prefill else False
                img_pos_info = self._get_decoding_position(token_sequence)

                if img_pos_info["is_in_image"]:
                    if img_pos_info["is_end_of_line"] and img_pos_info["num_of_lines"] >= ar_rows:
                        ar_row_done = True
                        break

            # fill kv cache for eol token 
            outputs = self.model(
                input_ids=input_ids, # EOL
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            # print("[Length check] token_sequence: ", token_sequence.shape, "KV cache length:", past_key_values[0][0].shape, "attention_mask: ", attention_mask.shape)

        assert self.img_h is not None and self.img_w is not None
        remain_rows = self.img_h - ar_rows
        
        if DO_EVAL_DRAFT is not None and DO_EVAL_DRAFT and remain_rows == 0:
            self.inner_cnt += 1
            generated_hidden_states = torch.cat(generated_hidden_states, dim=1).detach().cpu()
            generated_tokens = torch.cat(generated_tokens, dim=-1).detach().cpu()
            torch.save({
                "tokens": generated_tokens,
                "hidden_states": generated_hidden_states
            }, f"testing_fields/data_runtime/{RT_SAVE_NAME}/sample_{self.inner_cnt}.pt")
            logger.info(f"Full ar states saved at sample_{self.inner_cnt}.pt., tokens: {generated_tokens.shape}, hidden_states: {generated_hidden_states.shape}")


        assert self.img_w % block_size == 0, f"Image width {self.img_w} is not divisible by block_size"
        num_blocks_per_row = self.img_w // block_size

        for rows in range(remain_rows):
            input_ids = token_sequence[:, -(block_size+1):-1] # with eol token

            for blocks in range(num_blocks_per_row):
                # print(input_ids.shape)
                if blocks == num_blocks_per_row - 1:
                    # append eol token cause it is last block
                    input_ids = torch.cat([input_ids, self.image_end_line_token_tensor], dim=-1)
                input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
                ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                if draft_use_bi_mask:
                    mask4d = self._build_row_bidirectional_mask(attention_mask, input_ids.shape[-1])
                else:
                    logger.warning_once(
                        "Using causal mask in drafting."
                    )
                blk_output = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=mask4d if draft_use_bi_mask else attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                blk_logits = blk_output.logits
                if not parallel_as_draft:
                    generated_hidden_states.append(blk_logits)
                    
                past_key_values = blk_output.past_key_values
                if do_cfg:
                    cond, uncond = blk_logits.chunk(2, dim=0)
                    blk_logits = uncond + cfg_scale * (cond - uncond)
                scores = logits_processor(token_sequence, blk_logits)
                
                if do_sample:
                    probs = torch.nn.functional.softmax(scores / temperature, dim=-1)
                    next_blk_token = torch.multinomial(
                        probs.view(-1, probs.shape[-1]),
                        num_samples=1,
                        generator=self.generator,
                    ).view(1, -1)
                else:
                    next_blk_token = torch.argmax(scores, dim=-1, keepdim=True)

                if not parallel_as_draft:
                    generated_tokens.append(next_blk_token)
                    
                if parallel_as_draft:
                    rollback_kv_cache(past_key_values, input_ids.shape[-1])
                    input_ids = next_blk_token 
                    input_ids = torch.roll(input_ids, shifts=1, dims=0)
                    input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
                    with self.model.disable_adapter():
                        next_blk_token, outputs = self._forward_and_sample(
                            input_ids,
                            attention_mask,
                            past_key_values,
                            temperature,
                            logits_processor,
                            token_sequence,
                            do_sample,
                            do_cfg,
                            cfg_scale,
                            is_prefill=False,
                            is_multi_token=True
                        )
                        past_key_values = outputs.past_key_values
                        generated_hidden_states.append(outputs.logits)
                        
                        token_sequence = torch.cat([token_sequence, next_blk_token], dim=-1)
                        input_ids = next_blk_token
                else:
                    token_sequence = torch.cat([token_sequence, next_blk_token], dim=-1)
                    input_ids = next_blk_token
        

        # Ending: ensure sample is finished
        input_ids = token_sequence[:, -1]
        input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
        ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)

        with self.model.disable_adapter():
            while True:
                next_token, outputs = self._forward_and_sample(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    temperature,
                    logits_processor,
                    token_sequence,
                    do_sample,
                    do_cfg,
                    cfg_scale,
                    is_prefill=False,
                )
                past_key_values = outputs.past_key_values
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token
                ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                if (next_token.item() == eos_token_id) or (token_sequence.shape[-1] == max_length):
                    break
        
        if DO_EVAL_DRAFT is not None and DO_EVAL_DRAFT and remain_rows != 0:
            self.inner_cnt += 1
            generated_hidden_states = torch.cat(generated_hidden_states, dim=1).detach().cpu()
            generated_tokens = torch.cat(generated_tokens, dim=-1).detach().cpu()
            torch.save({
                "tokens": generated_tokens,
                "hidden_states": generated_hidden_states
            }, f"testing_fields/data_runtime/{RT_SAVE_NAME}/sample_{self.inner_cnt}.pt")
            logger.info(f"Row parallel states saved at sample_{self.inner_cnt}.pt., tokens: {generated_tokens.shape}, hidden_states: {generated_hidden_states.shape}")


        return token_sequence

    def _build_row_bidirectional_mask(self, attn_mask3d, row_len):
        device = attn_mask3d.device
        dtype = attn_mask3d.dtype

        key_valid = attn_mask3d.to(torch.bool).unsqueeze(2).expand(-1, -1, row_len, -1)

        neg_inf = torch.finfo(torch.float32).min
        attn = torch.zeros_like(key_valid, dtype=torch.float32)
        attn = attn.masked_fill(~key_valid, neg_inf)

        return attn

    def _get_decoding_position(self, token_sequence):
        self.num_image_start_tokens = (token_sequence[0] == self.image_start_token).sum()
        self.num_image_end_tokens = (token_sequence[0] == self.image_end_token).sum()

        is_in_image, is_end_of_line, num_of_lines = False, False, -1

        if self.num_image_start_tokens == self.num_image_end_tokens:
            is_in_image = False    
        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            is_in_image = True
            is_end_of_line = False
            if self.image_start_index is None:
                self.image_start_index = torch.where(token_sequence[0] == self.image_start_token)[0][-1].item()

            new_token_num = len(token_sequence[0][self.image_start_index + 1 :])
            if new_token_num >= 2:
                if self.img_h is None or self.img_w is None:
                    h_grids, w_grids = (
                        token_sequence[0][self.image_start_index + 1] - 8804,
                        token_sequence[0][self.image_start_index + 2] - 8804,
                    )
                    self.img_h, self.img_w = h_grids * 2, w_grids * 2
                
                new_img_token_num = len(token_sequence[0][self.image_start_index + 3 :]) # don't forget the new generated token
                if new_img_token_num % (self.img_w + 1) == 0:
                    is_end_of_line = True
                num_of_lines = new_img_token_num // (self.img_w + 1)
        
        return {
            "is_in_image": is_in_image,
            "is_end_of_line": is_end_of_line,
            "num_of_lines": num_of_lines
        } 

    def _init_image_position_info(self):
        self.image_start_index = None
        self.img_h = None
        self.img_w = None