import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache
# from peft import set_adapter, disable_adapter

import torch.nn.functional as F
from utils.logger import get_logger
from utils import rollback_kv_cache, snapshot_kv_cache

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
        attention_mask: torch.LongTensor = None,
        temperature: float = 1.0,
        max_length: int = 4096,
        eos_token_id: int = 8710,
        do_sample: bool = True,
        sample_mode: str = "baseline",
        do_cfg: bool = True,
        is_uncond_provided: bool = False,
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
        else:
            self.generator = None

        # Preparse everything
        input_ids = input_ids.contiguous()
        if is_uncond_provided:
            mask = attention_mask[0].bool()
            token_sequence = input_ids[0, mask].unsqueeze(0)
        else:
            token_sequence = input_ids

        # Use dynamic cache as default
        past_key_values = DynamicCache() if use_cache else None
        # past_key_values = None
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        # if attention_mask.dim() == 2:
        #     attention_mask = attention_mask.unsqueeze(1)

        # Prepare cfgs
        if do_cfg:
            if is_uncond_provided:
                # assert "position_ids" in kwargs, "need to provide position_ids for uncond and cond"
                pass
            else:
                input_ids = input_ids.repeat(2, 1)
                attention_mask = attention_mask.repeat(2, 1)
                attention_mask[1::2,  :prefill_length - 1] = 0
            
        generated_token_count = 0
        real_lens = kwargs["real_lens"] if "real_lens" in kwargs else None

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
                is_prefill,
                **kwargs
            )
            past_key_values = outputs.past_key_values

            token_sequence = torch.cat([token_sequence, next_token], dim=-1)
            
            input_ids = next_token.repeat(2, 1) if do_cfg else next_token
            
            ones = torch.ones(input_ids.shape[0], input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
            # ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, ones], dim=-1)

            generated_token_count += next_token.shape[-1]
            is_prefill = False if is_prefill else False
            
            if real_lens is not None:
                real_lens += next_token.shape[-1]
                position_ids = real_lens.view(real_lens.shape[0], 1) - 1            
                if "position_ids" in kwargs:
                    kwargs["position_ids"] = position_ids

            if (next_token.item() == eos_token_id) or (generated_token_count == max_length):
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
        is_multi_token: bool = False,
        **kwargs
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
        outputs.scores = scores
        probs = torch.nn.functional.softmax(scores / temperature, dim=-1)
        if do_sample:   
            probs = probs.view(-1, probs.shape[-1]) if is_multi_token else probs
            outputs.probs = probs
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
        tokenizer=None,
        *,
        image_start_token,
        image_end_token,
        image_end_line_token,
        **kwargs
    ):
        super().__init__(model=model, tokenizer=tokenizer)
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
        ar_rows: int = 1,
        parallel_as_draft: bool = False,
        draft_use_bi_mask: bool = True,
        block_size: int = 48,
        row_attention_mode: str = None,
        row_attention_window: int = 4,
        **kwargs
    ):
        # row_attention_mode overrides draft_use_bi_mask for backward compat
        if row_attention_mode is None:
            row_attention_mode = "full" if draft_use_bi_mask else "causal"

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
            attention_mask[1::2,:, :prefill_length - 1] = 0

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

                if row_attention_mode != "causal":
                    mask4d = build_row_mask(attention_mask, input_ids.shape[-1],
                                            mode=row_attention_mode, window=row_attention_window)
                else:
                    mask4d = None
                    logger.warning_once(
                        "Using causal mask in drafting."
                    )
                blk_output = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=mask4d if mask4d is not None else attention_mask,
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
                    next_blk_token = next_blk_token.view(-1, next_blk_token.shape[1])

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


class RowParallelSamplerTester(RowParallelSampler):
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        image_start_token,
        image_end_token,
        image_end_line_token,
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            image_start_token=image_start_token,
            image_end_token=image_end_token,
            image_end_line_token=image_end_line_token,
        )

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
        ar_rows: int = 1,
        parallel_as_draft: bool = False,
        draft_use_bi_mask: bool = True,
        block_size: int = 48,
        row_attention_mode: str = None,
        row_attention_window: int = 4,
        **kwargs
    ):
        # row_attention_mode overrides draft_use_bi_mask for backward compat
        if row_attention_mode is None:
            row_attention_mode = "full" if draft_use_bi_mask else "causal"

        anything_dict = {}
        if "return_anything_dict" in kwargs.keys() and kwargs["return_anything_dict"]:
            return_anything_dict = kwargs["return_anything_dict"]
        else:
            return_anything_dict = False

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
            attention_mask[1::2,:, :prefill_length - 1] = 0

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
        
        gt_ranks, draft_ranks = [], []
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

                if row_attention_mode != "causal":
                    mask4d = build_row_mask(attention_mask, input_ids.shape[-1],
                                            mode=row_attention_mode, window=row_attention_window)
                else:
                    mask4d = None
                    logger.warning_once(
                        "Using causal mask in drafting."
                    )
                blk_output = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=mask4d if mask4d is not None else attention_mask,
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
                    
                # if parallel_as_draft:
                #     rollback_kv_cache(past_key_values, input_ids.shape[-1])
                #     input_ids = next_blk_token 
                #     input_ids = torch.roll(input_ids, shifts=1, dims=0)
                #     input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
                #     with self.model.disable_adapter():
                #         next_blk_token, outputs = self._forward_and_sample(
                #             input_ids,
                #             attention_mask,
                #             past_key_values,
                #             temperature,
                #             logits_processor,
                #             token_sequence,
                #             do_sample,
                #             do_cfg,
                #             cfg_scale,
                #             is_prefill=False,
                #             is_multi_token=True
                #         )
                #         past_key_values = outputs.past_key_values
                #         generated_hidden_states.append(outputs.logits)
                        
                #         token_sequence = torch.cat([token_sequence, next_blk_token], dim=-1)
                #         input_ids = next_blk_token
                # else:
                #     token_sequence = torch.cat([token_sequence, next_blk_token], dim=-1)
                #     input_ids = next_blk_token
                # Got row-wise token
            # Check row-token if it's valid
            # Rollbacks: 
            
            rollback_kv_cache(past_key_values, input_ids.shape[-1])
            attention_mask = attention_mask[..., :-input_ids.shape[-1]]

            one_step_input = token_sequence[:, -1] # should be eol
            one_step_input = one_step_input.repeat(2, 1) if do_cfg else one_step_input
            # print("one_step_input: ", one_step_input)
            gt_tokens, gt_scores = [],[]
            tv_distances = []
            with self.model.disable_adapter():
                for iii in range(self.img_w+1):
                    ones = torch.ones(one_step_input.shape[0], 1, one_step_input.shape[1], dtype=torch.long, device=one_step_input.device)
                    attention_mask = torch.cat([attention_mask, ones], dim=-1)
                    # kwargs["position_ids"] = real_lens.view(real_lens.shape[0], 1) - 1

                    next_token, outputs = self._forward_and_sample(
                        one_step_input,
                        attention_mask,
                        past_key_values,
                        temperature,
                        logits_processor,
                        token_sequence,
                        do_sample,
                        do_cfg,
                        cfg_scale,
                        is_prefill,
                        **kwargs
                    )
                    
                    gt_tokens.append(next_token)
                    gt_scores.append(outputs.probs)

                    past_key_values = outputs.past_key_values
                    one_step_input = next_token.repeat(2, 1) if do_cfg else next_token
            
                    token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                    # real_lens += next_token.shape[-1]
            gt_tokens = torch.cat(gt_tokens, dim=-1)
            gt_scores = torch.cat(gt_scores, dim=0)
            p_draft = probs.view(-1, probs.shape[-1])
            p_target = gt_scores.view(-1, gt_scores.shape[-1])
        
            # print("gt_tokens: ", gt_tokens)
            
            gt_rank = (scores.squeeze(0).argsort(dim=-1, descending=True) == gt_tokens.view(-1, 1)).nonzero()[:,1]
            draft_rank = (gt_scores.argsort(dim=-1, descending=True) == next_blk_token.view(-1, 1)).nonzero()[:,1]
            tv_dist = 0.5 * torch.abs(p_draft - p_target).sum(dim=-1)
            
            tv_distances.append(tv_dist)
            gt_ranks.append(gt_rank)
            draft_ranks.append(draft_rank)

        gt_ranks = torch.cat(gt_ranks, dim=-1).to(torch.float)
        # print("Mean gt rank: {:.4f}, Std: {:.4f}".format(gt_ranks.mean(), gt_ranks.std()))
        draft_ranks = torch.cat(draft_ranks, dim=-1).to(torch.float)
        # print("Mean draft rank: {:.4f}, Std: {:.4f}".format(draft_ranks.mean(), draft_ranks.std()))
        tv_distances = torch.cat(tv_distances, dim=-1).to(torch.float)
        
        anything_dict["gt_ranks.mean"] = float(gt_ranks.mean())
        anything_dict["gt_ranks.std"] = float(gt_ranks.std())
        anything_dict["draft_ranks.mean"] = float(draft_ranks.mean())
        anything_dict["draft_ranks.std"] = float(draft_ranks.std())    
        anything_dict["tv_distance.mean"] = float(tv_distances.mean())
        anything_dict["tv_distance.std"] = float(tv_distances.std())

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

        if return_anything_dict:
            return token_sequence, anything_dict
        else:
            return token_sequence


def build_row_bidirectional_mask(attn_mask, row_len):
    if attn_mask.dim() == 3:
        attn_mask3d = attn_mask
    else:
        attn_mask3d = attn_mask.unsqueeze(1)

    key_valid = attn_mask3d.to(torch.bool).unsqueeze(2).expand(-1, -1, row_len, -1)

    neg_inf = torch.finfo(torch.float32).min
    attn = torch.zeros_like(key_valid, dtype=torch.float32)
    attn = attn.masked_fill(~key_valid, neg_inf)

    return attn


def build_row_mask(attn_mask, row_len, mode="full", window=4):
    """
    Build 4D attention mask for row-parallel inference.

    Starts from full bidirectional (all row tokens see all valid keys),
    then restricts the intra-row region (last row_len keys) based on mode.

    Modes:
        "full"                  - original bidirectional (all row tokens see each other)
        "bidirectional_window"  - token i sees [i-w, i+w] within the row
        "causal_window"         - token i sees [i-w, i] within the row
        "no_intrarow"           - token i sees only itself within the row

    Previous context (KV cache) visibility is unchanged in all modes.
    """
    # Start with full bidirectional base
    if attn_mask.dim() == 3:
        attn_mask3d = attn_mask
    else:
        attn_mask3d = attn_mask.unsqueeze(1)

    key_valid = attn_mask3d.to(torch.bool).unsqueeze(2).expand(-1, -1, row_len, -1)

    neg_inf = torch.finfo(torch.float32).min
    attn = torch.zeros_like(key_valid, dtype=torch.float32)
    attn = attn.masked_fill(~key_valid, neg_inf)

    if mode == "full":
        return attn

    # Restrict intra-row attention (the last row_len columns)
    total_len = attn.shape[-1]
    row_start = total_len - row_len

    q_idx = torch.arange(row_len, device=attn.device).unsqueeze(1)  # (row_len, 1)
    k_idx = torch.arange(row_len, device=attn.device).unsqueeze(0)  # (1, row_len)

    if mode == "no_intrarow":
        intra_visible = (q_idx == k_idx)
    elif mode == "bidirectional_window":
        intra_visible = (torch.abs(q_idx - k_idx) <= window)
    elif mode == "causal_window":
        intra_visible = (k_idx <= q_idx) & (k_idx >= q_idx - window)
    else:
        raise ValueError(f"Unknown row mask mode: {mode}")

    # Mask out non-visible intra-row positions
    intra_block = attn[:, :, :, row_start:]  # (B, 1, row_len, row_len)
    intra_block = intra_block.masked_fill(~intra_visible.unsqueeze(0).unsqueeze(0), neg_inf)
    attn[:, :, :, row_start:] = intra_block

    return attn


class RowVerifySampler(RowParallelSampler):
    """
    Safe-first verifier:
      1) LoRA drafts a full row
      2) Verify row left-to-right in small chunks with relaxed teacher score
      3) If a chunk is rejected, DO NOT repaint / keep stale future chunks
         -> fallback to base model for the rest of this row
    """
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        image_start_token,
        image_end_token,
        image_end_line_token,
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            image_start_token=image_start_token,
            image_end_token=image_end_token,
            image_end_line_token=image_end_line_token,
        )
        self._build_token_neighbors()

    # -----------------------------
    # helpers
    # -----------------------------
    def _merge_cfg_logits(self, logits, do_cfg: bool, cfg_scale: float):
        if not do_cfg:
            return logits
        cond, uncond = logits.chunk(2, dim=0)
        return uncond + cfg_scale * (cond - uncond)

    def _sample_scores(
        self,
        scores: torch.Tensor,
        temperature: float,
        do_sample: bool,
    ) -> torch.LongTensor:
        # scores: [1, T, V]
        if do_sample:
            probs = torch.softmax(scores / temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]),
                num_samples=1,
                generator=self.generator,
            ).view(scores.shape[0], scores.shape[1])
            return sampled
        else:
            return torch.argmax(scores, dim=-1)

    def _append_tokens_with_base(
        self,
        token_sequence: torch.LongTensor,
        past_key_values,
        attention_mask: torch.Tensor,
        tokens: torch.LongTensor,   # [1, L]
        do_cfg: bool,
    ):
        if tokens.numel() == 0:
            return token_sequence, past_key_values, attention_mask

        model_input = tokens.repeat(2, 1) if do_cfg else tokens
        ones = torch.ones(
            model_input.shape[0], 1, model_input.shape[-1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, ones], dim=-1)

        with self.model.disable_adapter():
            out = self.model(
                input_ids=model_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = out.past_key_values
        token_sequence = torch.cat([token_sequence, tokens], dim=-1)
        return token_sequence, past_key_values, attention_mask

    def _base_decode_n_tokens(
        self,
        token_sequence: torch.LongTensor,
        past_key_values,
        attention_mask: torch.Tensor,
        n_tokens: int,
        logits_processor: LogitsProcessorList,
        temperature: float,
        do_sample: bool,
        do_cfg: bool,
        cfg_scale: float,
    ):
        """
        Conservative fallback: decode the next n tokens with base model only.
        """
        if n_tokens <= 0:
            return token_sequence, past_key_values, attention_mask

        input_ids = token_sequence[:, -1:]
        input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids

        with self.model.disable_adapter():
            for _ in range(n_tokens):
                ones = torch.ones(
                    input_ids.shape[0], 1, input_ids.shape[-1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                next_token, outputs = self._forward_and_sample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    temperature=temperature,
                    logits_processor=logits_processor,
                    token_sequence=token_sequence,
                    do_sample=do_sample,
                    do_cfg=do_cfg,
                    cfg_scale=cfg_scale,
                    is_prefill=False,
                )
                past_key_values = outputs.past_key_values
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token

        return token_sequence, past_key_values, attention_mask

    def _draft_full_row(
        self,
        token_sequence: torch.LongTensor,
        past_key_values,
        attention_mask: torch.Tensor,
        logits_processor: LogitsProcessorList,
        temperature: float,
        do_sample: bool,
        do_cfg: bool,
        cfg_scale: float,
        block_size: int,
        num_blocks_per_row: int,
        row_attention_mode: str,
        row_attention_window: int,
        device,
    ):
        """
        Keep your original LoRA row proposal logic, but run it on snapshots so
        the real cache/state is untouched.
        Returns: draft_row [1, img_w + 1] (image tokens + EOL)
        """
        kv_seq_len = past_key_values.get_seq_length()
        local_kv = snapshot_kv_cache(past_key_values, kv_seq_len)
        local_mask = attention_mask.clone()

        # same as your original code
        draft_input_ids = token_sequence[:, -(block_size + 1):-1]
        draft_tokens_list = []

        for blk in range(num_blocks_per_row):
            blk_input = draft_input_ids
            if blk == num_blocks_per_row - 1:
                blk_input = torch.cat([blk_input, self.image_end_line_token_tensor], dim=-1)

            model_input = blk_input.repeat(2, 1) if do_cfg else blk_input

            ones = torch.ones(
                model_input.shape[0], 1, model_input.shape[1],
                dtype=torch.long, device=device
            )
            local_mask = torch.cat([local_mask, ones], dim=-1)

            if row_attention_mode != "causal":
                mask4d = build_row_mask(
                    local_mask,
                    model_input.shape[-1],
                    mode=row_attention_mode,
                    window=row_attention_window,
                )
            else:
                mask4d = None

            out = self.model(
                input_ids=model_input,
                past_key_values=local_kv,
                attention_mask=mask4d if mask4d is not None else local_mask,
                use_cache=True,
                return_dict=True,
            )
            local_kv = out.past_key_values

            logits = self._merge_cfg_logits(out.logits, do_cfg, cfg_scale)
            scores = logits_processor(token_sequence, logits)
            draft_blk = self._sample_scores(scores, temperature, do_sample)  # [1, L]

            draft_tokens_list.append(draft_blk)
            draft_input_ids = draft_blk

        draft_row = torch.cat(draft_tokens_list, dim=-1)
        return draft_row

    def _score_chunk_relaxed(
        self,
        token_sequence: torch.LongTensor,
        past_key_values,
        attention_mask: torch.Tensor,
        chunk_tokens: torch.LongTensor,   # [1, L]
        logits_processor: LogitsProcessorList,
        do_cfg: bool,
        cfg_scale: float,
        verify_rank_k: int,
        verify_mean_logprob_thresh: float,
        verify_min_logprob_thresh: float,
        verify_topk_frac_thresh: float,
    ):
        """
        Teacher-force this chunk through BASE model on a snapshot, then score it
        with relaxed criteria instead of exact token match.
        """
        teacher_input = chunk_tokens.repeat(2, 1) if do_cfg else chunk_tokens
        local_mask = attention_mask.clone()

        ones = torch.ones(
            teacher_input.shape[0], 1, teacher_input.shape[1],
            dtype=torch.long, device=teacher_input.device
        )
        local_mask = torch.cat([local_mask, ones], dim=-1)

        with self.model.disable_adapter():
            out = self.model(
                input_ids=teacher_input,
                attention_mask=local_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = self._merge_cfg_logits(out.logits, do_cfg, cfg_scale)
        scores = logits_processor(token_sequence, logits)    # [1, L, V]
        logprob = F.log_softmax(scores, dim=-1)

        token_scores = torch.gather(
            scores, dim=-1, index=chunk_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [1, L]

        token_logprob = torch.gather(
            logprob, dim=-1, index=chunk_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [1, L]

        # rank = 1 + #classes with larger score
        ranks = (scores > token_scores.unsqueeze(-1)).sum(dim=-1) + 1  # [1, L]

        mean_logprob = token_logprob.mean().item()
        min_logprob = token_logprob.min().item()
        topk_frac = (ranks <= verify_rank_k).float().mean().item()

        accepted = (
            (mean_logprob >= verify_mean_logprob_thresh)
            and (min_logprob >= verify_min_logprob_thresh)
            and (topk_frac >= verify_topk_frac_thresh)
        )

        stats = {
            "mean_logprob": mean_logprob,
            "min_logprob": min_logprob,
            "topk_frac": topk_frac,
            "avg_rank": ranks.float().mean().item(),
            "accepted": accepted,
        }
        return accepted, stats

    def _score_chunk_group_relaxed(
        self,
        token_sequence,
        past_key_values,
        attention_mask,
        chunk_tokens,
        logits_processor,
        do_cfg,
        cfg_scale,
        teacher_topm=4,
        neighbor_topk=16,
        chunk_accept_frac=0.6,
    ):
        teacher_input = chunk_tokens.repeat(2, 1) if do_cfg else chunk_tokens
        local_mask = attention_mask.clone()

        ones = torch.ones(
            teacher_input.shape[0], 1, teacher_input.shape[1],
            dtype=torch.long, device=teacher_input.device
        )
        local_mask = torch.cat([local_mask, ones], dim=-1)

        with self.model.disable_adapter():
            out = self.model(
                input_ids=teacher_input,
                attention_mask=local_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = self._merge_cfg_logits(out.logits, do_cfg, cfg_scale)
        scores = logits_processor(token_sequence, logits)   # [1, L, V]

        teacher_topm_idx = scores.topk(k=teacher_topm, dim=-1).indices   # [1,L,M]

        accepted_pos = []
        avg_rank = []

        for i in range(chunk_tokens.shape[1]):
            cand = chunk_tokens[0, i].item()
            topm = teacher_topm_idx[0, i] -4  # [M]

            ok = False
            for t in topm.tolist():
                neigh = self.token_neighbors[t, :neighbor_topk]
                if (neigh == cand).any():
                    ok = True
                    break
            accepted_pos.append(float(ok))

            cand_score = scores[0, i, cand]
            rank = (scores[0, i] > cand_score).sum().item() + 1
            avg_rank.append(rank)

        accept_frac = sum(accepted_pos) / len(accepted_pos)
        mean_rank = sum(avg_rank) / len(avg_rank)

        accepted = accept_frac >= chunk_accept_frac
        stats = {
            "accept_frac": accept_frac,
            "avg_rank": mean_rank,
            "accepted": accepted,
        }
        return accepted, stats

    def _build_token_neighbors(self, topk=32):
        model = self.model
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model") and hasattr(model.model, "vqmodel"):
            # Lumina: VQ-VAE codebook
            W = model.model.vqmodel.quantize.embedding.weight.detach()
        elif hasattr(model, "gen_embed"):
            # Janus
            W = model.gen_embed.weight.detach()
        else:
            raise RuntimeError("Cannot locate VQ codebook or gen_embed on model")
        W = F.normalize(W, dim=-1)
        sim = W @ W.T
        self.token_neighbors = sim.topk(k=topk, dim=-1).indices

    # -----------------------------
    # main sample
    # -----------------------------
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
        ar_rows: int = 1,
        draft_use_bi_mask: bool = True,
        block_size: int = 48,                 # keep as row-draft unit (typically full row width)
        row_attention_mode: str = None,
        row_attention_window: int = 4,

        # NEW verify knobs
        verify_chunk_size: int = 4,
        verify_rank_k: int = 8,
        verify_mean_logprob_thresh: float = -3.0,
        verify_min_logprob_thresh: float = -7.0,
        verify_topk_frac_thresh: float = 0.75,
        verify_tail_fallback: str = "base",  # "base" only for now; safe-first
        verify_fallback_do_sample: bool = False,
        **kwargs
    ):
        if row_attention_mode is None:
            row_attention_mode = "full" if draft_use_bi_mask else "causal"

        anything_dict = {}
        return_anything_dict = kwargs.get("return_anything_dict", False)

        is_prefill = True
        device = input_ids.device
        prefill_length = input_ids.shape[-1]
        self._init_image_position_info()

        if seed is not None:
            set_seed(seed)
            self.generator = torch.Generator(device).manual_seed(seed)

        input_ids = input_ids.contiguous()
        token_sequence = input_ids
        past_key_values = DynamicCache() if use_cache else None

        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)

        if do_cfg:
            input_ids = input_ids.repeat(2, 1)
            attention_mask = attention_mask.repeat(2, 1, 1)
            attention_mask[1::2, :, :prefill_length - 1] = 0

        # ------------------------------------------------
        # Phase 1: AR rows with base model
        # ------------------------------------------------
        with self.model.disable_adapter():
            while True:
                next_token, outputs = self._forward_and_sample(
                    input_ids, attention_mask, past_key_values,
                    temperature, logits_processor, token_sequence,
                    do_sample, do_cfg, cfg_scale, is_prefill
                )
                past_key_values = outputs.past_key_values
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token
                ones = torch.ones(
                    input_ids.shape[0], 1, input_ids.shape[-1],
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                is_prefill = False

                img_pos_info = self._get_decoding_position(token_sequence)
                if img_pos_info["is_in_image"]:
                    if img_pos_info["is_end_of_line"] and img_pos_info["num_of_lines"] >= ar_rows:
                        break

            # fill kv for last EOL token
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

        assert self.img_h is not None and self.img_w is not None
        remain_rows = self.img_h - ar_rows
        assert self.img_w % block_size == 0, "block_size should match your trained row draft unit"
        num_blocks_per_row = self.img_w // block_size

        assert self.img_w % verify_chunk_size == 0, "verify_chunk_size should divide row width"

        verify_total = 0
        verify_accepted = 0
        verify_corrected_rows = 0
        verify_rejected_chunks = 0
        verify_accept_chunks = 0

        verify_accept_logp = []
        verify_reject_logp = []

        # ------------------------------------------------
        # Phase 2: safe-first row verify
        # ------------------------------------------------
        for row_idx in range(remain_rows):
            # 1) LoRA drafts FULL row once (same row-level draft as before)
            draft_row = self._draft_full_row(
                token_sequence=token_sequence,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                logits_processor=logits_processor,
                temperature=temperature,
                do_sample=do_sample,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                block_size=block_size,
                num_blocks_per_row=num_blocks_per_row,
                row_attention_mode=row_attention_mode,
                row_attention_window=row_attention_window,
                device=device,
            )

            draft_img = draft_row[:, :self.img_w]   # verify image tokens only
            row_had_reject = False

            # anchor the first chunk with base model, do not verify it
            anchor = verify_chunk_size
            token_sequence, past_key_values, attention_mask = self._base_decode_n_tokens(
                token_sequence=token_sequence,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                n_tokens=anchor,
                logits_processor=logits_processor,
                temperature=temperature,
                do_sample=False,   # anchor稳一点
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
            )

            # 2) Frontier chunk verify
            for start in range(anchor, self.img_w, verify_chunk_size):
                end = start + verify_chunk_size
                chunk = draft_img[:, start:end]

                kv_seq_len = past_key_values.get_seq_length()
                kv_snapshot = snapshot_kv_cache(past_key_values, kv_seq_len)
                mask_snapshot = attention_mask.clone()

                accepted, stats = self._score_chunk_group_relaxed(
                    token_sequence=token_sequence,
                    past_key_values=kv_snapshot,
                    attention_mask=mask_snapshot,
                    chunk_tokens=chunk,
                    logits_processor=logits_processor,
                    do_cfg=do_cfg,
                    cfg_scale=cfg_scale,
                    teacher_topm=16,
                    neighbor_topk=32,
                    chunk_accept_frac=0.6,
                )

                verify_total += chunk.shape[-1]

                if accepted:
                    verify_accepted += chunk.shape[-1]
                    verify_accept_chunks += 1
                    # verify_accept_logp.append(stats["mean_logprob"])

                    # commit accepted chunk with BASE model cache
                    token_sequence, past_key_values, attention_mask = self._append_tokens_with_base(
                        token_sequence=token_sequence,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        tokens=chunk,
                        do_cfg=do_cfg,
                    )
                else:
                    row_had_reject = True
                    verify_rejected_chunks += 1
                    # verify_reject_logp.append(stats["mean_logprob"])

                    # SAFE-FIRST POLICY:
                    # Once current frontier chunk is rejected,
                    # do NOT keep future stale draft chunks.
                    # Fall back to base model for the rest of this row.
                    # only fallback current chunk
                    chunk_len = end - start
                    token_sequence, past_key_values, attention_mask = self._base_decode_n_tokens(
                        token_sequence=token_sequence,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        n_tokens=chunk_len,
                        logits_processor=logits_processor,
                        temperature=temperature,
                        do_sample=False,
                        do_cfg=do_cfg,
                        cfg_scale=cfg_scale,
                    )
                
                    # then continue; later you should redraft suffix, but even this is already less brutal
                    continue
                if row_had_reject:
                    verify_corrected_rows += 1
            # 3) append fixed EOL after image tokens of this row
            token_sequence, past_key_values, attention_mask = self._append_tokens_with_base(
                token_sequence=token_sequence,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                tokens=self.image_end_line_token_tensor,   # [1,1]
                do_cfg=do_cfg,
            )

            if row_had_reject:
                logger.info(f"[verify] row {row_idx}: rejected at least one chunk, fallback to base tail")
            else:
                logger.info(f"[verify] row {row_idx}: all chunks accepted")

        # ------------------------------------------------
        # Phase 3: ending loop
        # ------------------------------------------------
        input_ids = token_sequence[:, -1:]
        input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids

        while True:
            ones = torch.ones(
                input_ids.shape[0], 1, input_ids.shape[-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, ones], dim=-1)

            with self.model.disable_adapter():
                next_token, outputs = self._forward_and_sample(
                    input_ids, attention_mask, past_key_values,
                    temperature, logits_processor, token_sequence,
                    do_sample, do_cfg, cfg_scale, is_prefill=False,
                )

            past_key_values = outputs.past_key_values
            token_sequence = torch.cat([token_sequence, next_token], dim=-1)
            input_ids = next_token.repeat(2, 1) if do_cfg else next_token

            if (next_token.item() == eos_token_id) or (token_sequence.shape[-1] == max_length):
                break

        # ------------------------------------------------
        # stats
        # ------------------------------------------------
        if verify_total > 0:
            anything_dict["verify_total_tokens"] = verify_total
            anything_dict["verify_accepted_tokens"] = verify_accepted
            anything_dict["verify_acceptance_rate"] = verify_accepted / verify_total
            anything_dict["verify_corrected_rows"] = verify_corrected_rows
            anything_dict["verify_rejected_chunks"] = verify_rejected_chunks
            anything_dict["verify_accepted_chunks"] = verify_accept_chunks

            # if len(verify_accept_logp) > 0:
            #     anything_dict["verify_accept_mean_logprob"] = float(sum(verify_accept_logp) / len(verify_accept_logp))
            # if len(verify_reject_logp) > 0:
            #     anything_dict["verify_reject_mean_logprob"] = float(sum(verify_reject_logp) / len(verify_reject_logp))

            logger.info(
                f"Verify: accepted {verify_accepted}/{verify_total} tokens "
                f"({verify_accepted/verify_total:.1%}), "
                f"accepted_chunks={verify_accept_chunks}, "
                f"rejected_chunks={verify_rejected_chunks}, "
                f"corrected_rows={verify_corrected_rows}/{remain_rows}"
            )

        if return_anything_dict:
            return token_sequence, anything_dict
        return token_sequence