import os
import time
import torch
import random
import numpy as np

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache
# from peft import set_adapter, disable_adapter

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
        block_size: int = 48,
        row_attention_mode: str = None,
        row_attention_window: int = 4,
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

        # ── Phase 1: AR rows with base model ──
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
                ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                is_prefill = False
                img_pos_info = self._get_decoding_position(token_sequence)
                if img_pos_info["is_in_image"]:
                    if img_pos_info["is_end_of_line"] and img_pos_info["num_of_lines"] >= ar_rows:
                        break

            # fill kv cache for the last EOL token
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True, return_dict=True,
            )
            past_key_values = outputs.past_key_values

        assert self.img_h is not None and self.img_w is not None
        remain_rows = self.img_h - ar_rows
        assert self.img_w % block_size == 0
        num_blocks_per_row = self.img_w // block_size

        verify_total = 0
        verify_accepted = 0
        verify_corrected_rows = 0

        # ── Phase 2: Verify-and-correct loop ──
        for row_idx in range(remain_rows):
            kv_seq_len = past_key_values.get_seq_length()
            kv_snapshot = snapshot_kv_cache(past_key_values, kv_seq_len)
            mask_snapshot = attention_mask.clone()

            # 2a. LoRA proposes full row (block by block, same as parent)
            draft_input_ids = token_sequence[:, -(block_size + 1):-1]
            draft_tokens_list = []

            for blk in range(num_blocks_per_row):
                blk_input = draft_input_ids
                if blk == num_blocks_per_row - 1:
                    blk_input = torch.cat([blk_input, self.image_end_line_token_tensor], dim=-1)
                blk_input = blk_input.repeat(2, 1) if do_cfg else blk_input

                ones = torch.ones(blk_input.shape[0], 1, blk_input.shape[1], dtype=torch.long, device=device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                if row_attention_mode != "causal":
                    mask4d = build_row_mask(attention_mask, blk_input.shape[-1],
                                            mode=row_attention_mode, window=row_attention_window)
                else:
                    mask4d = None

                blk_out = self.model(
                    input_ids=blk_input,
                    past_key_values=past_key_values,
                    attention_mask=mask4d if mask4d is not None else attention_mask,
                    use_cache=True, return_dict=True,
                )
                past_key_values = blk_out.past_key_values
                blk_logits = blk_out.logits
                if do_cfg:
                    cond, uncond = blk_logits.chunk(2, dim=0)
                    blk_logits = uncond + cfg_scale * (cond - uncond)
                scores = logits_processor(token_sequence, blk_logits)

                if do_sample:
                    probs = torch.nn.functional.softmax(scores / temperature, dim=-1)
                    draft_blk = torch.multinomial(
                        probs.view(-1, probs.shape[-1]), num_samples=1,
                        generator=self.generator,
                    ).view(1, -1)
                else:
                    draft_blk = torch.argmax(scores, dim=-1, keepdim=True).view(1, -1)

                draft_tokens_list.append(draft_blk)
                draft_input_ids = draft_blk

            # draft_row: [1, img_w + 1] (image tokens + EOL)
            draft_row = torch.cat(draft_tokens_list, dim=-1)

            # 2b. Rollback KV to snapshot for teacher-force
            past_key_values = kv_snapshot
            attention_mask = mask_snapshot

            # 2c. Teacher-force draft through base model (causal)
            teacher_input = draft_row.repeat(2, 1) if do_cfg else draft_row
            ones = torch.ones(teacher_input.shape[0], 1, teacher_input.shape[1], dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, ones], dim=-1)

            with self.model.disable_adapter():
                teacher_out = self.model(
                    input_ids=teacher_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True, return_dict=True,
                )
            past_key_values = teacher_out.past_key_values
            teacher_logits = teacher_out.logits
            if do_cfg:
                cond, uncond = teacher_logits.chunk(2, dim=0)
                teacher_logits = uncond + cfg_scale * (cond - uncond)
            teacher_scores = logits_processor(token_sequence, teacher_logits)
            teacher_choice = torch.argmax(teacher_scores, dim=-1)  # [1, W+1]

            # 2d. Compare and correct (don't touch the EOL at the end)
            corrected = draft_row.clone()
            n_corrected = 0
            row_img_len = self.img_w  # only compare image tokens, not EOL
            for c in range(row_img_len):
                if teacher_choice[0, c] != draft_row[0, c]:
                    corrected[0, c] = teacher_choice[0, c]
                    n_corrected += 1

            verify_total += row_img_len
            verify_accepted += (row_img_len - n_corrected)
            if n_corrected > 0:
                verify_corrected_rows += 1

            # 2e. If any correction, rebuild KV cache with corrected tokens
            if n_corrected > 0:
                rollback_kv_cache(past_key_values, draft_row.shape[-1])
                attention_mask = attention_mask[..., :-draft_row.shape[-1]]

                corrected_input = corrected.repeat(2, 1) if do_cfg else corrected
                ones = torch.ones(corrected_input.shape[0], 1, corrected_input.shape[1], dtype=torch.long, device=device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                with self.model.disable_adapter():
                    rebuild_out = self.model(
                        input_ids=corrected_input,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True, return_dict=True,
                    )
                past_key_values = rebuild_out.past_key_values

            token_sequence = torch.cat([token_sequence, corrected], dim=-1)

        # ── Phase 3: Ending loop ──
        input_ids = token_sequence[:, -1:]
        input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
        ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)

        with self.model.disable_adapter():
            while True:
                next_token, outputs = self._forward_and_sample(
                    input_ids, attention_mask, past_key_values,
                    temperature, logits_processor, token_sequence,
                    do_sample, do_cfg, cfg_scale, is_prefill=False,
                )
                past_key_values = outputs.past_key_values
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token
                ones = torch.ones(input_ids.shape[0], 1, input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                if (next_token.item() == eos_token_id) or (token_sequence.shape[-1] == max_length):
                    break

        if verify_total > 0:
            anything_dict["verify_total_tokens"] = verify_total
            anything_dict["verify_accepted_tokens"] = verify_accepted
            anything_dict["verify_acceptance_rate"] = verify_accepted / verify_total
            anything_dict["verify_corrected_rows"] = verify_corrected_rows
            logger.info(
                f"Verify: {verify_accepted}/{verify_total} accepted "
                f"({verify_accepted/verify_total:.1%}), "
                f"{verify_corrected_rows}/{remain_rows} rows corrected"
            )

        if return_anything_dict:
            return token_sequence, anything_dict
        return token_sequence
