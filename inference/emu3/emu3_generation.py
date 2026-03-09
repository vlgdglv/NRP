import os, sys
import argparse
import gc
import math
from PIL import Image
import PIL.Image
import torch
import time

from datetime import datetime
import random
import numpy as np

from inference.sampler import SamplerEngine, RowParallelSampler
from utils.logger import get_logger

from model.emu3_arch.mllm.processing_emu3 import Emu3Processor
from model.emu3_arch.mllm.modeling_emu3 import Emu3ForCausalLM
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
from transformers.cache_utils import DynamicCache

from utils.logger import get_logger
from utils import rollback_kv_cache

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


def unbatched_cfg_sample(
    prompt_text: str,
    neg_prompt_text: str,
    model,
    processor,
    sampler,
    device,
    cfg_guidance_scale: float,
    image_area: int,
    max_new_tokens: int,
    seed: int,
    ar_rows: int = 1,
    do_decode_image: bool = True,
    return_tokens: bool = False
):
    pos_inputs = processor(
        text=[prompt_text],
        mode="G",
        ratio=["1:1"],
        image_area=image_area,
        return_tensors="pt",
        padding="longest",
    )
    neg_inputs = processor(
        text=[neg_prompt_text],
        mode="G",
        ratio=["1:1"],
        image_area=image_area,
        return_tensors="pt",
        padding="longest",
    )

    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)

    logits_processor = LogitsProcessorList([
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            cfg_guidance_scale,
            model,
            unconditional_ids=neg_inputs.input_ids.to(device),
        ),
        PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
    ])

    # Offload: GPU -> CPU
    processor.vision_tokenizer.to("cpu")
    model.to(device)

    torch.cuda.synchronize()
    time_start = time.time()

    with torch.no_grad():
        outputs = sampler.sample(
            pos_inputs.input_ids.to(device),
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(device),
            eos_token_id=model.config.eos_token_id,
            max_length=max_new_tokens,
            do_cfg=False,
            seed=seed
        )
    
    if do_decode_image:
        # Offload: CPU -> GPU
        model.to("cpu")
        processor.vision_tokenizer.to(device)
        mm_list = processor.decode(outputs[0])

        torch.cuda.synchronize()
        time_end = time.time()

        time_uesd = time_end - time_start
        if return_tokens:
            return mm_list, time_uesd, outputs
        else:
            return mm_list, time_uesd
    else:
        torch.cuda.synchronize()
        time_end = time.time()
        time_uesd = time_end - time_start
        return outputs, time_uesd


def batched_cfg_sample(
    prompt_text: str,
    neg_prompt_text: str,
    model,
    processor,
    sampler,
    device,
    cfg_guidance_scale: float,
    image_area: int,
    pad_token_id: int,
    max_new_tokens: int,
    seed: int,
    ar_rows: int = 1,
    do_decode_image: bool = True,
    return_tokens: bool = False,
):
    pos_inputs = processor(
        text=[prompt_text],
        mode="G",
        ratio=["1:1"],
        image_area=image_area,
        return_tensors="pt",
        padding="longest",
    )
    neg_inputs = processor(
        text=[neg_prompt_text],
        mode="G",
        ratio=["1:1"],
        image_area=image_area,
        return_tensors="pt",
        padding="longest",
    )
    # print("Input pos shape: ", pos_inputs.input_ids.shape, " neg shape: ", neg_inputs.input_ids.shape)

    pos_inputs_ids, pos_mask = pos_inputs.input_ids, pos_inputs.attention_mask
    neg_inputs_ids, neg_mask = neg_inputs.input_ids, neg_inputs.attention_mask
    Bc, Lc = pos_inputs_ids.shape
    Bu, Lu = neg_inputs_ids.shape

    assert Bc == Bu == 1, "shit we can't handle batch != 1"

    L = max(Lc, Lu)
    def padding(ids, mask, length, pad_direction="right"):
        pad_len = length - ids.shape[1]
        if pad_len <= 0:
            pos = mask.long().cumsum(-1) - 1
            pos = pos.clamp(min=0)
            real_lens = mask.sum(-1)  
            return ids, mask, pos, real_lens

        if pad_direction == "left":
            ids_pad = torch.full((1, pad_len), pad_token_id, dtype=ids.dtype, device=ids.device)
            mask_pad = torch.zeros((1, pad_len), dtype=mask.dtype, device=mask.device)
            input_ids_padded = torch.cat([ids_pad, ids], dim=1)
            attention_mask_padded = torch.cat([mask_pad, mask], dim=1)
            position_ids = attention_mask_padded.long().cumsum(-1) - 1
            # pad_pos = torch.ones((1, pad_len), dtype=valid_pos.dtype, device=valid_pos.device)
            # position_ids = torch.cat([pad_pos, valid_pos], dim=1)
            position_ids = position_ids.clamp(min=0)
            real_lens = attention_mask_padded.sum(-1)
        elif pad_direction == "right":
            ids_pad = torch.full((1, pad_len), pad_token_id, dtype=ids.dtype, device=ids.device) 
            mask_pad = torch.zeros((1, pad_len), dtype=mask.dtype, device=mask.device)
            input_ids_padded = torch.cat([ids, ids_pad], dim=1)
            attention_mask_padded = torch.cat([mask, mask_pad], dim=1)
            # position_ids = torch.arange(length, dtype=torch.long, device=ids.device).unsqueeze(0)
            position_ids = attention_mask_padded.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask_padded == 0, 0) 
            position_ids = position_ids.clamp(min=0)
            real_lens = mask.sum(-1)
        else:
            raise NotImplementedError

        return input_ids_padded, attention_mask_padded, position_ids, real_lens

    pos_inputs_ids, pos_mask, pos_position_ids, pos_real_lens = padding(pos_inputs_ids, pos_mask, L)
    neg_inputs_ids, neg_mask, neg_position_ids, neg_real_lens = padding(neg_inputs_ids, neg_mask, L)
    
    batch_ids, batch_mask = torch.cat([pos_inputs_ids, neg_inputs_ids], dim=0), torch.cat([pos_mask, neg_mask], dim=0)
    batch_position_ids, batch_real_lens = torch.cat([pos_position_ids, neg_position_ids], dim=0), torch.cat([pos_real_lens, neg_real_lens], dim=0)
    # print("Input pos shape: ", batch_ids.shape, " neg shape: ", batch_mask.shape)
    # print("position_ids: ", batch_position_ids)
    # print("real_lens: ", batch_real_lens)
    # print(batch_position_ids)
    # print(batch_real_lens)
    # print(batch_mask)

    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)

    logits_processor = LogitsProcessorList([
        # UnbatchedClassifierFreeGuidanceLogitsProcessor(
        #     cfg_guidance_scale,
        #     model,
        #     unconditional_ids=neg_inputs.input_ids.to(device),
        # ),
        # SimplePrefixMaskLogitsProcessor(constrained_fn),
        BatchPrefixMaskLogitsProcessor(constrained_fn)
    ])

    # Offload: GPU -> CPU
    processor.vision_tokenizer.to("cpu")
    model.to(device)

    torch.cuda.synchronize()
    time_start = time.time()

    with torch.no_grad():
        outputs = sampler.sample(
            batch_ids.to(device),
            logits_processor=logits_processor,
            attention_mask=batch_mask.to(device),
            eos_token_id=model.config.eos_token_id,
            max_length=max_new_tokens,
            do_cfg=True,
            cfg_scale=cfg_guidance_scale,
            seed=seed,
            is_uncond_provided=True,
            position_ids=batch_position_ids.to(device),
            real_lens=batch_real_lens.to(device),
            ar_rows=ar_rows
        )
        
    if do_decode_image:
        # Offload: CPU -> GPU
        model.to("cpu")
        processor.vision_tokenizer.to(device)
        mm_list = processor.decode(outputs[0])

        torch.cuda.synchronize()
        time_end = time.time()

        time_uesd = time_end - time_start
        if return_tokens:
            return mm_list, time_uesd, outputs
        else:
            return mm_list, time_uesd
    else:
        torch.cuda.synchronize()
        time_end = time.time()
        time_uesd = time_end - time_start
        return outputs, time_uesd


class SimplePrefixMaskLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert input_ids.shape[0] == 1
        assert scores.shape[0] == 1

        sent = input_ids[0]  # [seq_len]
        mask = torch.full_like(scores, -math.inf)
        allowed_tokens = self.prefix_allowed_tokens_fn(0, sent)

        if len(allowed_tokens) == 0:
            raise ValueError(
                "`prefix_allowed_tokens_fn` returned empty list — constraint unsatisfiable."
            )

        mask[0, allowed_tokens] = 0.0
        return scores + mask


class BatchPrefixMaskLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0) 
        
        if scores.dim() == 3:
            scores = scores.squeeze(0)

        seq_len = scores.shape[0]
        
        mask = torch.full_like(scores, -math.inf)

        for i in range(seq_len):
            if i > 0:
                place_holder_token = torch.ones((i,), dtype=input_ids.dtype, device=input_ids.device)
                current_context = torch.cat([input_ids, place_holder_token], dim=-1)
            else:
                current_context = input_ids

            allowed_tokens = self.prefix_allowed_tokens_fn(0, current_context)

            if len(allowed_tokens) == 0:
                mask[i, :] = 0.0 
            else:
                mask[i, allowed_tokens] = 0.0

        return scores + mask


class Emu3Sampler(SamplerEngine):
    def __init__(self, model, tokenizer = None):
        super().__init__(model, tokenizer)

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
        position_ids = kwargs["position_ids"] if "position_ids" in kwargs else None
        # print("Attention mask in sampler: ", attention_mask)
        print("position_ids in sampler: ", position_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_hidden_states=False,
            position_ids=position_ids,
        )
        if is_multi_token:
            last_logits = outputs.logits
        else:
            last_logits = outputs.logits[:, -1, :]
        
        # print("before cfg: ", last_logits.shape)
        if do_cfg:
            cond, uncond = last_logits.chunk(2, dim=0)
            cond = torch.nn.functional.log_softmax(cond, dim=-1)
            uncond = torch.nn.functional.log_softmax(uncond, dim=-1)
            # if is_prefill:
                # last_logits = cond
            # else:
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

    
class Emu3RowParallelSampler(Emu3Sampler):
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        image_start_token,
        image_end_token,
        image_end_line_token,
        latent_width,
        latent_height,
        **kwargs
    ):
        super().__init__(model=model, tokenizer=tokenizer)
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.image_end_line_token = image_end_line_token
        self.inner_cnt = 0
        self.image_end_line_token_tensor = torch.tensor([self.image_end_line_token,], dtype=torch.long, device=model.device).unsqueeze(0)
        self.img_w = latent_width - 1
        self.img_h= latent_height
        
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
        cfg_scale: float = 3.0,
        is_uncond_provided: bool = False,
        seed: int = None,
        use_cache: bool = True,
        ar_rows: int = 12,
        parallel_as_draft: bool = False,
        draft_use_bi_mask: bool = True,
        block_size: int = 90,
        **kwargs
    ):
        is_prefill = True
        device = input_ids.device
        prefill_length = input_ids.shape[-1]
        self._init_image_position_info()

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
        
        # Prepare cfgs
        if do_cfg:
            if is_uncond_provided:
                # assert "position_ids" in kwargs, "need to provide position_ids for uncond and cond"
                pass
            else:
                input_ids = input_ids.repeat(2, 1)
                attention_mask = attention_mask.repeat(2, 1)
                attention_mask[1::2,  :prefill_length - 1] = 0

        real_lens = kwargs["real_lens"] if "real_lens" in kwargs else None
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
                    is_prefill,
                    **kwargs
                )
                past_key_values = outputs.past_key_values
                generated_hidden_states.append(outputs.logits[:, -1:, :])
                generated_tokens.append(next_token)
                
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token
                
                ones = torch.ones(input_ids.shape[0], input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                is_prefill = False if is_prefill else False

                if real_lens is not None and "position_ids" in kwargs:
                    real_lens += next_token.shape[-1]
                    position_ids = real_lens.view(real_lens.shape[0], 1) - 1 
                    kwargs["position_ids"] = position_ids
                    
                img_pos_info = self._get_decoding_position(token_sequence)

                if img_pos_info["is_in_image"]:
                    # if img_pos_info["is_end_of_line"]:
                    #     print(img_pos_info["num_of_lines"], ar_rows, token_sequence.shape)
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
        
        # assert self.img_w % block_size == 0, f"Image width {self.img_w} is not divisible by block_size"
        num_blocks_per_row = self.img_w // block_size

        for rows in range(remain_rows):
            input_ids = token_sequence[:, -(self.img_w+1):] # with eol token
            for blocks in range(num_blocks_per_row):
                input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
                ones = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=input_ids.device)
                position_ids = real_lens.view(2, 1) - 1 + torch.arange(self.img_w+1, dtype=torch.long, device=input_ids.device)
                # print("position_ids in parallel: ", position_ids)
                # print(input_ids.shape, ones.shape, attention_mask.shape)                
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
                    position_ids=position_ids,
                )
                blk_logits = blk_output.logits
                
                if not parallel_as_draft:
                    generated_hidden_states.append(blk_logits)
                    
                past_key_values = blk_output.past_key_values
                if do_cfg:
                    cond, uncond = blk_logits.chunk(2, dim=0)
                    cond = torch.nn.functional.log_softmax(cond, dim=-1)
                    uncond = torch.nn.functional.log_softmax(uncond, dim=-1)
                    # if is_prefill:
                        # last_logits = cond
                    # else:
                    blk_logits = uncond + cfg_scale * (cond - uncond)
                blk_logits  = blk_logits.squeeze(0)
                
                scores = logits_processor(token_sequence, blk_logits).unsqueeze(0)
                
                if do_sample:
                    probs = torch.nn.functional.softmax(scores / temperature, dim=-1)
                    next_blk_token = torch.multinomial(
                        probs.view(-1, probs.shape[-1]),
                        num_samples=1,
                        generator=self.generator,
                    ).view(1, -1)
                else:
                    next_blk_token = torch.argmax(scores, dim=-1, keepdim=True)

                token_sequence = torch.cat([token_sequence, next_blk_token], dim=-1)
                real_lens += next_blk_token.shape[-1]

                # input_ids = next_blk_token
        # Ending: ensure sample is finished
        input_ids = token_sequence[:, -1]
        input_ids = input_ids.repeat(2, 1) if do_cfg else input_ids
        ones = torch.ones(input_ids.shape[0],  input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)
        kwargs["position_ids"] = real_lens.view(real_lens.shape[0], 1) - 1

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
                    **kwargs,
                )
                past_key_values = outputs.past_key_values
                token_sequence = torch.cat([token_sequence, next_token], dim=-1)
                input_ids = next_token.repeat(2, 1) if do_cfg else next_token
                ones = torch.ones(input_ids.shape[0], input_ids.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                if real_lens is not None and "position_ids" in kwargs:
                    real_lens += next_token.shape[-1]
                    position_ids = real_lens.view(real_lens.shape[0], 1) - 1 
                    kwargs["position_ids"] = position_ids
                if (next_token.item() == eos_token_id) or (token_sequence.shape[-1] == max_length):
                    break
        
        # if DO_EVAL_DRAFT is not None and DO_EVAL_DRAFT and remain_rows != 0:
        #     self.inner_cnt += 1
        #     generated_hidden_states = torch.cat(generated_hidden_states, dim=1).detach().cpu()
        #     generated_tokens = torch.cat(generated_tokens, dim=-1).detach().cpu()
        #     torch.save({
        #         "tokens": generated_tokens,
        #         "hidden_states": generated_hidden_states
        #     }, f"testing_fields/data_runtime/{RT_SAVE_NAME}/sample_{self.inner_cnt}.pt")
        #     logger.info(f"Row parallel states saved at sample_{self.inner_cnt}.pt., tokens: {generated_tokens.shape}, hidden_states: {generated_hidden_states.shape}")

        return token_sequence

    def _build_row_bidirectional_mask(self, attn_mask, row_len):
        if attn_mask.dim() == 3:
            attn_mask3d = attn_mask
        else:
            attn_mask3d = attn_mask.unsqueeze(1)

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

        
            new_img_token_num = len(token_sequence[0][self.image_start_index + 1 :]) # don't forget the new generated token
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
