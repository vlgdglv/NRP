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

from inference.sampler import SamplerEngine
from utils.logger import get_logger

from model.emu3_arch.mllm.processing_emu3 import Emu3Processor
from model.emu3_arch.mllm.modeling_emu3 import Emu3ForCausalLM
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor


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
        SimplePrefixMaskLogitsProcessor(constrained_fn),
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
        # print("position_ids in sampler: ", position_ids)
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

        # print(token_sequence.shape)
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