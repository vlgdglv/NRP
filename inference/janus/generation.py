import torch
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from inference.sampler import build_row_bidirectional_mask, build_row_mask
from utils import rollback_kv_cache

from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os, time
import PIL.Image

def build_prompt(desc: str, vl_chat_processor: VLChatProcessor) -> str:
    conv = [{"role": "User", "content": desc}, {"role": "Assistant", "content": ""}]
    s = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return s + vl_chat_processor.image_start_tag


@torch.inference_mode()
def decode_image(
    mmgpt, 
    generated_tokens,
    parallel_size: int = 1,
    img_size: int = 384,
    patch_size: int = 16,
    save_dir: str = None,
    save_name_base: str | None = None    
):
    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for i in range(parallel_size):
        name = save_name_base or "img"
        save_path = os.path.join(save_dir, f"{name}.jpg")
        PIL.Image.fromarray(dec[i]).save(save_path)
        saved_paths.append(save_path)
    return saved_paths


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    seed: int = 42,
    include_prefill: bool = False,
    do_sample: bool = True,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    
    if include_prefill:
        # print(input_ids.shape, generated_tokens.shape)
        full_tokens = torch.cat([input_ids.unsqueeze(dim=0).to(generated_tokens), generated_tokens], dim=1)
        return generated_tokens, full_tokens 
    else:
        return generated_tokens



@torch.inference_mode()
def gererate_row_parallel(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    cfg_weight=5.0, temperature=1.0,
    do_sample=True,
    image_width: int = 24,
    image_height: int = 24,
    ar_rows: int = 1,
    seed: int = 42,
    use_bi_mask: bool = True,
    row_attention_mode: str = None,
    row_attention_window: int = 4,
):
    # row_attention_mode overrides use_bi_mask for backward compat
    if row_attention_mode is None:
        row_attention_mode = "full" if use_bi_mask else "causal"
    generator = torch.Generator(device="cuda").manual_seed(seed)

    tokenizer = vl_chat_processor.tokenizer
    input_ids = torch.LongTensor(tokenizer.encode(prompt)).to("cuda")

    input_ids = torch.stack([input_ids, input_ids]).to("cuda")
    input_ids[1, 1:-1] = vl_chat_processor.pad_id

    input_embeddings = mmgpt.language_model.get_input_embeddings()(input_ids)  # [2, Lp, D]
    L_prefill = input_embeddings.shape[1]

    total_tokens = image_height * image_width
    gt_tokens, gt_logits = [], []

    generated_tokens = torch.zeros((total_tokens), dtype=torch.int32, device="cuda")

    def cfg_merge(logits, cfg_weight):
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        return logit_uncond + cfg_weight * (logit_cond - logit_uncond)

    attention_mask = torch.full_like(input_ids, 1, dtype=torch.int32).to("cuda")

    past = None
    pos_cur = L_prefill

    prev_row_embs = []
    with mmgpt.disable_adapter():
        for c in range(image_width * ar_rows):
            if c > 0:
                ones = torch.ones((input_embeddings.shape[0], 1), dtype=torch.int32, device="cuda")
                attention_mask = torch.cat([attention_mask, ones], dim=1)

            out = mmgpt.language_model.model(
                inputs_embeds=input_embeddings,
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past,
            )
            hidden_states, past = out.last_hidden_state, out.past_key_values
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logits = cfg_merge(logits, cfg_weight)
            probs = torch.softmax(logits / temperature, dim=-1)
            if do_sample:    
                nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)
            else:
                nxt = probs.argmax(dim=-1, keepdim=True)
            
            token_id = int(nxt)
            generated_tokens[c] = token_id
            gt_tokens.append(token_id)

            tok_emb = mmgpt.prepare_gen_img_embeds(torch.tensor([token_id, token_id], device="cuda").view(-1))       # [2, D]
            input_embeddings = tok_emb.unsqueeze(1)                      # [2,1,D]

            prev_row_embs.append(tok_emb[0].detach())
            pos_cur += 1

    prev_row_embs = prev_row_embs[-image_width:]
    # prev_row_embs.insert(0, prev_row_embs.pop())

    device = next(mmgpt.language_model.parameters()).device
    
    for r in range(ar_rows, image_height):
        row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
        # pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
        ones = torch.ones(attention_mask.shape[0], step_emb.shape[1], dtype=torch.long, device=input_ids.device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)
        if row_attention_mode != "causal":
            final_mask = build_row_mask(attention_mask, step_emb.shape[1],
                                        mode=row_attention_mode, window=row_attention_window).to(step_emb.dtype)
        else:
            final_mask = attention_mask
        out_prop = mmgpt.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past,
            attention_mask=final_mask,
            use_cache=True
        )
        log_q = cfg_merge(mmgpt.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
        q_probs  = F.softmax(log_q / (temperature), dim=-1)
        if do_sample:      # [W, V]
            proposal = torch.multinomial(q_probs, 1, generator=generator).squeeze(-1)      # [W]
        else:
            proposal = q_probs.argmax(dim=-1)

        prev_row_embs = list(mmgpt.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
        final_row = proposal.clone()

        for c in range(image_width):
            generated_tokens[r * image_width + c] = int(final_row[c])

    return generated_tokens


@torch.inference_mode()
def gererate_row_parallel_with_probe(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    cfg_weight=5.0, temperature=1.0,
    do_sample=True,
    image_width: int = 24,
    image_height: int = 24,
    ar_rows: int = 1,
    seed: int = 42,
    use_bi_mask: bool = True,
    row_attention_mode: str = None,
    row_attention_window: int = 4,
):
    # row_attention_mode overrides use_bi_mask for backward compat
    if row_attention_mode is None:
        row_attention_mode = "full" if use_bi_mask else "causal"
    generator = torch.Generator(device="cuda").manual_seed(seed)

    tokenizer = vl_chat_processor.tokenizer
    input_ids = torch.LongTensor(tokenizer.encode(prompt)).to("cuda")

    input_ids = torch.stack([input_ids, input_ids]).to("cuda")
    input_ids[1, 1:-1] = vl_chat_processor.pad_id

    input_embeddings = mmgpt.language_model.get_input_embeddings()(input_ids)  # [2, Lp, D]
    L_prefill = input_embeddings.shape[1]

    total_tokens = image_height * image_width
    gt_tokens, gt_logits = [], []

    generated_tokens = torch.zeros((total_tokens), dtype=torch.int32, device="cuda")

    def cfg_merge(logits, cfg_weight):
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        return logit_uncond + cfg_weight * (logit_cond - logit_uncond)

    attention_mask = torch.full_like(input_ids, 1, dtype=torch.int32).to("cuda")

    past = None
    pos_cur = L_prefill
    
    anything_dict = {}

    prev_row_embs = []
    with mmgpt.disable_adapter():
        for c in range(image_width * ar_rows):
            if c > 0:
                ones = torch.ones((input_embeddings.shape[0], 1), dtype=torch.int32, device="cuda")
                attention_mask = torch.cat([attention_mask, ones], dim=1)
            out = mmgpt.language_model.model(
                inputs_embeds=input_embeddings,
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past,
            )
            hidden_states, past = out.last_hidden_state, out.past_key_values
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logits = cfg_merge(logits, cfg_weight)
            probs = torch.softmax(logits / temperature, dim=-1)
            if do_sample:    
                nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)
            else:
                nxt = probs.argmax(dim=-1, keepdim=True)
            
            token_id = int(nxt)
            generated_tokens[c] = token_id
            gt_tokens.append(token_id)

            tok_emb = mmgpt.prepare_gen_img_embeds(torch.tensor([token_id, token_id], device="cuda").view(-1))       # [2, D]
            input_embeddings = tok_emb.unsqueeze(1)                      # [2,1,D]            
            prev_row_embs.append(tok_emb[0].detach())

            pos_cur += 1

    prev_row_embs = prev_row_embs[-image_width:]
    # prev_row_embs.insert(0, prev_row_embs.pop())

    device = next(mmgpt.language_model.parameters()).device
    
    gt_ranks, draft_ranks = [], []
    tv_distances = []
    for r in range(ar_rows, image_height):
        row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
        ones = torch.ones(attention_mask.shape[0], step_emb.shape[1], dtype=torch.long, device=input_ids.device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)
        if row_attention_mode != "causal":
            final_mask = build_row_mask(attention_mask, step_emb.shape[1],
                                        mode=row_attention_mode, window=row_attention_window).to(step_emb.dtype)
        else:
            final_mask = attention_mask
        out_prop = mmgpt.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past,
            attention_mask=final_mask,
            use_cache=True
        )
        log_q = cfg_merge(mmgpt.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)

        q_probs  = F.softmax(log_q / (temperature), dim=-1)   

        if do_sample:      # [W, V]
            proposal = torch.multinomial(q_probs, 1, generator=generator).squeeze(-1)      # [W]
        else:
            proposal = q_probs.argmax(dim=-1)
        
        prev_row_embs = []
        rollback_kv_cache(past, step_emb.shape[1])
        attention_mask = attention_mask[..., :-step_emb.shape[1]]
        one_step_token = token_id
        one_step_emb = mmgpt.prepare_gen_img_embeds(torch.tensor([one_step_token, one_step_token], device="cuda").view(-1)).unsqueeze(1)       # [2, 1, D]

        gt_tokens, gt_scores = [],[]
        with mmgpt.disable_adapter():
            for iii in range(image_width):
                ones = torch.ones(attention_mask.shape[0],  one_step_emb.shape[1], dtype=torch.long, device=input_ids.device)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                out = mmgpt.language_model.model(
                    inputs_embeds=one_step_emb,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past,
                )
                hidden_states, past = out.last_hidden_state, out.past_key_values
                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logits = cfg_merge(logits, cfg_weight)
                probs = torch.softmax(logits / temperature, dim=-1)
                if do_sample:    
                    nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)
                else:
                    nxt = probs.argmax(dim=-1, keepdim=True)
            
                gt_tokens.append(nxt)
                gt_scores.append(probs)

                token_id = int(nxt)
                generated_tokens[r * image_width + iii] = token_id

                tok_emb = mmgpt.prepare_gen_img_embeds(torch.tensor([token_id, token_id], device="cuda").view(-1))       # [2, D]
                one_step_emb = tok_emb.unsqueeze(1)                      # [2,1,D]            
                prev_row_embs.append(tok_emb[0].detach())
        
        gt_tokens = torch.cat(gt_tokens, dim=-1)
        gt_scores = torch.cat(gt_scores, dim=0)
        p_draft = q_probs.view(-1, q_probs.shape[-1])
        p_target = gt_scores.view(-1, gt_scores.shape[-1])
        
        gt_rank = (q_probs.squeeze(0).argsort(dim=-1, descending=True) == gt_tokens.view(-1, 1)).nonzero()[:,1]
        draft_rank = (gt_scores.argsort(dim=-1, descending=True) == proposal.view(-1, 1)).nonzero()[:,1]
        gt_ranks.append(gt_rank)
        draft_ranks.append(draft_rank)
        
        tv_dist = 0.5 * torch.abs(p_draft - p_target).sum(dim=-1)
        tv_distances.append(tv_dist)

    gt_ranks = torch.cat(gt_ranks, dim=-1).to(torch.float)
    # print("Mean gt rank: {:.4f}, Std: {:.4f}".format(gt_ranks.mean(), gt_ranks.std()))
    draft_ranks = torch.cat(draft_ranks, dim=-1).to(torch.float)
    # print("Mean draft rank: {:.4f}, Std: {:.4f}".format(draft_ranks.mean(), draft_ranks.std()))
    tv_distances = torch.cat(tv_distances, dim=-1).to(torch.float)
    
    anything_dict["gt_ranks.mean"] = gt_ranks.mean()
    anything_dict["gt_ranks.std"] = gt_ranks.std()
    anything_dict["draft_ranks.mean"] = draft_ranks.mean()
    anything_dict["draft_ranks.std"] = draft_ranks.std()
    anything_dict["tv_distance.mean"] = tv_distances.mean()
    anything_dict["tv_distance.std"] = tv_distances.std()
        # prev_row_embs = list(mmgpt.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
        # final_row = proposal.clone()
    
        # for c in range(image_width):
        #     generated_tokens[r * image_width + c] = int(final_row[c])

    return generated_tokens, anything_dict

@torch.inference_mode()
def gererate_row_parallel_tinyar(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    tinyar_head,
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
    do_sample: bool = True,
    image_width: int = 24,
    image_height: int = 24,
    ar_rows: int = 1,
    seed: int = 42,
    row_attention_mode: str = None,
    row_attention_window: int = 4,
):
    """Row-parallel sampling with TinyAR within-row coupling.

    Per row (after AR warmup):
      1. Backbone parallel forward over the W positions given previous row as input
         -> hidden states h_r[0..W-1] at the current row positions.
      2. TinyAR sequential rollout of W steps, conditioned on h_r and previously
         generated tokens within this row. Much cheaper than backbone (1-layer head).
      3. Feed generated tokens back as next row's input via prepare_gen_img_embeds.
    """
    from model.tinyar import get_output_head

    if row_attention_mode is None:
        row_attention_mode = "full"
    generator = torch.Generator(device="cuda").manual_seed(seed)

    tokenizer = vl_chat_processor.tokenizer
    input_ids = torch.LongTensor(tokenizer.encode(prompt)).to("cuda")
    input_ids = torch.stack([input_ids, input_ids]).to("cuda")
    input_ids[1, 1:-1] = vl_chat_processor.pad_id

    input_embeddings = mmgpt.language_model.get_input_embeddings()(input_ids)
    total_tokens = image_height * image_width
    generated_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device="cuda")

    def cfg_merge(logits):
        cond = logits[0::2, :]
        uncond = logits[1::2, :]
        return uncond + cfg_weight * (cond - uncond)

    attention_mask = torch.full_like(input_ids, 1, dtype=torch.int32)
    past = None

    out_head = get_output_head(mmgpt)
    prev_row_embs = []

    # Phase 1: AR warmup (same as the standard sampler)
    for c in range(image_width * ar_rows):
        if c > 0:
            ones = torch.ones((input_embeddings.shape[0], 1), dtype=torch.int32, device="cuda")
            attention_mask = torch.cat([attention_mask, ones], dim=1)
        out = mmgpt.language_model.model(
            inputs_embeds=input_embeddings,
            use_cache=True, attention_mask=attention_mask, past_key_values=past,
        )
        past = out.past_key_values
        logits = cfg_merge(mmgpt.gen_head(out.last_hidden_state[:, -1, :]))
        probs = torch.softmax(logits / temperature, dim=-1)
        if do_sample:
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(0)
        else:
            nxt = probs.argmax(dim=-1, keepdim=True)
        token_id = int(nxt)
        generated_tokens[c] = token_id
        tok_emb = mmgpt.prepare_gen_img_embeds(
            torch.tensor([token_id, token_id], device="cuda")
        )
        input_embeddings = tok_emb.unsqueeze(1)
        prev_row_embs.append(tok_emb[0].detach())

    prev_row_embs = prev_row_embs[-image_width:]

    # Phase 2: row-parallel + tinyAR
    W = image_width
    device = next(mmgpt.language_model.parameters()).device
    dtype = next(mmgpt.language_model.parameters()).dtype

    for r in range(ar_rows, image_height):
        # 2a. Backbone parallel forward for this row
        row_cond = torch.stack(prev_row_embs, dim=0)                    # [W, D]
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
        ones = torch.ones(attention_mask.shape[0], W, dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)

        if row_attention_mode != "causal":
            final_mask = build_row_mask(
                attention_mask, W, mode=row_attention_mode, window=row_attention_window,
            ).to(step_emb.dtype)
        else:
            final_mask = attention_mask

        out_row = mmgpt.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past, attention_mask=final_mask, use_cache=True,
        )
        past = out_row.past_key_values
        row_hidden = out_row.last_hidden_state                          # [2, W, D]
        # Collapse CFG later at tinyAR logit level; keep both pathways in tinyAR too.
        h_cond = row_hidden[0:1]                                         # [1, W, D]
        h_uncond = row_hidden[1:2]                                       # [1, W, D]

        # 2b. TinyAR sequential rollout (head is d_trunk-dim; O(W^2) recompute per row)
        D = h_cond.shape[-1]
        # Position-0 placeholder: zeros (head overwrites with learned BOS internally).
        prev_cond = torch.zeros(1, 1, D, device=device, dtype=dtype)
        prev_uncond = torch.zeros(1, 1, D, device=device, dtype=dtype)

        new_prev_row_embs = []
        for c in range(W):
            head_h_cond = tinyar_head.step(h_cond, prev_cond)          # [1, 1, D]
            head_h_uncond = tinyar_head.step(h_uncond, prev_uncond)    # [1, 1, D]

            logit_cond = out_head(head_h_cond[:, -1, :])               # [1, V]
            logit_uncond = out_head(head_h_uncond[:, -1, :])
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            probs = torch.softmax(logits / temperature, dim=-1)
            if do_sample:
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            else:
                nxt = probs.argmax(dim=-1)
            tid = int(nxt)
            generated_tokens[r * W + c] = tid

            tok_emb_full = mmgpt.prepare_gen_img_embeds(
                torch.tensor([tid], device=device)
            ).to(dtype=dtype)                                          # [1, D]
            tok_emb_step = tok_emb_full.unsqueeze(0)                   # [1, 1, D]
            prev_cond = torch.cat([prev_cond, tok_emb_step], dim=1)
            prev_uncond = torch.cat([prev_uncond, tok_emb_step], dim=1)
            new_prev_row_embs.append(tok_emb_full[0])

        prev_row_embs = new_prev_row_embs

    return generated_tokens
