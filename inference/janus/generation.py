import torch
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F


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
    include_prefill: bool = False
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

        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    
    if include_prefill:
        print(input_ids.shape, generated_tokens.shape)
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
):
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

    past = None
    pos_cur = L_prefill

    prev_row_embs = []
    for c in range(image_width * ar_rows):
        out = mmgpt.language_model.model(
            inputs_embeds=input_embeddings,
            use_cache=True,
            past_key_values=past,
        )
        hidden_states, past = out.last_hidden_state, out.past_key_values
        # log = cfg_merge(mmgpt.gen_head(last_hidden_state[:, -1, :]), cfg_weight).squeeze(0) # [V]
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
        # prev_row_embs.append(tok_emb[0].detach())

        pos_cur += 1

    prev_row_embs = prev_row_embs[-image_width:]
    # prev_row_embs.insert(0, prev_row_embs.pop())

    device = next(mmgpt.language_model.parameters()).device
    
    for r in range(ar_rows, image_height):
        # T_k = past[0][0].shape[2]
        # pos_base = L_prefill + r * image_width
        # Just use default attention mask
        # attn_mask = torch.full((2, 1, W, K_total), torch.finfo(torch.float16).min, device=device) #
        # for c in range(W):
        #     vis = _build_visible_indices_for_column(L_prefill, W, r, c, mask_policy, neighbor_k)
        #     if len(vis):
        #         idx = torch.as_tensor(vis, device=device, dtype=torch.long)
        #         attn_mask[0, 0, c, idx] = 0.0
        #         attn_mask[1, 0, c, idx] = 0.0
        
        row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
        # pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
        out_prop = mmgpt.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past,
            # position_ids=pos_ids,
            use_cache=True
        )
        log_q = cfg_merge(mmgpt.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)

        q_probs  = F.softmax(log_q / (temperature), dim=-1)   
        # for prob in q_probs:
        #     probs_all.append(prob.float().detach().cpu())

        if do_sample:      # [W, V]
            proposal = torch.multinomial(q_probs, 1, generator=generator).squeeze(-1)      # [W]
        else:
            proposal = q_probs.argmax(dim=-1)
        
        prev_row_embs = list(mmgpt.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
        final_row = proposal.clone()
    
        for c in range(image_width):
            generated_tokens[r * image_width + c] = int(final_row[c])

        prev_row_embs.insert(0, prev_row_embs.pop())

    return generated_tokens