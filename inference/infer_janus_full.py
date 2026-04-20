import torch
import argparse
from datetime import datetime
import torch.nn.functional as F
from inference.sampler import build_row_mask
from inference.janus.generation import build_prompt, decode_image

from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
from evaluation.eval_prompt_load import prompt_loader
import numpy as np
import os, time, re

from utils.logger import get_logger
logger = get_logger(__name__)


def slugify_first_words(text: str, n=6) -> str:
    words = text.strip().split()
    base = "-".join(words[:n]).lower()
    base = re.sub(r"[^a-z0-9\-]+", "", base)
    return base or "img"


@torch.inference_mode()
def generate_ar(
    mmgpt, vl_chat_processor, prompt,
    cfg_weight=5.0, temperature=1.0, do_sample=True,
    image_width=24, image_height=24, seed=42,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokenizer = vl_chat_processor.tokenizer
    input_ids = torch.LongTensor(tokenizer.encode(prompt)).to("cuda")
    input_ids = torch.stack([input_ids, input_ids]).to("cuda")
    input_ids[1, 1:-1] = vl_chat_processor.pad_id

    input_embeddings = mmgpt.language_model.get_input_embeddings()(input_ids)
    total_tokens = image_height * image_width
    generated_tokens = torch.zeros(total_tokens, dtype=torch.int32, device="cuda")
    attention_mask = torch.ones_like(input_ids, dtype=torch.int32)

    def cfg_merge(logits):
        return logits[1::2] + cfg_weight * (logits[0::2] - logits[1::2])

    past = None
    for c in range(total_tokens):
        if c > 0:
            ones = torch.ones((2, 1), dtype=torch.int32, device="cuda")
            attention_mask = torch.cat([attention_mask, ones], dim=1)
        out = mmgpt.language_model.model(
            inputs_embeds=input_embeddings, use_cache=True,
            attention_mask=attention_mask, past_key_values=past,
        )
        past = out.past_key_values
        logits = cfg_merge(mmgpt.gen_head(out.last_hidden_state[:, -1, :]))
        probs = torch.softmax(logits / temperature, dim=-1)
        if do_sample:
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            nxt = probs.argmax(dim=-1)
        token_id = int(nxt)
        generated_tokens[c] = token_id
        tok_emb = mmgpt.prepare_gen_img_embeds(
            torch.tensor([token_id, token_id], device="cuda")
        )
        input_embeddings = tok_emb.unsqueeze(1)

    return generated_tokens


@torch.inference_mode()
def generate_row_parallel(
    mmgpt, vl_chat_processor, prompt,
    cfg_weight=5.0, temperature=1.0, do_sample=True,
    image_width=24, image_height=24, ar_rows=1, seed=42,
    row_attention_mode="full", row_attention_window=4,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokenizer = vl_chat_processor.tokenizer
    input_ids = torch.LongTensor(tokenizer.encode(prompt)).to("cuda")
    input_ids = torch.stack([input_ids, input_ids]).to("cuda")
    input_ids[1, 1:-1] = vl_chat_processor.pad_id

    input_embeddings = mmgpt.language_model.get_input_embeddings()(input_ids)
    total_tokens = image_height * image_width
    generated_tokens = torch.zeros(total_tokens, dtype=torch.int32, device="cuda")

    def cfg_merge(logits):
        return logits[1::2] + cfg_weight * (logits[0::2] - logits[1::2])

    attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
    past = None
    prev_row_embs = []

    # Phase 1: AR warmup rows
    for c in range(image_width * ar_rows):
        if c > 0:
            ones = torch.ones((2, 1), dtype=torch.int32, device="cuda")
            attention_mask = torch.cat([attention_mask, ones], dim=1)
        out = mmgpt.language_model.model(
            inputs_embeds=input_embeddings, use_cache=True,
            attention_mask=attention_mask, past_key_values=past,
        )
        past = out.past_key_values
        logits = cfg_merge(mmgpt.gen_head(out.last_hidden_state[:, -1, :]))
        probs = torch.softmax(logits / temperature, dim=-1)
        if do_sample:
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            nxt = probs.argmax(dim=-1)
        token_id = int(nxt)
        generated_tokens[c] = token_id
        tok_emb = mmgpt.prepare_gen_img_embeds(
            torch.tensor([token_id, token_id], device="cuda")
        )
        input_embeddings = tok_emb.unsqueeze(1)
        prev_row_embs.append(tok_emb[0].detach())

    prev_row_embs = prev_row_embs[-image_width:]

    # Phase 2: Row-parallel drafting
    for r in range(ar_rows, image_height):
        row_cond = torch.stack(prev_row_embs, dim=0)
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous()
        ones = torch.ones(2, image_width, dtype=torch.long, device="cuda")
        attention_mask = torch.cat([attention_mask, ones], dim=-1)

        if row_attention_mode != "causal":
            final_mask = build_row_mask(
                attention_mask, image_width,
                mode=row_attention_mode, window=row_attention_window,
            ).to(step_emb.dtype)
        else:
            final_mask = attention_mask

        out = mmgpt.language_model.model(
            inputs_embeds=step_emb, past_key_values=past,
            attention_mask=final_mask, use_cache=True,
        )
        past = out.past_key_values
        logits = cfg_merge(mmgpt.gen_head(out.last_hidden_state)).squeeze(0)
        probs = F.softmax(logits / temperature, dim=-1)
        if do_sample:
            proposal = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            proposal = probs.argmax(dim=-1)

        prev_row_embs = list(
            mmgpt.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(0)
        )
        generated_tokens[r * image_width:(r + 1) * image_width] = proposal

    return generated_tokens


image_content_prompts = [
    'A golden retriever lying peacefully on a wooden porch, with autumn leaves scattered around.',
    'A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.',
    'close-up two birds on a tree branch, background of blue sky with cumulus clouds and rising sun, 4k, realistic',
    'Three penguins in yellow construction helmets, building a sandcastle on a tropical beach, one holding a blueprint, the ocean behind them glowing in soft blue hues under the setting sun, hyperrealistic textures, playful and cinematic',
    'Deep in the jungle where a rusty robot is abndoned , 4k ,realistic, photography',
    'animation art work, A cheese burger on the sky with birds, bright, detailed',
    'Apple castle on the grass, realistic, 4k, detailed, photography',
    'A mischievous hippo playing soccer, realistic, 4k, detailed, photography',
    'Truck full of vegetables, afternoon, 4k, photography, bright color,',
    'Masterpiece, 4k, photography, bright background, market selling fresh fruits',
    'photo, photography, realistic, very detailed, Amsterdam, center fancy sports car, afternoon, realistic. sharp, bright, film grain, high contrast',
    'dystopic civilization beautiful landscape, morning, woman, very intricate, very detailed, sharp, bright, colorful',
    'A single coffee on a dinner plate on a table, 4k, detailed, photography',
    'A cat in a lab coat, standing in front of a chalkboard full of complex equations, realistic, 4k',
    'Pixel art, A mushroom kingdom, glowing, masterpiece',
    'Japanese woman in a floral-pattern summer dress sitting on an old boat beached on a tropical island, overlooking a majestic azure blue ocean with gentle waves, landscape, sunset. Impressionistic',
    'abstract oil painting, gradient vibrant neon colour, rough, textural, broad brush strokes, a sleek spaceship traversing interstellar space, detailed night sky with stars and nebulas',
    'photo, photography, Fujifilm XT-4 Viltrox, Budapest, Hungary landscape, sunset, very intricate, very detailed, realistic. sharp, bright, colorful, film grain, high contrast',
    'A stylized clay cartoon character, a small, adorable humanoid figure with a skull head, riding a miniature motorcycle., detailed',
    'animation art work, cute cat boxing with silly dog, bright',
    'Pumpkin carraige on the road, 4k, realistic, photography',
    'photography, photo of a war pilot walking to his war plane on sunset, taken from behind, 4k, realistic',
    'animation art work, huge sand castle made by dwarfs, 4k, realistic',
    '4k, realistic, photography, Giant Tree on the hill, afternoon',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="inference_outputs/janus_full")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vl_processor_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--prompt_start_idx", type=int, default=0)
    parser.add_argument("--prompt_end_idx", type=int, default=-1)
    parser.add_argument("--target_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--cfg_guidance_scale", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_decode", action="store_true")
    parser.add_argument("--row_parallel", action="store_true")
    parser.add_argument("--ar_rows", type=int, default=1)
    parser.add_argument("--infer_count", type=int, default=-1)
    parser.add_argument("--row_attention_mode", type=str, default="full",
                        choices=["full", "bidirectional_window", "causal_window", "no_intrarow"])
    parser.add_argument("--row_attention_window", type=int, default=4)
    args = parser.parse_args()

    model_path = args.model_path
    vl_processor_path = args.vl_processor_path or model_path

    if args.save_name is not None:
        folder_name = args.save_name
    else:
        folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_dir = os.path.join(args.save_dir, folder_name)

    vl_chat_processor = VLChatProcessor.from_pretrained(vl_processor_path)
    vl_gpt = MultiModalityCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    if args.prompt_path is not None:
        prompts_iter = prompt_loader(args.prompt_path, args.prompt_start_idx, args.prompt_end_idx)
    else:
        prompts_iter = image_content_prompts

    time_cost = []
    collected_images = []

    for seed in args.seeds:
        for idx, item in enumerate(prompts_iter):
            if args.infer_count > 0 and idx >= args.infer_count:
                break

            if isinstance(item, dict):
                desc = item["prompt"]
                name = str(item["name"])
            else:
                desc = item
                name = f"{idx:04d}-{slugify_first_words(desc)}"

            prompt = build_prompt(desc, vl_chat_processor)

            t0 = time.perf_counter()
            if args.row_parallel:
                token_seq = generate_row_parallel(
                    vl_gpt, vl_chat_processor, prompt,
                    cfg_weight=args.cfg_guidance_scale,
                    temperature=args.temperature,
                    ar_rows=args.ar_rows, seed=seed,
                    row_attention_mode=args.row_attention_mode,
                    row_attention_window=args.row_attention_window,
                )
            else:
                token_seq = generate_ar(
                    vl_gpt, vl_chat_processor, prompt,
                    cfg_weight=args.cfg_guidance_scale,
                    temperature=args.temperature, seed=seed,
                )
            dt = time.perf_counter() - t0

            if idx >= 2:
                time_cost.append(dt)

            paths = [None]
            if args.do_decode:
                save_name = f"{name}_seed{seed}"
                paths = decode_image(
                    vl_gpt, token_seq,
                    img_size=args.target_size, patch_size=args.patch_size,
                    save_dir=save_dir, save_name_base=save_name,
                )
            logger.info(f"[{idx}] {name}  {dt:.2f}s  | {paths[0]}")

    if time_cost:
        logger.info(f"Timing (excluding warmup): mean={np.mean(time_cost):.2f}s std={np.std(time_cost):.2f}s  ar_rows={args.ar_rows}")
