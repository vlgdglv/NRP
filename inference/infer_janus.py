
import torch
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM

from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os, time
import PIL.Image

from utils.logger import get_logger
logger = get_logger(__name__)


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
    'a_skynet_cyberdyne_craft, the image is featuring a futuristic, highly advanced jet fighter drone flying rapidly at altitude thporugh stormclouds, silhouetted, chiascuro, sunset., realistic, 4k',

    'abstract oil painting, gradient vibrant neon colour, rough, textural, broad brush strokes, a sleek spaceship traversing interstellar space, detailed night sky with stars and nebulas',
    'photo, photography, Fujifilm XT-4 Viltrox, Budapest, Hungary landscape, sunset, very intricate, very detailed, realistic. sharp, bright, colorful, film grain, high contrast',
    'A stylized clay cartoon character, a small, adorable humanoid figure with a skull head, riding a miniature motorcycle., detailed',
    'animation art work, cute cat boxing with silly dog, bright',
    'Pumpkin carraige on the road, 4k, realistic, photography',
    'photography, photo of a war pilot walking to his war plane on sunset, taken from behind, 4k, realistic',

    'animation art work, huge sand castle made by dwarfs, 4k, realistic',
    '4k, realistic, photography, Giant Tree on the hill, afternoon',
]

def build_prompt(desc: str) -> str:
    conv = [{"role": "User", "content": desc}, {"role": "Assistant", "content": ""}]
    s = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return s + vl_chat_processor.image_start_tag

def slugify_first_words(text: str, n=6) -> str:
    import re
    words = text.strip().split()
    base = "-".join(words[:n]).lower()
    base = re.sub(r"[^a-z0-9\-]+", "", base)
    return base or "img"

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    save_dir: str = None,
    save_name_base: str | None = None,
):
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

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for i in range(parallel_size):
        name = save_name_base or "img"
        save_path = os.path.join(save_dir, f"{name}_{i}.jpg")
        PIL.Image.fromarray(dec[i]).save(save_path)
        saved_paths.append(save_path)
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="inference_outputs/janus")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Janus-Pro-7B/")
    parser.add_argument("--vl_processor_path", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_token_num_per_image", type=int, default=576)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--image_top_k", type=int, default=2000)
    parser.add_argument("--text_top_k", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=48)
    parser.add_argument("--cfg_guidance_scale", type=float, default=5.0)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--row_parallel", action="store_true")
    parser.add_argument("--do_warmup", type=int, default=1)
    parser.add_argument("--infer_count", type=int, default=-1, help="number of inference")
    parser.add_argument("--draft_use_causal_mask", action="store_true")
    args = parser.parse_args()
    
    model_path = args.model_path
    vl_processor_path = args.vl_processor_path if args.vl_processor_path is not None else model_path
    target_size = args.target_size
    image_token_num_per_image = args.image_token_num_per_image
    patch_size = args.patch_size

    if args.save_name is not None:
        folder_name = args.save_name
    else:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        folder_name = f"{current_time}"
    save_dir = os.path.join(args.save_dir, folder_name)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(vl_processor_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    gen_kwargs = dict(
        parallel_size=1,
        image_token_num_per_image=image_token_num_per_image,
        img_size=target_size,
        patch_size=patch_size,
    )

    for idx, desc in enumerate(image_content_prompts):
        prompt = build_prompt(desc)
        base = slugify_first_words(desc, n=6)

        t0 = time.perf_counter()
        paths = generate(vl_gpt, vl_chat_processor, prompt, save_dir=save_dir,save_name_base=f"{idx:02d}-{base}", **gen_kwargs)
        dt = time.perf_counter() - t0

        logger.info(f"[{idx}] {base}  ->  {dt:.2f}s  | saved: {paths[0]}")
