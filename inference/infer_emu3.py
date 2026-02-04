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

from inference.emu3.emu3_generation import Emu3Sampler, batched_cfg_sample, unbatched_cfg_sample
from utils.logger import get_logger

from model.emu3_arch.mllm.processing_emu3 import Emu3Processor
from model.emu3_arch.mllm.modeling_emu3 import Emu3ForCausalLM
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor


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


### Prompt for appendix - more qualitative results
image_content_prompts = [
    'A black Honda motorcycle parked in front of a garage.',
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


POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="inference_outputs/emu3")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Emu3-Gen/")
    parser.add_argument("--vq_model_path", type=str, default="/home/ffc3/bht/model_home/Emu3-VisionTokenizer/")
    parser.add_argument("--image_area", type=int, default=720*720)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--image_top_k", type=int, default=1024)
    parser.add_argument("--text_top_k", type=int, default=10)
    parser.add_argument("--cfg_guidance_scale", type=float, default=3.0)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--batched_cfg", action="store_true")
    parser.add_argument("--row_parallel", action="store_true")
    parser.add_argument("--do_warmup", type=int, default=1)
    parser.add_argument("--infer_count", type=int, default=-1, help="number of inference")
    parser.add_argument("--draft_use_causal_mask", action="store_true")
    args = parser.parse_args()
    
    emu_model_path = args.model_path
    emu_vq_model_path = args.vq_model_path
    device = "cuda:0"
    seeds = args.seeds
    max_new_tokens = args.max_new_tokens
    image_top_k = args.image_top_k
    cfg_guidance_scale = args.cfg_guidance_scale
    
    row_parallel = args.row_parallel
    lora_path = None
    if row_parallel and args.lora_path is not None:
        lora_path = args.lora_path

    # Load models
    model = Emu3ForCausalLM.from_pretrained(
        emu_model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", #"flash_attention_2" if not args.batched_cfg else "eager",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(emu_model_path, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(emu_vq_model_path, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(emu_vq_model_path, device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    image_area = args.image_area
    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=image_top_k,
    )

    sampler = Emu3Sampler(model, tokenizer)

    if args.save_name is not None:
        folder_name = args.save_name
    else:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        folder_name = f"{current_time}"
    save_dir = os.path.join(args.save_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    collected_images = []
    time_collector = []
    logger.info("start inference ...")
    for seed in seeds:
        set_seed(seed)
        for i, q_image_content_condition in enumerate(image_content_prompts):
            if args.infer_count > 0 and i >= args.infer_count:
                break

            prompt_text = q_image_content_condition + POSITIVE_PROMPT
            
            if args.batched_cfg:
                mm_list, time_uesd = batched_cfg_sample(
                    prompt_text,
                    NEGATIVE_PROMPT,
                    model,
                    processor,
                    sampler,
                    device,
                    cfg_guidance_scale,
                    image_area,
                    pad_token_id=model.config.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    seed=seed
                )
            else:
                mm_list, time_uesd = unbatched_cfg_sample(
                    prompt_text,
                    NEGATIVE_PROMPT,
                    model,
                    processor,
                    sampler,
                    device,
                    cfg_guidance_scale,
                    image_area,
                    max_new_tokens=max_new_tokens,
                    seed=seed
                )

            for img in mm_list:
                if isinstance(img, Image.Image):
                    result_image = img
                    break
            
            time_collector.append(time_uesd)
            logger.info(f"Image {i} generation time elapsed: {time_uesd:.2f} s")

            output_file_name = 'img_' + str(i) +'_seed' + str(seed) + ".png"
            if args.infer_count > 0:
                collected_images.append(np.array(result_image))
            else:
                result_image.save(os.path.join(save_dir, output_file_name))
                logger.info(f"Saved {output_file_name} to {save_dir}") # <|image|>
    
    if args.infer_count > 0 and len(collected_images) > 0:
        row_image = np.concatenate(collected_images, axis=1)
        output_file_name = 'img_' + str(args.infer_count) + "samples" +'_seed' + str(seed) + ".png"
        PIL.Image.fromarray(row_image).save(os.path.join(save_dir, output_file_name))
        logger.info(f"Saved {len(collected_images)} samples to {output_file_name} to {save_dir}")

    logger.info(f"Average time: {(sum(time_collector) / len(time_collector)):.2f} seconds")
    gc.collect()
