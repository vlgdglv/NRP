import argparse
import os
import sys
import gc
from PIL import Image
import PIL.Image
import torch
import time

from datetime import datetime
import random
import numpy as np


from inference.lumina.inference_solver import FlexARInferenceSolver


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="inference_outputs/lumina")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Lumina-mGPT-7B-768")
    parser.add_argument("--vae_tokenizer_path", type=str, default="/home/ffc3/bht/GSD/ckpts/chameleon/tokenizer")
    parser.add_argument("--target_size", type=int, default=768)
    parser.add_argument("--target_size_h", type=int, default=768)
    parser.add_argument("--target_size_w", type=int, default=768)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--image_top_k", type=int, default=2000)
    parser.add_argument("--text_top_k", type=int, default=10)
    parser.add_argument("--cfg_guidance_scale", type=float, default=3.0)
    args = parser.parse_args()
    
    model_path = args.model_path
    vae_tokenizer_path = args.vae_tokenizer_path
    target_size = args.target_size
    target_size_h, target_size_w = args.target_size_h, args.target_size
    device = "cuda:0"
    seeds = args.seeds
    image_top_k = args.image_top_k
    text_top_k = args.text_top_k
    cfg_guidance_scale = args.cfg_guidance_scale

    template_condition_sentences = f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n"

    inference_solver = FlexARInferenceSolver(
        model_path=model_path,
        vae_tokenizer_path=vae_tokenizer_path,
        precision=args.dtype,
        target_size=target_size,
        device = device,
    )

    collected_images = []

    if args.save_name is not None:
        folder_name = args.save_name
    else:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        folder_name = f"{current_time}"
    save_dir = os.path.join(args.save_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    image_logits_processor = inference_solver.create_logits_processor(cfg=cfg_guidance_scale, image_top_k=image_top_k)

    for seed in seeds:
        inference_solver.model.seed = seed
        for i, q_image_content_condition in enumerate(image_content_prompts):
        
            prompt_text = template_condition_sentences + q_image_content_condition
        
            output_file_name = 'img_' + str(i) +'_seed' + str(seed) + ".png"
            time_start = time.time()
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

            t1.record()
            with torch.no_grad():
                generated = inference_solver.generate(
                    images=[],
                    qas=[[prompt_text, None]],
                    max_gen_len=8192,
                    temperature=1.0,
                    cfg_guidance_scale=cfg_guidance_scale,
                    logits_processor=image_logits_processor,
                    seed=seed   
                )
            t2.record()
            torch.cuda.synchronize()

            t = t1.elapsed_time(t2) / 1000
            time_end = time.time()
            print("Token generation time elapsed: ", t, time_end - time_start)

            a1, new_image = generated[0], generated[1][0]
            result_image = inference_solver.create_image_grid([new_image], 1, 1)
            
            collected_images.append(np.array(result_image))
            result_image.save(os.path.join(save_dir, output_file_name))
            print(a1, 'saved', output_file_name) # <|image|>

    del inference_solver
    gc.collect()
