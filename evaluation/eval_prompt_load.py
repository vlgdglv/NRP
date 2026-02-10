from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Union


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


def _iter_jsonl_coco(
    path: str, 
    start_idx: int = 0,
    end_idx: int = -1,
    *, 
    caption_key: str = "caption"
):
    with open(path, "r") as f:
        all_prompts = json.load(f)
        if start_idx > 0:
            all_prompts = all_prompts[start_idx:]
        if end_idx > 0:
            all_prompts = all_prompts[:end_idx]

        for idx, p in enumerate(all_prompts):
            yield {
                "prompt": p[caption_key],
                "name": p["image_id"],
                "global_idx": start_idx + idx
            }


def _iter_default_prompts(prompts: List[str]):
    for p in prompts:
        if isinstance(p, str) and p.strip():
            yield p.strip()


def prompt_loader(
    path: Optional[str] = None,
    start_idx: int = 0,
    end_idx: int = -1,
    *,
    default_prompts: Optional[List[str]] = None,
    caption_key: str = "caption",
):
    if default_prompts is None:
        default_prompts = image_content_prompts

    use_file = bool(path) and os.path.isfile(path)

    if use_file:
        return _iter_jsonl_coco(path, caption_key=caption_key, start_idx=start_idx, end_idx=end_idx)
    else:
        return _iter_default_prompts(default_prompts)


if __name__ == "__main__":
    it = prompt_loader("datasets/coco2017_val_prompts.json")
    print_cnt = 0

    for thing in it:
        # if print_cnt > 5:
        #     break
        print_cnt += 1
    print("Total in coco: ", print_cnt)
    print("--"* 20," default ", "--"* 20)
    print_cnt = 0
    it2 = prompt_loader(None)  # fallback default list + infinite
    for thing in it2:
        print(thing)
        if print_cnt > 5:
            break
        print_cnt += 1

