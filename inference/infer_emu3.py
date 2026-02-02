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


def unbatched_cfg_sample(
    prompt_text: str,
    model,
    processor,
    sampler,
    device,
    cfg_guidance_scale: float,
    image_area: int
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
        text=[NEGATIVE_PROMPT],
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

        # print(outputs.shape)
                
        # assert outputs_old.shape[-1] == outputs.shape[-1]
        # for j in range(outputs_old.shape[-1]):
        #     assert outputs[0, j] == outputs_old[0, j], f"outputs_new != outputs_old at step {j}"
        
    # Offload: CPU -> GPU
    model.to("cpu")
    processor.vision_tokenizer.to(device)
    mm_list = processor.decode(outputs[0])

    torch.cuda.synchronize()
    time_end = time.time()

    time_uesd = time_end - time_start
    return mm_list, time_uesd

class SimplePrefixMaskLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # input_ids: [1, seq_len]
        # scores:    [1, vocab_size]

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
        is_multi_token: bool = False
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
        
        # print("before cfg: ", last_logits.shape)
        if do_cfg: 
            cond, uncond = last_logits.chunk(2, dim=0)
            if is_prefill:
                last_logits = cond
            else:
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

def batched_cfg_sample(
    prompt_text: str,
    model,
    processor,
    sampler,
    device,
    cfg_guidance_scale: float,
    image_area: int,
    pad_token_id: int,
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
        text=[NEGATIVE_PROMPT],
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
    def left_pad(ids, mask, length):
        pad_len = length - ids.shape[1]
        if pad_len <= 0:
            return ids, mask
        ids_pad = torch.full((1, pad_len), pad_token_id, dtype=ids.dtype, device=ids.device)
        mask_pad = torch.zeros((1, pad_len), dtype=mask.dtype, device=mask.device)
        return torch.cat([ids_pad, ids], dim=1), torch.cat([mask_pad, mask], dim=1)

    pos_inputs_ids, pos_mask = left_pad(pos_inputs_ids, pos_mask, L)
    neg_inputs_ids, neg_mask = left_pad(neg_inputs_ids, neg_mask, L)

    batch_ids, batch_mask = torch.cat([pos_inputs_ids, neg_inputs_ids], dim=0), torch.cat([pos_mask, neg_mask], dim=0)

    print("Input pos shape: ", batch_ids.shape, " neg shape: ", batch_mask.shape)

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
        )
    # Offload: CPU -> GPU
    model.to("cpu")
    processor.vision_tokenizer.to(device)
    mm_list = processor.decode(outputs[0])

    torch.cuda.synchronize()
    time_end = time.time()

    time_uesd = time_end - time_start
    return mm_list, time_uesd


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
    model = AutoModelForCausalLM.from_pretrained(
        emu_model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not args.batched_cfg else "eager",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(emu_model_path, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(emu_vq_model_path, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(emu_vq_model_path, device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    image_area = model.config.image_area
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
                    model,
                    processor,
                    sampler,
                    device,
                    cfg_guidance_scale,
                    image_area,
                    pad_token_id=model.config.pad_token_id
                )
            else:
                mm_list, time_uesd = unbatched_cfg_sample(
                    prompt_text,
                    model,
                    processor,
                    sampler,
                    device,
                    cfg_guidance_scale,
                    image_area,
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
