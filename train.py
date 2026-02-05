import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import argparse
from transformers import Trainer, TrainingArguments

from data.dataset import TokenDataset, ImageRowCollator
from model.base_model import load_lumina_with_lora, load_emu3_with_lora
from model.modeling_draft import RowExpertModel
from model import lumina_img_token_config, emu3_img_token_config

from utils.logger import get_logger

logger = get_logger(__name__)


def train(args):
    model_name = args.model_name.lower().strip()
    model_path = args.model_path
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lerarning_rate = args.learning_rate
    image_width = args.image_width
    image_height = args.image_height
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    use_standard_causal = args.use_standard_causal

    
    run_name = args.run_name or f"lora_{lora_rank}_{lora_alpha}_{epochs}_{batch_size}_{lerarning_rate}_{'cm' if use_standard_causal else 'bm'}"
    output_dir = os.path.join(args.output_dir, run_name)
    
    device = "cuda"
    
    if model_name == "lumina":
        base_model = load_lumina_with_lora(
            model_path=model_path,
            device=device,
            lora_rank=lora_rank, 
            lora_alpha=lora_alpha
        )
        img_token_config = lumina_img_token_config
    elif model_name == "emu3":
        base_model = load_emu3_with_lora(
            model_path=model_path,
            # device=device,
            lora_rank=lora_rank, 
            lora_alpha=lora_alpha
        )
        img_token_config = emu3_img_token_config
    else:
        raise NotImplementedError
    
    losses = args.losses
    use_ce = True if "ce" in losses else False
    use_kd = True if "kd" in losses else False

    model = RowExpertModel(
        base_model,
        use_ce=use_ce,
        ce_weight=args.ce_weight,
        use_kd=use_kd,
        kd_weight=args.kd_weight,
        kd_temp=args.kd_temp        
    )
    
    model.base_model.config.use_cache = False
    model.base_model.gradient_checkpointing_enable()
    model.base_model.enable_input_require_grads()

    train_dataset = TokenDataset(
        data_path, 
        use_teacher=args.use_teacher, 
        teacher_data_dir=args.teacher_data_dir,
        # temperary setting for ensuring fair comparison
        # start_idx=0,
        # end_idx=10000,
    )
    collator = ImageRowCollator(
        image_width=image_width, 
        image_height=image_height, 
        use_standard_causal=use_standard_causal, 
        block_size=args.block_size,
        use_teacher=args.use_teacher,
        **img_token_config,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lerarning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        bf16=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        run_name=run_name,
        logging_first_step=True,
        logging_steps=10,
        report_to="wandb" if args.enable_wandb else "none",
        deepspeed=args.deepspeed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )

    trainer.train()

    model.base_model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lumina")
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Lumina-mGPT-7B-768")
    parser.add_argument("--data_path", type=str, default="/home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train")
    parser.add_argument("--teacher_data_dir", type=str, default="/home/ffc3/bht/NRP/datasets/COCO_Lumina7B_training")
    parser.add_argument("--output_dir", type=str, default="training_outputs/lumina")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_strategy", type=str, default="steps") # "no", "epoch", "steps"
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--image_width", type=int, default=49, help="include end-of-line token") # include end-of-line token
    parser.add_argument("--image_height", type=int, default=48)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--losses", type=str, nargs="+", choices=["ce", "kd"], )
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--kd_temp", type=float, default=1.0)
    parser.add_argument("--use_teacher", action="store_true")
    parser.add_argument("--use_standard_causal", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    train(args)