import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import argparse
from transformers import Trainer, TrainingArguments

from data.dataset import TokenDataset, ImageRowCollator, create_dataloader
from model.base_model import load_lumina_with_lora
from model.modeling_draft import RowExpertModel 
from utils.logger import get_logger

logger = get_logger(__name__)


def train(args):
    model_path = args.model_path
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lerarning_rate = args.learning_rate
    image_width = args.image_width
    image_height = args.image_height
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha

    run_name = args.run_name or f"lora_{lora_rank}_{lora_alpha}_{epochs}_{batch_size}_{lerarning_rate}"
    output_dir = os.path.join(args.output_dir, run_name)
    
    device = "cuda"

    base_model = load_lumina_with_lora(
        model_path=model_path,
        device=device,
        lora_rank=lora_rank, 
        lora_alpha=lora_alpha
    )
    model = RowExpertModel(base_model)

    model.base_model.gradient_checkpointing_enable()
    model.base_model.enable_input_require_grads()

    train_dataset = TokenDataset(data_path)
    collator = ImageRowCollator(image_width=image_width, image_height=image_height)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lerarning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        bf16=True,
        save_strategy=args.save_strategy,
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
    parser.add_argument("--model_path", type=str, default="/home/ffc3/bht/model_home/Lumina-mGPT-7B-768")
    parser.add_argument("--data_path", type=str, default="/home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train")
    parser.add_argument("--output_dir", type=str, default="training_outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_strategy", type=str, default="no") # "no", "epoch", "steps"
    parser.add_argument("--image_width", type=int, default=49) # include end-of-line token
    parser.add_argument("--image_height", type=int, default=48)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    train(args)