import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional

from data import TokenDataset, ImageRowCollator
from model.lumina_full import load_lumina_full, LuminaFullModel
from model import lumina_img_token_config
from utils.logger import get_logger

logger = get_logger(__name__)


class LuminaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_losses = {"steps": 0}

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if self.args.process_index == 0 and isinstance(outputs, dict):
            for key, value in outputs.items():
                if "loss" in key and key != "loss":
                    val = value.item() if hasattr(value, "item") else value
                    self.custom_losses[key] = self.custom_losses.get(key, 0.0) + val
            self.custom_losses["steps"] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self.args.process_index == 0 and self.custom_losses.get("steps", 0) > 0:
            steps = self.custom_losses["steps"]
            for key, total_val in self.custom_losses.items():
                if key != "steps":
                    logs[f"train/{key}"] = total_val / steps
            self.custom_losses = {"steps": 0}
        super().log(logs, start_time)


def train(args):
    model_path = args.model_path
    image_width = args.image_width
    image_height = args.image_height

    run_name = args.run_name or f"lumina_full_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    base_model = load_lumina_full(
        model_path=model_path,
        checkpoint_path=args.checkpoint_path,
    )

    losses = args.losses if args.losses is not None else []

    model = LuminaFullModel(
        base_model,
        use_ce="ce" in losses,
        ce_weight=args.ce_weight,
        use_kd="kd" in losses,
        kd_weight=args.kd_weight,
        kd_temp=args.kd_temp,
        use_topk_mass="topk_mass" in losses,
        topk_mass_topk=args.topk_mass_topk,
        topk_mass_weight=args.topk_mass_weight,
        use_topk_cover="topk_cover" in losses,
        topk_cover_weight=args.topk_cover_weight,
        topk_cover_temp=args.topk_cover_temp,
        topk_cover_topk=args.topk_cover_topk,
        image_latent_width=image_width,
        image_latent_height=image_height,
    )

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    dataset_names = args.dataset_name
    data_paths = args.data_path
    if len(dataset_names) == 1:
        dataset_cfg = {"dataset_names": dataset_names[0], "data_dir": data_paths[0]}
    elif len(data_paths) == 1:
        dataset_cfg = {"dataset_names": dataset_names, "data_dir": data_paths[0]}
    elif len(dataset_names) == len(data_paths):
        dataset_cfg = {"dataset_names": dataset_names, "data_dir": data_paths}
    else:
        raise ValueError(f"Invalid dataset config: {dataset_names}, {data_paths}")

    train_dataset = TokenDataset(
        data_dir=dataset_cfg["data_dir"],
        dataset_name=dataset_cfg["dataset_names"],
        use_teacher=args.use_teacher,
        teacher_data_dir=args.teacher_data_dir,
    )

    collator = ImageRowCollator(
        image_width=image_width,
        image_height=image_height,
        use_standard_causal=args.use_standard_causal,
        block_size=args.block_size,
        use_teacher=args.use_teacher,
        row_attention_mode=args.row_attention_mode,
        row_attention_window=args.row_attention_window,
        **lumina_img_token_config,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
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
        deepspeed=args.deepspeed,
    )

    trainer = LuminaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.base_model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, nargs="+", default=["COCO"])
    parser.add_argument("--data_path", type=str, nargs="+", required=True)
    parser.add_argument("--teacher_data_dir", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="training_outputs/lumina_full")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--image_width", type=int, default=49)
    parser.add_argument("--image_height", type=int, default=48)
    parser.add_argument("--losses", type=str, nargs="+",
                        choices=["ce", "kd", "topk_cover", "topk_mass"], default=["ce"])
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--kd_temp", type=float, default=1.0)
    parser.add_argument("--topk_cover_weight", type=float, default=1.0)
    parser.add_argument("--topk_cover_temp", type=float, default=1.0)
    parser.add_argument("--topk_cover_topk", type=int, default=128)
    parser.add_argument("--topk_mass_weight", type=float, default=1.0)
    parser.add_argument("--topk_mass_topk", type=int, default=64)
    parser.add_argument("--use_teacher", action="store_true")
    parser.add_argument("--use_standard_causal", action="store_true")
    parser.add_argument("--row_attention_mode", type=str, default="full",
                        choices=["full", "bidirectional_window", "causal_window", "no_intrarow"])
    parser.add_argument("--row_attention_window", type=int, default=4)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    train(args)
