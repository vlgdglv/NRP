import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
from transformers import Trainer, TrainingArguments, TrainerCallback

from typing import Dict, Optional

from data import TokenDataset, ImageRowCollator, JanusImageRowCollator
from model.base_model import load_lumina_with_lora, load_emu3_with_lora, load_janus_with_lora
from model.modeling_draft import RowExpertModel
from model import lumina_img_token_config, emu3_img_token_config, janus_img_token_config

from utils.logger import get_logger

logger = get_logger(__name__)


class GlatAnnealCallback(TrainerCallback):
    """Linearly anneal model.glat_ratio from `start` to `end` over training.

    Final state (end, usually 0) matches inference (no reveals), closing the
    train/inference distribution gap GLAT otherwise suffers from.
    """
    def __init__(self, model, start: float, end: float):
        self.model = model
        self.start = start
        self.end = end

    def on_step_begin(self, args, state, control, **kwargs):
        total = max(state.max_steps, 1)
        progress = min(state.global_step / total, 1.0)
        self.model.glat_ratio = self.start + (self.end - self.start) * progress


class MyTrainer(Trainer):
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

def normalize_dataset_cfg(args):
    dataset_names = args.dataset_name
    data_paths = args.data_path

    if len(dataset_names) == 1:
        return {
            "dataset_names": dataset_names[0],
            "data_dir": data_paths[0],
        }
    if len(data_paths) == 1:
        return {
            "dataset_names": dataset_names,
            "data_dir": data_paths[0],
        }
    if len(dataset_names) == len(data_paths):
        for name, path in zip(dataset_names, data_paths):
            assert name.lower() in path.lower()
        return {
            "dataset_names": dataset_names,
            "data_dir": data_paths,
        }
    raise ValueError(f"Invalid dataset config: dataset_name={dataset_names}, data_path={data_paths}")

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
    warmup_ratio = args.warmup_ratio
    
    run_name = args.run_name or f"lora_{lora_rank}_{lora_alpha}_{epochs}_{batch_size}_{lerarning_rate}_{'cm' if use_standard_causal else 'bm'}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda"
    
    if model_name == "lumina":
        base_model = load_lumina_with_lora(
            model_path=model_path,
            device=device,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_checkpoint_path=args.lora_checkpoint_path,
            strict_loading=args.strict_checkpoint_loading,
        )
        img_token_config = lumina_img_token_config
    elif model_name == "emu3":
        base_model = load_emu3_with_lora(
            model_path=model_path,
            # device=device,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_checkpoint_path=args.lora_checkpoint_path,
            strict_loading=args.strict_checkpoint_loading,
        )
        img_token_config = emu3_img_token_config
    elif model_name == "janus":
        base_model = load_janus_with_lora(
            model_path=model_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_checkpoint_path=args.lora_checkpoint_path,
            strict_loading=args.strict_checkpoint_loading,
        )
        # img_token_config = janus_img_token_config
    else:
        raise NotImplementedError
    
    losses = args.losses if args.losses is not None else []
    use_ce = True if "ce" in losses else False
    use_kd = True if "kd" in losses else False
    use_tv = True if "tv" in losses else False
    use_acc = True if "acc" in losses else False
    use_topk_cover = True if "topk_cover" in losses else False
    use_mse = True if "mse" in losses else False
    use_topk_mass = True if "topk_mass" in losses else False
    use_rel_loss = True if "rel_loss" in losses else False

    model = RowExpertModel(
        base_model,
        use_ce=use_ce,
        ce_weight=args.ce_weight,
        use_kd=use_kd,
        use_tv=use_tv,
        use_acc=use_acc,
        use_topk_cover=use_topk_cover,
        kd_weight=args.kd_weight,
        kd_temp=args.kd_temp,
        tv_weight=args.tv_weight,
        tv_temp=args.tv_temp,
        acc_weight=args.acc_weight,
        acc_temp=args.acc_temp,
        topk_cover_weight=args.topk_cover_weight,
        topk_cover_temp=args.topk_cover_temp,
        topk_cover_topk=args.topk_cover_topk,
        use_mse=use_mse,
        mse_weight=args.mse_weight,
        image_latent_width=image_width,
        image_latent_height=image_height,
        use_gumbel=args.use_gumbel,
        use_topk_mass=use_topk_mass,
        topk_mass_topk=args.topk_mass_topk,
        topk_mass_weight=args.topk_mass_weight,
        use_row_rel=use_rel_loss,
        row_rel_weight=args.row_rel_weight,
        refine_mode=args.refine_mode,
        refine_weight=args.refine_weight,
        refine_tau=args.refine_tau,
        refine_topk=args.refine_topk,
        refine_full_sequence=args.refine_full_sequence,
        use_glat=args.use_glat,
        glat_ratio=args.glat_ratio,
    )
    
    if model_name == "janus":
        model.base_model.language_model.config.use_cache = False
        model.base_model.language_model.gradient_checkpointing_enable()
        model.base_model.language_model.enable_input_require_grads()
    else:
        model.base_model.config.use_cache = False
        model.base_model.gradient_checkpointing_enable()
        model.base_model.enable_input_require_grads()

    dataset_cfg = normalize_dataset_cfg(args)
    train_dataset = TokenDataset(
        data_dir=dataset_cfg["data_dir"],
        dataset_name=dataset_cfg["dataset_names"], 
        use_teacher=args.use_teacher, 
        teacher_data_dir=args.teacher_data_dir,
        # temperary setting for ensuring fair comparison
        # start_idx=0,
        # end_idx=10000,
    )

    if model_name == "janus":
        collator = JanusImageRowCollator(
            image_width=image_width,
            image_height=image_height,
            use_standard_causal=use_standard_causal,
            use_teacher=args.use_teacher,
            row_attention_mode=args.row_attention_mode,
            row_attention_window=args.row_attention_window,
        )
    else:
        # assert args.block_size+1 == image_width, "Do not support block-wise prediction in single row."
        collator = ImageRowCollator(
            image_width=image_width,
            image_height=image_height,
            use_standard_causal=use_standard_causal,
            block_size=args.block_size,
            use_teacher=args.use_teacher,
            row_attention_mode=args.row_attention_mode,
            row_attention_window=args.row_attention_window,
            **img_token_config,
        )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lerarning_rate,
        warmup_ratio=warmup_ratio,
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

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )

    if args.use_glat and args.glat_anneal:
        # initial glat_ratio at step 0 = glat_ratio_start
        model.glat_ratio = args.glat_ratio_start
        trainer.add_callback(GlatAnnealCallback(
            model=model,
            start=args.glat_ratio_start,
            end=args.glat_ratio_end,
        ))
        logger.info(
            f"GLAT anneal enabled: ratio {args.glat_ratio_start} -> {args.glat_ratio_end} "
            f"linearly over training."
        )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.base_model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lumina")
    parser.add_argument("--model_path", type=str, default="/jizhicfs/pkuhetu/bht/model_home/Lumina-mGPT-7B-768")
    parser.add_argument("--dataset_name", type=str, nargs="+", default=["COCO"])
    parser.add_argument("--data_path", type=str, nargs="+", default=["/home/ffc3/bht/GSD/COCO_Lumina7B_tokens_for_train"])
    parser.add_argument("--teacher_data_dir", type=str, nargs="+", default=["/home/ffc3/bht/NRP/datasets/COCO_Lumina7B_training"])
    parser.add_argument("--output_dir", type=str, default="training_outputs/lumina")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_strategy", type=str, default="steps") # "no", "epoch", "steps"
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--image_width", type=int, default=49, help="include end-of-line token") # include end-of-line token
    parser.add_argument("--image_height", type=int, default=48)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--losses", type=str, nargs="+", choices=["ce", "kd", "tv", "acc", "topk_cover", "mse", "topk_mass", "rel_loss"], )
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--kd_temp", type=float, default=1.0)
    parser.add_argument("--tv_weight", type=float, default=1.0)
    parser.add_argument("--tv_temp", type=float, default=1.0)
    parser.add_argument("--acc_weight", type=float, default=1.0)
    parser.add_argument("--acc_temp", type=float, default=1.0)
    parser.add_argument("--topk_cover_weight", type=float, default=1.0)
    parser.add_argument("--topk_cover_temp", type=float, default=1.0)
    parser.add_argument("--topk_cover_topk", type=int, default=128)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--use_teacher", action="store_true")
    parser.add_argument("--use_gumbel", action="store_true")
    parser.add_argument("--topk_mass_weight", type=float, default=1.0)
    parser.add_argument("--topk_mass_topk", type=int, default=128)
    parser.add_argument("--row_rel_weight", type=float, default=1.0)
    # Two-stage repairable draft training
    parser.add_argument("--refine_mode", type=str, default="none",
                        choices=["none", "deterministic_soft_topk", "soft_gumbel", "straight_through_hard"],
                        help="Draft interface for two-stage refine training")
    parser.add_argument("--refine_weight", type=float, default=1.0,
                        help="Weight for refine loss")
    parser.add_argument("--refine_tau", type=float, default=1.0,
                        help="Temperature for draft probability computation")
    parser.add_argument("--refine_topk", type=int, default=256,
                        help="Top-k for deterministic_soft_topk mode")
    parser.add_argument("--refine_full_sequence", action="store_true",
                        help="Replace ALL positions with draft (ablation baseline)")
    parser.add_argument("--use_standard_causal", action="store_true")
    parser.add_argument("--use_glat", action="store_true",
                        help="Glancing training: reveal random label tokens at input to encourage within-row modeling")
    parser.add_argument("--glat_ratio", type=float, default=0.3,
                        help="Fixed reveal ratio (used when --glat_anneal is NOT set)")
    parser.add_argument("--glat_anneal", action="store_true",
                        help="Linearly anneal glat_ratio from --glat_ratio_start to --glat_ratio_end over training. "
                             "Recommended so the final state matches inference (no reveals).")
    parser.add_argument("--glat_ratio_start", type=float, default=0.8,
                        help="Initial reveal ratio when --glat_anneal is set")
    parser.add_argument("--glat_ratio_end", type=float, default=0.0,
                        help="Final reveal ratio when --glat_anneal is set")
    # Row attention mode for research experiments
    parser.add_argument("--row_attention_mode", type=str, default="full",
                        choices=["full", "bidirectional_window", "causal_window", "no_intrarow"],
                        help="Row attention pattern: full (default), bidirectional_window, causal_window, no_intrarow")
    parser.add_argument("--row_attention_window", type=int, default=4,
                        help="Window size for bidirectional_window and causal_window modes")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_checkpoint_path", type=str, default=None,
                       help="Path to previously saved LoRA checkpoint to continue training from")
    parser.add_argument("--strict_checkpoint_loading", action="store_true", default=False,
                       help="Fail if checkpoint loading encounters any errors")
    args = parser.parse_args()

    train(args)