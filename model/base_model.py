import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
from utils.logger import get_logger
from utils import is_rank0

def import_lumina():
    from model.lumina_arch.chameleon.modeling_chameleon import ChameleonForConditionalGeneration
    return ChameleonForConditionalGeneration

def import_emu3():
    from model.emu3_arch.mllm.processing_emu3 import Emu3Processor
    from model.emu3_arch.mllm.modeling_emu3 import Emu3ForCausalLM
    from model.emu3_arch.tokenizer.modeling_emu3visionvq import Emu3VisionVQModel
    return Emu3Processor, Emu3ForCausalLM, Emu3VisionVQModel

def import_janus():
    from model.janus_arch.models import MultiModalityCausalLM, VLChatProcessor
    return MultiModalityCausalLM, VLChatProcessor 

from transformers import AutoModelForCausalLM, AutoConfig, AutoImageProcessor
from transformers import BitsAndBytesConfig


logger = get_logger(__name__)


def load_lumina_with_lora(
    model_path,
    device,
    lora_rank=64,
    lora_alpha=128,
    lora_checkpoint_path=None,
    strict_loading=False
):
    try:
        ChameleonForConditionalGeneration = import_lumina()
    except Exception as e:
        raise ImportError(
            "Emu3 backend is not available in this environment. "
            "Likely transformers/torch version mismatch or missing optional deps. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=None,
        # local_files_only=True
    )
    
    model.model.vqmodel.to("cpu")
    
    if lora_checkpoint_path is not None:
        # Load from existing LoRA checkpoint
        try:
            if is_rank0():
                logger.info(f"Loading LoRA checkpoint from: {lora_checkpoint_path}")
            model = PeftModel.from_pretrained(model, lora_checkpoint_path, is_trainable=True)

            # Verify checkpoint compatibility
            loaded_config = model.peft_config[model.active_adapter]
            if loaded_config.r != lora_rank or loaded_config.lora_alpha != lora_alpha:
                warning_msg = (f"Checkpoint LoRA config mismatch: "
                             f"checkpoint has r={loaded_config.r}, alpha={loaded_config.lora_alpha}, "
                             f"but requested r={lora_rank}, alpha={lora_alpha}")
                if strict_loading:
                    raise ValueError(warning_msg)
                elif is_rank0():
                    logger.warning(warning_msg + " - Continuing with checkpoint config")

        except Exception as e:
            error_msg = f"Failed to load LoRA checkpoint from {lora_checkpoint_path}: {e}"
            if strict_loading:
                raise RuntimeError(error_msg)
            else:
                if is_rank0():
                    logger.warning(error_msg + " - Falling back to fresh LoRA initialization")
                lora_checkpoint_path = None

    if lora_checkpoint_path is None:
        # Create fresh LoRA configuration
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            # "lm_head"
                            ],
            modules_to_save=["lm_head"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_config)
    
    if is_rank0():
        print("model type:", type(model))
        print("has peft_config:", hasattr(model, "peft_config"))
        print("peft_config:", getattr(model, "peft_config", None))
        model.print_trainable_parameters()

    return model


def load_emu3_with_lora(
    model_path,
    vq_model_path=None,
    device=None,
    lora_rank=64,
    lora_alpha=128,
    lora_checkpoint_path=None,
    strict_loading=False
):
    try:
        Emu3Processor, Emu3ForCausalLM, Emu3VisionVQModel = import_emu3()
    except Exception as e:
        raise ImportError(
            "Emu3 backend is not available in this environment. "
            "Likely transformers/torch version mismatch or missing optional deps. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Emu3ForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
        # quantization_config=bnb_config
    )

    model = prepare_model_for_kbit_training(model)
 
    if lora_checkpoint_path is not None:
        # Load from existing LoRA checkpoint
        try:
            if is_rank0():
                logger.info(f"Loading LoRA checkpoint from: {lora_checkpoint_path}")
            model = PeftModel.from_pretrained(model, lora_checkpoint_path, is_trainable=True)

            # Verify checkpoint compatibility
            loaded_config = model.peft_config[model.active_adapter]
            if loaded_config.r != lora_rank or loaded_config.lora_alpha != lora_alpha:
                warning_msg = (f"Checkpoint LoRA config mismatch: "
                             f"checkpoint has r={loaded_config.r}, alpha={loaded_config.lora_alpha}, "
                             f"but requested r={lora_rank}, alpha={lora_alpha}")
                if strict_loading:
                    raise ValueError(warning_msg)
                elif is_rank0():
                    logger.warning(warning_msg + " - Continuing with checkpoint config")

        except Exception as e:
            error_msg = f"Failed to load LoRA checkpoint from {lora_checkpoint_path}: {e}"
            if strict_loading:
                raise RuntimeError(error_msg)
            else:
                if is_rank0():
                    logger.warning(error_msg + " - Falling back to fresh LoRA initialization")
                lora_checkpoint_path = None

    if lora_checkpoint_path is None:
        # Create fresh LoRA configuration
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            # "gate_proj",
                            # "up_proj",
                            # "down_proj",
                            # "lm_head"
                            ],
            # modules_to_save=["lm_head"],
            # lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_config)
    
    if is_rank0():
        print("model type:", type(model))
        print("has peft_config:", hasattr(model, "peft_config"))
        print("peft_config:", getattr(model, "peft_config", None))
        model.print_trainable_parameters()

    return model


def load_janus_with_lora(
    model_path,
    lora_rank=64,
    lora_alpha=128,
    lora_checkpoint_path=None,
    strict_loading=False
):
    try:
        MultiModalityCausalLM, VLChatProcessor = import_janus()
    except Exception as e:
        raise ImportError(
            "Janus backend is not available in this environment. "
            "Likely transformers/torch version mismatch or missing optional deps. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    # cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # print(type(cfg), cfg.__class__.__name__)

    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    )
    
    if lora_checkpoint_path is not None:
        # Load from existing LoRA checkpoint
        try:
            if is_rank0():
                logger.info(f"Loading LoRA checkpoint from: {lora_checkpoint_path}")
            model = PeftModel.from_pretrained(model, lora_checkpoint_path, is_trainable=True)

            # Verify checkpoint compatibility
            loaded_config = model.peft_config[model.active_adapter]
            if loaded_config.r != lora_rank or loaded_config.lora_alpha != lora_alpha:
                warning_msg = (f"Checkpoint LoRA config mismatch: "
                             f"checkpoint has r={loaded_config.r}, alpha={loaded_config.lora_alpha}, "
                             f"but requested r={lora_rank}, alpha={lora_alpha}")
                if strict_loading:
                    raise ValueError(warning_msg)
                elif is_rank0():
                    logger.warning(warning_msg + " - Continuing with checkpoint config")

        except Exception as e:
            error_msg = f"Failed to load LoRA checkpoint from {lora_checkpoint_path}: {e}"
            if strict_loading:
                raise RuntimeError(error_msg)
            else:
                if is_rank0():
                    logger.warning(error_msg + " - Falling back to fresh LoRA initialization")
                lora_checkpoint_path = None

    if lora_checkpoint_path is None:
        # Create fresh LoRA configuration
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            ],
            modules_to_save=["gen_head"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_config)

    if is_rank0():
        print("model type:", type(model))
        print("has peft_config:", hasattr(model, "peft_config"))
        print("peft_config:", getattr(model, "peft_config", None))
        model.print_trainable_parameters()

    return model

    

if __name__ == "__main__":
    # lumina_model_path = "/home/ffc3/bht/model_home/Lumina-mGPT-7B-768"
    # peft_model = load_lumina_with_lora(model_path=lumina_model_path, lora_rank=64, lora_alpha=128)

    # emu3_model_path = "/home/ffc3/bht/model_home/Emu3-Gen/"
    # peft_model = load_emu3_with_lora(model_path=emu3_model_path, lora_rank=8, lora_alpha=16)

    janus_model_path = "/home/ffc3/bht/model_home/Janus-Pro-7B/"
    peft_model = load_janus_with_lora(model_path=janus_model_path, lora_rank=8, lora_alpha=16)