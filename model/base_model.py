import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
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

from transformers import AutoTokenizer, AutoImageProcessor
from transformers import BitsAndBytesConfig


logger = get_logger(__name__)


def load_lumina_with_lora(
    model_path,
    device, 
    lora_rank=64, 
    lora_alpha=128
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
        local_files_only=True
    )
    
    model.model.vqmodel.to("cpu")
    
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
    lora_alpha=128
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

if __name__ == "__main__":
    # lumina_model_path = "/home/ffc3/bht/model_home/Lumina-mGPT-7B-768"
    # peft_model = load_lumina_with_lora(
    #     model_path=lumina_model_path,
    #     device="cuda",
    #     lora_rank=64, 
    #     lora_alpha=128
    # )

    emu3_model_path = "/home/ffc3/bht/model_home/Emu3-Gen/"
    peft_model = load_emu3_with_lora(model_path=emu3_model_path, lora_rank=8, lora_alpha=16)