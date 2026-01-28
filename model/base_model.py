import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from utils.logger import get_logger
from utils import is_rank0

from model.lumina_arch.chameleon import ChameleonForConditionalGeneration

logger = get_logger(__name__)


def load_lumina_with_lora(
    model_path,
    device, 
    lora_rank=64, 
    lora_alpha=128
):

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=None,
        local_files_only=True
    )
    
    model.model.vqmodel.to("cpu")
    
    # print("Checking module names for LoRA...")
    # for name, module in model.named_modules():
    #     if "attn" in name and isinstance(module, torch.nn.Linear):
    #         print(f"Found Linear Layer: {name}")
    #         break
    
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
                        "lm_head"
                        ],
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
    model_path = "/home/ffc3/bht/model_home/Lumina-mGPT-7B-768"
    peft_model = load_lumina_with_lora(
        model_path=model_path,
        device="cuda",
        lora_rank=64, 
        lora_alpha=128
    )
