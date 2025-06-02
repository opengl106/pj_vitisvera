from transformers import AutoModelForCausalLM

from src.utils.models.quant import BNB_CONFIG


def create_causal_model(model_name: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=BNB_CONFIG
    )
