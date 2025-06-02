from transformers import pipeline

from src.utils.models.causal import create_causal_model
from src.utils.models.tokenizer import create_tokenizer


def create_generater_pipeline(model_name: str) -> pipeline:

    model = create_causal_model(model_name)
    tokenizer = create_tokenizer(model_name)

    return pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
