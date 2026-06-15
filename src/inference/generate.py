"""Inference helpers for single-turn and multi-turn generation."""

from unsloth import FastLanguageModel


def load_model_for_inference(model_path: str, max_seq_length: int = 2048):
    """Load a model for inference (with LoRA adapters if needed)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_single_turn(
    model,
    tokenizer,
    user_message: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single-turn response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )

    return tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)


def generate_multi_turn(
    model,
    tokenizer,
    conversation: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response in a multi-turn conversation.

    Args:
        conversation: List of {"role": "user"/"assistant"/"system", "content": str}
    """
    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )

    return tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
