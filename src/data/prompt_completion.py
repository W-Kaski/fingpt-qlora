"""Utilities for TRL prompt-completion SFT datasets."""


ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def normalize_role(role: str) -> str:
    """Map ShareGPT roles to chat-template roles."""
    if role not in ROLE_MAP:
        raise ValueError(f"Unknown conversation role: {role}")
    return ROLE_MAP[role]


def sharegpt_to_prompt_completion(example: dict) -> dict:
    """Convert one ShareGPT example to TRL prompt-completion format."""
    prompt = []
    completion = None

    for turn in example["conversations"]:
        role = normalize_role(turn["from"])
        message = {"role": role, "content": turn["value"]}
        if role == "assistant" and completion is None:
            completion = message
        else:
            prompt.append(message)

    if completion is None:
        raise ValueError("Example does not contain an assistant completion")

    return {
        "prompt": prompt,
        "completion": [completion],
        "task_type": example.get("task_type", ""),
    }


def convert_sharegpt_dataset(examples: list[dict], max_examples: int | None = None) -> list[dict]:
    """Convert ShareGPT examples to prompt-completion records."""
    if max_examples is not None:
        examples = examples[:max_examples]
    return [sharegpt_to_prompt_completion(example) for example in examples]
