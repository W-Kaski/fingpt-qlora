"""LoRA configuration presets for experiments."""

from src.config import LORA


PRESETS = {
    "baseline": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "high_rank": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "low_rank": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "attention_only": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}


def get_lora_config(preset: str = "baseline") -> dict:
    """Get LoRA config for a named preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Must be one of {list(PRESETS)}")
    return PRESETS[preset]


def get_unsloth_lora_kwargs(preset: str = "baseline") -> dict:
    """Get kwargs ready for FastLanguageModel.get_peft_model()."""
    config = get_lora_config(preset)
    return {
        "r": config["r"],
        "target_modules": config["target_modules"],
        "lora_alpha": config["alpha"],
        "lora_dropout": config["dropout"],
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 42,
    }
