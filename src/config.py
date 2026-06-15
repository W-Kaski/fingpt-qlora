from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class DataConfig:
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05
    task_weights: dict = field(default_factory=lambda: {
        "sentiment": 0.30,
        "summary": 0.25,
        "qa": 0.20,
        "reasoning": 0.15,
        "analysis": 0.10,
    })
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"


@dataclass
class TrainConfig:
    output_dir: str = "outputs/fingpt-qlora"
    batch_size: int = 2
    grad_accum: int = 8
    epochs: int = 3
    lr: float = 2e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 5
    save_total_limit: int = 3
    seed: int = 42


@dataclass
class EvalConfig:
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


# Singleton instances
MODEL = ModelConfig()
LORA = LoRAConfig()
DATA = DataConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()
