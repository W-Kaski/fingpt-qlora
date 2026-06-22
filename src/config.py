from dataclasses import dataclass, field
from typing import Tuple
import yaml


@dataclass
class ModelConfig:
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
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
    output_dir: str = "outputs/v3_completion_only"
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 1
    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 5
    save_total_limit: int = 3
    seed: int = 42
    max_steps: int = -1
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    completion_only_loss: bool = True
    packing: bool = False
    max_train_examples: int = 5000
    max_eval_examples: int = 1000


@dataclass
class EvalConfig:
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_configs_from_yaml(config_path: str = "configs/base.yaml"):
    """Create config instances from YAML file."""
    config = load_config(config_path)

    model = ModelConfig(
        base_model=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    lora = LoRAConfig(
        r=config["lora"]["r"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=tuple(config["lora"]["target_modules"]),
    )

    data = DataConfig(
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        task_weights=config["data"]["task_weights"],
    )

    train = TrainConfig(
        output_dir=f"outputs/{config['training'].get('experiment_name', 'v3_completion_only')}",
        batch_size=config["training"]["batch_size"],
        grad_accum=config["training"]["gradient_accumulation_steps"],
        epochs=config["training"]["num_epochs"],
        lr=config["training"]["learning_rate"],
        scheduler=config["training"]["scheduler"],
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        warmup_steps=config["training"].get("warmup_steps", 0),
        weight_decay=config["training"]["weight_decay"],
        optim=config["training"]["optim"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        save_steps=config["evaluation"]["save_steps"],
        eval_steps=config["evaluation"]["eval_steps"],
        logging_steps=config["evaluation"]["logging_steps"],
        save_total_limit=config["evaluation"]["save_total_limit"],
        max_steps=config["training"].get("max_steps", -1),
        save_strategy=config["training"].get("save_strategy", "steps"),
        eval_strategy=config["training"].get("eval_strategy", "steps"),
        load_best_model_at_end=config["training"].get("load_best_model_at_end", True),
        metric_for_best_model=config["training"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=config["training"].get("greater_is_better", False),
        completion_only_loss=config["training"].get("completion_only_loss", True),
        packing=config["training"].get("packing", False),
        max_train_examples=config["training"].get("max_train_examples", 5000),
        max_eval_examples=config["training"].get("max_eval_examples", 1000),
    )

    eval_config = EvalConfig()

    return model, lora, data, train, eval_config


# Default singleton instances (from base.yaml)
MODEL = ModelConfig()
LORA = LoRAConfig()
DATA = DataConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()
