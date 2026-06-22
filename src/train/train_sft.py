"""SFT training script for QLoRA fine-tuning with Unsloth."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from src.config import DATA, MODEL, TRAIN, create_configs_from_yaml
from src.data.prompt_completion import convert_sharegpt_dataset
from src.train.lora_config import get_unsloth_lora_kwargs


def load_training_data(split: str, data_config=DATA, train_config=TRAIN):
    """Load ShareGPT data as prompt-completion records."""
    from datasets import Dataset

    path = Path(data_config.splits_dir) / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    max_examples = (
        train_config.max_train_examples if split == "train" else train_config.max_eval_examples
    )
    records = convert_sharegpt_dataset(data, max_examples=max_examples)
    return Dataset.from_list(records)


def setup_model(model_config=MODEL, preset: str = "baseline"):
    """Load base model and attach LoRA adapters."""
    from unsloth import FastLanguageModel

    print(f"[model] Loading {model_config.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.base_model,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
    )

    lora_kwargs = get_unsloth_lora_kwargs(preset)
    print(f"[lora] Applying LoRA with r={lora_kwargs['r']}, alpha={lora_kwargs['lora_alpha']}")
    model = FastLanguageModel.get_peft_model(model, **lora_kwargs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def get_training_args(model_config=MODEL, train_config=TRAIN, output_suffix: str = "v3"):
    """Build SFT training arguments."""
    from trl import SFTConfig

    output_dir = f"{train_config.output_dir}-{output_suffix}"

    # warmup_steps takes precedence over warmup_ratio when > 0
    warmup_kwargs = {}
    if train_config.warmup_steps > 0:
        warmup_kwargs["warmup_steps"] = train_config.warmup_steps
    else:
        warmup_kwargs["warmup_ratio"] = train_config.warmup_ratio

    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.grad_accum,
        num_train_epochs=train_config.epochs,
        learning_rate=train_config.lr,
        max_steps=train_config.max_steps,
        lr_scheduler_type=train_config.scheduler,
        **warmup_kwargs,
        weight_decay=train_config.weight_decay,
        optim=train_config.optim,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        eval_strategy=train_config.eval_strategy,
        eval_steps=train_config.eval_steps,
        seed=train_config.seed,
        report_to="none",
        completion_only_loss=train_config.completion_only_loss,
        packing=train_config.packing,
        save_strategy=train_config.save_strategy,
        load_best_model_at_end=train_config.load_best_model_at_end,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
    )


def current_git_commit() -> str:
    """Return the current git commit SHA, or unknown outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def save_run_artifacts(output_dir: str, config_path: str, trainer, train_size: int, val_size: int) -> None:
    """Save reproducibility metadata next to the adapter."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config_used.yaml")
    trainer.state.save_to_json(str(output / "trainer_state.json"))

    manifest = {
        "experiment": output.name,
        "git_commit": current_git_commit(),
        "config_path": config_path,
        "config_copy": "config_used.yaml",
        "train_examples": train_size,
        "val_examples": val_size,
        "created_at_unix": int(time.time()),
    }
    with open(output / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def train(config_path: str = "configs/experiments/v3_completion_only.yaml", preset: str = "baseline", output_suffix: str = "v3"):
    """Run full training pipeline."""
    model_config, _, data_config, train_config, _ = create_configs_from_yaml(config_path)
    model, tokenizer = setup_model(model_config, preset)

    train_dataset = load_training_data("train", data_config, train_config)
    val_dataset = load_training_data("val", data_config, train_config)
    print(f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("[data] Format: prompt-completion")

    training_args = get_training_args(model_config, train_config, output_suffix)
    print(f"[train] completion_only_loss={training_args.completion_only_loss}")

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    print("[train] Starting training...")
    trainer.train()

    output_dir = training_args.output_dir
    print(f"[save] Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_run_artifacts(output_dir, config_path, trainer, len(train_dataset), len(val_dataset))

    return model, tokenizer, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v3_completion_only.yaml")
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--output-suffix", default="v3")
    args = parser.parse_args()
    train(config_path=args.config, preset=args.preset, output_suffix=args.output_suffix)
