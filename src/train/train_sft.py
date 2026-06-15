"""SFT training script for QLoRA fine-tuning with Unsloth."""

import json
from pathlib import Path

from unsloth import FastLanguageModel
from trl import SFTConfig
from datasets import Dataset

from src.config import MODEL, TRAIN, DATA
from src.train.lora_config import get_unsloth_lora_kwargs


def load_training_data(split: str, tokenizer) -> Dataset:
    """Load training data, apply chat template to produce text strings."""
    path = Path(DATA.splits_dir) / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for example in data:
        conversations = example["conversations"]
        messages = []
        for turn in conversations:
            role = turn["from"]
            if role == "gpt":
                role = "assistant"
            messages.append({"role": role, "content": turn["value"]})

        # Convert message list to a single text string via chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append({"text": text, "task_type": example.get("task_type", "")})

    return Dataset.from_list(texts)


def setup_model(preset: str = "baseline"):
    """Load base model and attach LoRA adapters."""
    print(f"[model] Loading {MODEL.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL.base_model,
        max_seq_length=MODEL.max_seq_length,
        load_in_4bit=MODEL.load_in_4bit,
    )

    lora_kwargs = get_unsloth_lora_kwargs(preset)
    print(f"[lora] Applying LoRA with r={lora_kwargs['r']}, alpha={lora_kwargs['lora_alpha']}")
    model = FastLanguageModel.get_peft_model(model, **lora_kwargs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def get_training_args(output_suffix: str = "v1") -> SFTConfig:
    """Build SFT training arguments."""
    output_dir = f"{TRAIN.output_dir}-{output_suffix}"
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN.batch_size,
        gradient_accumulation_steps=TRAIN.grad_accum,
        num_train_epochs=TRAIN.epochs,
        learning_rate=TRAIN.lr,
        lr_scheduler_type=TRAIN.scheduler,
        warmup_ratio=TRAIN.warmup_ratio,
        weight_decay=TRAIN.weight_decay,
        optim=TRAIN.optim,
        max_seq_length=MODEL.max_seq_length,
        fp16=TRAIN.fp16,
        bf16=TRAIN.bf16,
        logging_steps=TRAIN.logging_steps,
        save_steps=TRAIN.save_steps,
        save_total_limit=TRAIN.save_total_limit,
        eval_strategy="steps",
        eval_steps=TRAIN.eval_steps,
        seed=TRAIN.seed,
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )


def train(preset: str = "baseline", output_suffix: str = "v1"):
    """Run full training pipeline."""
    model, tokenizer = setup_model(preset)

    train_dataset = load_training_data("train", tokenizer)
    val_dataset = load_training_data("val", tokenizer)
    print(f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    training_args = get_training_args(output_suffix)

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
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

    return model, tokenizer, trainer


if __name__ == "__main__":
    train()
