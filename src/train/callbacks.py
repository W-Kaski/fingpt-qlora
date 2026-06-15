"""Custom training callbacks for logging and monitoring."""

import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class LossLoggerCallback(TrainerCallback):
    """Log training loss to a JSON file."""

    def __init__(self, output_dir: str = "results/training_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
            "timestamp": time.time(),
        }
        self.logs.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        output_path = self.output_dir / "training_logs.json"
        with open(output_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"[callback] Saved {len(self.logs)} log entries to {output_path}")


class SampleGeneratorCallback(TrainerCallback):
    """Generate sample outputs during training to monitor quality."""

    SAMPLE_PROMPTS = [
        "Analyze the sentiment of this financial news: 'Apple reported record Q4 revenue of $89.5B, beating analyst estimates.'",
        "What are the key risks of investing in emerging market bonds?",
        "Summarize: Company X reported EPS of $2.50 vs $2.30 expected, revenue up 15% YoY.",
    ]

    def __init__(self, tokenizer, every_n_steps: int = 500):
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.samples = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return

        model.eval()
        for prompt in self.SAMPLE_PROMPTS[:1]:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(input_ids=inputs, max_new_tokens=150, temperature=0.7)

            response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            self.samples.append({
                "step": state.global_step,
                "prompt": prompt,
                "response": response,
            })
            print(f"\n[step {state.global_step}] Sample: {response[:200]}...")

        model.train()
