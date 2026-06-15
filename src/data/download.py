"""Download financial datasets from Hugging Face Hub."""

import os
import json
from pathlib import Path

from datasets import load_dataset
from src.config import DATA


DATASETS_CONFIG = {
    "fingpt_sentiment": {
        "hf_id": "FinGPT/fingpt-sentiment-train",
        "task_type": "sentiment",
        "split": "train",
    },
    "financial_phrasebank": {
        "hf_id": "financial_phrasebank",
        "task_type": "sentiment",
        "split": "train",
        "kwargs": {"name": "sentences_allagree"},
    },
    "convfinqa": {
        "hf_id": "FinGPT/ConvFinQA",
        "task_type": "reasoning",
        "split": "train",
    },
}


def download_dataset(name: str, config: dict, output_dir: str) -> Path:
    """Download a single dataset and save as JSON."""
    output_path = Path(output_dir) / f"{name}.json"
    if output_path.exists():
        print(f"[skip] {name} already exists at {output_path}")
        return output_path

    print(f"[download] {config['hf_id']}...")
    kwargs = config.get("kwargs", {})
    ds = load_dataset(config["hf_id"], split=config["split"], **kwargs)

    records = []
    for row in ds:
        record = dict(row)
        record["_source"] = name
        record["_task_type"] = config["task_type"]
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[done] {name}: {len(records)} examples -> {output_path}")
    return output_path


def download_all(output_dir: str = None):
    """Download all configured datasets."""
    output_dir = output_dir or DATA.raw_dir
    results = {}
    for name, config in DATASETS_CONFIG.items():
        try:
            path = download_dataset(name, config, output_dir)
            results[name] = path
        except Exception as e:
            print(f"[error] Failed to download {name}: {e}")
    return results


if __name__ == "__main__":
    download_all()
