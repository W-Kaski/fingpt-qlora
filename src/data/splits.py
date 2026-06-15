"""Split merged dataset into train/val/test with stratification."""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.config import DATA


def stratified_split(
    data: list[dict],
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data into train/val/test, stratified by task_type."""
    train_ratio = train_ratio or DATA.train_split
    val_ratio = val_ratio or DATA.val_split
    test_ratio = test_ratio or DATA.test_split

    rng = random.Random(seed)
    by_task = defaultdict(list)
    for ex in data:
        by_task[ex.get("task_type", "unknown")].append(ex)

    train, val, test = [], [], []
    for task_type, examples in by_task.items():
        rng.shuffle(examples)
        n = len(examples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(examples[:n_train])
        val.extend(examples[n_train:n_train + n_val])
        test.extend(examples[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"[split] train={len(train)}, val={len(val)}, test={len(test)}")
    for task_type in sorted(by_task.keys()):
        n_train = sum(1 for ex in train if ex.get("task_type") == task_type)
        n_val = sum(1 for ex in val if ex.get("task_type") == task_type)
        n_test = sum(1 for ex in test if ex.get("task_type") == task_type)
        print(f"  {task_type}: train={n_train}, val={n_val}, test={n_test}")

    return train, val, test


def split_all(merged_path: str = None, output_dir: str = None, seed: int = 42):
    """Load merged data and create train/val/test splits."""
    merged_path = merged_path or str(Path(DATA.processed_dir) / "merged.json")
    output_dir = Path(output_dir or DATA.splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(merged_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[info] Loaded {len(data)} examples")

    train, val, test = stratified_split(data, seed=seed)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"[saved] {split_name}: {len(split_data)} examples -> {path}")


if __name__ == "__main__":
    split_all()
