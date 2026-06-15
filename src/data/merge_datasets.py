"""Merge datasets with stratified weighting by task type."""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.config import DATA


def load_sharegpt_data(path: str) -> list[dict]:
    """Load ShareGPT format data."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stratified_merge(
    datasets: dict[str, list[dict]],
    weights: dict[str, float] = None,
    total_size: int = None,
    seed: int = 42,
) -> list[dict]:
    """
    Merge multiple datasets with stratified weighting.

    Args:
        datasets: Dict mapping task_type -> list of examples
        weights: Dict mapping task_type -> target weight (should sum to 1.0)
        total_size: Total number of examples in merged dataset. None = use all.
        seed: Random seed for reproducibility.
    """
    weights = weights or DATA.task_weights
    rng = random.Random(seed)

    # Normalize weights to only include tasks that have data
    available_weights = {k: v for k, v in weights.items() if k in datasets}
    if not available_weights:
        print("[error] No matching tasks between weights and datasets")
        return []
    total_weight = sum(available_weights.values())
    available_weights = {k: v / total_weight for k, v in available_weights.items()}
    print(f"[merge] Normalized weights: {available_weights}")

    if total_size is None:
        total_size = sum(len(v) for v in datasets.values())

    merged = []
    for task_type, weight in available_weights.items():
        task_data = datasets[task_type]
        target_count = int(total_size * weight)
        actual_count = min(target_count, len(task_data))

        if actual_count < target_count:
            print(f"[warn] {task_type}: wanted {target_count}, have {len(task_data)}")

        sampled = rng.sample(task_data, actual_count)
        merged.extend(sampled)
        print(f"[merge] {task_type}: {actual_count}/{len(task_data)} examples (weight={weight:.2f})")

    rng.shuffle(merged)
    print(f"[merge] Total: {len(merged)} examples")
    return merged


def merge_all(sharegpt_path: str = None, output_path: str = None):
    """Load ShareGPT data, group by task type, merge with weights."""
    sharegpt_path = sharegpt_path or str(Path(DATA.splits_dir) / "sharegpt_all.json")
    output_path = output_path or str(Path(DATA.processed_dir) / "merged.json")

    data = load_sharegpt_data(sharegpt_path)

    by_task = defaultdict(list)
    for ex in data:
        task = ex.get("task_type", "unknown")
        by_task[task].append(ex)

    print(f"[info] Task distribution before merge:")
    for task, examples in sorted(by_task.items()):
        print(f"  {task}: {len(examples)}")

    merged = stratified_merge(dict(by_task))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {len(merged)} examples -> {output_path}")
    return merged


if __name__ == "__main__":
    merge_all()
