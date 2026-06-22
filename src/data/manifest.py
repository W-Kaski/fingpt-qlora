"""Build reproducibility manifests for prepared data splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def file_sha256(path: Path) -> str:
    """Return the sha256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def example_fingerprint(example: dict) -> str:
    """Stable exact-match fingerprint for a formatted training example."""
    payload = json.dumps(example.get("conversations", []), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_split(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_split(path: Path) -> tuple[dict, set[str]]:
    """Summarize one split and return its exact-match fingerprints."""
    data = load_split(path)
    task_counts = Counter(example.get("task_type", "unknown") for example in data)
    fingerprints = {example_fingerprint(example) for example in data}
    return (
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "num_examples": len(data),
            "task_distribution": dict(sorted(task_counts.items())),
            "unique_examples": len(fingerprints),
        },
        fingerprints,
    )


def count_intersection(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left) & set(right))


def build_data_manifest(
    splits_dir: str | Path = "data/splits",
    dataset_sources: dict[str, str] | None = None,
    seed: int = 42,
) -> dict:
    """Build a manifest for train/val/test split files."""
    splits_dir = Path(splits_dir)
    split_summaries = {}
    fingerprints = {}

    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        summary, split_fingerprints = summarize_split(path)
        split_summaries[split] = summary
        fingerprints[split] = split_fingerprints

    duplicate_report = {
        "train_val_exact_duplicates": count_intersection(fingerprints["train"], fingerprints["val"]),
        "train_test_exact_duplicates": count_intersection(fingerprints["train"], fingerprints["test"]),
        "val_test_exact_duplicates": count_intersection(fingerprints["val"], fingerprints["test"]),
    }

    return {
        "split_seed": seed,
        "dataset_sources": dataset_sources or {},
        "splits": split_summaries,
        "duplicate_report": duplicate_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output", default="results/data_manifest_v3.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_data_manifest(splits_dir=args.splits_dir, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] Saved data manifest to {output}")


if __name__ == "__main__":
    main()
