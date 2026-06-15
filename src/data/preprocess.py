"""Clean and filter raw datasets."""

import json
import re
from pathlib import Path

from src.config import DATA

# Map each dataset filename to the correct text field
TEXT_KEY_MAP = {
    "fingpt_sentiment": "input",
    "financial_phrasebank": "sentence",
    "convfinqa": "question",
}


def detect_text_key(records: list[dict]) -> str:
    """Auto-detect the main text field from the first record."""
    if not records:
        return "text"
    sample = records[0]
    for candidate in ["text", "input", "sentence", "question"]:
        if candidate in sample:
            return candidate
    return list(sample.keys())[0]


def deduplicate_exact(records: list[dict], text_key: str) -> list[dict]:
    """Remove exact duplicate texts."""
    seen = set()
    unique = []
    for rec in records:
        text = rec.get(text_key, "")
        if text and text not in seen:
            seen.add(text)
            unique.append(rec)
    removed = len(records) - len(unique)
    if removed:
        print(f"[dedup] Removed {removed} exact duplicates")
    return unique


def filter_short_texts(records: list[dict], text_key: str, min_length: int = 20) -> list[dict]:
    """Remove records where the main text is too short."""
    filtered = [r for r in records if len(str(r.get(text_key, ""))) >= min_length]
    removed = len(records) - len(filtered)
    if removed:
        print(f"[filter] Removed {removed} records shorter than {min_length} chars")
    return filtered


def clean_text(text: str) -> str:
    """Basic text cleaning. Preserves currency symbols (€£¥$)."""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove control characters but keep currency symbols and common Unicode
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def clean_records(records: list[dict], text_key: str) -> list[dict]:
    """Apply text cleaning to all records."""
    cleaned = []
    for rec in records:
        rec = dict(rec)
        if text_key in rec:
            rec[text_key] = clean_text(rec[text_key])
        if rec.get(text_key):
            cleaned.append(rec)
    return cleaned


def preprocess_dataset(records: list[dict], text_key: str = None, source_name: str = "") -> list[dict]:
    """Full preprocessing pipeline for a single dataset."""
    # Resolve text key: explicit > map > auto-detect
    if text_key is None:
        text_key = TEXT_KEY_MAP.get(source_name) or detect_text_key(records)
    print(f"[preprocess] Using text_key='{text_key}' for {source_name or 'unknown'}")

    records = clean_records(records, text_key)

    # ConvFinQA is structured Q&A; skip dedup and length filter
    if source_name == "convfinqa":
        print(f"[preprocess] Skipping dedup/length filter for structured Q&A data")
        return records

    records = deduplicate_exact(records, text_key)
    records = filter_short_texts(records, text_key)
    return records


def preprocess_all(raw_dir: str = None, output_dir: str = None):
    """Preprocess all raw datasets."""
    raw_dir = Path(raw_dir or DATA.raw_dir)
    output_dir = Path(output_dir or DATA.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in raw_dir.glob("*.json"):
        source_name = raw_file.stem
        print(f"\n[process] {raw_file.name} (source={source_name})")
        with open(raw_file, encoding="utf-8") as f:
            records = json.load(f)

        processed = preprocess_dataset(records, source_name=source_name)

        output_path = output_dir / raw_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"[done] {len(records)} -> {len(processed)} records")


if __name__ == "__main__":
    preprocess_all()
