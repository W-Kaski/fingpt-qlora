"""Convert raw datasets to ShareGPT conversational format."""

import json
from pathlib import Path

from src.config import DATA
from src.data.templates import get_system_prompt


LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "pos": "Positive",
    "neg": "Negative",
    "neu": "Neutral",
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def normalize_label(label) -> str:
    """Normalize sentiment label to Positive/Negative/Neutral."""
    if isinstance(label, (int, float)):
        label = int(label)
    label_str = str(label).strip().lower()
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    return str(label).capitalize()


def format_sentiment_response(label: str, text: str) -> str:
    """Create a structured sentiment analysis response."""
    label = normalize_label(label)
    return (
        f"## Sentiment Analysis\n\n"
        f"**Sentiment: {label}**\n\n"
        f"**Key Factors:**\n"
        f"- {text[:200]}\n\n"
        f"**Implications:** The overall sentiment is {label.lower()}, which suggests "
        f"{'positive' if label == 'Positive' else 'negative' if label == 'Negative' else 'neutral'} "
        f"market outlook based on the given text."
    )


def format_summary_response(text: str) -> str:
    """Create a structured financial summary response."""
    return (
        f"## Financial Summary\n\n"
        f"**Key Points:**\n"
        f"- {text[:500]}\n\n"
        f"**Metrics:** Refer to the source filing for detailed numerical data.\n\n"
        f"**Outlook:** Based on the available information, the overall financial position "
        f"appears stable. Further analysis recommended with additional data sources."
    )


def format_reasoning_response(question: str, answer: str) -> str:
    """Create a structured reasoning response."""
    return (
        f"## Financial Reasoning\n\n"
        f"**Problem:** {question}\n\n"
        f"**Analysis:**\n"
        f"Based on the given financial data, I'll work through this step by step.\n\n"
        f"**Answer:** {answer}"
    )


def convert_sentiment_row(row: dict) -> dict:
    """Convert a sentiment dataset row to ShareGPT format."""
    text = row.get("text", row.get("sentence", row.get("input", "")))
    label = row.get("label", row.get("output", row.get("answer", "Neutral")))
    system = get_system_prompt("sentiment")
    instruction = f"Analyze the sentiment of this financial text:\n\n{text}"
    response = format_sentiment_response(label, text)

    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ],
        "task_type": "sentiment",
    }


def convert_reasoning_row(row: dict) -> dict:
    """Convert a ConvFinQA row to ShareGPT format."""
    question = row.get("question", row.get("input", ""))
    answer = str(row.get("answer", row.get("output", "")))
    # pre_text/post_text may be lists (table rows) or strings
    pre_raw = row.get("pre_text", "")
    post_raw = row.get("post_text", "")
    pre_text = " ".join(pre_raw) if isinstance(pre_raw, list) else str(pre_raw)
    post_text = " ".join(post_raw) if isinstance(post_raw, list) else str(post_raw)
    context = f"{pre_text}\n{post_text}".strip()

    system = get_system_prompt("reasoning")
    instruction = f"{context}\n\nQuestion: {question}" if context else question
    response = format_reasoning_response(question, answer)

    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ],
        "task_type": "reasoning",
    }


CONVERTERS = {
    "fingpt_sentiment": convert_sentiment_row,
    "financial_phrasebank": convert_sentiment_row,
    "convfinqa": convert_reasoning_row,
}


def convert_dataset(records: list[dict], source_name: str) -> list[dict]:
    """Convert all records in a dataset to ShareGPT format."""
    converter = CONVERTERS.get(source_name)
    if not converter:
        print(f"[warn] No converter for {source_name}, skipping")
        return []

    converted = []
    for rec in records:
        try:
            conv = converter(rec)
            if all(turn["value"].strip() for turn in conv["conversations"]):
                converted.append(conv)
        except Exception as e:
            print(f"[warn] Failed to convert record: {e}")
            continue

    print(f"[convert] {source_name}: {len(records)} -> {len(converted)} valid conversations")
    return converted


def convert_all(processed_dir: str = None, output_dir: str = None):
    """Convert all processed datasets to ShareGPT format."""
    processed_dir = Path(processed_dir or DATA.processed_dir)
    output_dir = Path(output_dir or DATA.splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_converted = []
    for json_file in processed_dir.glob("*.json"):
        if json_file.name.startswith("sharegpt_"):
            continue
        with open(json_file, encoding="utf-8") as f:
            records = json.load(f)

        source_name = json_file.stem
        converted = convert_dataset(records, source_name)
        all_converted.extend(converted)

    output_path = output_dir / "sharegpt_all.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_converted, f, ensure_ascii=False, indent=2)

    print(f"\n[total] {len(all_converted)} conversations -> {output_path}")
    return all_converted


if __name__ == "__main__":
    convert_all()
