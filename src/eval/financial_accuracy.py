"""Domain-specific financial accuracy metrics."""

import re
from collections import Counter


def extract_sentiment_from_output(text: str) -> str:
    """Extract sentiment label from structured model output."""
    match = re.search(r"\*\*Sentiment:\s*(\w+)\*\*", text, re.IGNORECASE)
    if match:
        label = match.group(1).capitalize()
        if label in ("Positive", "Negative", "Neutral"):
            return label

    text_lower = text.lower()
    if "positive" in text_lower and "negative" not in text_lower:
        return "Positive"
    if "negative" in text_lower and "positive" not in text_lower:
        return "Negative"

    return "Neutral"


def extract_number_from_output(text: str) -> str:
    """Extract the primary numerical answer from model output."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[0] if numbers else ""


def financial_sentiment_accuracy(predictions: list[str], references: list[str]) -> dict:
    """Compute sentiment accuracy specifically for financial text."""
    pred_labels = [extract_sentiment_from_output(p) for p in predictions]
    ref_labels = [extract_sentiment_from_output(r) for r in references]

    correct = sum(1 for p, r in zip(pred_labels, ref_labels) if p == r)
    total = len(ref_labels)

    per_class = {}
    for label in ["Positive", "Negative", "Neutral"]:
        mask = [i for i, r in enumerate(ref_labels) if r == label]
        if mask:
            class_correct = sum(1 for i in mask if pred_labels[i] == ref_labels[i])
            per_class[label] = {
                "accuracy": class_correct / len(mask),
                "support": len(mask),
            }

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


def numerical_exact_match(predictions: list[str], references: list[str]) -> dict:
    """Compute exact match for numerical financial answers."""
    pred_nums = [extract_number_from_output(p) for p in predictions]
    ref_nums = [extract_number_from_output(r) for r in references]

    correct = sum(1 for p, r in zip(pred_nums, ref_nums) if p == r and p != "")
    total = sum(1 for r in ref_nums if r != "")

    return {
        "exact_match": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
