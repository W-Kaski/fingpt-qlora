"""Evaluation metrics for financial NLP tasks."""

import numpy as np
from collections import Counter


def compute_sentiment_accuracy(predictions: list[str], references: list[str]) -> dict:
    """Compute exact-match accuracy for sentiment labels."""
    from src.data.format_chat import normalize_label

    pred_labels = [normalize_label(extract_sentiment_label(p)) for p in predictions]
    ref_labels = [normalize_label(extract_sentiment_label(r)) for r in references]

    correct = sum(1 for p, r in zip(pred_labels, ref_labels) if p == r)
    total = len(ref_labels)
    accuracy = correct / total if total > 0 else 0.0

    macro_f1 = compute_macro_f1(pred_labels, ref_labels)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "correct": correct,
        "total": total,
    }


def compute_macro_f1(predictions: list[str], references: list[str]) -> float:
    """Compute macro F1 score."""
    labels = set(references)
    f1_scores = []

    for label in labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return np.mean(f1_scores) if f1_scores else 0.0


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L F1 score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return {
        "rouge_l_f1": np.mean(scores),
        "std": np.std(scores),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore F1."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    return {
        "bertscore_f1": F1.mean().item(),
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
    }


def compute_exact_match(predictions: list[str], references: list[str]) -> dict:
    """Compute exact match for numerical answers."""
    def extract_number(text: str) -> str:
        import re
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[0] if numbers else ""

    pred_nums = [extract_number(p) for p in predictions]
    ref_nums = [extract_number(r) for r in references]

    correct = sum(1 for p, r in zip(pred_nums, ref_nums) if p == r)
    total = len(ref_nums)
    return {
        "exact_match": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


def extract_sentiment_label(text: str) -> str:
    """Extract sentiment label from model output."""
    import re
    text_lower = text.lower()

    match = re.search(r"\*\*sentiment:\s*(\w+)\*\*", text_lower)
    if match:
        return match.group(1).capitalize()

    for label in ["positive", "negative", "neutral"]:
        if label in text_lower:
            return label.capitalize()

    return "Neutral"


def compute_bootstrap_ci(
    metric_fn, predictions: list, references: list,
    n_iterations: int = 1000, confidence: float = 0.95, seed: int = 42,
    score_key: str = None,
) -> dict:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function returning a dict with the score.
        score_key: Key to extract from the result dict. Auto-detected if None.
    """
    rng = np.random.RandomState(seed)
    n = len(predictions)

    # Auto-detect score key from first call
    if score_key is None:
        sample = metric_fn(predictions[:1], references[:1])
        for candidate in ["accuracy", "exact_match", "rouge_l_f1", "bertscore_f1", "macro_f1"]:
            if candidate in sample:
                score_key = candidate
                break
        if score_key is None:
            raise ValueError(f"Cannot detect score key from metric function output: {list(sample.keys())}")

    scores = []
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        boot_preds = [predictions[i] for i in indices]
        boot_refs = [references[i] for i in indices]
        result = metric_fn(boot_preds, boot_refs)
        scores.append(result[score_key])

    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return {
        "mean": np.mean(scores),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": np.std(scores),
    }
