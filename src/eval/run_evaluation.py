"""Kaggle evaluation runner and artifact writer."""

import argparse
import json
import re
import time
from pathlib import Path

from src.eval.benchmark import build_prompt
from src.eval.metrics import compute_bootstrap_ci, compute_macro_f1, compute_rouge_l
from src.eval.metrics import compute_sentiment_accuracy, extract_sentiment_label


SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


def format_compliance(predictions: list[str]) -> dict:
    """Measure whether generations contain a recognized sentiment label.

    Accepts any of the 5 output format variants (v4) as well as the
    original v3 structured template.
    """
    total = len(predictions)
    label_pattern = r"\*\*(?:Sentiment:\s*)?(?:Positive|Negative|Neutral)\*\*"
    compliant = sum(
        1
        for prediction in predictions
        if re.search(label_pattern, prediction, re.IGNORECASE)
    )
    return {
        "compliant": compliant,
        "total": total,
        "rate": compliant / total if total else 0.0,
    }


def confusion_matrix(predictions: list[str], references: list[str], labels: list[str]) -> list[list[int]]:
    """Return confusion matrix with rows=reference labels, cols=prediction labels."""
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for prediction, reference in zip(predictions, references):
        if reference not in label_index or prediction not in label_index:
            continue
        matrix[label_index[reference]][label_index[prediction]] += 1
    return matrix


def per_class_report(predictions: list[str], references: list[str], labels: list[str]) -> dict:
    """Compute precision, recall, F1, and support for each label."""
    report = {}
    for label in labels:
        tp = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref == label)
        fp = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref != label)
        fn = sum(1 for pred, ref in zip(predictions, references) if pred != label and ref == label)
        support = sum(1 for ref in references if ref == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return report


def _sentiment_metric_block(base_predictions: list[str], ft_predictions: list[str], references: list[str]) -> dict:
    base = compute_sentiment_accuracy(base_predictions, references)
    fine_tuned = compute_sentiment_accuracy(ft_predictions, references)

    base_labels = [extract_sentiment_label(prediction) for prediction in base_predictions]
    ft_labels = [extract_sentiment_label(prediction) for prediction in ft_predictions]
    ref_labels = [extract_sentiment_label(reference) for reference in references]

    return {
        "labels": SENTIMENT_LABELS,
        "accuracy": {
            "base": base["accuracy"],
            "fine_tuned": fine_tuned["accuracy"],
            "delta": fine_tuned["accuracy"] - base["accuracy"],
        },
        "macro_f1": {
            "base": compute_macro_f1(base_labels, ref_labels),
            "fine_tuned": compute_macro_f1(ft_labels, ref_labels),
            "delta": compute_macro_f1(ft_labels, ref_labels) - compute_macro_f1(base_labels, ref_labels),
        },
        "per_class": {
            "base": per_class_report(base_labels, ref_labels, SENTIMENT_LABELS),
            "fine_tuned": per_class_report(ft_labels, ref_labels, SENTIMENT_LABELS),
        },
        "confusion_matrix": {
            "base": confusion_matrix(base_labels, ref_labels, SENTIMENT_LABELS),
            "fine_tuned": confusion_matrix(ft_labels, ref_labels, SENTIMENT_LABELS),
        },
        "format_compliance": format_compliance(ft_predictions),
        "bootstrap_ci": {
            "fine_tuned_accuracy": compute_bootstrap_ci(
                compute_sentiment_accuracy,
                ft_predictions,
                references,
                n_iterations=200,
                score_key="accuracy",
            )
        },
    }


def compute_metrics(results: list[dict], experiment_name: str, adapter_dataset: str) -> dict:
    """Compute evaluation metrics from base/fine-tuned prediction records."""
    metrics = {
        "experiment": experiment_name,
        "adapter_dataset": adapter_dataset,
        "created_at_unix": int(time.time()),
        "tasks": {},
    }

    task_types = sorted({result.get("task_type", "unknown") for result in results})
    for task_type in task_types:
        task_results = [result for result in results if result.get("task_type", "unknown") == task_type]
        refs = [result["reference"] for result in task_results]
        base_predictions = [result["base_prediction"] for result in task_results]
        ft_predictions = [result["ft_prediction"] for result in task_results]

        task_metrics = {"num_examples": len(task_results)}
        try:
            task_metrics["rouge_l"] = {
                "base": compute_rouge_l(base_predictions, refs)["rouge_l_f1"],
                "fine_tuned": compute_rouge_l(ft_predictions, refs)["rouge_l_f1"],
            }
            task_metrics["rouge_l"]["delta"] = (
                task_metrics["rouge_l"]["fine_tuned"] - task_metrics["rouge_l"]["base"]
            )
        except ImportError as exc:
            task_metrics["rouge_l"] = {
                "base": None,
                "fine_tuned": None,
                "delta": None,
                "error": str(exc),
            }

        if task_type == "sentiment":
            task_metrics.update(_sentiment_metric_block(base_predictions, ft_predictions, refs))

        metrics["tasks"][task_type] = task_metrics

    return metrics


def comparison_table(metrics: dict) -> str:
    """Render a Markdown comparison table."""
    lines = [
        f"# {metrics['experiment']} Evaluation",
        "",
        "| Task | Metric | Base | Fine-tuned | Delta | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for task, task_metrics in metrics["tasks"].items():
        if "accuracy" in task_metrics:
            acc = task_metrics["accuracy"]
            lines.append(
                f"| {task} | Sentiment accuracy | {acc['base']:.3f} | {acc['fine_tuned']:.3f} | {acc['delta']:+.3f} | Primary |"
            )
            f1 = task_metrics["macro_f1"]
            lines.append(
                f"| {task} | Macro F1 | {f1['base']:.3f} | {f1['fine_tuned']:.3f} | {f1['delta']:+.3f} | Primary |"
            )
            class_f1 = task_metrics["per_class"]["fine_tuned"]
            f1_summary = ", ".join(
                f"{label}={values['f1']:.3f}" for label, values in class_f1.items()
            )
            lines.append(f"| {task} | Per-class F1 | n/a | n/a | n/a | {f1_summary} |")
            fmt = task_metrics["format_compliance"]
            lines.append(
                f"| {task} | Format compliance | n/a | {fmt['rate']:.3f} | n/a | {fmt['compliant']}/{fmt['total']} |"
            )
        rouge = task_metrics["rouge_l"]
        if rouge["base"] is None:
            lines.append(f"| {task} | ROUGE-L | n/a | n/a | n/a | Missing dependency |")
        else:
            lines.append(
                f"| {task} | ROUGE-L | {rouge['base']:.3f} | {rouge['fine_tuned']:.3f} | {rouge['delta']:+.3f} | Secondary |"
            )
    lines.append("")
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_evaluation_artifacts(
    results: list[dict],
    output_dir: str | Path,
    experiment_name: str,
    adapter_dataset: str,
) -> dict:
    """Write metrics, predictions, and comparison table artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(results, experiment_name, adapter_dataset)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "predictions.jsonl", results)
    (output_dir / "comparison_table.md").write_text(comparison_table(metrics), encoding="utf-8")
    return metrics


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate one deterministic response."""
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)


def load_unsloth_model(model_path: str, max_seq_length: int):
    """Load an Unsloth model lazily so report-only tests do not require GPU deps."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_model_comparison(
    test_data_path: str | Path,
    base_model_path: str,
    adapter_path: str,
    max_seq_length: int = 2048,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Generate base and fine-tuned predictions for a test split."""
    import gc
    import torch

    with open(test_data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    base_model, base_tokenizer = load_unsloth_model(base_model_path, max_seq_length)
    base_predictions = [
        generate_response(base_model, base_tokenizer, build_prompt(example["conversations"]), max_new_tokens)
        for example in test_data
    ]
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    ft_model, ft_tokenizer = load_unsloth_model(adapter_path, max_seq_length)
    ft_predictions = [
        generate_response(ft_model, ft_tokenizer, build_prompt(example["conversations"]), max_new_tokens)
        for example in test_data
    ]
    del ft_model
    del ft_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    results = []
    for index, example in enumerate(test_data):
        results.append(
            {
                "task_type": example.get("task_type", "unknown"),
                "reference": example["conversations"][-1]["value"],
                "base_prediction": base_predictions[index],
                "ft_prediction": ft_predictions[index],
                "index": index,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", help="Existing JSON list of evaluation records.")
    parser.add_argument("--test-data", default="data/splits/test.json")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--adapter-path", default="outputs/lora_adapter")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="results/eval_outputs/v3_completion_only")
    parser.add_argument("--experiment-name", default="v3_completion_only")
    parser.add_argument("--adapter-dataset", default="ericwang7717/fingpt-lora-adapter-v3")
    args = parser.parse_args()

    if args.results_json:
        with open(args.results_json, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = run_model_comparison(
            test_data_path=args.test_data,
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )

    write_evaluation_artifacts(
        results=results,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        adapter_dataset=args.adapter_dataset,
    )
    print(f"[eval] Wrote evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
