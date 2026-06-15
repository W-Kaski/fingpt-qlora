"""Run model inference on evaluation set and collect outputs."""

import json
from pathlib import Path


def build_prompt(conversations: list[dict]) -> str:
    """Build a prompt from conversation turns (excluding the last assistant turn)."""
    messages = []
    for turn in conversations:
        role = turn["from"]
        if role == "system":
            messages.append({"role": "system", "content": turn["value"]})
        elif role == "human":
            messages.append({"role": "user", "content": turn["value"]})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": turn["value"]})
    return messages[:-1]  # Exclude last assistant turn


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate a single response from the model."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return response


def run_benchmark(
    model,
    tokenizer,
    test_data: list[dict],
    task_type: str = None,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Run model on test data, collect (reference, prediction) pairs."""
    results = []
    for i, example in enumerate(test_data):
        if task_type and example.get("task_type") != task_type:
            continue

        messages = build_prompt(example["conversations"])
        reference = example["conversations"][-1]["value"]
        prediction = generate_response(model, tokenizer, messages, max_new_tokens)

        results.append({
            "reference": reference,
            "prediction": prediction,
            "task_type": example.get("task_type", "unknown"),
            "index": i,
        })

        if (i + 1) % 50 == 0:
            print(f"[progress] {i + 1}/{len(test_data)} examples processed")

    return results


def save_results(results: list[dict], output_path: str):
    """Save benchmark results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {len(results)} results -> {output_path}")


def load_results(path: str) -> list[dict]:
    """Load benchmark results from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
