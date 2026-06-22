"""Convert raw datasets to ShareGPT conversational format."""

import json
import hashlib
import re
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
    # 5-point scale from financial_phrasebank
    "strong positive": "Positive",
    "moderately positive": "Positive",
    "mildly positive": "Positive",
    "strong negative": "Negative",
    "moderately negative": "Negative",
    "mildly negative": "Negative",
}

# Sentiment-specific implication templates (15 per class for output diversity)
IMPLICATIONS = {
    "Positive": [
        "This suggests bullish momentum and potential upside for the stock.",
        "Investors may view this as a buying opportunity with growth potential.",
        "The positive developments could attract institutional interest.",
        "Market sentiment appears favorable, supporting upward price movement.",
        "Strong fundamentals point to sustained investor confidence.",
        "The rally may continue as buyers absorb available supply.",
        "This signals improving operational efficiency and margin expansion.",
        "Earnings beats like this typically lead to analyst upgrades.",
        "The positive guidance suggests management sees smooth sailing ahead.",
        "Sector rotation into this space appears to be accelerating.",
        "Momentum indicators confirm the strength of this move.",
        "This could trigger re-rating across the peer group.",
        "Dividend sustainability looks solid given these results.",
        "The market is rewarding execution, not just promise.",
        "Technical breakout confirms fundamental strength.",
    ],
    "Negative": [
        "This indicates bearish pressure and potential downside risk.",
        "Investors should exercise caution and monitor for further deterioration.",
        "The negative signals may trigger selling pressure in the near term.",
        "Risk-off sentiment could weigh on the stock in the coming sessions.",
        "Continued weakness here could breach key support levels.",
        "The earnings miss raises questions about forward guidance.",
        "Debt covenants may come under pressure if conditions persist.",
        "Institutional holders may reduce exposure on this news.",
        "Short interest is likely to spike as a result.",
        "The sector headwinds suggest limited near-term recovery potential.",
        "Management credibility takes a hit with this kind of miss.",
        "The selloff appears justified by deteriorating fundamentals.",
        "Further downside exists if support at current levels breaks.",
        "Competitive pressure is squeezing margins faster than expected.",
        "Regulatory risk adds an additional layer of uncertainty.",
    ],
    "Neutral": [
        "The market is likely to remain range-bound until clearer signals emerge.",
        "Investors may adopt a wait-and-see approach pending further developments.",
        "The balanced outlook suggests limited near-term price movement.",
        "Current conditions warrant a neutral stance with close monitoring.",
        "The market has largely priced in these developments.",
        "No clear catalyst exists to drive significant movement either way.",
        "Consolidation is likely before the next directional move.",
        "Volume patterns suggest neither bulls nor bears have conviction.",
        "The results are in line with consensus expectations.",
        "Traders will be watching for the next data point to break the deadlock.",
        "The muted reaction suggests the news was largely expected.",
        "Range-bound trading reflects uncertainty about the macro outlook.",
        "Institutional positioning remains balanced at current levels.",
        "The setup favors patience over aggressive positioning.",
        "Risk/reward appears roughly symmetric at this juncture.",
    ],
}


def normalize_label(label) -> str:
    """Normalize sentiment label to Positive/Negative/Neutral."""
    # Check original label first (handles int keys like 0, 1, 2)
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    label_str = str(label).strip().lower()
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    return str(label).capitalize()


def extract_key_factors(text: str, max_factors: int = 3) -> list[str]:
    """Extract key factors from financial text by splitting into sentences."""
    sentences = re.split(r'(?<!\d)\.(?!\d)\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    financial_keywords = [
        'revenue', 'profit', 'loss', 'earnings', 'growth', 'decline',
        'surge', 'drop', 'rise', 'fall', 'increase', 'decrease',
        'stock', 'shares', 'market', 'trading', 'investor',
        'billion', 'million', 'percent', '%', '$',
        'rates', 'inflation', 'gdp', 'unemployment', 'fed',
        'guidance', 'outlook', 'forecast', 'upgrade', 'downgrade',
        'miss', 'beat', 'estimate', 'consensus', 'restructuring',
        'debt', 'cash flow', 'margin', 'acquisition', 'merger',
    ]

    scored = []
    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for kw in financial_keywords if kw in s_lower)
        if re.search(r'\d+', s):
            score += 1
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    factors = [s for _, s in scored[:max_factors]]

    if not factors:
        factors = sentences[:max_factors]

    # Truncate overly long factors to keep outputs concise
    return [f[:150] + "..." if len(f) > 150 else f for f in factors]


def format_sentiment_response(label: str, text: str) -> str:
    """Create a structured sentiment analysis response with extracted factors.

    Uses 5 output format variants (selected by stable hash) so the model
    learns analytical reasoning rather than a single rigid template.
    """
    label = normalize_label(label)
    factors = extract_key_factors(text)
    factors_text = "\n".join(f"- {f}" for f in factors)

    # Stable hash: same text always produces the same output (reproducibility)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    impl_idx = int(digest[:8], 16) % len(IMPLICATIONS[label])
    implication = IMPLICATIONS[label][impl_idx]
    variant = int(digest[8:16], 16) % 5

    if variant == 0:
        # Classic structured format
        return (
            f"## Sentiment Analysis\n\n"
            f"**Sentiment: {label}**\n\n"
            f"**Key Factors:**\n{factors_text}\n\n"
            f"**Implications:** {implication}"
        )
    elif variant == 1:
        # Paragraph style
        return (
            f"The overall sentiment is **{label.lower()}**. "
            f"{' '.join(factors[:2])}. "
            f"{implication}"
        )
    elif variant == 2:
        # Driver-first analysis
        main_factor = factors[0] if factors else text[:200]
        return (
            f"**{label}**\n\n"
            f"The primary driver is: {main_factor}\n\n"
            f"Looking ahead: {implication}"
        )
    elif variant == 3:
        # Conclusion first
        return (
            f"Sentiment: **{label}**\n\n"
            f"Key takeaway: {implication}\n\n"
            f"Supporting points:\n{factors_text}"
        )
    else:
        # Concise
        return (
            f"**{label}** — {implication}"
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
