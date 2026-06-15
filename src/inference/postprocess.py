"""Post-process model outputs: clean, extract structured fields."""

import re


def clean_output(text: str) -> str:
    """Remove artifacts from model output."""
    text = re.sub(r"<\|.*?\|>", "", text)  # Remove special tokens
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse multiple newlines
    return text.strip()


def extract_sentiment(text: str) -> str:
    """Extract sentiment label from model output."""
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


def extract_key_metrics(text: str) -> list[str]:
    """Extract key financial metrics mentioned in the text."""
    patterns = [
        r"\$[\d,.]+[BMK]?",
        r"\d+\.?\d*%",
        r"EPS.*?\$[\d,.]+",
        r"revenue.*?\$[\d,.]+",
    ]
    metrics = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        metrics.extend(matches)
    return list(set(metrics))


def extract_sections(text: str) -> dict[str, str]:
    """Extract markdown sections from structured output."""
    sections = {}
    current_section = "intro"
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        elif line.startswith("**") and line.endswith("**"):
            current_content.append(line)
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections
