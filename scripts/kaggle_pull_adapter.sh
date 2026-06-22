#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
command -v kaggle >/dev/null || { echo "kaggle CLI not found. Install with: pip install -U kaggle" >&2; exit 1; }

DATASET="${1:-ericwang7717/fingpt-lora-adapter-v3}"
OUTPUT_DIR="${2:-outputs/lora_adapter}"

mkdir -p "$OUTPUT_DIR"
kaggle datasets download -d "$DATASET" -p "$OUTPUT_DIR" --unzip

echo "Adapter downloaded to $OUTPUT_DIR"
