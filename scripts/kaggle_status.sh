#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
command -v kaggle >/dev/null || { echo "kaggle CLI not found. Install with: pip install -U kaggle" >&2; exit 1; }

NOTEBOOK_DIR="${1:-notebooks/04_training}"
METADATA="$NOTEBOOK_DIR/kernel-metadata.json"

if [[ ! -f "$METADATA" ]]; then
  echo "Missing $METADATA" >&2
  exit 1
fi

KERNEL_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$METADATA")"
kaggle kernels status "$KERNEL_ID"
