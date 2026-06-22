#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
command -v kaggle >/dev/null || { echo "kaggle CLI not found. Install with: pip install -U kaggle" >&2; exit 1; }

NOTEBOOK_DIR="${1:?usage: scripts/kaggle_push_notebook.sh <notebook-dir>}"

if [[ ! -f "$NOTEBOOK_DIR/kernel-metadata.json" ]]; then
  echo "Missing kernel-metadata.json in $NOTEBOOK_DIR" >&2
  exit 1
fi

kaggle kernels push -p "$NOTEBOOK_DIR"
