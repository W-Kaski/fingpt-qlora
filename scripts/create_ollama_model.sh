#!/bin/bash
# Create Ollama model from GGUF file
# Usage: ./scripts/create_ollama_model.sh [GGUF_PATH]

set -e

GGUF_PATH=${1:-"outputs/v1_baseline/gguf"}
MODEL_NAME="fingpt"

echo "=== Creating Ollama model: $MODEL_NAME ==="

# Find the GGUF file
GGUF_FILE=$(find "$GGUF_PATH" -name "*.gguf" -type f | head -1)

if [ -z "$GGUF_FILE" ]; then
    echo "Error: No .gguf file found in $GGUF_PATH"
    exit 1
fi

echo "Using GGUF file: $GGUF_FILE"

# Create Modelfile
cat > /tmp/Modelfile << EOF
FROM $GGUF_FILE

TEMPLATE """{{ .System }}
{{ .Prompt }}"""

SYSTEM """You are FinGPT, a financial analysis assistant. You provide clear, structured analysis of financial data, reports, and market information."""

PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER top_p 0.9
EOF

echo "Modelfile created at /tmp/Modelfile"

# Create Ollama model
ollama create "$MODEL_NAME" -f /tmp/Modelfile

echo "=== Done ==="
echo "Test with: ollama run $MODEL_NAME \"Analyze the sentiment: Apple beats earnings\""
