#!/bin/bash
# Push model artifacts to Hugging Face Hub
# Usage: ./scripts/push_to_hub.sh [HF_TOKEN]

set -e

HF_TOKEN=${1:-$HF_TOKEN}
EXPERIMENT_NAME=${2:-v1_baseline}
MODEL_PATH="outputs/${EXPERIMENT_NAME}/merged"
REPO_NAME="W-Kaski/fingpt-qlora-qwen2.5-7b"

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set. Pass as argument or set env var."
    echo "Usage: ./scripts/push_to_hub.sh hf_xxxxxxxx"
    exit 1
fi

echo "=== Pushing model to Hugging Face Hub ==="
echo "Model path: $MODEL_PATH"
echo "Repo: $REPO_NAME"

python -c "
from huggingface_hub import login
login(token='$HF_TOKEN')
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='$MODEL_PATH',
    max_seq_length=2048,
    load_in_4bit=True,
)
model.push_to_hub_merged('$REPO_NAME', tokenizer, save_method='merged_16bit')
print('Pushed to Hub successfully!')
"

echo "=== Done ==="
echo "Model available at: https://huggingface.co/$REPO_NAME"
