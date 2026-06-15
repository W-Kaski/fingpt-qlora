#!/bin/bash
# Kaggle environment setup script
# Run this at the start of each Kaggle notebook

set -e

echo "=== FinGPT-QLoRA Kaggle Setup ==="

# Install Unsloth
pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# Install other dependencies
pip install rouge_score bert_score datasets gradio

# Clone project repo (if not already present)
if [ ! -d "fingpt-qlora" ]; then
    git clone https://github.com/W-Kaski/fingpt-qlora.git
fi

cd fingpt-qlora

# Verify GPU
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "=== Setup complete ==="
