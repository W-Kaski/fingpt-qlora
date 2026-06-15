# FinGPT-QLoRA

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Model-yellow.svg)](https://huggingface.co/)

**Financial analysis chatbot fine-tuned with QLoRA on Qwen2.5-7B-Instruct.**

Fine-tuned for financial sentiment analysis, report summarization, financial Q&A, and investment analysis using Unsloth on Kaggle T4 GPUs.

## Results

<!-- Update after training -->
| Task | Metric | Base Model | Fine-tuned | Delta |
|------|--------|-----------|------------|-------|
| Sentiment | Accuracy | TBD | TBD | TBD |
| Sentiment | ROUGE-L | TBD | TBD | TBD |
| Summary | ROUGE-L | TBD | TBD | TBD |
| QA | BERTScore F1 | TBD | TBD | TBD |
| Reasoning | Exact Match | TBD | TBD | TBD |

## Architecture

```
Qwen2.5-7B-Instruct (4-bit NF4)
        │
        ▼
  QLoRA Adapters (r=16, α=32)
  ┌─────────────────────────────┐
  │ q_proj  k_proj  v_proj      │
  │ o_proj  gate_proj            │
  │ up_proj  down_proj           │
  └─────────────────────────────┘
        │
        ▼
  Financial Instruction Dataset
  (~10k examples, 5 task types)
        │
        ▼
  Deployed: HF Spaces (Gradio)
            GGUF (Ollama)
```

## Quick Start

```bash
# Clone
git clone https://github.com/W-Kaski/fingpt-qlora.git
cd fingpt-qlora

# Install
pip install -r demo/requirements.txt

# Run demo locally (requires Ollama with fingpt model)
ollama run fingpt "Analyze sentiment: Apple beats earnings estimates"
```

## Project Structure

```
fingpt-qlora/
├── notebooks/                  # Each folder = one Kaggle kernel
│   ├── 01_data_exploration/    # Data profiling (no GPU)
│   ├── 02_data_preprocessing/  # Data pipeline (no GPU)
│   ├── 03_training_smoke_test/ # Quick validation: 100 examples, 1 epoch
│   ├── 03_training_qlora/      # Full training (T4, ~2h)
│   ├── 04_evaluation/          # Base vs fine-tuned comparison
│   └── 05_gguf_export_ollama/  # GGUF export for Ollama
├── src/                        # Python modules
│   ├── config.py               # All hyperparameters
│   ├── data/                   # Data pipeline
│   ├── eval/                   # Evaluation metrics
│   ├── train/                  # Training scripts
│   └── inference/              # Inference helpers
├── demo/                       # Gradio demo app
└── scripts/                    # Utility scripts
```

## Methodology

### Data

- **FinGPT Sentiment** (`FinGPT/fingpt-sentiment-train`): ~7.7k financial sentiment examples
- **Financial PhraseBank** (`financial_phrasebank`): ~2.2k sentences with unanimous agreement
- **ConvFinQA** (`FinGPT/ConvFinQA`): ~5k numerical reasoning questions
- Custom-curated financial Q&A and earnings summaries

All data converted to ShareGPT conversational format with task-specific system prompts.

### Training

- **Base Model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Method:** QLoRA (4-bit NF4 quantization + LoRA adapters)
- **LoRA Config:** r=16, α=32, all attention + MLP layers
- **Framework:** Unsloth (2x faster, 70% less VRAM)
- **Effective Batch Size:** 16 (2 × 8 gradient accumulation)
- **Learning Rate:** 2e-4 with cosine scheduler
- **Epochs:** 3
- **Training Time:** ~1.5-2 hours on Kaggle T4

### Evaluation

- Sentiment accuracy and macro F1
- ROUGE-L for summarization quality
- BERTScore F1 for semantic similarity
- Exact match for numerical reasoning
- Bootstrap 95% confidence intervals
- McNemar's test for statistical significance

## Reproduction

### On Kaggle (Recommended)

1. Create a new Kaggle Notebook with GPU (T4) enabled
2. Enable Internet in notebook settings
3. Run notebooks in order: `01` → `02` → `03` → `04` → `05`
4. Total GPU time: ~4-5 hours for full pipeline

### Ablation Experiments

Run `03_training_qlora.ipynb` with different configs:

| Experiment | LORA_R | LEARNING_RATE | NUM_EPOCHS |
|-----------|--------|---------------|------------|
| v1_baseline | 16 | 2e-4 | 3 |
| v2_high_rank | 32 | 1e-4 | 3 |
| v3_few_epochs | 16 | 2e-4 | 1 |
| v4_low_rank | 8 | 2e-4 | 3 |

## Deployment

### HuggingFace Spaces

```bash
# Push model to Hub
./scripts/push_to_hub.sh YOUR_HF_TOKEN

# Create HF Space with Gradio SDK
# Upload demo/app.py and demo/requirements.txt
```

### Ollama (Local)

```bash
# Export GGUF
# Run notebook 05_gguf_export_ollama.ipynb

# Create Ollama model
./scripts/create_ollama_model.sh

# Run
ollama run fingpt "Your financial question here"
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Qwen2.5-7B-Instruct |
| Fine-tuning | QLoRA via Unsloth |
| Framework | HuggingFace PEFT + TRL |
| Training | Kaggle T4 GPU |
| Demo | Gradio ChatInterface |
| Local Inference | GGUF + Ollama |
| Evaluation | ROUGE, BERTScore, Bootstrap CI |

## License

MIT License - see [LICENSE](LICENSE)

## Author

Eric Wang - [GitHub](https://github.com/W-Kaski) | [Website](https://www.anio.me/)
