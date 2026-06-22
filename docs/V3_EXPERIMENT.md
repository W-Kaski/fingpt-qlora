# v3_completion_only Experiment Contract

## Status

`v3_completion_only` is the baseline experiment. `v4_improved` is now the active experiment.

## v3 → v4 Changes

| Aspect | v3 | v4 |
| --- | --- | --- |
| IMPLICATIONS pool | 4 per sentiment | 15 per sentiment |
| Output formats | 1 fixed template | 5 variants (hash-selected) |
| Key factor keywords | 13 | 23 (added rates, inflation, guidance, etc.) |
| Epochs | 1 | 2 |
| Diagnostic config | `v3_diagnostic.yaml` | `v4_diagnostic.yaml` |
| Active config | `v3_completion_only.yaml` | `v4_improved.yaml` |

## Training Contract

| Field | Value |
| --- | --- |
| Config | `configs/experiments/v3_completion_only.yaml` |
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Runtime | Kaggle T4 GPU |
| Data format | prompt-completion |
| Loss | `completion_only_loss=True` |
| Train examples | 5000 |
| Eval examples | 1000 maximum |
| Epochs | 1 |
| Learning rate | `0.0001` |
| Warmup ratio | `0.1` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Packing | false |

## Adapter

Current adapter source:

```text
ericwang7717/fingpt-lora-adapter-v3
```

## Required Artifacts Per Run

Diagnostic run must complete before full training.

Each production run must produce:

- LoRA adapter directory
- `trainer_state.json`
- training log
- evaluation prediction JSON
- metrics table
- `results/data_manifest_v3.json` with split hashes and duplicate checks
- `run_manifest.json` with git commit and config copy
- Kaggle notebook version or dataset version

## Evaluation Contract

Primary metrics:

- sentiment accuracy
- macro F1
- format compliance

Secondary metrics:

- ROUGE-L
- bootstrap confidence interval

ROUGE-L must not be presented as a standalone capability metric because the task uses a fixed structured output template.
