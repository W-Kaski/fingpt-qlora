# FinGPT-QLoRA

Financial sentiment assistant fine-tuned from `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` with QLoRA on Kaggle T4.

## Current Experiment

The repository tracks two experiments: `v3_completion_only` (baseline) and `v4_improved` (active).

### v4_improved (active)

| Item | Value |
| --- | --- |
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Training backend | Kaggle Notebook with T4 GPU |
| Dataset format | prompt-completion |
| Loss | `completion_only_loss=True` |
| Train size | 5000 examples |
| Epochs | 2 |
| Effective batch | 16 |
| Learning rate | `1e-4` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Key change | 15 implications/sentiment, 5 output format variants |

### v3_completion_only (baseline)

| Metric | Base | Fine-tuned | Delta |
| --- | ---: | ---: | ---: |
| Sentiment accuracy | 0.581 | 0.833 | +0.252 |
| ROUGE-L | 0.230 | 0.981 | +0.751 |
| Format compliance | n/a | 20/20 | n/a |

ROUGE-L is treated as a format/text similarity signal, not as proof of financial reasoning.

v4 diagnostic-first workflow: run `v4_diagnostic.yaml` (10 steps) first, then `v4_improved.yaml`.

## Repository Role

Kaggle is the remote training backend. This repository is the control plane:

- source-controlled configs
- data formatting code
- notebook and Kaggle CLI workflow
- result manifests and evaluation tables
- demo and inference helpers

Large data, checkpoints, and adapters are not committed.

## Local Setup

```bash
pip install -r requirements.txt
```

Run local validation:

```bash
python -m unittest discover tests
python -m compileall -q src tests
```

## Data Pipeline

Canonical implementation lives in `src.data`. The preprocessing notebook is only a Kaggle wrapper.

```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.data.format_chat
python -m src.data.merge_datasets
python -m src.data.splits
python -m src.data.manifest --splits-dir data/splits --output results/data_manifest_v3.json
```

Generated files go under `data/` and are ignored by git.

## Training

Active config:

```bash
configs/experiments/v4_improved.yaml
```

Diagnostic config:

```bash
configs/experiments/v4_diagnostic.yaml
```

Run the diagnostic first:

```bash
python -m src.train.train_sft \
  --config configs/experiments/v4_diagnostic.yaml \
  --output-suffix diagnostic
```

Inspect diagnostic artifacts before full training:

- `outputs/v4_diagnostic-diagnostic/run_manifest.json`
- `outputs/v4_diagnostic-diagnostic/trainer_state.json`
- `outputs/v4_diagnostic-diagnostic/config_used.yaml`

Full training entrypoint:

```bash
python -m src.train.train_sft \
  --config configs/experiments/v4_improved.yaml \
  --output-suffix kaggle
```

For actual GPU training, use the Kaggle workflow in `docs/KAGGLE_WORKFLOW.md`.

Each training run writes reproducibility artifacts next to the adapter:

- `config_used.yaml`
- `trainer_state.json`
- `run_manifest.json`

The data pipeline writes:

- `results/data_manifest_v3.json`

## Evaluation

Evaluation entrypoint:

```bash
python -m src.eval.run_evaluation \
  --test-data data/splits/test.json \
  --base-model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
  --adapter-path outputs/v4_improved-kaggle \
  --output-dir results/eval_outputs/v4_improved
```

The script writes:

- `metrics.json`
- `predictions.jsonl`
- `comparison_table.md`

`notebooks/05_evaluation/` is a Kaggle wrapper around this script.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_data_exploration/` | optional exploratory inspection only |
| `notebooks/02_data_preprocessing/` | Kaggle wrapper for `src.data` pipeline |
| `notebooks/04_training/` | v3 QLoRA diagnostic wrapper |
| `notebooks/05_evaluation/` | base vs fine-tuned evaluation |
| `notebooks/06_demo/` | adapter-backed demo notebook |

Old diagnostic, baseline, and GGUF notebooks were removed from the active workflow.

## Demo

```bash
MODEL_PATH=outputs/lora_adapter python demo/app.py
```

`MODEL_PATH` can point to a local adapter folder or a hosted adapter/model path.

## License

MIT License.
