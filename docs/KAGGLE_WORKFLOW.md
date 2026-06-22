# Kaggle Remote Workflow

Kaggle is the source of truth for training runs. Local code controls configs, notebooks, and result ingestion.

## Prerequisites

```bash
kaggle config view
```

The expected Kaggle account is the one that owns:

```text
ericwang7717/fingpt-lora-adapter-v3
```

## Standard Flow

1. Validate local code.

```bash
python -m unittest discover tests
python -m compileall -q src tests
```

2. Optionally push or update the data preprocessing notebook.

```bash
bash scripts/kaggle_push_notebook.sh notebooks/02_data_preprocessing
bash scripts/kaggle_status.sh notebooks/02_data_preprocessing
```

This notebook is a wrapper around `src.data`; it is not a separate data implementation.

3. Push or update the Kaggle diagnostic training notebook.

```bash
bash scripts/kaggle_push_notebook.sh notebooks/04_training
```

4. Check remote diagnostic status.

```bash
bash scripts/kaggle_status.sh notebooks/04_training
```

Inspect the diagnostic output before any full run. Required diagnostic checks:

- model loads
- data pipeline finishes
- prompt-completion dataset is accepted
- 10 training steps finish
- loss is finite
- adapter and run artifacts are saved
- no OOM or TRL argument error occurs

5. After diagnostic passes, run full training from the same commit:

```bash
python -m src.train.train_sft \
  --config configs/experiments/v3_completion_only.yaml \
  --output-suffix kaggle
```

6. Push or update the evaluation notebook after the adapter is published.

```bash
bash scripts/kaggle_push_notebook.sh notebooks/05_evaluation
bash scripts/kaggle_status.sh notebooks/05_evaluation
```

7. Pull adapter and evaluation outputs.

```bash
bash scripts/kaggle_pull_adapter.sh
bash scripts/kaggle_pull_results.sh
```

8. Update result docs only from downloaded artifacts.

Evaluation artifacts are expected under:

```text
results/eval_outputs/v3_completion_only/
```

## Rules

- Do not manually type metrics into README unless they are copied from a committed result artifact.
- Do not create a new active experiment without adding a new YAML config under `configs/experiments/`.
- Do not train from notebook-only hardcoded hyperparameters.
- Do not commit `data/`, `outputs/`, checkpoints, adapters, or raw Kaggle credentials.
