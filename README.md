# Fine-Grained Banking Intent Classification

This project studies how different NLP models interpret short banking customer queries. It compares a simple lexical baseline, a BiLSTM built in PyTorch, and several DistilBERT transfer-learning setups on BANKING77.

## What has been completed

- BANKING77 loading with a stratified validation split and an untouched official test set
- Exploratory analysis of class sizes, missing values, duplicate queries, and query length
- E0: TF-IDF + Logistic Regression
- E1: PyTorch BiLSTM
- E2: frozen DistilBERT
- E3: fully fine-tuned DistilBERT
- E4: controlled learning-rate comparison
- Final test evaluation, per-class metrics, confusion matrix, and error analysis

## Results

The generated results are in [`results/metrics.csv`](results/metrics.csv). The strongest measured experiment was fine-tuned DistilBERT, with a test Macro-F1 of 0.9106.

## Setup

Use the existing Conda environment described by `environment.yml`, or create a compatible environment and install `requirements.txt`. Datasets is pinned to 3.6.0 because the BANKING77 repository uses its official remote dataset script.

```bash
conda env create -f environment.yml
conda run -n nlp-banking jupyter lab
```

## Where to look

- `AGENTS.md` — project rules and scientific safeguards
- `PROJECT_PLAN.md` — the experiment roadmap and current status
- `notebooks/` — the runnable experiments and their executed versions
- `src/` — reusable data, model, metric, and reproducibility code
- `results/` — generated metrics, predictions, logs, and evaluation artifacts
- `report/final_report.md` — the concise final write-up
