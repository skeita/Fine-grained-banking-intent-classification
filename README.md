# Fine-Grained Banking Intent Classification

An independent Deep Learning NLP project comparing TF-IDF, a PyTorch BiLSTM, and DistilBERT transfer-learning strategies on BANKING77.

## Completed workflow

- BANKING77 loading with a stratified validation split and untouched official test set
- EDA: counts, missing values, duplicates, class distribution, and query length
- E0: TF-IDF + Logistic Regression
- E1: PyTorch BiLSTM
- E2: frozen DistilBERT
- E3: fully fine-tuned DistilBERT
- E4: controlled learning-rate comparison
- Final test evaluation, per-class metrics, confusion matrix, and error analysis

## Reproduced test results

See [`results/metrics.csv`](results/metrics.csv) for the generated metrics. The strongest measured experiment was E3 fine-tuned DistilBERT with test Macro-F1 0.9106.

## Setup

Use the existing Conda environment described by `environment.yml`, or create a compatible environment and install `requirements.txt`. Datasets is pinned to 3.6.0 because the BANKING77 repository uses its official remote dataset script.

```bash
conda env create -f environment.yml
conda run -n nlp-banking jupyter lab
```

## Important files

- `AGENTS.md` — persistent project and scientific rules
- `PROJECT_PLAN.md` — workflow status
- `notebooks/` — executed experiment notebooks
- `src/` — reusable data, model, metric, and utility code
- `results/` — generated metrics, predictions, logs, and checkpoints
- `report/final_report.md` — concise evidence-based report
