# NLP Banking — Project Plan

## Objective

Compare classical, recurrent, and pretrained Transformer approaches for
fine-grained intent classification on BANKING77.

## Experiments

- E0 — TF-IDF + Logistic Regression
- E1 — PyTorch BiLSTM
- E2 — Frozen DistilBERT
- E3 — Fully fine-tuned DistilBERT
- E4 — Controlled learning-rate comparison: `2e-5` vs `5e-5`

## Main metric

Macro-F1.

## Required notebooks

1. `01_eda.ipynb`
2. `02_baseline.ipynb`
3. `03_bilstm.ipynb`
4. `04_distilbert_frozen.ipynb`
5. `05_distilbert_finetuning.ipynb`
6. `06_experiments.ipynb`
7. `07_evaluation.ipynb`

## Status

- [x] Repository and environment
- [x] Dataset and EDA
- [x] Classical baseline
- [x] BiLSTM
- [x] Frozen DistilBERT
- [x] Fine-tuned DistilBERT
- [x] Controlled experiment
- [x] Final evaluation
- [x] Error analysis
- [x] Report
- [ ] Reproducibility check
