# Fine-Grained Banking Intent Classification

## 1. Project overview

This project evaluates fine-grained banking customer-query classification on the public BANKING77 dataset. The official test set was kept untouched while a stratified 10% validation split was created from the original training set.

## 2. Questions and hypotheses

The comparison asks whether task-trained sequential representations and pretrained contextual representations improve intent classification, and whether full fine-tuning is worth its computational cost compared with a frozen encoder.

## 3. Data and EDA

The executed loader produced 9,002 training examples, 1,001 validation examples, and 3,080 official test examples across 77 intents. The training split contained no missing text/label values and no duplicate texts. Class and query-length figures are in `figures/`.

## 4. Methods

E0 used TF-IDF bigrams with Logistic Regression. E1 used a PyTorch BiLSTM with a training-only vocabulary, 128-dimensional embeddings, a bidirectional 128-hidden-unit LSTM, dropout, Adam, and seed 42. E2 used a frozen `distilbert/distilbert-base-uncased` encoder with a trainable classification head. E3 fine-tuned all DistilBERT parameters. E4 repeated E3 with learning rate 5e-5 instead of 2e-5.

## 5. Results

| Experiment | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| E0 TF-IDF + Logistic Regression | 0.8562 | 0.8558 | 0.8558 |
| E1 BiLSTM | 0.8088 | 0.8092 | 0.8092 |
| E2 Frozen DistilBERT | 0.1565 | 0.6705 | 0.6705 |
| E3 Fine-tuned DistilBERT | 0.9104 | 0.9106 | 0.9106 |
| E4 Fine-tuned DistilBERT, 5e-5 | 0.9104 | 0.9104 | 0.9104 |

These values are taken from `results/metrics.csv` after evaluation on the official test split.

## 6. Error analysis

The most frequent errors include `pending_transfer` predicted as `balance_not_updated_after_bank_transfer`, `why_verify_identity` predicted as `verify_my_identity`, and `card_arrival` predicted as `card_delivery_estimate`. The full top-pair table is in `results/top_confusions.csv`; per-class scores and the confusion matrix are also saved under `results/`.

## 7. Discussion and limitations

The fine-tuned Transformer achieved the strongest measured Macro-F1. The frozen model was substantially weaker in this configuration, indicating that the classification head did not sufficiently adapt from the frozen representations during the short run. The BiLSTM was useful but below the lexical baseline and the fine-tuned Transformer. The 5e-5 run was effectively tied with 2e-5, so this single controlled comparison does not establish a meaningful learning-rate winner.

The runs were performed on the available CUDA device. GPU memory pressure warnings occurred during fine-tuning, so peak GPU memory was not reported as a reliable metric. Additional seeds, calibration analysis, and a more systematic error review would strengthen the study.

## 8. Conclusion

The executed evidence supports fine-tuned DistilBERT as the best of the tested approaches on BANKING77, while preserving the important distinction between measured results and interpretation. All reported numbers are generated artifacts, not estimates.
