# Sources for the project techniques

This file links each technique used in the project to a practical implementation source and, where useful, an original paper. The local course files are included in the repository under `references/course/` and `references/papers/`.

## 1. Dataset and data splitting

- [BANKING77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) — dataset used by this project.
- [BANKING77 dataset information](https://huggingface.co/datasets/PolyAI/banking77/blob/main/dataset_infos.json) — labels, split sizes, and dataset metadata.
- [Hugging Face Datasets: loading datasets](https://huggingface.co/docs/datasets/en/loading) — `load_dataset`.
- [Hugging Face Datasets: `train_test_split`](https://huggingface.co/docs/datasets/v2.5.1/en/package_reference/main_classes#datasets.Dataset.train_test_split) — seeded and stratified splitting.
- Local project code: [`src/data/load_data.py`](../src/data/load_data.py).

## 2. TF-IDF baseline

- [scikit-learn `TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — converts text into TF-IDF features and supports word n-grams.
- [scikit-learn `LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — classification model used after TF-IDF.
- Local course reference: [`references/course/char_rnn_classification.ipynb`](course/char_rnn_classification.ipynb) for the broader text-classification context.
- Project notebook: [`notebooks/02_baseline.ipynb`](../notebooks/02_baseline.ipynb).

## 3. Tokenisation, vocabulary, and embeddings

- [PyTorch `Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — learns a vector representation for each vocabulary item.
- Local course notebook: [`5_Embeddings.ipynb`](course/5_Embeddings.ipynb).
- Project implementation: [`src/models/bilstm.py`](../src/models/bilstm.py), including `<PAD>` and `<UNK>`.

## 4. BiLSTM sequence classifier

- [PyTorch `LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — recurrent sequence model used in the BiLSTM.
- [PyTorch `pack_padded_sequence`](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html) — prevents padded tokens from affecting the recurrent computation.
- [Hochreiter and Schmidhuber, Long Short-Term Memory](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory) — original LSTM paper.
- Local course notebook: [`6_LSTM.ipynb`](course/6_LSTM.ipynb).
- Project implementation: [`src/models/bilstm.py`](../src/models/bilstm.py).

## 5. Transformer architecture

- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original Transformer paper.
- Local paper copy: [`Vaswani et al (2017) Attention si all you need.pdf`](papers/Vaswani%20et%20al%20(2017)%20Attention%20si%20all%20you%20need.pdf).
- Local course material: [`Transformers_course_DSTI_ch02_architecture_and_functioning.pdf`](course/Transformers_course_DSTI_ch02_architecture_and_functioning.pdf).
- Local notebook: [`7_Transformers_introduction.ipynb`](course/7_Transformers_introduction.ipynb).

## 6. BERT and DistilBERT

- [Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — BERT pretraining and contextual representations.
- [Sanh et al., DistilBERT](https://arxiv.org/abs/1910.01108) — the distillation method behind DistilBERT.
- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) — checkpoint, intended use, and loading examples.
- Local paper copy: [`Jawahar et al - 2019 - What does BERT learn about the structure of language.pdf`](papers/Jawahar%20et%20al%20-%202019%20-%20What%20does%20BERT%20learn%20about%20the%20structure%20of%20language.pdf).
- Project model identifier: `distilbert/distilbert-base-uncased`.

## 7. Sequence classification and fine-tuning

- [Hugging Face sequence-classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) — tokenizer, sequence-classification head, training, and evaluation workflow.
- [Hugging Face Auto Classes](https://huggingface.co/docs/transformers/en/model_doc/auto) — `AutoTokenizer` and `AutoModelForSequenceClassification`.
- [Hugging Face fine-tuning guide](https://huggingface.co/docs/transformers/en/training) — what fine-tuning means and how tokenized batches are prepared.
- Project implementation: [`src/models/transformer.py`](../src/models/transformer.py).

## 8. Evaluation metrics and error analysis

- [scikit-learn classification metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html) — accuracy, precision, recall, F1, classification reports, and confusion matrices.
- [scikit-learn `confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) — confusion-pair analysis.
- Project implementation: [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py).
- Project outputs: [`results/metrics.csv`](../results/metrics.csv), [`results/per_class_metrics.csv`](../results/per_class_metrics.csv), and [`results/top_confusions.csv`](../results/top_confusions.csv).

## 9. Reproducibility

- [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) — random seeds and reproducibility limitations.
- Project utility: [`src/utils/__init__.py`](../src/utils/__init__.py).
