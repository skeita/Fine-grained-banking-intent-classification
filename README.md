# Fine-Grained Banking Intent Classification

This individual project studies banking-intent classification on BANKING77. It compares a lexical baseline, a PyTorch BiLSTM, frozen DistilBERT, fully fine-tuned DistilBERT, and one controlled learning-rate experiment.

**Author:** Saibou KEITA  
**Primary metric:** Macro-F1  
**Reproducibility seed:** 42

## Dataset

This project uses the public [PolyAI/banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77) through the Hugging Face datasets library. BANKING77 contains 77 fine-grained banking intents and customer support queries.

The official training split is divided into training and validation data using seed 42. The official test split is kept untouched until final evaluation. Raw dataset files are downloaded when the notebooks run and are not stored in this repository.

## Results

The strongest measured experiment is fully fine-tuned DistilBERT with test Macro-F1 **0.9106**. All reported values come from executed artifacts; the official test split remains untouched until final evaluation.

| Experiment | Test Macro-F1 |
|---|---:|
| E0 TF-IDF + Logistic Regression | 0.8558 |
| E1 BiLSTM | 0.8092 |
| E2 Frozen DistilBERT | 0.6705 |
| E3 Fine-tuned DistilBERT | 0.9106 |
| E4 Fine-tuned DistilBERT, learning rate 5e-5 | 0.9104 |

The complete table is in [`results/metrics.csv`](results/metrics.csv). The main report is [`report/final_report.md`](report/final_report.md). An extended Word version is available as [`report/final_report_8_pages.docx`](report/final_report_8_pages.docx) when that document is included in the repository.

## Environment setup

Run all commands from the repository root:

```bash
cd /home/etudiant-keita/NLP-Banking
```

Create the documented Conda environment:

```bash
conda env create -f environment.yml
```

Or install the Python dependencies into an existing environment:

```bash
conda run -n nlp-banking python -m pip install -r requirements.txt
```

The project uses Python 3.11. The `requirements.txt` file pins `datasets` to `3.6.0` for the BANKING77 loader.

Optional Hugging Face authentication for higher download limits:

```bash
export HF_TOKEN=your_token_here
```

The token is read by the project and is not printed or stored.

## Run the notebooks

Start Jupyter Lab:

```bash
conda run -n nlp-banking jupyter lab
```

Run the notebooks in order:

1. `notebooks/00_setup_reproducibility_executed.ipynb`
2. `notebooks/01_eda_executed.ipynb`
3. `notebooks/02_baseline_executed.ipynb`
4. `notebooks/03_bilstm_executed.ipynb`
5. `notebooks/04_distilbert_frozen_executed.ipynb`
6. `notebooks/05_distilbert_finetuning_executed.ipynb`
7. `notebooks/06_experiments_lr_5e-5_executed.ipynb`
8. `notebooks/07_evaluation_executed.ipynb`

## Start the Streamlit application

Start the local inference dashboard in another terminal:

```bash
cd /home/etudiant-keita/NLP-Banking
conda run -n nlp-banking streamlit run app/app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.fileWatcherType none
```

Open the deployed app at [https://fine-grained-banking-intent-classification-a4g9nex9ivsdxoxskgc.streamlit.app/](https://fine-grained-banking-intent-classification-a4g9nex9ivsdxoxskgc.streamlit.app/). For local use, open [http://127.0.0.1:8501](http://127.0.0.1:8501).

The Streamlit interface contains:

- **Data explorer:** dataset sizes, class balance, query length, and EDA figures;
- **Predict intent:** editable banking query box, suggestions, confidence, and intent name;
- **Intent atlas:** per-intent metrics, real examples, and categorized confusion pairs;
- **Model lab:** final comparison and training/validation curves;
- **Error analysis:** top confusion pairs and their counts.

To stop Streamlit, press `Ctrl+C` in its terminal.

## Validation and quality checks

Run the available tests and code quality checks:

```bash
conda run -n nlp-banking python -m pytest -q
conda run -n nlp-banking ruff check src tests app
conda run -n nlp-banking ruff format --check src tests app
```

The Streamlit interface can also be smoke-tested with:

```bash
conda run -n nlp-banking python -m py_compile app/app.py
```

## Project structure
- `PROJECT_PLAN.md` — workflow and completion status;
- `configs/base.yaml` — canonical experiment configuration;
- `notebooks/` — sequential executed labs;
- `src/` — reusable data, feature, model, evaluation, and tracking code;
- `results/` — metrics, predictions, logs, and model artifacts;
- `figures/` — EDA, comparison, and training-history plots;
- `app/app.py` — interactive Streamlit dashboard;
- `report/` — Markdown and Word final reports.

## Scientific safeguards

The project keeps the official test split untouched until final evaluation, creates validation data only from the original training split, uses seed 42, records configuration and package information, separates observations from interpretations, and does not fabricate metrics, timings, GPU measurements, or conclusions.
