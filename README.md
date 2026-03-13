# 🇻🇳 Vietnamese Sentiment Analysis with PhoBERT

End-to-end sentiment analysis pipeline for Vietnamese text, from data exploration to production deployment. Fine-tunes [PhoBERT](https://github.com/VinAIResearch/PhoBERT) (VinAI) on the [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) dataset and serves predictions via FastAPI + Streamlit.

## Highlights

- **93.97% accuracy** and **0.83 macro F1** on the UIT-VSFC test set
- Full ML pipeline: EDA → preprocessing → training → hyperparameter tuning → evaluation → error analysis → deployment
- Hyperparameter search with Optuna, experiment tracking with MLflow
- REST API with FastAPI (Swagger docs included)
- Interactive web UI with Streamlit
- Docker support for one-command deployment

## Project Structure

```
phobert-sentiment-analysis/
├── notebooks/
│   ├── 01_data_pipeline.ipynb        # EDA, preprocessing, tokenization
│   ├── 02_training_and_evaluate.ipynb # Baseline + PhoBERT + Optuna tuning
│   └── 03_error_analysis.ipynb       # Error analysis + model comparison
├── app/
│   ├── __init__.py
│   ├── model.py                      # Shared model loading & prediction
│   ├── api.py                        # FastAPI REST API
│   └── streamlit_app.py              # Streamlit web UI
├── models/
│   └── phobert-sentiment/            # Fine-tuned checkpoint (not in repo)
├── scripts/
│   └── clean_notebook.sh             # Strip widget metadata before commit
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Results

| Model | Test Accuracy | Test F1 (Macro) |
|-------|:------------:|:---------------:|
| TF-IDF + Logistic Regression | 0.8850 | 0.7210 |
| PhoBERT (fine-tuned) | 0.9397 | 0.8272 |
| **PhoBERT (Optuna-tuned)** | **TBD** | **TBD** |

> Update the Optuna-tuned row with your actual results after running the notebook.

### Per-Class Performance (PhoBERT)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Negative | 0.95 | 0.97 | 0.96 | 1,409 |
| Neutral | 0.70 | 0.48 | 0.57 | 167 |
| Positive | 0.95 | 0.97 | 0.96 | 1,590 |

> The Neutral class has the lowest F1 due to its small size (5.3% of test set) and inherent ambiguity.

## Quick Start

### Prerequisites

- Python 3.10+
- ~2 GB disk space for model checkpoint
- GPU optional (runs on CPU, slower inference)

### 1. Clone and Setup

```bash
git clone https://github.com/nguyenxuanha20210301/phobert-sentiment-analysis.git
cd phobert-sentiment-analysis

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Model Checkpoint

The fine-tuned model (~515 MB) is not included in the repo. Download it from one of these options:

**Option A: From Google Drive (if you trained the model yourself)**

Copy your checkpoint files into the models directory:

```bash
cp /path/to/your/checkpoints/* models/phobert-sentiment/
```

**Option B: From Hugging Face (recommended for others)**

```bash
# TODO: Upload checkpoint to Hugging Face Hub and update this command
# huggingface-cli download your-username/phobert-sentiment --local-dir models/phobert-sentiment
```

After downloading, verify the checkpoint:

```bash
ls models/phobert-sentiment/
```

Expected files:

```
config.json
model.safetensors
special_tokens_map.json
tokenizer_config.json
vocab.txt
bpe.codes
added_tokens.json
```

### 3. Verify Model

```bash
python3 -c "from app.model import predict; print(predict('Môn học này rất hay'))"
```

Expected output:

```
Model loaded on cpu
{'text': 'Môn học này rất hay', 'predicted_label': 'Positive', 'confidence': 0.9945, ...}
```

### 4. Run the API

```bash
uvicorn app.api:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger documentation.

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and model info |
| POST | `/predict` | Single text prediction |
| POST | `/predict/batch` | Batch prediction (up to 32 texts) |

#### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Thầy dạy rất hay và dễ hiểu"}'
```

#### Example Response

```json
{
  "text": "Thầy dạy rất hay và dễ hiểu",
  "cleaned_text": "Thầy dạy rất hay và dễ hiểu",
  "predicted_label": "Positive",
  "confidence": 0.9965,
  "probabilities": {
    "Negative": 0.0020,
    "Neutral": 0.0015,
    "Positive": 0.9965
  }
}
```

### 5. Run the Web UI

Open a separate terminal:

```bash
cd phobert-sentiment-analysis
source venv/bin/activate
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** — enter Vietnamese text and get instant sentiment predictions.

### 6. Run with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI API | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

Stop with `docker-compose down`.

## Dataset

**UIT-VSFC** (Vietnamese Students' Feedback Corpus) — a benchmark dataset for Vietnamese sentiment analysis published by UIT-NLP group.

| Split | Samples | Negative | Neutral | Positive |
|-------|:-------:|:--------:|:-------:|:--------:|
| Train | 11,426 | 3,037 | 1,522 | 6,867 |
| Validation | 1,583 | 421 | 211 | 951 |
| Test | 3,166 | 1,409 | 167 | 1,590 |

The dataset contains student feedback about university courses, labeled with three sentiment classes. It is loaded directly from [Hugging Face](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback).

## Pipeline Overview

### Phase 1 — Data Pipeline (`01_data_pipeline.ipynb`)

- Load UIT-VSFC dataset from Hugging Face
- Exploratory data analysis: class distribution, text length analysis, top words per class
- Data quality checks: duplicates, missing values, cross-split leakage
- Vietnamese text preprocessing: URL removal, punctuation normalization, word segmentation with `underthesea`
- PhoBERT tokenization analysis and PyTorch Dataset/DataLoader setup

### Phase 2 — Training & Evaluation (`02_training_and_evaluate.ipynb`)

- TF-IDF + Logistic Regression baseline
- PhoBERT fine-tuning with AdamW optimizer, linear warmup, gradient clipping
- Early stopping on validation F1 (patience=2)
- Hyperparameter tuning with Optuna (learning rate, warmup ratio, weight decay)
- Retrain with best Optuna params as the final production model
- All runs tracked with MLflow for experiment comparison
- Training curves, confusion matrix, per-class metrics

### Phase 3 — Error Analysis (`03_error_analysis.ipynb`)

- Misclassification breakdown by confusion pair
- Error rate vs text length analysis
- Prediction confidence analysis (correct vs incorrect)
- High-confidence error inspection
- Baseline vs PhoBERT per-class F1 comparison

### Phase 4 — Deployment

- `app/model.py` — Preprocessing + inference pipeline (identical to training preprocessing)
- `app/api.py` — FastAPI with health check, single and batch prediction
- `app/streamlit_app.py` — Interactive web UI with example texts
- `Dockerfile` + `docker-compose.yml` — Containerized deployment

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language Model | PhoBERT-base (VinAI) |
| Word Segmentation | underthesea |
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers |
| Baseline | scikit-learn (TF-IDF + LogReg) |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Notebooks | Google Colab |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `vinai/phobert-base` |
| Max sequence length | 128 tokens |
| Batch size | 32 |
| Learning rate | 2e-5 (tuned with Optuna) |
| Weight decay | 0.01 (tuned with Optuna) |
| Epochs | 4 (with early stopping) |
| Warmup ratio | 10% (tuned with Optuna) |
| Optimizer | AdamW |
| Scheduler | Linear warmup |
| Gradient clipping | max_norm=1.0 |

## References

- Nguyen, D.Q. & Nguyen, A.T. (2020). [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744). *Findings of EMNLP 2020*.
- Nguyen, K.V. et al. (2018). [UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis](https://arxiv.org/abs/1911.09339). *KSE 2018*.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.