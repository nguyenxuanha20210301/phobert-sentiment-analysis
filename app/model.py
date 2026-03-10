# ============================================================
# Shared model loading — used by both FastAPI and Streamlit
# Loads PhoBERT checkpoint once and provides a predict function
# ============================================================
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import re

MODEL_DIR = Path(__file__).parent.parent / "models" / "phobert-sentiment"
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
MAX_LENGTH = 128

_model = None
_tokenizer = None
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model():
    """Load model and tokenizer once, cache globally."""
    global _model, _tokenizer
    if _model is None:
        device = get_device()
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.to(device)
        _model.eval()
        print(f"Model loaded on {device}")
    return _model, _tokenizer


def preprocess_vietnamese(text: str) -> str:
    """Same preprocessing pipeline as Phase 1 — must be identical."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = text.strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = re.sub(
        r'[^\w\s.,!?;:\-àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
        ' ', text, flags=re.IGNORECASE
    )
    text = word_tokenize(text, format="text")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(text: str) -> dict:
    """Run full pipeline: preprocess → tokenize → predict → return results."""
    model, tokenizer = load_model()
    device = get_device()

    cleaned = preprocess_vietnamese(text)
    if not cleaned:
        return {"error": "Empty text after preprocessing"}

    encoding = tokenizer(
        cleaned,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probs))

    return {
        "text": text,
        "cleaned_text": cleaned,
        "predicted_label": LABEL_NAMES[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            label: float(prob) for label, prob in zip(LABEL_NAMES, probs)
        }
    }