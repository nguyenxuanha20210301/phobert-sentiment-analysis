# ============================================================
# ONNX inference — faster alternative to PyTorch runtime
# Uses onnxruntime for hardware-optimized prediction
# ============================================================
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer
from app.model import preprocess_vietnamese

MODEL_DIR = Path(__file__).parent.parent / "models" / "phobert-sentiment"
ONNX_PATH = MODEL_DIR / "model.onnx"
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
MAX_LENGTH = 128

_session = None
_tokenizer = None


def load_onnx_model():
    """Load ONNX model and tokenizer once, cache globally."""
    global _session, _tokenizer
    if _session is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"]
        )
        print("ONNX model loaded.")
    return _session, _tokenizer


def predict_onnx(text: str) -> dict:
    """Run prediction using ONNX runtime."""
    session, tokenizer = load_onnx_model()

    cleaned = preprocess_vietnamese(text)
    if not cleaned:
        return {"error": "Empty text after preprocessing"}

    encoding = tokenizer(
        cleaned,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    logits = session.run(
        ["logits"],
        {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64)
        }
    )[0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs = probs[0]

    predicted_class = int(np.argmax(probs))

    return {
        "text": text,
        "cleaned_text": cleaned,
        "predicted_label": LABEL_NAMES[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            label: float(prob) for label, prob in zip(LABEL_NAMES, probs)
        },
        "runtime": "onnx"
    }