# ============================================================
# FastAPI REST API — serves PhoBERT sentiment predictions
# Endpoints: health check, single prediction, batch prediction
# ONNX inference endpoint — faster alternative to PyTorch
# ============================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from app.model import predict, load_model, LABEL_NAMES
from pathlib import Path

app = FastAPI(
    title="PhoBERT Vietnamese Sentiment API",
    description="Sentiment analysis for Vietnamese text using fine-tuned PhoBERT",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Môn học này rất hay và bổ ích")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32)

class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    predicted_label: str
    confidence: float
    probabilities: dict


@app.on_event("startup")
async def startup():
    load_model()
    print("API ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "phobert-base", "labels": LABEL_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: TextRequest):
    result = predict(request.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    results = []
    for text in request.texts:
        result = predict(text)
        results.append(result)
    return {"predictions": results, "count": len(results)}

ONNX_PATH = Path(__file__).parent.parent / "models" / "phobert-sentiment" / "model.onnx"

if ONNX_PATH.exists():
    from app.model_onnx import predict_onnx, load_onnx_model

    @app.on_event("startup")
    async def startup_onnx():
        load_onnx_model()
        print("ONNX model ready.")

    @app.post("/predict/onnx", response_model=PredictionResponse)
    async def predict_sentiment_onnx(request: TextRequest):
        result = predict_onnx(request.text)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result