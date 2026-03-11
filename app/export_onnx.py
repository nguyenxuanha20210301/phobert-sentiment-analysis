# ============================================================
# Export PhoBERT checkpoint to ONNX format for faster inference
# ONNX enables hardware-optimized inference without PyTorch
# ============================================================
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).parent.parent / "models" / "phobert-sentiment"
ONNX_PATH = MODEL_DIR / "model.onnx"
MAX_LENGTH = 128

def export():
    print("Loading PyTorch model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    dummy_text = "Môn học này rất hay"
    inputs = tokenizer(
        dummy_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    print("Exporting to ONNX (legacy exporter)...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )

    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved: {ONNX_PATH}")
    print(f"Size: {size_mb:.1f} MB")
    print("Model verification passed.")


if __name__ == "__main__":
    export()