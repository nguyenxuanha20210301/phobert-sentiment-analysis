# ============================================================
# Benchmark: PyTorch vs ONNX inference speed
# Demonstrates deployment optimization for production
# ============================================================
import time
from app.model import predict as predict_pytorch
from app.model_onnx import predict_onnx

test_texts = [
    "Thầy dạy rất hay và dễ hiểu",
    "Môn học này quá khó, không hiểu gì cả",
    "Bình thường, không có gì đặc biệt",
    "Giảng viên nhiệt tình, tài liệu đầy đủ",
    "Phòng học nóng, máy chiếu hỏng liên tục",
]

N_RUNS = 20

# Warmup
predict_pytorch(test_texts[0])
predict_onnx(test_texts[0])

# Benchmark PyTorch
start = time.time()
for _ in range(N_RUNS):
    for text in test_texts:
        predict_pytorch(text)
pytorch_time = time.time() - start

# Benchmark ONNX
start = time.time()
for _ in range(N_RUNS):
    for text in test_texts:
        predict_onnx(text)
onnx_time = time.time() - start

total_predictions = N_RUNS * len(test_texts)

print("=" * 50)
print("INFERENCE BENCHMARK")
print("=" * 50)
print(f"Total predictions: {total_predictions}")
print(f"PyTorch: {pytorch_time:.2f}s ({total_predictions/pytorch_time:.1f} pred/s)")
print(f"ONNX:    {onnx_time:.2f}s ({total_predictions/onnx_time:.1f} pred/s)")
print(f"Speedup: {pytorch_time/onnx_time:.2f}x")

# Verify predictions match
for text in test_texts:
    pt = predict_pytorch(text)
    ox = predict_onnx(text)
    match = pt["predicted_label"] == ox["predicted_label"]
    print(f"{'✅' if match else '❌'} {text[:40]}  PT={pt['predicted_label']} ONNX={ox['predicted_label']}")