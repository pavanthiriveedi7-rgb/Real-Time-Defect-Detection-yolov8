import time

import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort


# ==========================================
# Configuration
# ==========================================

PT_MODEL = "models/best_32epoch.pt"
ONNX_MODEL = "models/best_32epoch.onnx"

IMG_SIZE = 320
NUM_WARMUP = 20
NUM_RUNS = 100


# ==========================================
# Create fixed test input
# ==========================================

image = np.zeros(
    (IMG_SIZE, IMG_SIZE, 3),
    dtype=np.uint8
)


# ==========================================
# Prepare ONNX input
# ==========================================

onnx_input = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

onnx_input = (
    onnx_input.astype(np.float32)
    / 255.0
)

onnx_input = np.transpose(
    onnx_input,
    (2, 0, 1)
)

onnx_input = np.expand_dims(
    onnx_input,
    axis=0
)


# ==========================================
# Load PyTorch model
# ==========================================

print()
print("=" * 55)
print("Loading PyTorch model")
print("=" * 55)

pt_model = YOLO(PT_MODEL)

print("PyTorch model loaded.")


# ==========================================
# PyTorch benchmark
# ==========================================

print()
print("=" * 55)
print("PyTorch YOLOv8 Benchmark")
print("=" * 55)

# Warm-up
for _ in range(NUM_WARMUP):

    pt_model.predict(
        source=image,
        imgsz=IMG_SIZE,
        device="cpu",
        verbose=False
    )


pt_times = []

for _ in range(NUM_RUNS):

    start = time.perf_counter()

    pt_model.predict(
        source=image,
        imgsz=IMG_SIZE,
        device="cpu",
        verbose=False
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    pt_times.append(elapsed)


pt_avg = float(np.mean(pt_times))
pt_min = float(np.min(pt_times))
pt_max = float(np.max(pt_times))

print(f"Warm-up runs       : {NUM_WARMUP}")
print(f"Measured runs      : {NUM_RUNS}")
print(f"Average latency    : {pt_avg:.2f} ms")
print(f"Minimum latency    : {pt_min:.2f} ms")
print(f"Maximum latency    : {pt_max:.2f} ms")
print(f"Approx FPS         : {1000 / pt_avg:.2f}")


# ==========================================
# Load ONNX model
# ==========================================

print()
print("=" * 55)
print("Loading ONNX model")
print("=" * 55)

session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded.")
print("Input name:", input_name)


# ==========================================
# ONNX benchmark
# ==========================================

print()
print("=" * 55)
print("ONNX Runtime Benchmark")
print("=" * 55)

# Warm-up
for _ in range(NUM_WARMUP):

    session.run(
        None,
        {
            input_name: onnx_input
        }
    )


onnx_times = []

for _ in range(NUM_RUNS):

    start = time.perf_counter()

    session.run(
        None,
        {
            input_name: onnx_input
        }
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    onnx_times.append(elapsed)


onnx_avg = float(np.mean(onnx_times))
onnx_min = float(np.min(onnx_times))
onnx_max = float(np.max(onnx_times))

print(f"Warm-up runs       : {NUM_WARMUP}")
print(f"Measured runs      : {NUM_RUNS}")
print(f"Average latency    : {onnx_avg:.2f} ms")
print(f"Minimum latency    : {onnx_min:.2f} ms")
print(f"Maximum latency    : {onnx_max:.2f} ms")
print(f"Approx FPS         : {1000 / onnx_avg:.2f}")


# ==========================================
# Comparison
# ==========================================

print()
print("=" * 55)
print("Performance Comparison")
print("=" * 55)

print(
    f"PyTorch average   : {pt_avg:.2f} ms"
)

print(
    f"ONNX average      : {onnx_avg:.2f} ms"
)

if onnx_avg < pt_avg:

    improvement = (
        (pt_avg - onnx_avg)
        / pt_avg
        * 100
    )

    speedup = (
        pt_avg / onnx_avg
    )

    print(
        f"ONNX improvement  : {improvement:.2f}%"
    )

    print(
        f"ONNX speedup      : {speedup:.2f}x"
    )

else:

    difference = (
        (onnx_avg - pt_avg)
        / pt_avg
        * 100
    )

    print(
        f"ONNX difference   : {difference:.2f}% slower"
    )


print()
print("=" * 55)
print("Benchmark complete")
print("=" * 55)