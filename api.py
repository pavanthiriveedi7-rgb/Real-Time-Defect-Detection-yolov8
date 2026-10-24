"""Steel Surface Inspector API.  Run:  uvicorn api:app --host 0.0.0.0 --port 8000
Env: MODEL_PATH (default models/best_v2.onnx), IMGSZ (default 640)."""
import os, threading, time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best_v2.onnx"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Steel Surface Inspector")
model = YOLO(str(MODEL_PATH), task="detect")
lock = threading.Lock()  # one inference at a time; the model is not thread-safe
model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, verbose=False)  # warm-up


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "imgsz": IMGSZ, "classes": list(model.names.values())}


@app.post("/predict")
def predict(file: UploadFile = File(...), conf: float = Form(0.25), iou: float = Form(0.7)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read this file as an image. Use JPG or PNG.")
    start = time.perf_counter()
    with lock:
        r = model.predict(img, imgsz=IMGSZ, conf=conf, iou=iou, verbose=False)[0]
    ms = (time.perf_counter() - start) * 1000
    dets = [
        {"cls": int(c), "name": model.names[int(c)], "conf": round(float(p), 4),
         "box": [round(float(v), 1) for v in b]}
        for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())
    ]
    dets.sort(key=lambda d: -d["conf"])
    h, w = img.shape[:2]
    return {"width": w, "height": h, "inference_ms": round(ms, 1), "detections": dets}


STATIC = ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
