import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from ultralytics import YOLO

MODEL_PATH = r"models\best_32epoch.onnx"
IMAGE_SIZE = 320
CONFIDENCE = 0.25

app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0"
)

model = YOLO(MODEL_PATH, task="detect")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "task": "detect"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    annotate: bool = Query(False)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    start_time = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        task="detect",
        verbose=False
    )

    inference_ms = (time.perf_counter() - start_time) * 1000
    detections = results[0].boxes
    predictions = []

    if detections is not None and len(detections) > 0:
        for class_id, confidence, box in zip(
            detections.cls.tolist(),
            detections.conf.tolist(),
            detections.xyxy.tolist()
        ):
            class_index = int(class_id)

            predictions.append({
                "class": model.names[class_index],
                "confidence": round(float(confidence), 4),
                "box": [
                    round(float(value), 2)
                    for value in box
                ]
            })

    if annotate:
        annotated_image = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated_image)

        return Response(
            content=bytes(buffer),
            media_type="image/jpeg",
            headers={
                "X-Inference-Ms": str(round(inference_ms, 2)),
                "X-Detections-Count": str(len(predictions))
            }
        )

    return {
        "filename": file.filename,
        "inference_ms": round(inference_ms, 2),
        "detections_count": len(predictions),
        "detections": predictions
    }