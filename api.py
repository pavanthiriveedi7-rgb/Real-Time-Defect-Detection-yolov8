import logging
import time
from io import BytesIO
from typing import List

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

logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "task": "detect"
    }


def _predict_single_image(
    image_bytes: bytes,
    filename: str
):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"The file '{filename}' is not a valid image."
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

    return {
        "filename": filename,
        "inference_ms": round(inference_ms, 2),
        "detections_count": len(predictions),
        "detections": predictions
    }, results[0]


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

    result_dict, result_obj = _predict_single_image(image_bytes, file.filename)

    logger.info(
        f"POST /predict | file={file.filename} | "
        f"inference_ms={result_dict['inference_ms']} | "
        f"detections={result_dict['detections_count']} | annotate={annotate}"
    )

    if annotate:
        annotated_image = result_obj.plot()
        _, buffer = cv2.imencode(".jpg", annotated_image)

        return Response(
            content=bytes(buffer),
            media_type="image/jpeg",
            headers={
                "X-Inference-Ms": str(result_dict["inference_ms"]),
                "X-Detections-Count": str(result_dict["detections_count"])
            }
        )

    return result_dict


@app.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    annotate: bool = Query(False)
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were uploaded."
        )

    results_list = []
    annotated_buffers = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not an image."
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is empty."
            )

        result_dict, result_obj = _predict_single_image(image_bytes, file.filename)
        results_list.append(result_dict)

        if annotate:
            annotated_image = result_obj.plot()
            _, buffer = cv2.imencode(".jpg", annotated_image)
            annotated_buffers.append((file.filename, buffer))

    total_inference_ms = sum(r["inference_ms"] for r in results_list)

    logger.info(
        f"POST /predict-batch | files={len(files)} | "
        f"total_inference_ms={round(total_inference_ms, 2)} | annotate={annotate}"
    )

    if annotate and len(annotated_buffers) == 1:
        filename, buffer = annotated_buffers[0]
        return Response(
            content=bytes(buffer),
            media_type="image/jpeg",
            headers={
                "X-Inference-Ms": str(results_list[0]["inference_ms"]),
                "X-Detections-Count": str(results_list[0]["detections_count"])
            }
        )

    return {
        "files_processed": len(files),
        "total_inference_ms": round(total_inference_ms, 2),
        "results": results_list
    }