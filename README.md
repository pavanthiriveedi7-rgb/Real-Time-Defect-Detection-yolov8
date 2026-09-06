# Real-Time Defect Detection (YOLOv8 + ONNX + FastAPI)

Industrial surface-defect detection system built over four weeks:

- Week 1: NEU-DET dataset preparation and YOLO conversion.
- Week 2: YOLOv8 training and validation.
- Week 3: ONNX export and real-time OpenCV inference.
- Week 4: FastAPI service with single/batch prediction and web UI.

## Dataset (Week 1)

This project uses the NEU-DET steel surface defect dataset with six classes:

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled_in_scale`
- `scratches`

### Install

```powershell
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1. Convert and split annotations

The converter expects image files and Pascal-VOC XML files anywhere below `--src`. It creates YOLO labels and a train/validation/test split. YOLO detection labels are one row per object in normalized `class cx cy width height` format, as required by Ultralytics. [Ultralytics dataset format](https://docs.ultralytics.com/datasets/detect)

```powershell
python prepare_dataset.py --src /path/to/NEU-DET --out data/neu
```

### 2. Audit the data

```powershell
python validate_dataset.py --root data/neu
```

Fix every issue before training. Inspect random images with their boxes.

### 3. Augment training only

```powershell
python augment_train.py --root data/neu --copies 1
```

Do not run this script more than once on the same output unless you want repeated augmentation.

### 4. Check the YAML

Edit `data/neu/data.yaml` if necessary. Never augment validation or test data.

**Notes**

- This assumes Pascal-VOC XML annotations. If your download contains only class folders and no XML, it is a classification dataset and bounding boxes must be annotated first with CVAT, Label Studio, or Roboflow.
- Keep the test split untouched until final evaluation.
- The split uses a fixed seed for reproducibility.
- The six class names and aliases match the common NEU-DET naming convention.

## Training (Week 2)

Train a YOLOv8 detection model on the prepared dataset:

```powershell
yolo detect train data=data/neu/data.yaml model=yolov8n.pt epochs=32 imgsz=320
```

Export the trained model to ONNX:

```powershell
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=320
```

The exported file is used for CPU inference:

```text
models/best_32epoch.onnx
```

## Real-time inference (Week 3)

OpenCV-based real-time inference script:

```powershell
python realtime_inference.py
```

Displays:

- Detected defect class.
- Confidence score.
- FPS.
- Inference latency in milliseconds.

## FastAPI service (Week 4)

### Endpoints

#### `GET /health`

Returns model status.

Example response:

```json
{
  "status": "ok",
  "model": "models\\best_32epoch.onnx",
  "task": "detect"
}
```

#### `POST /predict`

Single-image prediction.

Request (multipart/form-data):

- `file`: image file.
- `annotate`: `true` or `false`.

Example JSON response (`annotate=false`):

```json
{
  "filename": "crazing_103_c85c40689d.jpg",
  "inference_ms": 310.22,
  "detections_count": 0,
  "detections": []
}
```

If `annotate=true`, the response is a JPEG image with bounding boxes.

#### `POST /predict-batch`

Batch prediction for multiple images.

Request (multipart/form-data):

- `files`: list of image files.
- `annotate`: `true` or `false`.

Example JSON response:

```json
{
  "files_processed": 2,
  "total_inference_ms": 620.45,
  "results": [
    {
      "filename": "crazing_103_c85c40689d.jpg",
      "inference_ms": 310.22,
      "detections_count": 0,
      "detections": []
    },
    {
      "filename": "scratches_50.jpg",
      "inference_ms": 310.23,
      "detections_count": 0,
      "detections": []
    }
  ]
}
```

### Example logs (`api.log`)

```text
2026-09-05 12:21:35,634 | INFO | POST /predict | file=crazing_103_c85c40689d.jpg | inference_ms=7636.59 | detections=0 | annotate=False
2026-09-05 12:21:37,852 | INFO | POST /predict | file=crazing_103_c85c40689d.jpg | inference_ms=277.59 | detections=0 | annotate=False
2026-09-05 12:21:47,949 | INFO | POST /predict | file=crazing_103_c85c40689d.jpg | inference_ms=333.6 | detections=0 | annotate=False
2026-09-05 12:30:10,123 | INFO | POST /predict-batch | files=2 | total_inference_ms=140.31 | annotate=False
```

### Web UI

- Accessible at `http://127.0.0.1:8000` when the server is running.
- Supports:
  - Single-image prediction with annotated output.
  - Batch prediction with summary results.
  - Display of inference time and detection count.

## How to run the API

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Start the server:

```powershell
uvicorn api:app --reload
```

3. Open the UI:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

Or use the API docs:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Repository structure

- `prepare_dataset.py`, `validate_dataset.py`, `augment_train.py` – Week 1 data pipeline.
- `data/neu/` – prepared dataset and `data.yaml`.
- `models/best_32epoch.onnx` – exported YOLOv8 ONNX model.
- `realtime_inference.py` – Week 3 OpenCV inference.
- `api.py` – Week 4 FastAPI application.
- `static/index.html` – web UI.
- `test_batch.ps1` – PowerShell script for batch testing.
- `api.log` – request logs.
- `WEEK4_OUTPUT.md` – detailed API output examples.

## Notes

- The current model is CPU-optimized but may produce few or no detections at high confidence thresholds depending on training quality.
- For better accuracy, retrain or fine-tune the model with high-quality bounding-box labels.
 ## Team / Collaborators

This project was developed as part of the **zaalima** engineering program.

- **Pavan Chowdary** – End-to-end pipeline, model training, API, and UI  
  [GitHub](https://github.com/pavanthiriveedi7-rgb) | [LinkedIn](https://www.linkedin.com/in/pavan-kumar-tiruveedhi/)

- **Veeramsetty Naga Malleswari** – Data processing, augmentation, and evaluation  
  [GitHub](https://github.com/nagamalleswari7) | [LinkedIn](https://www.linkedin.com/in/naga-malleswari-veeramsetty-435a54323?utm_source=share_via&utm_content=profile&utm_medium=member_android)

- **Vaithees** – Real-time inference, optimization, and deployment docs  
  [GitHub](https://github.com/Vaithees-R)
