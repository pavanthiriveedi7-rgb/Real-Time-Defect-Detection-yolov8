\# Project 2 – Real-Time Defect Detection (YOLOv8 + ONNX + FastAPI)  

\*\*Final Report\*\*



\## 1. Project overview



This project implements an industrial surface-defect detection system for steel surfaces using the NEU-DET dataset. The system covers the full pipeline: data preparation, model training, ONNX export, real-time inference, and a FastAPI-based inference service with a web UI.



\*\*Goal:\*\* Build a CPU-friendly defect detection system that can:



\- Detect six types of surface defects.

\- Run real-time inference on CPU.

\- Expose predictions through a simple API and web interface.



\## 2. Dataset



\- \*\*Name:\*\* NEU-DET steel surface defect dataset.

\- \*\*Classes (6):\*\*

&#x20; - `crazing`

&#x20; - `inclusion`

&#x20; - `patches`

&#x20; - `pitted\_surface`

&#x20; - `rolled\_in\_scale`

&#x20; - `scratches`

\- \*\*Annotation format:\*\* Pascal-VOC XML → converted to YOLO detection format (`class cx cy width height`, normalized).

\- \*\*Split:\*\* Train / validation / test with a fixed random seed for reproducibility.



\## 3. System architecture



The system has four main components:



1\. \*\*Data pipeline (Week 1)\*\*

&#x20;  - `prepare\_dataset.py`: Converts Pascal-VOC XML to YOLO labels and creates splits.

&#x20;  - `validate\_dataset.py`: Checks label integrity and image/label consistency.

&#x20;  - `augment\_train.py`: Applies augmentation to training images only.



2\. \*\*Model training (Week 2)\*\*

&#x20;  - YOLOv8 detection model trained using Ultralytics.

&#x20;  - Configuration: `yolov8n`, 32 epochs, image size 320.

&#x20;  - Exported to ONNX for efficient CPU inference.



3\. \*\*Real-time inference (Week 3)\*\*

&#x20;  - OpenCV-based script (`realtime\_inference.py`) using the ONNX model.

&#x20;  - Displays:

&#x20;    - Detected defect class.

&#x20;    - Confidence score.

&#x20;    - FPS.

&#x20;    - Inference latency (ms).



4\. \*\*FastAPI service + UI (Week 4)\*\*

&#x20;  - `api.py` implements:

&#x20;    - `GET /health`

&#x20;    - `POST /predict` (single image, optional annotated output).

&#x20;    - `POST /predict-batch` (multiple images).

&#x20;  - Logging to `api.log`.

&#x20;  - Simple web UI (`static/index.html`) for:

&#x20;    - Single and batch uploads.

&#x20;    - Viewing annotated images and detection tables.

&#x20;    - Displaying inference time and detection counts.



\## 4. Implementation details



\### Environment



\- Python 3.10

\- Virtual environment: `.venv`

\- Key libraries:

&#x20; - `ultralytics` (YOLOv8)

&#x20; - `onnx`, `onnxruntime`

&#x20; - `fastapi`, `uvicorn`

&#x20; - `opencv-python`

&#x20; - `numpy`



\### Model



\- Base model: `yolov8n.pt` (nano).

\- Training command (example):



```powershell

yolo detect train data=data/neu/data.yaml model=yolov8n.pt epochs=32 imgsz=320

```



\- Export command:



```powershell

yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=320

```



\- Inference backend: ONNX Runtime with CPUExecutionProvider.



\### API endpoints



\- `GET /health`  

&#x20; Returns model path and task.



\- `POST /predict`  

&#x20; - Input: single image (`file`), `annotate` (bool).

&#x20; - Output:

&#x20;   - JSON with `filename`, `inference\_ms`, `detections\_count`, `detections`.

&#x20;   - Or JPEG image with bounding boxes if `annotate=true`.



\- `POST /predict-batch`  

&#x20; - Input: list of images (`files`), `annotate` (bool).

&#x20; - Output: JSON with `files\_processed`, `total\_inference\_ms`, and per-file results.



\### Performance (CPU)



Typical observations on a standard laptop CPU:



\- Single-image inference: \~150–350 ms after warm-up.

\- First request may be slower due to model loading.

\- Batch inference scales roughly linearly with number of images.



Exact numbers depend on hardware and image size.



\## 5. Results and observations



\- The pipeline successfully:

&#x20; - Converts and validates NEU-DET annotations.

&#x20; - Trains a YOLOv8 detection model.

&#x20; - Exports to ONNX and runs on CPU.

&#x20; - Serves predictions via FastAPI and a web UI.



\- Detection quality:

&#x20; - At `CONFIDENCE = 0.25`, some test images return zero detections.

&#x20; - At very low thresholds (e.g. `0.01`), many weak boxes appear.

&#x20; - This indicates the model can localize defects but may need:

&#x20;   - More training epochs.

&#x20;   - A larger model (e.g. `yolov8s` or `yolov8m`).

&#x20;   - Better or more consistent bounding-box annotations.



\- The system is functionally complete and ready for:

&#x20; - Further model tuning.

&#x20; - Integration into a larger inspection workflow.

&#x20; - Deployment on a CPU-only server for demo or pilot use.



\## 6. How to run



\### 1. Setup



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



\### 2. Run the API



```powershell

uvicorn api:app --reload

```



\### 3. Open the UI



\[http://127.0.0.1:8000](http://127.0.0.1:8000)



Or use API docs:



\[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)



\### 4. Run real-time OpenCV inference



```powershell

python realtime\_inference.py

```



Press `q` to exit.



\## 7. Repository structure



\- `prepare\_dataset.py`, `validate\_dataset.py`, `augment\_train.py` – Week 1 data pipeline.

\- `data/neu/` – prepared dataset and `data.yaml`.

\- `models/best\_32epoch.onnx` – exported YOLOv8 ONNX model.

\- `realtime\_inference.py` – Week 3 OpenCV inference.

\- `api.py` – Week 4 FastAPI application.

\- `static/index.html` – web UI.

\- `test\_batch.ps1` – batch test script.

\- `api.log` – request logs.

\- `WEEK4\_OUTPUT.md` – API output examples.

\- `README.md` – full project documentation.

\- `REPORT.md` – this final report.



\## 8. Conclusion



Project 2 delivers an end-to-end, CPU-compatible defect detection system:



\- Data preparation and validation.

\- Model training and export.

\- Real-time inference.

\- REST API and web UI.



The current implementation is a solid foundation. Future work can focus on improving detection accuracy, adding authentication, and deploying the service in a production environment.

