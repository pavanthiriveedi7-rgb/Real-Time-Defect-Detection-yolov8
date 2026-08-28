import time
import cv2
from ultralytics import YOLO

MODEL_PATH = r"models\best_32epoch.onnx"
IMAGE_SIZE = 320
CONFIDENCE = 0.25

model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

previous_time = time.perf_counter()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a camera frame.")
        break

    start_time = time.perf_counter()

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    inference_ms = (time.perf_counter() - start_time) * 1000
    annotated_frame = results[0].plot()

    current_time = time.perf_counter()
    fps = 1 / max(current_time - previous_time, 0.0001)
    previous_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f} | Inference: {inference_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    detections = results[0].boxes

    if detections is not None and len(detections) > 0:
        for class_id, confidence in zip(
            detections.cls.tolist(),
            detections.conf.tolist()
        ):
            class_name = model.names[int(class_id)]
            print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

    cv2.imshow("Industrial Defect Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()