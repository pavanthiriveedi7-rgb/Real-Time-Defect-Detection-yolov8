import cv2
from ultralytics import YOLO

MODEL_PATH = r"models\best_32epoch.onnx"

model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a camera frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Industrial Defect Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()