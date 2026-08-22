import time

import cv2
import numpy as np
import onnxruntime as ort


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/best_32epoch.onnx"

IMG_SIZE = 320

CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


# ==========================================
# Load ONNX model
# ==========================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded.")
print("Input:", input_name)
print("Input shape:", session.get_inputs()[0].shape)


# ==========================================
# Preprocess
# ==========================================

def preprocess(frame):
    """
    Convert OpenCV BGR image into
    YOLOv8 ONNX input format.
    """

    image = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # CHW -> BCHW
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================================
# IoU
# ==========================================

def calculate_iou(box, boxes):

    if len(boxes) == 0:
        return np.array([])

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    box = np.asarray(
        box,
        dtype=np.float32
    )

    x1 = np.maximum(
        box[0],
        boxes[:, 0]
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1]
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2]
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3]
    )

    intersection_width = np.maximum(
        0,
        x2 - x1
    )

    intersection_height = np.maximum(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    box_area = max(
        0,
        box[2] - box[0]
    ) * max(
        0,
        box[3] - box[1]
    )

    boxes_area = (
        np.maximum(
            0,
            boxes[:, 2] - boxes[:, 0]
        )
        *
        np.maximum(
            0,
            boxes[:, 3] - boxes[:, 1]
        )
    )

    union = (
        box_area +
        boxes_area -
        intersection
    )

    return intersection / (
        union + 1e-6
    )


# ==========================================
# NMS
# ==========================================

def nms(boxes, scores, iou_threshold):

    if len(boxes) == 0:
        return []

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        current = int(order[0])

        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]

        ious = calculate_iou(
            boxes[current],
            boxes[remaining]
        )

        order = remaining[
            ious < iou_threshold
        ]

    return keep


# ==========================================
# Post-processing
# ==========================================

def postprocess(
    output,
    original_width,
    original_height
):

    # --------------------------------------
    # Get model output
    # --------------------------------------

    predictions = np.asarray(
        output[0]
    )

    print_shape = False

    if print_shape:
        print(
            "Raw output shape:",
            predictions.shape
        )

    # Expected:
    #
    # [1, 10, 2100]
    #
    # Remove batch dimension.
    if predictions.ndim == 3:
        predictions = predictions[0]

    # Now:
    #
    # [10, 2100]
    #
    # Convert to:
    #
    # [2100, 10]

    predictions = predictions.T

    # --------------------------------------
    # Scaling
    # --------------------------------------

    x_scale = (
        original_width /
        IMG_SIZE
    )

    y_scale = (
        original_height /
        IMG_SIZE
    )

    boxes = []
    scores = []
    class_ids = []

    # --------------------------------------
    # Process each detection
    # --------------------------------------

    for prediction in predictions:

        # Make sure prediction is a
        # one-dimensional NumPy array.

        prediction = np.asarray(
            prediction
        ).reshape(-1)

        # First four values:
        #
        # x_center
        # y_center
        # width
        # height

        x_center = float(
            prediction[0]
        )

        y_center = float(
            prediction[1]
        )

        width = float(
            prediction[2]
        )

        height = float(
            prediction[3]
        )

        # Six class scores.
        #
        # Total values:
        #
        # 4 box values
        # +
        # 6 classes
        # =
        # 10 values

        class_scores = np.asarray(
            prediction[4:10],
            dtype=np.float32
        ).reshape(-1)

        if len(class_scores) != len(CLASS_NAMES):
            continue

        # Best class

        class_id = int(
            np.argmax(class_scores)
        )

        # Convert NumPy value to
        # normal Python float.

        confidence = float(
            class_scores[class_id].item()
        )

        # Confidence filtering

        if confidence < CONF_THRESHOLD:
            continue

        # ----------------------------------
        # Convert box format
        # ----------------------------------

        x1 = (
            x_center -
            width / 2
        ) * x_scale

        y1 = (
            y_center -
            height / 2
        ) * y_scale

        x2 = (
            x_center +
            width / 2
        ) * x_scale

        y2 = (
            y_center +
            height / 2
        ) * y_scale

        # ----------------------------------
        # Keep boxes inside image
        # ----------------------------------

        x1 = max(
            0,
            min(
                original_width - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                original_height - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                original_width - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                original_height - 1,
                y2
            )
        )

        boxes.append(
            [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]
        )

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    # --------------------------------------
    # NMS
    # --------------------------------------

    keep = nms(
        boxes,
        scores,
        IOU_THRESHOLD
    )

    detections = []

    for index in keep:

        detections.append(
            {
                "box": boxes[index],
                "score": scores[index],
                "class_id": class_ids[index]
            }
        )

    return detections


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


print()
print(
    "Real-time defect detection started."
)
print(
    "Press Q to quit."
)
print()


# ==========================================
# FPS
# ==========================================

previous_time = time.perf_counter()

fps = 0.0


# ==========================================
# Main loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Failed to read camera frame."
        )

        break

    height, width = frame.shape[:2]

    # --------------------------------------
    # Preprocess
    # --------------------------------------

    input_tensor = preprocess(
        frame
    )

    # --------------------------------------
    # ONNX inference
    # --------------------------------------

    start_time = time.perf_counter()

    output = session.run(
        None,
        {
            input_name:
            input_tensor
        }
    )

    inference_time = (
        time.perf_counter()
        - start_time
    ) * 1000

    # --------------------------------------
    # Postprocess
    # --------------------------------------

    detections = postprocess(
        output,
        width,
        height
    )

    # --------------------------------------
    # Draw detections
    # --------------------------------------

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )

        score = detection["score"]

        class_id = detection[
            "class_id"
        ]

        class_name = CLASS_NAMES[
            class_id
        ]

        label = (
            f"{class_name}: "
            f"{score:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    y1 - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # --------------------------------------
    # Calculate FPS
    # --------------------------------------

    current_time = (
        time.perf_counter()
    )

    elapsed = (
        current_time -
        previous_time
    )

    if elapsed > 0:

        fps = 1.0 / elapsed

    previous_time = current_time

    # --------------------------------------
    # Display information
    # --------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {inference_time:.1f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Defects: {len(detections)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # --------------------------------------
    # Show frame
    # --------------------------------------

    cv2.imshow(
        "Real-Time Defect Detection - YOLOv8 ONNX",
        frame
    )

    # --------------------------------------
    # Quit with Q
    # --------------------------------------

    if (
        cv2.waitKey(1) & 0xFF
    ) == ord("q"):

        break


# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()

print(
    "Real-time detection stopped."
)