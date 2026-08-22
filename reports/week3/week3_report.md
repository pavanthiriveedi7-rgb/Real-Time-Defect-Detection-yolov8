# Week 3 - Model Optimization and Real-Time Inference

## 1. Objective

The objective of Week 3 was to optimize the trained YOLOv8 defect detection model for inference and implement real-time detection using ONNX Runtime and OpenCV.

The trained model used for this work was:

`models/best_32epoch.pt`

---

## 2. Model Export

The trained YOLOv8 PyTorch model was exported to ONNX format.

### Original model

`models/best_32epoch.pt`

### Optimized inference model

`models/best_32epoch.onnx`

The ONNX export completed successfully using Ultralytics.

The exported model uses an input resolution of:

`320 x 320`

The ONNX Runtime model input shape was verified as:

`[1, 3, 320, 320]`

The output shape was verified as:

`[1, 10, 2100]`

---

## 3. ONNX Runtime Verification

The exported ONNX model was loaded successfully using ONNX Runtime.

The model was executed using:

`CPUExecutionProvider`

The ONNX model successfully accepted the expected input tensor and produced the expected output tensor.

---

## 4. Real-Time OpenCV Inference

A real-time inference application was implemented using:

- OpenCV
- ONNX Runtime
- NumPy
- YOLOv8 ONNX model

The application performs the following pipeline:

Camera frame
-> Preprocessing
-> ONNX Runtime inference
-> YOLO output processing
-> Confidence filtering
-> Non-Maximum Suppression
-> Bounding box drawing
-> Class and confidence display
-> FPS measurement

The application uses the laptop webcam as the real-time input source.

The six defect classes used by the model are:

1. crazing
2. inclusion
3. patches
4. pitted_surface
5. rolled_in_scale
6. scratches

---

## 5. Real-Time Test

The real-time OpenCV application successfully opened the webcam and performed inference.

During the test, the application displayed:

- FPS
- inference latency
- number of detected objects
- defect class
- confidence score
- bounding boxes

A sample webcam run produced approximately:

FPS: 22.6

Inference latency: 10.1 ms

Detections: 2

The webcam test demonstrates that the complete real-time inference pipeline is functioning.

The webcam environment was not a controlled steel-defect dataset, so these detections should not be treated as an accuracy evaluation.

---

## 6. Performance Benchmark

A controlled benchmark was performed using:

- 20 warm-up runs
- 100 measured runs
- 320 x 320 input
- CPU execution

### PyTorch Model

Average latency:

28.41 ms

Minimum latency:

21.55 ms

Maximum latency:

45.35 ms

Approximate FPS:

35.20

### ONNX Runtime Model

Average latency:

13.72 ms

Minimum latency:

8.91 ms

Maximum latency:

73.83 ms

Approximate FPS:

72.91

---

## 7. Performance Comparison

| Metric | PyTorch | ONNX Runtime |
|---|---:|---:|
| Average latency | 28.41 ms | 13.72 ms |
| Minimum latency | 21.55 ms | 8.91 ms |
| Maximum latency | 45.35 ms | 73.83 ms |
| Approx. FPS | 35.20 | 72.91 |

The benchmark reported:

- ONNX improvement: 51.72%
- ONNX speedup: 2.07x

Therefore, ONNX Runtime provided substantially lower average inference latency than the PyTorch model in the tested CPU environment.

---

## 8. Limitations

The benchmark was performed on a CPU-only environment.

The system reported:

`PyTorch 2.13.0+cpu`

CUDA was not available on the test machine.

Therefore, GPU-specific acceleration such as TensorRT was not evaluated locally.

The real-time webcam test was also not an accuracy evaluation because the webcam was not displaying controlled NEU defect test samples.

---

## 9. Week 3 Deliverables

The following Week 3 components were completed:

- [x] ONNX environment setup
- [x] YOLOv8 PyTorch model export to ONNX
- [x] ONNX model verification
- [x] ONNX Runtime inference
- [x] OpenCV real-time inference
- [x] Bounding box visualization
- [x] Defect class prediction
- [x] Confidence display
- [x] FPS measurement
- [x] Inference latency measurement
- [x] PyTorch benchmark
- [x] ONNX benchmark
- [x] Performance comparison
- [x] Week 3 documentation

---

## 10. Conclusion

Week 3 successfully converted the trained YOLOv8 defect detection model into ONNX format and integrated it with ONNX Runtime for real-time inference.

The ONNX model demonstrated lower average inference latency and approximately 2.07x speedup compared with the PyTorch model in the tested CPU benchmark.

The real-time OpenCV pipeline is now ready to serve as the inference component for the next stage of the project.