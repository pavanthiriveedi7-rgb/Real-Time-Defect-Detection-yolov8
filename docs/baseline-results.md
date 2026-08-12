\# YOLOv8 Baseline Results



\## Model



\- Model: YOLOv8n

\- Epochs: 50

\- Image size: 320

\- Batch size: 2

\- Device: CPU



\## Validation results



\- Precision: 0.504

\- Recall: 0.477

\- mAP50: 0.442

\- mAP50-95: 0.198



\## Speed



\- Preprocessing: 0.3 ms

\- Inference: 19.4 ms

\- Post-processing: 1.6 ms



\## Per-class results



| Class | Precision | Recall | mAP50 | mAP50-95 |

|---|---:|---:|---:|---:|

| crazing | 1.000 | 0.000 | 0.153 | 0.0447 |

| inclusion | 0.701 | 0.356 | 0.559 | 0.235 |

| patches | 0.506 | 0.819 | 0.678 | 0.349 |

| pitted\_surface | 0.254 | 0.721 | 0.617 | 0.325 |

| rolled\_in\_scale | 0.200 | 0.570 | 0.275 | 0.103 |

| scratches | 0.362 | 0.396 | 0.369 | 0.135 |



\## Observations



\- The dataset and training pipeline completed successfully.

\- Patches had the strongest performance.

\- Crazing had zero recall.

\- Rolled-in scale and scratches need improvement.

\- The model is not ready for production deployment.

