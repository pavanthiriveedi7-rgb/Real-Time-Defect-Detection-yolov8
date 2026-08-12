# Week 1 NEU defect pipeline

## Install
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Convert and split annotations
The converter expects image files and Pascal-VOC XML files anywhere below `--src`. It creates YOLO labels and a train/validation/test split. YOLO detection labels are one row per object in normalized `class cx cy width height` format, as required by Ultralytics. [Ultralytics dataset format](https://docs.ultralytics.com/datasets/detect)

```bash
python prepare_dataset.py --src /path/to/NEU-DET --out data/neu
```

## 2. Audit the data
```bash
python validate_dataset.py --root data/neu
```
Fix every issue before training. Inspect random images with their boxes.

## 3. Augment training only
```bash
python augment_train.py --root data/neu --copies 1
```
Do not run this script more than once on the same output unless you want repeated augmentation.

## 4. Check the YAML
Edit `data/neu/data.yaml` if necessary. Never augment validation or test data.

## Notes
- This assumes Pascal-VOC XML annotations. If your download contains only class folders and no XML, it is a classification dataset and bounding boxes must be annotated first with CVAT, Label Studio, or Roboflow.
- Keep the test split untouched until final evaluation.
- The split uses a fixed seed for reproducibility.
- The six class names and aliases match the common NEU-DET naming convention.
