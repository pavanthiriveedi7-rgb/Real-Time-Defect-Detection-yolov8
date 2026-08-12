"""Convert NEU Pascal-VOC XML annotations to YOLO format and split without leakage.
Usage: python prepare_dataset.py --src /path/to/NEU-DET --out data/neu --val 0.15 --test 0.15
Expected source: images/**/*.jpg and annotations/**/*.xml (or image/annotation dirs).
"""
from __future__ import annotations
import argparse, hashlib, random, shutil
from pathlib import Path
import cv2, yaml

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
ALIASES = {"cr":"crazing", "in":"inclusion", "pa":"patches", "ps":"pitted_surface", "rs":"rolled_in_scale", "scr":"scratches"}

def find_pairs(src: Path):
    xmls = list(src.rglob("*.xml"))
    for xml in xmls:
        stem = xml.stem
        candidates = list(src.rglob(stem + ext) for ext in (".jpg", ".jpeg", ".png", ".bmp"))
        imgs = [p for group in candidates for p in group]
        if imgs: yield imgs[0], xml

def read_voc(xml: Path, w: int, h: int):
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    result = []
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip().lower().replace(" ", "_")
        name = ALIASES.get(raw, raw)
        if name not in CLASSES: continue
        box = obj.find("bndbox")
        if box is None: continue
        x1 = float(box.findtext("xmin", "0")); y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0")); y2 = float(box.findtext("ymax", "0"))
        x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
        y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
        if x2 > x1 and y2 > y1:
            result.append((CLASSES.index(name), x1, y1, x2, y2))
    return result

def yolo_line(cls, x1, y1, x2, y2, w, h):
    return f"{cls} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/neu")); ap.add_argument("--val", type=float, default=.15)
    ap.add_argument("--test", type=float, default=.15); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); pairs = list(find_pairs(args.src))
    if not pairs: raise SystemExit("No image/XML pairs found. Check --src layout.")
    rng = random.Random(args.seed); rng.shuffle(pairs)
    n = len(pairs); ntest = round(n*args.test); nval = round(n*args.val)
    groups = {"test": pairs[:ntest], "val": pairs[ntest:ntest+nval], "train": pairs[ntest+nval:]}
    for split, items in groups.items():
        (args.out/"images"/split).mkdir(parents=True, exist_ok=True); (args.out/"labels"/split).mkdir(parents=True, exist_ok=True)
        for image, xml in items:
            # Stable unique name avoids collisions between folders.
            uid = hashlib.sha1(str(image.relative_to(args.src)).encode()).hexdigest()[:10]
            name = f"{image.stem}_{uid}"
            dst = args.out/"images"/split/(name + image.suffix.lower()); shutil.copy2(image, dst)
            im = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if im is None: continue
            h, w = im.shape[:2]; boxes = read_voc(xml, w, h)
            (args.out/"labels"/split/(name+".txt")).write_text("\n".join(yolo_line(*b, w, h) for b in boxes) + ("\n" if boxes else ""))
    cfg = {"path": str(args.out.resolve()), "train":"images/train", "val":"images/val", "test":"images/test", "names": {i:n for i,n in enumerate(CLASSES)}}
    (args.out/"data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Prepared {n} images: train={len(groups['train'])}, val={len(groups['val'])}, test={len(groups['test'])}")
if __name__ == "__main__": main()
