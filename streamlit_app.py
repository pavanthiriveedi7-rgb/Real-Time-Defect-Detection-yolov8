"""Steel Surface Inspector (Streamlit). Run: python -m streamlit run streamlit_app.py"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_v2.pt"
IMGSZ = 640

st.set_page_config(page_title="Steel Surface Inspector", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

st.title("Steel Surface Inspector")
st.caption(f"Model: {MODEL_PATH.name} | image size {IMGSZ} | test mAP50 0.750")

with st.sidebar:
    st.header("Thresholds")
    conf = st.slider("Show detections above", 0.05, 0.90, 0.33, 0.01,
                     help="F1 is highest at about 0.33 for this model.")
    reject = st.slider("Reject when any defect is above", 0.30, 0.95, 0.60, 0.01)

file = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png", "bmp"])

if file is None:
    st.info("Upload an image to start. NEU-DET test images work well.")
    st.stop()

img = Image.open(file).convert("RGB")
bgr = np.array(img)[..., ::-1]  # ultralytics expects BGR arrays
with st.spinner("Inspecting..."):
    res = model.predict(bgr, imgsz=IMGSZ, conf=conf, verbose=False)[0]

dets = [
    {"Defect": model.names[int(c)].replace("_", " "), "Confidence": round(float(p), 3)}
    for c, p in zip(res.boxes.cls, res.boxes.conf)
]
dets.sort(key=lambda d: -d["Confidence"])

left, right = st.columns([2, 1])
with left:
    st.image(res.plot()[..., ::-1], caption="Detections", use_container_width=True)
with right:
    if not dets:
        st.success("CLEAR: no defects above the display threshold.")
    elif any(d["Confidence"] >= reject for d in dets):
        st.error(f"REJECT: {len(dets)} defect(s), at least one above {reject:.2f}.")
    else:
        st.warning(f"REVIEW: {len(dets)} defect(s), none above {reject:.2f}.")
    if dets:
        st.dataframe(dets, use_container_width=True, hide_index=True)
        counts = {}
        for d in dets:
            counts[d["Defect"]] = counts.get(d["Defect"], 0) + 1
        st.bar_chart(counts)
