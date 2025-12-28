import os
import numpy as np
import streamlit as st
from PIL import Image
import cv2

import tensorflow as tf
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="Anemia Conjunctiva Screening", page_icon="🩸", layout="wide")

CROPPER_PATH = "models/cropper/model_optimized.h5"
VIT_DIR = "models/classifier_vit"

CROPPER_INPUT = (256, 256)     
VIT_IMAGE_SIZE = (224, 224)

# CSS
def load_css(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# Load models (cached)
@st.cache_resource
def load_cropper():
    if not os.path.exists(CROPPER_PATH):
        raise FileNotFoundError(f"Cropper model not found: {CROPPER_PATH}")
    return tf.keras.models.load_model(CROPPER_PATH, compile=False)

@st.cache_resource
def load_vit():
    if not os.path.exists(VIT_DIR):
        raise FileNotFoundError(f"ViT model folder not found: {VIT_DIR}")

    processor = AutoImageProcessor.from_pretrained(VIT_DIR)
    model = ViTForImageClassification.from_pretrained(VIT_DIR)
    model.eval()
    return processor, model

# Conjunctiva crop pipeline
def predict_mask(cropper, pil_img: Image.Image) -> np.ndarray:
    """
    Returns mask in HxW float [0..1] in cropper input size.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img)
    arr = cv2.resize(arr, CROPPER_INPUT, interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # [1,H,W,3]

    pred = cropper.predict(arr, verbose=0)

    # Handle common shapes:
    # [1,H,W,1] or [1,H,W] -> squeeze to [H,W]
    pred = np.array(pred)
    if pred.ndim == 4:
        pred = pred[0, :, :, 0]
    elif pred.ndim == 3:
        pred = pred[0, :, :]
    else:
        raise ValueError(f"Unexpected mask output shape: {pred.shape}")

    pred = np.clip(pred, 0.0, 1.0)
    return pred

def crop_from_mask(original_pil: Image.Image, mask_256: np.ndarray, thr=0.5, pad=12) -> Image.Image:
    """
    Convert mask bbox (in 256 space) -> crop bbox on original image.
    If mask empty, fallback to original image center crop.
    """
    Hm, Wm = mask_256.shape[:2]
    binmask = (mask_256 >= thr).astype(np.uint8)

    ys, xs = np.where(binmask > 0)
    orig_w, orig_h = original_pil.size

    if len(xs) < 10 or len(ys) < 10:
        # Fallback: center crop square-ish
        side = min(orig_w, orig_h)
        cx, cy = orig_w // 2, orig_h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(orig_w, x1 + side)
        y2 = min(orig_h, y1 + side)
        return original_pil.crop((x1, y1, x2, y2))

    x1m, x2m = xs.min(), xs.max()
    y1m, y2m = ys.min(), ys.max()

    # padding in mask space
    x1m = max(0, x1m - pad); y1m = max(0, y1m - pad)
    x2m = min(Wm - 1, x2m + pad); y2m = min(Hm - 1, y2m + pad)

    # map to original resolution
    sx = orig_w / float(Wm)
    sy = orig_h / float(Hm)

    x1 = int(x1m * sx); x2 = int((x2m + 1) * sx)
    y1 = int(y1m * sy); y2 = int((y2m + 1) * sy)

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(orig_w, x2); y2 = min(orig_h, y2)

    return original_pil.crop((x1, y1, x2, y2))

# ViT classify
def classify_anemia(processor, vit_model, pil_img: Image.Image):
    """
    Returns: (pred_label:str, prob_anemic:float, probs:dict)
    Label mapping from your config:
      0: Anemic
      1: Non-Anemic
    """
    inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = vit_model(**inputs).logits  # [1,2]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # id2label per config.json: 0 Anemic, 1 Non-Anemic
    prob_anemic = float(probs[0])
    prob_non = float(probs[1])

    pred_id = int(np.argmax(probs))
    pred_label = "Anemic" if pred_id == 0 else "Non-Anemic"
    return pred_label, prob_anemic, {"Anemic": prob_anemic, "Non-Anemic": prob_non}

# UI
with st.sidebar:
    st.markdown("### Anemia Screening")
    st.markdown('<div class="muted">Upload → Crop conjunctiva → Classify (ViT)</div>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Menu",
        options=["Dashboard", "Predict", "About"],
        icons=["speedometer2", "cloud-upload", "info-circle"],
        default_index=1,
    )

st.markdown("## Anemia Conjunctiva Screening Dashboard")
st.markdown('<div class="muted">Deteksi kemungkinan anemia dari conjunctiva (bukan diagnosis medis).</div>', unsafe_allow_html=True)

if selected == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Cropper", "Keras .h5")
    with c2: st.metric("Classifier", "ViT (safetensors)")
    with c3: st.metric("ViT Input", "224×224")
    with c4: st.metric("Labels", "Anemic / Non-Anemic")
    style_metric_cards()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Pipeline**")
    st.write("1) Upload foto mata")
    st.write("2) Model cropper membuat mask conjunctiva dan crop ROI")
    st.write("3) ROI diproses oleh ViT image processor (resize + normalize) lalu diklasifikasi")
    st.markdown('<span class="badge">Disclaimer</span> <span class="muted">Hasil ini untuk screening/riset.</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Predict":
    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload foto mata (jpg/png)", type=["jpg", "jpeg", "png"])
        thr = st.slider("Mask threshold (crop)", 0.1, 0.9, 0.5, 0.01)
        run = st.button("Run Detection", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Tips supaya hasil stabil**")
        st.write("- Foto fokus di area mata, tidak blur")
        st.write("- Pencahayaan cukup (tidak terlalu gelap/overexposed)")
        st.write("- Jarak relatif konsisten dengan data training")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if uploaded is None:
            st.info("Upload gambar untuk melihat preview & hasil.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            original = Image.open(uploaded)
            st.image(original, caption="Original", use_container_width=True)

            if run:
                with st.spinner("Loading models & processing..."):
                    cropper = load_cropper()
                    processor, vit_model = load_vit()

                    mask = predict_mask(cropper, original)
                    roi = crop_from_mask(original, mask, thr=thr, pad=12)

                    pred_label, prob_anemic, probs = classify_anemia(processor, vit_model, roi)

                st.markdown("### Result")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Prediction", pred_label)
                with m2:
                    st.metric("Prob. Anemic", f"{prob_anemic:.3f}")
                style_metric_cards()

                st.image(roi, caption="Cropped conjunctiva ROI", use_container_width=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_anemic,
                    gauge={"axis": {"range": [0, 1]}},
                    title={"text": "Anemia Probability (class 0)"},
                ))
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

                st.write("**Class probabilities**")
                st.write(probs)

                st.warning("Disclaimer: Ini bukan diagnosis medis. Untuk konfirmasi anemia, lakukan pemeriksaan Hb/lab dan konsultasi tenaga kesehatan.")

            st.markdown("</div>", unsafe_allow_html=True)

elif selected == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("App ini menggunakan 2 model:")
    st.write("- `model_optimized.h5`: crop/segment conjunctiva dari foto mata")
    st.write("- ViT classifier (`model.safetensors` + config): klasifikasi Anemic vs Non-Anemic")
    st.markdown("</div>", unsafe_allow_html=True)
