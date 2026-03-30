import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from class_labels import CLASS_NAMES

st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")
st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image to classify it.")


# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()


# ─────────────────────────────────────────────
# Preprocessing  (must match training exactly)
# ─────────────────────────────────────────────
def preprocess_image(image):
    image = np.array(image)                      

   
    image = cv2.resize(image, (48, 48))         

    image = image.astype(np.float32) / 255.0   
    image = np.expand_dims(image, axis=0)       
    return image


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  

    processed = preprocess_image(image)
    preds = model.predict(processed)[0]         

    # ── Top prediction ──
    class_id   = int(np.argmax(preds))
    confidence = float(preds[class_id]) * 100
    label      = CLASS_NAMES.get(class_id, f"Class {class_id}")

    st.success(f"**Predicted Sign:** {label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # ── Top-3 predictions (handy when confidence is low) ──
    st.subheader("Top 3 Predictions")
    top3_ids = np.argsort(preds)[::-1][:3]     

    for rank, idx in enumerate(top3_ids, 1):
        name = CLASS_NAMES.get(int(idx), f"Class {idx}")
        prob = float(preds[idx]) * 100

        col1, col2 = st.columns([3, 1])
        col1.text(f"{rank}. {name}")
        col2.text(f"{prob:.2f}%")

        # progress bar as a quick visual confidence indicator
        st.progress(min(prob / 100, 1.0))