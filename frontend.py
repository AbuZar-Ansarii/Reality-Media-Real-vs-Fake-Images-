import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import uuid
import random

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Real vs Fake Image Feed",
    layout="wide"
)

IMAGE_DIR = "pin images"
MODEL_PATH = "fake_image_detector_model.h5"
CLASS_NAMES = ["Fake", "Real"]  

os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------------------
# LOAD MODEL (CACHED)
# -----------------------------------
@st.cache_resource
def load_fake_detector():
    return load_model(MODEL_PATH)

model = load_fake_detector()

# -----------------------------------
# PREPROCESS FUNCTION
# -----------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------
def predict_image(image_path):
    processed = preprocess_image(image_path)
    prediction = model.predict(processed, verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return CLASS_NAMES[class_index], confidence

# -----------------------------------
# SIDEBAR - UPLOAD
# -----------------------------------
st.sidebar.title("📤 Upload Image")

if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()

uploaded_files = st.sidebar.file_uploader(
    "Add image(s) to feed",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    new_upload = False

    for file in uploaded_files:
        file_hash = hash(file.getvalue())

        # Process only if not seen before
        if file_hash not in st.session_state.uploaded_hashes:
            image = Image.open(file).convert("RGB")
            filename = f"{uuid.uuid4()}.png"
            save_path = os.path.join(IMAGE_DIR, filename)
            image.save(save_path)

            st.session_state.uploaded_hashes.add(file_hash)
            new_upload = True

    if new_upload:
        st.sidebar.success("Image(s) uploaded successfully")


# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🖼️ Real vs Fake Image Feed</h1>
    <p style="text-align:center;color:gray;">
    Upload from sidebar • Auto AI analysis
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# LOAD IMAGES
# -----------------------------------
image_files = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

if not image_files:
    st.warning("No images available.")
    st.stop()

random.shuffle(image_files)
# ['fake', 'real']
# -----------------------------------
# DISPLAY GRID
# -----------------------------------
NUM_COLS = 3
cols = st.columns(NUM_COLS)

for idx, img_path in enumerate(image_files):
    with cols[idx % NUM_COLS]:
        image = Image.open(img_path)
        label, confidence = predict_image(img_path)

        st.image(image, use_container_width=True)

        color = "green" if label == "Real" else "red"
        icon = "✅" if label == "Real" else "❌"

        st.markdown(
            f"<p style='color:{color}; font-weight:bold;'>"
            f"{icon} {label} ({confidence:.2%})</p>",
            unsafe_allow_html=True
        )
