# ============================================================
# MNIST Digit Recognizer — Streamlit front-end for mnist_model.keras
# ============================================================
from pathlib import Path
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ---- IMPORTANT — re-declare `normalize` exactly as it was in the training notebook.
# mnist_artifacts.pkl stores artifacts["preprocess"] = normalize as a *function reference*
# (pickle records the module + name, not the body). When pickle.load() runs here, Python
# tries to resolve __main__.normalize — so this top-level definition must exist BEFORE
# load_artifacts() is called, otherwise you will hit:
#     AttributeError: Can't get attribute 'normalize' on <module '__main__' from 'app.py'>
def normalize(images: np.ndarray) -> np.ndarray:
    return images.astype("float32") / 255.0

# ---- paths (relative to this file so it works in Docker too) ----
ROOT          = Path(__file__).parent
MODEL_PATH    = ROOT / "mnist_model.keras"
ARTIFACT_PATH = ROOT / "mnist_artifacts.pkl"

# ---- page config ----
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="📝",
    layout="centered",
)

# ---- cached loader: runs once per server, not once per click ----
@st.cache_resource(show_spinner="Loading MNIST model…")
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return artifacts, model

artifacts, model = load_artifacts()
class_labels = artifacts["class_labels"]            # [0,1,…,9]
input_shape  = tuple(artifacts["input_shape"])      # (28, 28)

# ---- preprocess: rgba canvas -> (1, 28, 28) float32 in [0, 1] ----
def preprocess(rgba: np.ndarray) -> np.ndarray:
    img  = Image.fromarray(rgba.astype("uint8")).convert("L")
    img  = img.resize((input_shape[1], input_shape[0]), Image.LANCZOS)
    arr  = np.asarray(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]

# ============================== UI ==============================
st.title("📝  MNIST Digit Recognizer")
st.caption(
    "Draw a digit (0–9) in the black box. "
    "The model expects a thick white stroke on a black background — "
    "the same colour scheme it was trained on."
)

col_draw, col_pred = st.columns([1.05, 1])

with col_draw:
    st.subheader("1. Draw")
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="mnist-canvas",
    )
    predict_clicked = st.button("🚀  Predict", type="primary", use_container_width=True)

with col_pred:
    st.subheader("2. Prediction")
    if predict_clicked and canvas.image_data is not None:
        x = preprocess(canvas.image_data)

        if x.sum() < 1e-3:
            st.warning("The canvas looks empty — draw a digit and click Predict.")
        else:
            probs = model.predict(x, verbose=0)[0]
            top   = int(np.argmax(probs))

            st.metric(label="Predicted digit",
                      value=str(class_labels[top]),
                      delta=f"{probs[top]*100:.1f}% confident")
            st.progress(float(probs[top]))

            st.caption("Full softmax distribution")
            st.bar_chart(
                {str(c): float(p) for c, p in zip(class_labels, probs)}
            )
    else:
        st.info("Draw something on the left, then press **Predict**.")

# ---- sidebar diagnostics ----
with st.sidebar:
    st.header("Model card")
    st.write("**Architecture:** Flatten → Dense(256, ReLU) → Dropout → Dense(128, ReLU) → Dropout → Dense(10, softmax)")
    st.write(f"**TensorFlow:** `{artifacts.get('tensorflow_version', tf.__version__)}`")
    test_metrics = artifacts.get("test_metrics", {})
    if test_metrics:
        st.write(f"**Test accuracy:** `{test_metrics.get('accuracy', 0):.4f}`")
        st.write(f"**Test loss:**     `{test_metrics.get('loss', 0):.4f}`")