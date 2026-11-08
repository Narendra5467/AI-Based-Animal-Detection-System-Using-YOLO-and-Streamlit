import os
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Animal Detection",
    page_icon="🐾",
    layout="wide",
)

st.title("🐾 AI-Based Animal Detection System")
st.markdown(
    "Detect **cat, cow, dog, horse, person, and sheep** using the custom-trained "
    "YOLO model from the accompanying Google Colab notebook."
)

# The trained checkpoint from the notebook must be placed beside app.py.
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")


@st.cache_resource(show_spinner="Loading your trained YOLO model...")
def load_model(model_path: str):
    return YOLO(model_path)


if not os.path.isfile(MODEL_PATH):
    st.error(
        f"Model file `{MODEL_PATH}` was not found. Download `best.pt` from your "
        "Colab notebook and place it in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the YOLO checkpoint: {exc}")
    st.stop()

# Use the class labels stored in the checkpoint, matching the notebook's
# CLASS_NAMES = ["cat", "cow", "dog", "horse", "person", "sheep"].
model_names = model.names
if isinstance(model_names, dict):
    class_names = [model_names[k] for k in sorted(model_names)]
else:
    class_names = list(model_names)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Increase this value to hide lower-confidence detections.",
    )
    image_size = st.select_slider(
        "Image size",
        options=[320, 416, 512, 640, 768],
        value=640,
    )
    st.caption("Model classes")
    for class_name in class_names:
        st.write(f"• {class_name}")

def predict_image(image: Image.Image):
    """Run inference and return an annotated RGB image plus detection rows."""
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )[0]

    # Ultralytics plot() returns BGR; Streamlit/PIL expects RGB.
    annotated_bgr = result.plot()
    annotated_rgb = Image.fromarray(annotated_bgr[:, :, ::-1])

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            label = (
                model_names[class_id]
                if isinstance(model_names, dict)
                else model_names[class_id]
            )
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "Animal / Class": label,
                    "Confidence": round(score * 100, 2),
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1),
                }
            )

    return annotated_rgb, detections


def show_results(image: Image.Image, key_prefix: str):
    if st.button("🔍 Detect animals", type="primary", key=f"{key_prefix}_detect"):
        with st.spinner("Detecting objects with your custom YOLO model..."):
            annotated, detections = predict_image(image)

        st.subheader("Detection output")
        st.image(annotated, use_container_width=True)

        if detections:
            st.success(f"Detection completed: {len(detections)} object(s) found.")
            df = pd.DataFrame(detections)
            display_df = df.copy()
            display_df["Confidence"] = display_df["Confidence"].map(
                lambda value: f"{value:.2f}%"
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.subheader("Class-wise count")
            counts = (
                df.groupby("Animal / Class")
                .size()
                .reset_index(name="Number of detections")
                .sort_values("Number of detections", ascending=False)
            )
            st.dataframe(counts, use_container_width=True, hide_index=True)

            # Allow the user to save the annotated prediction image.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                annotated.save(tmp.name, format="JPEG", quality=95)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as result_file:
                st.download_button(
                    "⬇️ Download detected image",
                    data=result_file.read(),
                    file_name="animal_detection_result.jpg",
                    mime="image/jpeg",
                    key=f"{key_prefix}_download",
                )
            os.remove(tmp_path)
        else:
            st.warning(
                "No objects were detected. Try another image or lower the confidence threshold."
            )


upload_tab, camera_tab = st.tabs(["📁 Upload image", "📷 Camera"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP.",
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Input image")
        st.image(input_image, use_container_width=True)
        show_results(input_image, "upload")

with camera_tab:
    camera_file = st.camera_input("Capture an image")
    if camera_file is not None:
        camera_image = Image.open(camera_file).convert("RGB")
        st.image(camera_image, caption="Captured image", use_container_width=True)
        show_results(camera_image, "camera")

st.divider()
st.caption(
    "Model: custom YOLO checkpoint trained in Google Colab | "
    "Dataset classes: cat, cow, dog, horse, person, sheep"
)
st.info(
    "Prediction quality depends on the training data, annotation quality, class-ID "
    "mapping, and similarity between training and uploaded images."
)
