import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
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
    "Upload an image or use your camera to detect **cat, cow, dog, horse, "
    "person, and sheep** with your custom-trained YOLO model."
)

# ------------------------------------------------------------
# Find the trained model checkpoint
# ------------------------------------------------------------
# Recommended: keep best.pt in the same folder as this app.py.
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# You can override the path by setting YOLO_MODEL_PATH.
env_model_path = os.getenv("YOLO_MODEL_PATH")

candidate_paths = []
if env_model_path:
    candidate_paths.append(os.path.expanduser(env_model_path))

candidate_paths.extend(
    [
        os.path.join(APP_DIR, "best.pt"),
        os.path.join(APP_DIR, "weights", "best.pt"),
        os.path.join(APP_DIR, "runs", "animal_detection_corrected_cpu",
                     "weights", "best.pt"),
        os.path.join(APP_DIR, "runs", "animal_detection_cpu_5",
                     "weights", "best.pt"),
    ]
)

MODEL_PATH = next(
    (os.path.abspath(p) for p in candidate_paths if os.path.isfile(p)),
    None,
)

if MODEL_PATH is None:
    st.error(
        "The trained model file **best.pt** was not found. Download `best.pt` "
        "from the corrected Google Colab notebook and place it in the same "
        "folder as `app.py`."
    )
    st.info(
        "Expected project layout:\n\n"
        "```text\n"
        "Animal_Detection/\n"
        "├── app.py\n"
        "├── best.pt\n"
        "└── requirements.txt\n"
        "```\n\n"
        "Alternatively, set the `YOLO_MODEL_PATH` environment variable to "
        "the full path of your checkpoint."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading your trained YOLO model...")
def load_model(model_path: str, modified_time: float):
    # modified_time is included so Streamlit reloads the model if best.pt changes.
    del modified_time
    return YOLO(model_path)


try:
    model = load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))
except Exception as exc:
    st.error(f"Unable to load the YOLO checkpoint: {exc}")
    st.stop()

# Always use the class names saved inside the checkpoint. Do not hard-code
# a different class order: the checkpoint's class-ID mapping is authoritative.
model_names = model.names
if isinstance(model_names, dict):
    class_names = [model_names[k] for k in sorted(model_names)]
else:
    class_names = list(model_names)

st.success(f"Model loaded successfully: `{os.path.basename(MODEL_PATH)}`")
with st.expander("Model information"):
    st.write("Checkpoint:", MODEL_PATH)
    st.write("Classes stored in checkpoint:", class_names)
    st.caption(
        "The displayed labels come directly from best.pt, so the app follows "
        "the class-ID mapping used during training."
    )

# ------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------
with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help=(
            "Minimum confidence required to display a detection. Lowering this "
            "may show more detections, including false positives."
        ),
    )
    image_size = st.select_slider(
        "Image size",
        options=[320, 416, 512, 640, 768],
        value=640,
    )
    st.caption("Classes in the trained model")
    for class_name in class_names:
        st.write(f"• {class_name}")
    st.caption("Inference device: CPU")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
def predict_image(image: Image.Image):
    """Return an annotated RGB image and a list of detected objects."""
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        device="cpu",
        verbose=False,
    )[0]

    # Ultralytics result.plot() returns a BGR NumPy array.
    annotated_rgb = Image.fromarray(result.plot()[:, :, ::-1])

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())

            if isinstance(model_names, dict):
                label = model_names.get(class_id, f"class_{class_id}")
            else:
                label = (
                    model_names[class_id]
                    if 0 <= class_id < len(model_names)
                    else f"class_{class_id}"
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
        try:
            with st.spinner("Detecting objects with your custom YOLO model..."):
                annotated, detections = predict_image(image)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

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

            # Provide the annotated image as a downloadable JPEG.
            image_buffer = BytesIO()
            annotated.save(image_buffer, format="JPEG", quality=95)
            st.download_button(
                "⬇️ Download detected image",
                data=image_buffer.getvalue(),
                file_name="animal_detection_result.jpg",
                mime="image/jpeg",
                key=f"{key_prefix}_download",
            )
        else:
            st.warning(
                "No objects were detected. Try a clearer image or adjust the "
                "confidence threshold. A lower threshold can also increase "
                "false positives."
            )


# ------------------------------------------------------------
# Image upload and camera tabs
# ------------------------------------------------------------
upload_tab, camera_tab = st.tabs(["📁 Upload image", "📷 Camera"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP.",
        key="animal_image_upload",
    )
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file)
            input_image = ImageOps.exif_transpose(input_image).convert("RGB")
            st.subheader("Input image")
            st.image(input_image, use_container_width=True)
            show_results(input_image, "upload")
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

with camera_tab:
    camera_file = st.camera_input("Capture an image")
    if camera_file is not None:
        try:
            camera_image = Image.open(camera_file)
            camera_image = ImageOps.exif_transpose(camera_image).convert("RGB")
            st.image(camera_image, caption="Captured image", use_container_width=True)
            show_results(camera_image, "camera")
        except Exception as exc:
            st.error(f"Could not process the camera image: {exc}")

st.divider()
st.caption(
    "Model: custom YOLO checkpoint trained in Google Colab | "
    "Labels are read from the checkpoint's saved class mapping."
)
st.info(
    "Prediction quality depends on dataset size, annotation accuracy, class-ID "
    "mapping, training quality, and similarity between training and uploaded images. "
    "Changing the confidence threshold does not correct a model trained with "
    "incorrect labels."
)
