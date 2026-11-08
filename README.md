# AI-Based Animal Detection — Deployment Package

This Streamlit application is prepared for the uploaded notebook
`Custom_YOLO_Animal_Detection(1).ipynb`.

## Model details taken from the notebook

- YOLO checkpoint created by training `YOLO("yolo26n.pt")`
- Dataset YAML: `data.yaml`
- Class order: `cat`, `cow`, `dog`, `horse`, `person`, `sheep`
- Training: 20 epochs, image size 640
- Best checkpoint path in Colab:
  `/content/runs/animal_detection_20/weights/best.pt`

The app reads the class names from the checkpoint itself. The class list shown in
the app is therefore not used to remap or modify predictions.

## 1. Download your trained model from Colab

Run the final download cell in your notebook:

```python
files.download(BEST_PT)
```

If needed, the notebook defines:

```python
BEST_PT = "/content/runs/animal_detection_20/weights/best.pt"
```

Download the resulting `best.pt` file to your computer.

## 2. Prepare the project folder

Extract this ZIP and place the downloaded `best.pt` in the same folder as `app.py`:

```text
Animal_Detection_Deployment/
├── app.py
├── requirements.txt
├── README.md
└── best.pt
```

`best.pt` is not included in this ZIP because it must come from your own completed
Colab training run.

## 3. Run locally (optional)

Install Python 3.10 or 3.11, then run these commands in the project folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages and start the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the local application. You can upload an image or use the
camera tab, change the confidence threshold, view detections and class counts,
and download the annotated image.

## 4. Deploy using Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `best.pt`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**.
5. Choose the repository, branch, and `app.py` as the main file.
6. Select **Deploy**.

### Model file note

For a quick demo, you can add `best.pt` to the repository if its size is within
GitHub's file limits. Do not put a private model in a public repository. For a
large or private checkpoint, store it in private model/object storage and adapt
the app to retrieve it using secure secrets; do not expose access tokens in code.

## Important model-quality note

This deployment uses the trained checkpoint as-is. It does not retrain the model
or correct incorrect predictions. If predictions are wrong, first verify that
the dataset's YOLO class IDs really match the notebook's class order and inspect
the annotations. Then retrain and replace `best.pt` with the corrected checkpoint.
