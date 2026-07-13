# Hybrid Deep Learning-Based Cervical Cancer Severity Classification Backend

This backend provides Flask APIs for dataset upload, automatic ZIP extraction,
dataset cleaning, train/validation/test splitting, hybrid CNN + ResNet training,
YOLOv5-assisted region detection, prediction, and evaluation visualizations.

The system is designed for Pap smear image datasets such as SIPaKMeD or severity
folder datasets with class names like:

- Normal
- LSIL
- HSIL
- SCC

The dataset pipeline detects class folders automatically, so your ZIP can use
any class names as long as images are organized inside class folders.

## Folder Structure

```text
backend/
|-- app.py
|-- train.py
|-- predict.py
|-- evaluate.py
|-- config.py
|-- wsgi.py
|-- requirements.txt
|-- README.md
|-- models/
|   |-- cnn_model.py
|   |-- resnet_model.py
|   |-- hybrid_model.py
|   `-- yolov5/
|       |-- detector.py
|       `-- README.md
|-- dataset/
|   |-- raw/
|   `-- processed/
|-- uploads/
|-- static/
|-- templates/
|-- utils/
|   |-- dataset_utils.py
|   |-- logger.py
|   `-- preprocessing.py
|-- saved_models/
|-- results/
|   `-- graphs/
`-- logs/
```

## Setup

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

If TensorFlow or PyTorch GPU packages need a specific CUDA version, install the
matching builds from their official install pages before running training.

## Run the Flask Backend

```bash
python -m backend.app
```

On this Windows setup, use the included helper script to avoid OneDrive
bytecode-cache writes:

```powershell
cd "D:\AppData\OneDrive\Documents\New project"
.\backend\run_backend.ps1
```

Server URL:

```text
http://127.0.0.1:5000
```

Health check:

```text
GET http://127.0.0.1:5000/health
```

## API Endpoints

### 1. Upload Dataset

```text
POST /upload-dataset
```

Option A: Multipart upload in Postman

- Method: `POST`
- URL: `http://127.0.0.1:5000/upload-dataset`
- Body: `form-data`
- Key: `dataset`
- Type: `File`
- Value: select your ZIP dataset

Option B: Use a local ZIP path already on this machine

- Method: `POST`
- URL: `http://127.0.0.1:5000/upload-dataset`
- Body: `raw` JSON

```json
{
  "zip_path": "D:/archive.zip"
}
```

The backend will:

- Extract the ZIP safely
- Detect class folders
- Skip corrupted images
- Copy valid images into `backend/dataset/processed/all`
- Split images into `train`, `val`, and `test`
- Save a manifest at `backend/dataset/processed/manifest.json`

### 2. Train Model

```text
POST /train
```

Postman JSON body:

```json
{
  "epochs": 20,
  "batch_size": 16
}
```

Training includes:

- Custom CNN branch
- ResNet50 transfer learning branch with ImageNet weights
- CNN + ResNet feature fusion
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- TensorBoard
- Accuracy and loss graphs
- Confusion matrix
- Precision/Recall/F1 graph
- ROC curve

Outputs:

```text
C:\Users\<you>\cervical_backend_storage\saved_models\best_hybrid_model.keras
C:\Users\<you>\cervical_backend_storage\saved_models\class_indices.json
C:\Users\<you>\cervical_backend_storage\results\training_history.json
C:\Users\<you>\cervical_backend_storage\results\evaluation_metrics.json
C:\Users\<you>\cervical_backend_storage\results\graphs\
C:\Users\<you>\cervical_backend_storage\logs\tensorboard\
```

### 3. Predict Image

```text
POST /predict
```

Postman setup:

- Method: `POST`
- URL: `http://127.0.0.1:5000/predict`
- Body: `form-data`
- Key: `image`
- Type: `File`
- Value: select a Pap smear image

Example response:

```json
{
  "prediction": "HSIL",
  "confidence": "97.12%",
  "detected_cells": 4
}
```

The prediction workflow is:

1. Upload Pap smear image
2. YOLOv5 detects suspicious cervical cell regions
3. Detected regions are cropped
4. Hybrid CNN + ResNet classifier predicts each crop
5. Crop probabilities are averaged
6. API returns final class, confidence, detections, and per-class probabilities

## YOLOv5 Integration

Place custom YOLOv5 detection weights at:

```text
C:\Users\<you>\cervical_backend_storage\saved_models\yolov5_cells.pt
```

If no YOLOv5 weights exist, the backend falls back to classifying the full image.
This lets you train and test the classifier before your detection model is ready.

To train YOLOv5 separately, use bounding-box annotations for cervical cells and
export the trained `.pt` file to `backend/saved_models/yolov5_cells.pt`.

## Runtime Storage Location

By default, uploaded ZIPs, extracted datasets, generated graphs, logs, and saved
models are stored outside the OneDrive project folder at:

```text
C:\Users\<you>\cervical_backend_storage
```

This avoids Windows/OneDrive file-lock errors during large ZIP extraction. To
use another storage drive, set this environment variable before starting Flask:

```powershell
$env:CERVICAL_BACKEND_STORAGE = "E:\cervical_backend_storage"
python -m backend.app
```

## Command-Line Usage

Train:

```bash
python -m backend.train
```

Predict:

```bash
python -m backend.predict path/to/image.jpg
```

## TensorBoard

After training starts:

```bash
tensorboard --logdir backend/logs/tensorboard
```

Open:

```text
http://127.0.0.1:6006
```

## Deployment

For local production-style serving on Windows:

```bash
waitress-serve --listen=0.0.0.0:5000 backend.wsgi:application
```

For Linux servers, you can use Gunicorn:

```bash
gunicorn backend.wsgi:application --bind 0.0.0.0:5000 --workers 2
```

## Important Notes

- This is a research backend and is not a clinical diagnostic system.
- A 6 GB dataset can take significant time to extract and preprocess.
- ResNet50 ImageNet weights may download the first time training runs.
- GPU support is enabled automatically when TensorFlow detects a compatible GPU.
- Class folders must contain at least 3 valid images for splitting.
