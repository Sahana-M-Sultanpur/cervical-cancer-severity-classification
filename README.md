# Deep Learning-Based Severity Classification of Keratoconus

This project is a research prototype for classifying keratoconus severity from corneal topography data. It includes:

- A PyTorch training pipeline
- Multi-map corneal topography support
- Optional clinical/tabular feature fusion
- Evaluation with classification metrics
- A Streamlit GUI for upload, prediction, confidence display, and visual explanation

> This project is for academic/research use only. It is not a clinical diagnostic system.

## Problem Statement

Keratoconus is a progressive corneal ectatic disorder that causes corneal thinning and irregular curvature. Early identification and severity grading can support clinical decision-making, follow-up planning, and treatment selection. This project uses deep learning to classify corneal topography data into severity categories.

## Severity Classes

The default classes are:

1. Normal
2. Mild Keratoconus
3. Moderate Keratoconus
4. Severe Keratoconus

You can modify these in `src/config.py` if your dataset uses different labels.

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── demo/
├── models/
├── scripts/
│   └── create_demo_dataset.py
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── model.py
│   ├── predict.py
│   ├── splits.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Dataset Format

Create a CSV manifest with one row per eye.

Recommended columns:

```csv
patient_id,label,axial_path,anterior_elevation_path,posterior_elevation_path,pachymetry_path,kmax,central_pachymetry,thinnest_pachymetry,astigmatism,age
P001,Mild Keratoconus,images/P001_OD_axial.png,images/P001_OD_anterior.png,images/P001_OD_posterior.png,images/P001_OD_pachymetry.png,49.2,501,476,-2.8,24
P002,Normal,images/P002_OS_axial.png,images/P002_OS_anterior.png,images/P002_OS_posterior.png,images/P002_OS_pachymetry.png,43.1,532,520,-0.7,31
```

Paths may be absolute or relative to the manifest CSV location.

If you only have one image per eye, use:

```csv
patient_id,label,image_path,kmax,central_pachymetry,thinnest_pachymetry,astigmatism,age
P001,Moderate Keratoconus,images/P001_OD_topography.png,54.3,472,441,-4.0,22
```

## Demo Dataset

The repository includes a script that creates synthetic topography-like images for testing the pipeline:

```bash
python scripts/create_demo_dataset.py
```

This creates:

```text
data/demo/manifest.csv
data/demo/images/
```

The demo images are not medical data. They only help verify that the software works.

## Training

Train with multi-map images and clinical features:

```bash
python -m src.train --manifest data/demo/manifest.csv --epochs 5 --batch-size 4
```

The best checkpoint is saved to:

```text
models/best_model.pth
```

For single-map training:

```bash
python -m src.train --manifest data/demo/manifest.csv --single-map --epochs 5
```

For image-only training without clinical inputs:

```bash
python -m src.train --manifest data/demo/manifest.csv --no-tabular
```

## Evaluation

After training, evaluate on a manifest:

```bash
python -m src.evaluate --manifest models/splits/test.csv --checkpoint models/best_model.pth
```

The script prints precision, recall, F1-score, and a confusion matrix.

## Run the GUI

```bash
streamlit run app/streamlit_app.py
```

The GUI supports:

- Single image upload
- Multi-map upload
- Manual clinical values
- Severity prediction
- Confidence chart
- Demo heatmap overlay

If `models/best_model.pth` exists, the GUI uses the trained model. If no checkpoint exists, it uses a clearly labeled demo estimate based on clinical values.

## Suggested Methodology for Report

1. Collect corneal topography maps and clinical measurements.
2. Clean and anonymize the dataset.
3. Assign severity labels using clinician-provided grades or a validated grading system.
4. Split data at the patient level to avoid leakage.
5. Resize and normalize corneal maps.
6. Train a CNN-based classifier with optional clinical feature fusion.
7. Evaluate using accuracy, macro F1-score, precision, recall, and confusion matrix.
8. Build a GUI for image upload, prediction, confidence visualization, and explanation.

## Important Limitations

- Models trained on one topography device may not generalize to another device.
- Severity labels must be clinically reliable.
- Patient-level splitting is essential because both eyes from the same patient can leak information.
- Synthetic demo data must never be used to claim medical performance.
- This prototype requires clinical validation before any real-world medical use.

