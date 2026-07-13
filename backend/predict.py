"""Prediction pipeline for Pap smear images."""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from tensorflow.keras.models import load_model

from backend.config import CLASS_INDICES_PATH, HYBRID_MODEL_PATH
from backend.models.yolov5.detector import YOLOv5Detector
from backend.utils.preprocessing import crop_regions


def load_class_names() -> list:
    """Load class names saved during training."""
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError("Class index file missing. Train the model first.")

    with CLASS_INDICES_PATH.open("r", encoding="utf-8") as file:
        class_indices = json.load(file)

    return [
        class_name
        for class_name, _ in sorted(class_indices.items(), key=lambda item: item[1])
    ]


def predict_image(image_path: Path) -> Dict[str, object]:
    """Run YOLOv5 detection, crop regions, classify them, and average scores."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Prediction image not found: {image_path}")

    if not HYBRID_MODEL_PATH.exists():
        raise FileNotFoundError("Trained hybrid model not found. Call /train first.")

    class_names = load_class_names()
    detector = YOLOv5Detector()
    detections = detector.detect(image_path)
    crops = crop_regions(image_path, detections)

    model = load_model(HYBRID_MODEL_PATH)
    probabilities = model.predict(crops, verbose=0)
    average_probabilities = np.mean(probabilities, axis=0)

    predicted_index = int(np.argmax(average_probabilities))
    confidence = float(average_probabilities[predicted_index] * 100)

    return {
        "prediction": class_names[predicted_index],
        "confidence": f"{confidence:.2f}%",
        "detected_cells": len(detections),
        "detections": detections,
        "class_probabilities": {
            class_name: f"{float(score * 100):.2f}%"
            for class_name, score in zip(class_names, average_probabilities)
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict cervical cancer severity from an image.")
    parser.add_argument("image", type=str, help="Path to Pap smear image")
    args = parser.parse_args()
    print(json.dumps(predict_image(Path(args.image)), indent=2))
