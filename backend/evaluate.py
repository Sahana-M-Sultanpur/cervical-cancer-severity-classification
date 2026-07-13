"""Evaluation and visualization helpers for model training."""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from backend.config import GRAPHS_DIR, METRICS_PATH, RESULTS_DIR, ensure_directories


def _save_plot(path: Path) -> str:
    """Save the current Matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def save_training_graphs(history: Dict[str, List[float]]) -> Dict[str, str]:
    """Create accuracy and loss graphs from Keras training history."""
    ensure_directories()
    saved_paths = {}

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("accuracy", []), label="Training Accuracy")
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Accuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    saved_paths["accuracy_graph"] = _save_plot(GRAPHS_DIR / "accuracy_graph.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="Training Loss")
    plt.plot(history.get("val_loss", []), label="Validation Loss")
    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    saved_paths["loss_graph"] = _save_plot(GRAPHS_DIR / "loss_graph.png")

    return saved_paths


def save_precision_recall_graph(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """Create a per-class precision/recall/F1 bar chart."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")
    plt.xticks(x, class_names, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.title("Precision, Recall, and F1-score")
    plt.legend()
    return _save_plot(GRAPHS_DIR / "precision_recall_graph.png")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """Create and save a confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set_title("Confusion Matrix")
    return _save_plot(GRAPHS_DIR / "confusion_matrix.png")


def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> str:
    """Create a multi-class ROC curve plot."""
    y_true_binary = label_binarize(y_true, classes=list(range(len(class_names))))

    plt.figure(figsize=(8, 6))
    for class_index, class_name in enumerate(class_names):
        if y_true_binary.shape[1] <= class_index:
            continue
        fpr, tpr, _ = roc_curve(y_true_binary[:, class_index], y_prob[:, class_index])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    return _save_plot(GRAPHS_DIR / "roc_curve.png")


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, object]:
    """Save evaluation metrics and visualization files."""
    ensure_directories()
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    graph_paths = {
        "confusion_matrix": save_confusion_matrix(y_true, y_pred, class_names),
        "precision_recall_graph": save_precision_recall_graph(y_true, y_pred, class_names),
        "roc_curve": save_roc_curve(y_true, y_prob, class_names),
    }

    report_path = RESULTS_DIR / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    metrics = {
        "classification_report": report,
        "graphs": graph_paths,
        "classification_report_csv": str(report_path),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
