"""Central configuration for paths, model settings, and training defaults."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Large datasets and model outputs should not live inside a OneDrive-synced
# folder on Windows because file locking can interrupt ZIP extraction and pip.
# Override this location with CERVICAL_BACKEND_STORAGE if needed.
DEFAULT_STORAGE_DIR = Path.home() / "cervical_backend_storage"
STORAGE_DIR = Path(os.environ.get("CERVICAL_BACKEND_STORAGE", DEFAULT_STORAGE_DIR)).resolve()

UPLOAD_DIR = STORAGE_DIR / "uploads"
RAW_DATASET_DIR = STORAGE_DIR / "dataset" / "raw"
PROCESSED_DATASET_DIR = STORAGE_DIR / "dataset" / "processed"
SPLIT_DATASET_DIR = PROCESSED_DATASET_DIR / "splits"
ACTIVE_SPLIT_DIR_PATH = PROCESSED_DATASET_DIR / "active_split_dir.txt"
SAVED_MODELS_DIR = STORAGE_DIR / "saved_models"
RESULTS_DIR = STORAGE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
LOG_DIR = STORAGE_DIR / "logs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DEFAULT_EPOCHS = 20
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}

HYBRID_MODEL_PATH = SAVED_MODELS_DIR / "best_hybrid_model.keras"
CLASS_INDICES_PATH = SAVED_MODELS_DIR / "class_indices.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
YOLO_WEIGHTS_PATH = SAVED_MODELS_DIR / "yolov5_cells.pt"

# These are examples. The dataset pipeline detects class folders automatically.
DEFAULT_SEVERITY_CLASSES = ["Normal", "LSIL", "HSIL", "SCC"]


def ensure_directories() -> None:
    """Create all runtime directories used by the backend."""
    for directory in [
        UPLOAD_DIR,
        RAW_DATASET_DIR,
        PROCESSED_DATASET_DIR,
        SAVED_MODELS_DIR,
        RESULTS_DIR,
        GRAPHS_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_active_split_dir() -> Path:
    """Return the latest successful dataset split folder."""
    if ACTIVE_SPLIT_DIR_PATH.exists():
        saved_path = Path(ACTIVE_SPLIT_DIR_PATH.read_text(encoding="utf-8").strip())
        if saved_path.exists():
            return saved_path
    return SPLIT_DATASET_DIR
