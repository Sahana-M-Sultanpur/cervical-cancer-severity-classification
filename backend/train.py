"""Training script for the hybrid CNN + ResNet classifier."""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from backend.config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    DEFAULT_EPOCHS,
    LOG_DIR,
    HYBRID_MODEL_PATH,
    IMAGE_SIZE,
    RESULTS_DIR,
    TRAINING_HISTORY_PATH,
    ensure_directories,
    get_active_split_dir,
)
from backend.evaluate import evaluate_predictions, save_training_graphs
from backend.models.hybrid_model import build_hybrid_model
from backend.utils.logger import get_logger


logger = get_logger(__name__)


def configure_gpu() -> Dict[str, object]:
    """Enable TensorFlow memory growth when a GPU is available."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return {"gpu_available": bool(gpus), "gpu_count": len(gpus)}


def create_generators(batch_size: int = BATCH_SIZE):
    """Create train, validation, and test generators from split folders."""
    split_dataset_dir = get_active_split_dir()
    train_dir = split_dataset_dir / "train"
    val_dir = split_dataset_dir / "val"
    test_dir = split_dataset_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Dataset splits not found at {split_dataset_dir}. Call /upload-dataset before training."
        )

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=18,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, val_generator, test_generator


def build_callbacks() -> list:
    """Create training callbacks requested by the project."""
    ensure_directories()
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(HYBRID_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(RESULTS_DIR / "training_log.csv")),
    ]
    try:
        import tensorboard  # noqa: F401
        from tensorflow.keras.callbacks import TensorBoard

        callbacks.append(TensorBoard(log_dir=str(LOG_DIR / "tensorboard")))
    except Exception as exc:
        logger.warning("TensorBoard callback disabled: %s", exc)

    return callbacks


def train_model(epochs: int = DEFAULT_EPOCHS, batch_size: int = BATCH_SIZE) -> Dict[str, object]:
    """Train the hybrid classifier and save metrics, graphs, and best weights."""
    ensure_directories()
    gpu_info = configure_gpu()
    logger.info("Starting training. GPU info: %s", gpu_info)

    train_generator, val_generator, test_generator = create_generators(batch_size=batch_size)
    class_names = list(train_generator.class_indices.keys())

    with CLASS_INDICES_PATH.open("w", encoding="utf-8") as file:
        json.dump(train_generator.class_indices, file, indent=2)

    model = build_hybrid_model(num_classes=len(class_names))
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=build_callbacks(),
    )

    model.save(HYBRID_MODEL_PATH)

    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history_dict, file, indent=2)

    training_graphs = save_training_graphs(history_dict)
    y_prob = model.predict(test_generator, verbose=1)
    y_true = test_generator.classes
    metrics = evaluate_predictions(np.array(y_true), y_prob, class_names)

    return {
        "message": "Training completed successfully.",
        "classes": class_names,
        "gpu": gpu_info,
        "model_path": str(HYBRID_MODEL_PATH),
        "history_path": str(TRAINING_HISTORY_PATH),
        "training_graphs": training_graphs,
        "evaluation": metrics,
    }


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(result, indent=2))
