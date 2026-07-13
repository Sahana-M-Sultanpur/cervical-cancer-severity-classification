"""Image preprocessing helpers for training and prediction."""

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import IMAGE_SIZE


def is_valid_image(path: Path) -> bool:
    """Check whether an image can be opened and decoded."""
    try:
        with Image.open(path) as image:
            image.verify()

        decoded = cv2.imread(str(path))
        return decoded is not None
    except (OSError, UnidentifiedImageError):
        return False


def is_valid_image_bytes(data: bytes) -> bool:
    """Check whether image bytes can be decoded by Pillow and OpenCV."""
    try:
        import io

        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        image_array = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return decoded is not None
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def load_image(path: Path, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load an image as normalized RGB array with shape H x W x 3."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype("float32") / 255.0
    return image


def preprocess_batch(paths: Iterable[Path], image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load multiple image paths into a model-ready NumPy batch."""
    return np.stack([load_image(Path(path), image_size) for path in paths], axis=0)


def crop_regions(image_path: Path, boxes: List[dict], image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Crop YOLO-detected boxes. If no boxes exist, use the full image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    regions = []
    usable_boxes = boxes or [{"xmin": 0, "ymin": 0, "xmax": width, "ymax": height}]

    for box in usable_boxes:
        xmin = max(0, int(box["xmin"]))
        ymin = max(0, int(box["ymin"]))
        xmax = min(width, int(box["xmax"]))
        ymax = min(height, int(box["ymax"]))

        if xmax <= xmin or ymax <= ymin:
            continue

        crop = image[ymin:ymax, xmin:xmax]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, image_size)
        regions.append(crop.astype("float32") / 255.0)

    if not regions:
        regions.append(load_image(image_path, image_size))

    return np.stack(regions, axis=0)
