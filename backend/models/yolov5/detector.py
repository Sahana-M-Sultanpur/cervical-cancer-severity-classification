"""YOLOv5 detector wrapper.

The wrapper tries to load YOLOv5 through PyTorch Hub. If YOLO weights are not
available yet, it falls back to returning the full image as one detected region.
That keeps the classification pipeline usable before a detection model is added.
"""

from pathlib import Path
from typing import List

import cv2

from backend.config import YOLO_WEIGHTS_PATH
from backend.utils.logger import get_logger


logger = get_logger(__name__)


class YOLOv5Detector:
    """Detect suspicious cervical cell regions using YOLOv5."""

    def __init__(self, weights_path: Path = YOLO_WEIGHTS_PATH, confidence: float = 0.25):
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load a custom YOLOv5 model if weights exist, otherwise fallback."""
        try:
            import torch

            if self.weights_path.exists():
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(self.weights_path),
                    force_reload=False,
                )
                self.model.conf = self.confidence
                logger.info("Loaded custom YOLOv5 weights from %s", self.weights_path)
            else:
                logger.warning(
                    "YOLOv5 weights not found at %s. Using full-image fallback detection.",
                    self.weights_path,
                )
        except Exception as exc:
            logger.warning("YOLOv5 could not be loaded. Fallback detection active: %s", exc)
            self.model = None

    def detect(self, image_path: Path) -> List[dict]:
        """Return detected boxes as dictionaries with xmin/ymin/xmax/ymax."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image for YOLO detection: {image_path}")

        height, width = image.shape[:2]

        if self.model is None:
            return [
                {
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": width,
                    "ymax": height,
                    "confidence": 1.0,
                    "class_name": "full_image_fallback",
                }
            ]

        results = self.model(str(image_path))
        detections = []

        for _, row in results.pandas().xyxy[0].iterrows():
            detections.append(
                {
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                    "confidence": float(row["confidence"]),
                    "class_name": str(row.get("name", "cell")),
                }
            )

        if not detections:
            logger.info("YOLOv5 found no cells. Using full image for classification.")
            return [
                {
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": width,
                    "ymax": height,
                    "confidence": 1.0,
                    "class_name": "full_image_fallback",
                }
            ]

        return detections
