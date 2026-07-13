"""Logging helpers used across the backend."""

import logging
from logging.handlers import RotatingFileHandler

from backend.config import LOG_DIR, ensure_directories


def get_logger(name: str = "cervical_backend") -> logging.Logger:
    """Return a configured logger with console and rotating file handlers."""
    ensure_directories()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_DIR / "backend.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
