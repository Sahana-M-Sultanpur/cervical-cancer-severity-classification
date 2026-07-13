"""Dataset upload, extraction, cleaning, and train/validation/test splitting."""

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from sklearn.model_selection import train_test_split

from backend.config import (
    ACTIVE_SPLIT_DIR_PATH,
    ALLOWED_ARCHIVE_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    PROCESSED_DATASET_DIR,
    RAW_DATASET_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    ensure_directories,
)
from backend.utils.logger import get_logger
from backend.utils.preprocessing import is_valid_image, is_valid_image_bytes


logger = get_logger(__name__)


def _handle_remove_readonly(function, path, _exc_info):
    """Make read-only Windows files writable, then retry deletion."""
    os.chmod(path, 0o700)
    function(path)


def remove_tree(path: Path) -> None:
    """Remove a folder safely on Windows, including read-only files."""
    if path.exists():
        shutil.rmtree(path, onerror=_handle_remove_readonly)


def validate_zip_file(zip_path: Path) -> None:
    """Validate that the uploaded dataset is a ZIP archive."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")

    if zip_path.suffix.lower() not in ALLOWED_ARCHIVE_EXTENSIONS:
        raise ValueError("Only ZIP dataset uploads are supported.")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Uploaded file is not a valid ZIP archive.")


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extract a ZIP archive while preventing path traversal attacks.

    Some large Windows-created archives contain duplicate image entries. Python's
    extractall can fail with WinError 5 when it tries to overwrite one of those
    duplicate paths, so extraction is performed manually and duplicates are
    skipped.
    """
    validate_zip_file(zip_path)
    destination.mkdir(parents=True, exist_ok=True)

    extract_root = destination / zip_path.stem
    remove_tree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    extracted_paths = set()
    skipped_duplicates = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target_path = (extract_root / member.filename).resolve()
            if not str(target_path).startswith(str(extract_root.resolve())):
                raise ValueError(f"Blocked unsafe archive path: {member.filename}")

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if target_path in extracted_paths or target_path.exists():
                skipped_duplicates += 1
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_paths.add(target_path)

    logger.info(
        "Extracted dataset ZIP to %s. Skipped %s duplicate entries.",
        extract_root,
        skipped_duplicates,
    )
    return extract_root


def find_class_folders(root: Path) -> Dict[str, List[Path]]:
    """Find image files and map them to class names.

    SIPaKMeD commonly stores files like:
    im_Dyskeratotic/im_Dyskeratotic/CROPPED/001_01.bmp

    For that layout, the class should be Dyskeratotic, not CROPPED.
    For generic datasets, the immediate parent folder is used as the class.
    """
    class_map: Dict[str, List[Path]] = {}

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            continue

        class_name = file_path.parent.name
        for part in file_path.relative_to(root).parts:
            if part.startswith("im_") and part != "images":
                class_name = part.removeprefix("im_")
                break

        class_map.setdefault(class_name, []).append(file_path)

    if not class_map:
        raise ValueError("No class folders with supported image files were found.")

    return class_map


def clean_and_organize_images(class_map: Dict[str, List[Path]]) -> Dict[str, int]:
    """Copy valid images into a normalized dataset/all/class_name structure."""
    all_dir = PROCESSED_DATASET_DIR / "all"
    remove_tree(all_dir)
    all_dir.mkdir(parents=True, exist_ok=True)

    summary = {"valid_images": 0, "corrupted_images": 0, "classes": {}}

    for class_name, files in sorted(class_map.items()):
        class_dir = all_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_count = 0

        for index, file_path in enumerate(files, start=1):
            if not is_valid_image(file_path):
                summary["corrupted_images"] += 1
                logger.warning("Skipping corrupted image: %s", file_path)
                continue

            new_name = f"{class_name}_{index:06d}{file_path.suffix.lower()}"
            shutil.copy2(file_path, class_dir / new_name)
            class_count += 1
            summary["valid_images"] += 1

        summary["classes"][class_name] = class_count

    if summary["valid_images"] == 0:
        raise ValueError("All images were corrupted or unsupported.")

    return summary


def split_dataset(source_dir: Path | None = None) -> Dict[str, object]:
    """Split organized data into train, validation, and test folders."""
    source_dir = source_dir or PROCESSED_DATASET_DIR / "all"
    if not source_dir.exists():
        raise FileNotFoundError(f"Processed source folder does not exist: {source_dir}")

    split_dir = PROCESSED_DATASET_DIR / f"splits_{uuid4().hex[:8]}"
    for split_name in ["train", "val", "test"]:
        (split_dir / split_name).mkdir(parents=True, exist_ok=True)

    split_summary = {
        "train": {},
        "val": {},
        "test": {},
        "class_names": [],
        "split_dir": str(split_dir),
    }

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        files = sorted(
            file for file in class_dir.iterdir() if file.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        if len(files) < 3:
            raise ValueError(f"Class '{class_dir.name}' needs at least 3 valid images for splitting.")

        train_files, temp_files = train_test_split(
            files,
            test_size=VALIDATION_SIZE + TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=relative_test_size,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        split_summary["class_names"].append(class_dir.name)
        for split_name, split_files in {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }.items():
            target_dir = split_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in split_files:
                shutil.copy2(file_path, target_dir / file_path.name)
            split_summary[split_name][class_dir.name] = len(split_files)

    manifest_path = PROCESSED_DATASET_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(split_summary, file, indent=2)
    ACTIVE_SPLIT_DIR_PATH.write_text(str(split_dir), encoding="utf-8")

    return split_summary


def _class_name_from_zip_member(member_name: str) -> str:
    """Infer class name from a ZIP member path.

    SIPaKMeD paths contain folders like im_Dyskeratotic/.../CROPPED/file.bmp.
    Generic datasets usually use class_name/file.jpg.
    """
    parts = [part for part in Path(member_name).parts if part not in {"", "."}]

    for part in parts:
        if part.startswith("im_") and part != "images":
            return part.removeprefix("im_")

    # If the image lives in a generic CROPPED folder, use its parent as class.
    if len(parts) >= 3 and parts[-2].lower() == "cropped":
        parent = parts[-3]
        return parent.removeprefix("im_")

    if len(parts) >= 2:
        return parts[-2]

    return "unknown"


def prepare_dataset_directly_from_zip(zip_path: Path) -> Dict[str, object]:
    """Read images directly from ZIP into processed/all without raw extraction."""
    ensure_directories()
    validate_zip_file(zip_path)

    all_dir = PROCESSED_DATASET_DIR / f"all_{uuid4().hex[:8]}"
    all_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "valid_images": 0,
        "corrupted_images": 0,
        "skipped_duplicates": 0,
        "classes": {},
    }
    written_names = set()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            if member_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                continue

            class_name = _class_name_from_zip_member(member.filename)
            class_dir = all_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            data = archive.read(member)
            if not is_valid_image_bytes(data):
                summary["corrupted_images"] += 1
                logger.warning("Skipping corrupted image in ZIP: %s", member.filename)
                continue

            class_count = summary["classes"].get(class_name, 0) + 1
            target_name = f"{class_name}_{class_count:06d}{member_path.suffix.lower()}"
            target_path = class_dir / target_name

            if target_path in written_names:
                summary["skipped_duplicates"] += 1
                continue

            if target_path.exists():
                summary["skipped_duplicates"] += 1
                continue

            with target_path.open("wb") as target:
                target.write(data)
            written_names.add(target_path)
            summary["classes"][class_name] = class_count
            summary["valid_images"] += 1

    if summary["valid_images"] == 0:
        raise ValueError("No valid images were found inside the ZIP file.")

    split_summary = split_dataset(source_dir=all_dir)
    return {
        "message": "Dataset read from ZIP, cleaned, and split successfully.",
        "zip_path": str(zip_path),
        "processed_source": str(all_dir),
        "cleaning": summary,
        "splits": split_summary,
    }


def prepare_dataset_from_zip(zip_path: Path) -> Dict[str, object]:
    """Full dataset workflow: extract, detect classes, clean images, and split."""
    return prepare_dataset_directly_from_zip(zip_path)
