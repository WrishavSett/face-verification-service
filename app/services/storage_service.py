import shutil
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.exceptions import StudentNotFoundError
from app.utils.image_utils import decode_file_to_bgr

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_reference_image(student_id: str) -> np.ndarray:
    """
    Locate and return the reference image for a given student as a BGR array.

    Searches for {student_id}.jpg, {student_id}.jpeg, {student_id}.png
    under PHOTO_DIR, in that order.

    Args:
        student_id: Unique university identification number.

    Returns:
        BGR numpy array of the reference image.

    Raises:
        StudentNotFoundError: If no matching file is found for the student_id.
    """
    photo_path = _resolve_photo_path(student_id)
    return decode_file_to_bgr(photo_path)


def save_reference_image(student_id: str, contents: bytes, extension: str) -> Path:
    """
    Persist a reference image for a student to local storage.

    Removes any existing reference photo for the student before writing,
    ensuring there is always exactly one file per student_id.

    Args:
        student_id: Unique university identification number.
        contents:   Raw bytes of the image file.
        extension:  File extension including leading dot, e.g. '.jpg'.

    Returns:
        Path where the file was written.
    """
    _ensure_photo_dir()
    _remove_existing(student_id)

    dest = settings.PHOTO_DIR / f"{student_id}{extension}"
    dest.write_bytes(contents)

    return dest


def student_exists(student_id: str) -> bool:
    """
    Check whether a reference photo exists for the given student_id.

    Args:
        student_id: Unique university identification number.

    Returns:
        True if a reference photo is found, False otherwise.
    """
    try:
        _resolve_photo_path(student_id)
        return True
    except StudentNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_photo_path(student_id: str) -> Path:
    """
    Search PHOTO_DIR for a file matching student_id with any supported extension.

    Raises:
        StudentNotFoundError: If no matching file is found.
    """
    for ext in SUPPORTED_EXTENSIONS:
        candidate = settings.PHOTO_DIR / f"{student_id}{ext}"
        if candidate.exists():
            return candidate

    raise StudentNotFoundError(student_id)


def _ensure_photo_dir() -> None:
    """Create PHOTO_DIR and any missing parents if they do not exist."""
    settings.PHOTO_DIR.mkdir(parents=True, exist_ok=True)


def _remove_existing(student_id: str) -> None:
    """
    Delete any existing reference photo for the student across all supported
    extensions. Handles the hall ticket photo update case cleanly.
    """
    for ext in SUPPORTED_EXTENSIONS:
        candidate = settings.PHOTO_DIR / f"{student_id}{ext}"
        if candidate.exists():
            candidate.unlink()