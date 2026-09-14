import numpy as np
import cv2
from fastapi import UploadFile

from app.core.exceptions import InvalidImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


async def decode_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    """
    Read an UploadFile, validate its format and size,
    and decode it into a BGR numpy array for InsightFace.

    Args:
        upload: FastAPI UploadFile from a multipart form field.

    Returns:
        BGR numpy array of shape (H, W, 3).

    Raises:
        InvalidImageError: If content type is unsupported, file exceeds size
                           limit, or the bytes cannot be decoded as an image.
    """
    _validate_content_type(upload.content_type)

    contents = await upload.read()

    _validate_file_size(contents)

    return _decode_bytes_to_bgr(contents)


def decode_file_to_bgr(path) -> np.ndarray:
    """
    Read an image from a filesystem path and return it as a BGR numpy array.

    Args:
        path: pathlib.Path or str pointing to a JPEG/PNG file.

    Returns:
        BGR numpy array of shape (H, W, 3).

    Raises:
        InvalidImageError: If the file cannot be decoded as an image.
    """
    img = cv2.imread(str(path))

    if img is None:
        raise InvalidImageError(f"Could not decode image at path: {path}")

    return img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_content_type(content_type: str | None) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"Unsupported file type '{content_type}'. "
            f"Accepted types: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )


def _validate_file_size(contents: bytes) -> None:
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise InvalidImageError(
            f"File size {len(contents) / (1024 * 1024):.1f} MB "
            f"exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit"
        )


def _decode_bytes_to_bgr(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise InvalidImageError("Uploaded bytes could not be decoded as an image")

    return img