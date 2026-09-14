import threading
from functools import lru_cache

import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings
from app.core.exceptions import NoFaceDetectedError, MultipleFacesDetectedError

# Module-level lock to ensure thread-safe model initialisation.
# InsightFace model loading is not thread-safe; subsequent inference calls are.
_model_lock = threading.Lock()
_face_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    """
    Return the shared, initialised FaceAnalysis instance.

    Initialises once on first call (lazy singleton). Thread-safe via a module-level
    lock held only during initialisation, not during inference.

    Returns:
        Prepared FaceAnalysis instance.
    """
    global _face_app

    if _face_app is not None:
        return _face_app

    with _model_lock:
        # Re-check inside the lock to handle concurrent first calls.
        if _face_app is None:
            app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=settings.DET_SIZE)
            _face_app = app

    return _face_app


def get_embedding(image: np.ndarray, source: str = "image") -> np.ndarray:
    """
    Detect exactly one face in the image and return its L2-normalised embedding.

    Args:
        image:  BGR numpy array of shape (H, W, 3).
        source: Label used in exception messages to identify which image
                failed — e.g. "reference image" or "snapshot".

    Returns:
        L2-normalised embedding vector of shape (512,) as float32.

    Raises:
        NoFaceDetectedError:       If no face is found in the image.
        MultipleFacesDetectedError: If more than one face is found.
    """
    face_app = get_face_app()
    faces = face_app.get(image)

    if len(faces) == 0:
        raise NoFaceDetectedError(source=source)

    if len(faces) > 1:
        raise MultipleFacesDetectedError(source=source, count=len(faces))

    embedding = faces[0].embedding
    return _l2_normalise(embedding)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalise(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalise a vector so cosine similarity reduces to a dot product.

    If the vector norm is zero (degenerate case), the original vector is
    returned unchanged to avoid a division-by-zero.

    Args:
        vector: Raw embedding array from InsightFace.

    Returns:
        Unit-length float32 array of the same shape.
    """
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)