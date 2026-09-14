import numpy as np

from app.core.config import settings


def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised embedding vectors.

    Since both vectors are expected to be L2-normalised (enforced by
    face_service.get_embedding), cosine similarity reduces to a dot product,
    which is both faster and numerically stable.

    Result is clipped to [0.0, 1.0] to guard against floating point drift
    on near-identical vectors producing values marginally outside this range.

    Args:
        embedding_a: L2-normalised float32 array of shape (512,).
        embedding_b: L2-normalised float32 array of shape (512,).

    Returns:
        Similarity score in the range [0.0, 1.0].
    """
    score = float(np.dot(embedding_a, embedding_b))
    return float(np.clip(score, 0.0, 1.0))


def is_verified(score: float, threshold: float | None = None) -> bool:
    """
    Apply the similarity threshold to determine verification outcome.

    Args:
        score:     Cosine similarity score in [0.0, 1.0].
        threshold: Override the configured threshold. If None, uses
                   settings.SIMILARITY_THRESHOLD. Useful for testing.

    Returns:
        True if score >= threshold, False otherwise.
    """
    cutoff = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    return score >= cutoff


def evaluate(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """
    Compute similarity and apply threshold in one call.

    Convenience wrapper used by the verify route so it makes a single call
    rather than two sequential ones.

    Args:
        embedding_a: L2-normalised float32 array of shape (512,).
        embedding_b: L2-normalised float32 array of shape (512,).
        threshold:   Optional threshold override.

    Returns:
        Tuple of (verified: bool, score: float).
    """
    score = compute_similarity(embedding_a, embedding_b)
    verified = is_verified(score, threshold)
    return verified, score