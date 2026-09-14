import numpy as np
import pytest

from app.services.similarity_service import compute_similarity, evaluate, is_verified


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def unit_vector(size: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def identical_embeddings():
    v = unit_vector(seed=1)
    return v, v.copy()


@pytest.fixture
def similar_embeddings():
    """Two embeddings with a known high cosine similarity."""
    base = unit_vector(seed=2)
    noise = unit_vector(seed=3) * 0.05
    perturbed = base + noise
    perturbed = perturbed / np.linalg.norm(perturbed)
    return base, perturbed.astype(np.float32)


@pytest.fixture
def dissimilar_embeddings():
    v1 = unit_vector(seed=4)
    v2 = unit_vector(seed=5)
    return v1, v2


# ---------------------------------------------------------------------------
# compute_similarity
# ---------------------------------------------------------------------------

class TestComputeSimilarity:
    def test_identical_vectors_return_one(self, identical_embeddings):
        a, b = identical_embeddings
        score = compute_similarity(a, b)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_score_is_within_bounds(self, dissimilar_embeddings):
        a, b = dissimilar_embeddings
        score = compute_similarity(a, b)
        assert 0.0 <= score <= 1.0

    def test_similar_embeddings_score_high(self, similar_embeddings):
        a, b = similar_embeddings
        score = compute_similarity(a, b)
        assert score > 0.90

    def test_returns_float(self, identical_embeddings):
        a, b = identical_embeddings
        assert isinstance(compute_similarity(a, b), float)

    def test_symmetry(self, dissimilar_embeddings):
        a, b = dissimilar_embeddings
        assert compute_similarity(a, b) == pytest.approx(compute_similarity(b, a), abs=1e-6)


# ---------------------------------------------------------------------------
# is_verified
# ---------------------------------------------------------------------------

class TestIsVerified:
    def test_score_above_threshold_returns_true(self):
        assert is_verified(0.75, threshold=0.60) is True

    def test_score_equal_to_threshold_returns_true(self):
        assert is_verified(0.60, threshold=0.60) is True

    def test_score_below_threshold_returns_false(self):
        assert is_verified(0.45, threshold=0.60) is False

    def test_uses_settings_threshold_when_none(self):
        # Should not raise; outcome depends on configured threshold
        result = is_verified(0.80)
        assert isinstance(result, bool)

    def test_threshold_override(self):
        assert is_verified(0.50, threshold=0.40) is True
        assert is_verified(0.50, threshold=0.60) is False


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_tuple_of_bool_and_float(self, identical_embeddings):
        a, b = identical_embeddings
        verified, score = evaluate(a, b, threshold=0.60)
        assert isinstance(verified, bool)
        assert isinstance(score, float)

    def test_identical_embeddings_verified(self, identical_embeddings):
        a, b = identical_embeddings
        verified, score = evaluate(a, b, threshold=0.60)
        assert verified is True
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_dissimilar_embeddings_not_verified(self, dissimilar_embeddings):
        a, b = dissimilar_embeddings
        # Force a high threshold to ensure rejection
        verified, score = evaluate(a, b, threshold=0.99)
        assert verified is False

    def test_score_consistent_with_compute_similarity(self, similar_embeddings):
        a, b = similar_embeddings
        _, score_from_evaluate = evaluate(a, b, threshold=0.60)
        score_direct = compute_similarity(a, b)
        assert score_from_evaluate == pytest.approx(score_direct, abs=1e-6)