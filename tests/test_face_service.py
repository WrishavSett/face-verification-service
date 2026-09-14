import numpy as np
import pytest

from app.core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_face(embedding: np.ndarray):
    """Return a minimal object that mimics an InsightFace face result."""
    class MockFace:
        pass
    f = MockFace()
    f.embedding = embedding
    return f


def _unit_vector(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------

class TestGetEmbedding:
    def test_returns_normalised_embedding(self, mocker):
        raw = _unit_vector(seed=10) * 5.0  # deliberately not unit length
        mock_face = _make_mock_face(raw)

        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [mock_face]

        from app.services.face_service import get_embedding
        result = get_embedding(np.zeros((480, 640, 3), dtype=np.uint8))

        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_raises_no_face_detected_when_empty(self, mocker):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = []

        from app.services.face_service import get_embedding
        with pytest.raises(NoFaceDetectedError):
            get_embedding(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_raises_multiple_faces_when_more_than_one(self, mocker):
        faces = [_make_mock_face(_unit_vector(i)) for i in range(2)]
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = faces

        from app.services.face_service import get_embedding
        with pytest.raises(MultipleFacesDetectedError) as exc_info:
            get_embedding(np.zeros((480, 640, 3), dtype=np.uint8))

        assert exc_info.value.count == 2

    def test_source_label_appears_in_no_face_error(self, mocker):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = []

        from app.services.face_service import get_embedding
        with pytest.raises(NoFaceDetectedError) as exc_info:
            get_embedding(np.zeros((480, 640, 3), dtype=np.uint8), source="snapshot")

        assert "snapshot" in str(exc_info.value)

    def test_source_label_appears_in_multiple_faces_error(self, mocker):
        faces = [_make_mock_face(_unit_vector(i)) for i in range(3)]
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = faces

        from app.services.face_service import get_embedding
        with pytest.raises(MultipleFacesDetectedError) as exc_info:
            get_embedding(np.zeros((480, 640, 3), dtype=np.uint8), source="reference image")

        assert "reference image" in str(exc_info.value)

    def test_embedding_shape_is_512(self, mocker):
        raw = _unit_vector(seed=7)
        mock_face = _make_mock_face(raw)
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [mock_face]

        from app.services.face_service import get_embedding
        result = get_embedding(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result.shape == (512,)