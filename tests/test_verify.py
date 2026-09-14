import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_jpeg_bytes() -> bytes:
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _post_verify(student_id: str, image_bytes: bytes, content_type: str = "image/jpeg"):
    return client.post(
        "/api/v1/verify",
        data={"student_id": student_id},
        files={"image": ("snapshot.jpg", io.BytesIO(image_bytes), content_type)},
    )


def _unit_vector(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _mock_face(seed: int = 0):
    face = MagicMock()
    face.embedding = _unit_vector(seed)
    return face


# ---------------------------------------------------------------------------
# Success path — verified
# ---------------------------------------------------------------------------

class TestVerifySuccess:
    def test_verified_true_when_same_embedding(self, mocker, tmp_path):
        embedding = _unit_vector(seed=1)
        face = MagicMock()
        face.embedding = embedding

        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [face]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        # Write a dummy reference photo
        ref_path = tmp_path / "STU-100.jpg"
        ref_path.write_bytes(_minimal_jpeg_bytes())

        response = _post_verify("STU-100", _minimal_jpeg_bytes())

        assert response.status_code == 200
        data = response.json()
        assert data["verified"] is True
        assert data["score"] == pytest.approx(1.0, abs=1e-4)

    def test_response_echoes_student_id(self, mocker, tmp_path):
        face = _mock_face(seed=2)
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [face]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        ref_path = tmp_path / "STU-101.jpg"
        ref_path.write_bytes(_minimal_jpeg_bytes())

        response = _post_verify("STU-101", _minimal_jpeg_bytes())

        assert response.json()["student_id"] == "STU-101"

    def test_score_is_between_zero_and_one(self, mocker, tmp_path):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [_mock_face(seed=3)]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        ref_path = tmp_path / "STU-102.jpg"
        ref_path.write_bytes(_minimal_jpeg_bytes())

        response = _post_verify("STU-102", _minimal_jpeg_bytes())

        score = response.json()["score"]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

class TestVerifyFailures:
    def test_student_not_found_returns_404(self, mocker, tmp_path):
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        response = _post_verify("NONEXISTENT-999", _minimal_jpeg_bytes())

        assert response.status_code == 404
        assert response.json()["error"] == "STUDENT_NOT_FOUND"

    def test_invalid_content_type_returns_400(self):
        response = _post_verify("STU-200", b"not-an-image", content_type="text/plain")
        assert response.status_code == 400
        assert response.json()["error"] == "INVALID_IMAGE"

    def test_no_face_in_snapshot_returns_422(self, mocker, tmp_path):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = []
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        ref_path = tmp_path / "STU-201.jpg"
        ref_path.write_bytes(_minimal_jpeg_bytes())

        response = _post_verify("STU-201", _minimal_jpeg_bytes())

        assert response.status_code == 422
        assert response.json()["error"] == "NO_FACE_DETECTED"

    def test_multiple_faces_in_snapshot_returns_422(self, mocker, tmp_path):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [_mock_face(0), _mock_face(1)]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        ref_path = tmp_path / "STU-202.jpg"
        ref_path.write_bytes(_minimal_jpeg_bytes())

        response = _post_verify("STU-202", _minimal_jpeg_bytes())

        assert response.status_code == 422
        assert response.json()["error"] == "MULTIPLE_FACES_DETECTED"

    def test_blank_student_id_returns_422(self):
        response = _post_verify("   ", _minimal_jpeg_bytes())
        assert response.status_code == 422

    def test_verified_false_when_dissimilar_embeddings(self, mocker, tmp_path):
        """Force a low similarity by returning different embeddings for each call."""
        call_count = {"n": 0}

        def side_effect(image):
            face = MagicMock()
            face.embedding = _unit_vector(seed=call_count["n"])
            call_count["n"] += 1
            return [face]

        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.side_effect = side_effect
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)
        mocker.patch("app.core.config.settings.SIMILARITY_THRESHOLD", 0.99)

        ref_path = tmp_path / "STU-203.jpg"
        ref_path.write_bytes(_minimal_jpeg_bytes())

        response = _post_verify("STU-203", _minimal_jpeg_bytes())

        assert response.status_code == 200
        assert response.json()["verified"] is False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"