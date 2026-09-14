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
    """Return a minimal valid JPEG byte sequence for upload tests."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _post_enroll(student_id: str, image_bytes: bytes, content_type: str = "image/jpeg"):
    return client.post(
        "/api/v1/enroll",
        data={"student_id": student_id},
        files={"image": ("photo.jpg", io.BytesIO(image_bytes), content_type)},
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestEnrollSuccess:
    def test_returns_200_with_enrolled_true(self, mocker, tmp_path):
        mocker.patch("app.services.face_service.get_face_app").return_value.get.return_value = [
            _mock_face()
        ]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        response = _post_enroll("STU-001", _minimal_jpeg_bytes())

        assert response.status_code == 200
        assert response.json()["enrolled"] is True

    def test_response_echoes_student_id(self, mocker, tmp_path):
        mocker.patch("app.services.face_service.get_face_app").return_value.get.return_value = [
            _mock_face()
        ]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        response = _post_enroll("STU-002", _minimal_jpeg_bytes())

        assert response.json()["student_id"] == "STU-002"

    def test_reference_photo_written_to_disk(self, mocker, tmp_path):
        mocker.patch("app.services.face_service.get_face_app").return_value.get.return_value = [
            _mock_face()
        ]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        _post_enroll("STU-003", _minimal_jpeg_bytes())

        saved_files = list(tmp_path.glob("STU-003.*"))
        assert len(saved_files) == 1

    def test_re_enroll_overwrites_existing(self, mocker, tmp_path):
        mocker.patch("app.services.face_service.get_face_app").return_value.get.return_value = [
            _mock_face()
        ]
        mocker.patch("app.core.config.settings.PHOTO_DIR", tmp_path)

        _post_enroll("STU-004", _minimal_jpeg_bytes())
        _post_enroll("STU-004", _minimal_jpeg_bytes())

        saved_files = list(tmp_path.glob("STU-004.*"))
        assert len(saved_files) == 1


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

class TestEnrollFailures:
    def test_invalid_content_type_returns_400(self, mocker):
        mocker.patch("app.services.face_service.get_face_app").return_value.get.return_value = [
            _mock_face()
        ]
        response = _post_enroll("STU-005", b"not-an-image", content_type="text/plain")
        assert response.status_code == 400

    def test_no_face_detected_returns_422(self, mocker):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = []

        response = _post_enroll("STU-006", _minimal_jpeg_bytes())

        assert response.status_code == 422
        assert response.json()["error"] == "NO_FACE_DETECTED"

    def test_multiple_faces_returns_422(self, mocker):
        mocker.patch(
            "app.services.face_service.get_face_app"
        ).return_value.get.return_value = [_mock_face(), _mock_face()]

        response = _post_enroll("STU-007", _minimal_jpeg_bytes())

        assert response.status_code == 422
        assert response.json()["error"] == "MULTIPLE_FACES_DETECTED"

    def test_blank_student_id_returns_422(self, mocker):
        response = _post_enroll("   ", _minimal_jpeg_bytes())
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_face():
    face = MagicMock()
    face.embedding = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    return face