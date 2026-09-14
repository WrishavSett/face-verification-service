import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.exceptions import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    StudentNotFoundError,
)
from app.models.responses import ErrorResponse, VerifyResponse
from app.services import face_service, similarity_service, storage_service
from app.utils.image_utils import decode_upload_to_bgr

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        404: {"model": ErrorResponse, "description": "Student not found"},
        422: {"model": ErrorResponse, "description": "No face detected or multiple faces detected"},
    },
    summary="Verify a student's identity against their reference photo",
)
async def verify(
    student_id: str = Form(..., description="University identification number"),
    image: UploadFile = File(..., description="Webcam snapshot of the student's face"),
) -> VerifyResponse:
    """
    Verify a student's identity by comparing a webcam snapshot against
    their enrolled reference photo.

    Pipeline:
        1. Decode and validate the uploaded snapshot.
        2. Fetch the reference image from local storage using student_id.
        3. Run face detection and embedding generation on both images.
        4. Compute cosine similarity between the two embeddings.
        5. Return verified=True if score >= threshold, else False.
    """
    # Step 1 — Decode snapshot
    try:
        snapshot_bgr = await decode_upload_to_bgr(image)
    except InvalidImageError as e:
        logger.warning("Invalid snapshot upload for student '%s': %s", student_id, e)
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_IMAGE", "detail": str(e)},
        )

    # Step 2 — Fetch reference image
    try:
        reference_bgr = storage_service.get_reference_image(student_id)
    except StudentNotFoundError as e:
        logger.warning("Verification attempted for unknown student: '%s'", student_id)
        raise HTTPException(
            status_code=404,
            detail={"error": "STUDENT_NOT_FOUND", "detail": str(e)},
        )

    # Step 3 — Generate embeddings for both images
    try:
        snapshot_embedding = face_service.get_embedding(snapshot_bgr, source="snapshot")
    except NoFaceDetectedError as e:
        logger.warning("No face in snapshot for student '%s'", student_id)
        raise HTTPException(
            status_code=422,
            detail={"error": "NO_FACE_DETECTED", "detail": str(e)},
        )
    except MultipleFacesDetectedError as e:
        logger.warning("Multiple faces in snapshot for student '%s': %d faces", student_id, e.count)
        raise HTTPException(
            status_code=422,
            detail={"error": "MULTIPLE_FACES_DETECTED", "detail": str(e)},
        )

    try:
        reference_embedding = face_service.get_embedding(reference_bgr, source="reference image")
    except NoFaceDetectedError as e:
        logger.error("No face in reference image for student '%s' — data integrity issue", student_id)
        raise HTTPException(
            status_code=422,
            detail={"error": "NO_FACE_IN_REFERENCE", "detail": str(e)},
        )
    except MultipleFacesDetectedError as e:
        logger.error(
            "Multiple faces in reference image for student '%s' — data integrity issue", student_id
        )
        raise HTTPException(
            status_code=422,
            detail={"error": "MULTIPLE_FACES_IN_REFERENCE", "detail": str(e)},
        )

    # Step 4 — Compute similarity and apply threshold
    verified, score = similarity_service.evaluate(snapshot_embedding, reference_embedding)

    # Step 5 — Log outcome and return
    logger.info(
        "Verification for student '%s': verified=%s score=%.4f",
        student_id,
        verified,
        score,
    )

    return VerifyResponse(verified=verified, score=score, student_id=student_id)