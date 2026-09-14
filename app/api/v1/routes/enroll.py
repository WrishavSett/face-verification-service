import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.exceptions import InvalidImageError, MultipleFacesDetectedError, NoFaceDetectedError
from app.models.responses import EnrollResponse, ErrorResponse
from app.services import face_service, storage_service
from app.utils.image_utils import decode_upload_to_bgr

logger = logging.getLogger(__name__)

router = APIRouter()

_CONTENT_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        422: {"model": ErrorResponse, "description": "No face detected or multiple faces detected"},
    },
    summary="Enroll a student's reference photo",
)
async def enroll(
    student_id: str = Form(..., description="University identification number"),
    image: UploadFile = File(..., description="Passport photo of the student"),
) -> EnrollResponse:
    """
    Enroll a student's reference photo into local storage.

    Pipeline:
        1. Decode and validate the uploaded photo.
        2. Run face detection to confirm exactly one face is present.
        3. Persist the image to local storage keyed by student_id.
        4. Return confirmation.

    If a reference photo already exists for the student_id, it is
    replaced — this handles the hall ticket photo update case.
    """
    # Step 1 — Decode and validate the uploaded photo
    try:
        image_bgr = await decode_upload_to_bgr(image)
    except InvalidImageError as e:
        logger.warning("Invalid image upload during enroll for student '%s': %s", student_id, e)
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_IMAGE", "detail": str(e)},
        )

    # Step 2 — Confirm exactly one face is detectable
    try:
        face_service.get_embedding(image_bgr, source="reference image")
    except NoFaceDetectedError as e:
        logger.warning("No face detected in enroll image for student '%s'", student_id)
        raise HTTPException(
            status_code=422,
            detail={"error": "NO_FACE_DETECTED", "detail": str(e)},
        )
    except MultipleFacesDetectedError as e:
        logger.warning(
            "Multiple faces detected in enroll image for student '%s': %d faces",
            student_id,
            e.count,
        )
        raise HTTPException(
            status_code=422,
            detail={"error": "MULTIPLE_FACES_DETECTED", "detail": str(e)},
        )

    # Step 3 — Persist the image
    extension = _resolve_extension(image.content_type)

    await image.seek(0)
    contents = await image.read()

    saved_path = storage_service.save_reference_image(student_id, contents, extension)

    logger.info("Enrolled reference photo for student '%s' at '%s'", student_id, saved_path)

    # Step 4 — Return confirmation
    return EnrollResponse(
        enrolled=True,
        student_id=student_id,
        message=f"Reference photo enrolled successfully for student {student_id}",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_extension(content_type: str | None) -> str:
    """
    Map a MIME type to a file extension.

    Falls back to '.jpg' for any unrecognised type — image_utils will have
    already rejected genuinely unsupported types before this point.
    """
    return _CONTENT_TYPE_TO_EXTENSION.get(content_type or "", ".jpg")