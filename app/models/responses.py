from pydantic import BaseModel, Field


class VerifyResponse(BaseModel):
    """Response schema for the POST /verify endpoint."""

    verified: bool = Field(
        ...,
        description="True if the student's face matches the reference image above threshold.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between the two face embeddings.",
        examples=[0.7842],
    )
    student_id: str = Field(
        ...,
        description="Echo of the student_id from the request, for audit traceability.",
    )


class EnrollResponse(BaseModel):
    """Response schema for the POST /enroll endpoint."""

    enrolled: bool = Field(
        ...,
        description="True if the student's reference photo was successfully stored.",
    )
    student_id: str = Field(
        ...,
        description="Echo of the student_id from the request.",
    )
    message: str = Field(
        ...,
        description="Human-readable confirmation of the enrollment outcome.",
        examples=["Reference photo enrolled successfully for student STU-2024-00123"],
    )


class ErrorResponse(BaseModel):
    """Uniform error response schema for all 4xx responses."""

    error: str = Field(
        ...,
        description="Machine-readable error code.",
        examples=["NO_FACE_DETECTED", "STUDENT_NOT_FOUND", "INVALID_IMAGE"],
    )
    detail: str = Field(
        ...,
        description="Human-readable explanation of the error.",
        examples=["No face detected in snapshot"],
    )