from pydantic import BaseModel, Field, field_validator


class VerifyRequest(BaseModel):
    """
    Schema for the /verify endpoint.

    Note: image is handled separately as a FastAPI UploadFile in the route.
    This schema validates the non-file fields received via multipart form.
    """
    student_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique university identification number of the student.",
        examples=["STU-2024-00123"],
    )

    @field_validator("student_id")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("student_id must not be blank")
        return stripped


class EnrollRequest(BaseModel):
    """
    Schema for the /enroll endpoint.

    Note: image is handled separately as a FastAPI UploadFile in the route.
    This schema validates the non-file fields received via multipart form.
    """
    student_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique university identification number of the student.",
        examples=["STU-2024-00123"],
    )

    @field_validator("student_id")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("student_id must not be blank")
        return stripped