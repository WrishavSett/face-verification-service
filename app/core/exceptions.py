class StudentNotFoundError(Exception):
    """Raised when no reference photo exists for the given student_id."""
    def __init__(self, student_id: str):
        self.student_id = student_id
        super().__init__(f"No reference photo found for student: {student_id}")


class NoFaceDetectedError(Exception):
    """Raised when InsightFace detects no face in the provided image."""
    def __init__(self, source: str = "image"):
        self.source = source
        super().__init__(f"No face detected in {source}")


class MultipleFacesDetectedError(Exception):
    """Raised when InsightFace detects more than one face in the provided image."""
    def __init__(self, source: str = "image", count: int = 0):
        self.source = source
        self.count = count
        super().__init__(f"Expected 1 face in {source}, found {count}")


class InvalidImageError(Exception):
    """Raised when the uploaded file cannot be decoded as a valid image."""
    def __init__(self, detail: str = "Could not decode image"):
        super().__init__(detail)