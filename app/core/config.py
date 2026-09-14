from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Face detection
    INSIGHTFACE_MODEL: str = "buffalo_l"
    DET_SIZE: tuple[int, int] = (640, 640)

    # Verification
    SIMILARITY_THRESHOLD: float = 0.60

    # Storage
    PHOTO_DIR: Path = Path("storage/student_photos")

    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Face Verification Service"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()