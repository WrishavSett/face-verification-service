import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.routes import enroll, verify
from app.core.config import settings
from app.core.exceptions import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    StudentNotFoundError,
)
from app.services.face_service import get_face_app

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown events
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the InsightFace model once at startup so the first request
    does not bear the initialisation cost.
    """
    logger.info("Starting %s", settings.PROJECT_NAME)
    logger.info("Loading InsightFace model '%s'...", settings.INSIGHTFACE_MODEL)

    get_face_app()

    logger.info("InsightFace model loaded. Service ready.")
    logger.info("Similarity threshold: %.2f", settings.SIMILARITY_THRESHOLD)
    logger.info("Photo directory: %s", settings.PHOTO_DIR)

    yield

    logger.info("Shutting down %s", settings.PROJECT_NAME)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "Face verification service for examination centres. "
        "Compares a webcam snapshot against an enrolled reference photo "
        "and returns a verification decision."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(StudentNotFoundError)
async def student_not_found_handler(request: Request, exc: StudentNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "STUDENT_NOT_FOUND", "detail": str(exc)},
    )


@app.exception_handler(NoFaceDetectedError)
async def no_face_handler(request: Request, exc: NoFaceDetectedError):
    return JSONResponse(
        status_code=422,
        content={"error": "NO_FACE_DETECTED", "detail": str(exc)},
    )


@app.exception_handler(MultipleFacesDetectedError)
async def multiple_faces_handler(request: Request, exc: MultipleFacesDetectedError):
    return JSONResponse(
        status_code=422,
        content={"error": "MULTIPLE_FACES_DETECTED", "detail": str(exc)},
    )


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError):
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_IMAGE", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_SERVER_ERROR", "detail": "An unexpected error occurred"},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(
    verify.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Verification"],
)

app.include_router(
    enroll.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Enrollment"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Service health check")
async def health():
    return {
        "status": "ok",
        "model": settings.INSIGHTFACE_MODEL,
        "threshold": settings.SIMILARITY_THRESHOLD,
    }