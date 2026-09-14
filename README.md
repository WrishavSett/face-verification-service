# Face Verification Service

Face verification API for examination centres. Compares a webcam snapshot
against an enrolled reference photo and returns a verification decision.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs available at: `http://localhost:8000/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/enroll` | Enroll a student's reference photo |
| POST | `/api/v1/verify` | Verify a student against their reference photo |
| GET | `/health` | Service health check |

---

## Enroll

```bash
curl -X POST http://localhost:8000/api/v1/enroll \
  -F "student_id=STU-2024-00123" \
  -F "image=@/path/to/passport_photo.jpg"
```

Response:
```json
{
  "enrolled": true,
  "student_id": "STU-2024-00123",
  "message": "Reference photo enrolled successfully for student STU-2024-00123"
}
```

---

## Verify

```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "student_id=STU-2024-00123" \
  -F "image=@/path/to/snapshot.jpg"
```

Response:
```json
{
  "verified": true,
  "score": 0.7842,
  "student_id": "STU-2024-00123"
}
```

---

## Error Responses

All errors follow a uniform schema:

```json
{
  "error": "ERROR_CODE",
  "detail": "Human-readable explanation"
}
```

| Status | Error Code | Cause |
|--------|-----------|-------|
| 400 | `INVALID_IMAGE` | Unsupported format or corrupt file |
| 404 | `STUDENT_NOT_FOUND` | No reference photo for given student_id |
| 422 | `NO_FACE_DETECTED` | Face detection failed on snapshot |
| 422 | `MULTIPLE_FACES_DETECTED` | More than one face found in image |
| 500 | `INTERNAL_SERVER_ERROR` | Unexpected server-side failure |

---

## Configuration

All values are set via `.env`. See `.env.example` for full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `INSIGHTFACE_MODEL` | `buffalo_l` | Model pack for detection and embedding |
| `SIMILARITY_THRESHOLD` | `0.60` | Minimum score to return `verified: true` |
| `PHOTO_DIR` | `storage/student_photos` | Reference photo directory |
| `DET_SIZE` | `(640, 640)` | Face detection input resolution |