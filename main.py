"""FastAPI application entry point for the RideUP OCR backend."""

from __future__ import annotations

import base64
import binascii
import logging
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.api_core.exceptions import GoogleAPIError

from config import settings
from extractor import enrich_deliveroo_postcodes, extract_from_image
from models import (
    Base64ImageRequest,
    ErrorResponse,
    ExtractionResult,
    HealthResponse,
    RootResponse,
)

# ---------------------------------------------------------------------------
# Logging — structured, no print statements anywhere in the codebase.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("rideup_ocr")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION = "1.0.0"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RideUP OCR Backend",
    description=(
        "Production OCR API for UK gig-economy delivery screenshots. "
        "Extracts pay, distance, time, order count, and platform from a single "
        "image using Google Gemini Vision."
    ),
    version=API_VERSION,
)

# CORS — open by default; client controls origins at the edge.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log endpoint, response time, and confidence (when present)."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    confidence = response.headers.get("X-Confidence", "-")
    logger.info(
        "method=%s path=%s status=%d duration_ms=%.1f confidence=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        confidence,
    )
    return response


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _error_response(
    *, status_code: int, error: str, detail: str
) -> JSONResponse:
    """Build a uniform JSON error response."""
    body = ErrorResponse(error=error, detail=detail, status_code=status_code)
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Render ``HTTPException`` instances as the uniform error envelope."""
    return _error_response(
        status_code=exc.status_code,
        error=_status_to_error_name(exc.status_code),
        detail=str(exc.detail) if exc.detail is not None else "",
    )


def _status_to_error_name(code: int) -> str:
    return {
        400: "bad_request",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        500: "server_error",
        502: "bad_gateway",
    }.get(code, "error")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_upload(upload: UploadFile) -> None:
    """Validate the ``UploadFile`` content type and filename extension."""
    content_type = (upload.content_type or "").lower()
    filename = (upload.filename or "").lower()
    ext_ok = any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

    if content_type and content_type not in ALLOWED_CONTENT_TYPES and not ext_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type or filename}'. "
                "Allowed: JPG, PNG, WEBP."
            ),
        )
    if not content_type and not ext_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to determine file type. Allowed: JPG, PNG, WEBP.",
        )


def _validate_size(num_bytes: int) -> None:
    """Reject payloads larger than the configured max."""
    if num_bytes > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image too large: {num_bytes / (1024 * 1024):.2f} MB exceeds "
                f"limit of {settings.max_file_size_mb} MB."
            ),
        )
    if num_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload.",
        )


def _run_extraction(image_bytes: bytes) -> ExtractionResult:
    """Invoke the extractor, mapping known errors to safe HTTP responses."""
    try:
        return extract_from_image(image_bytes)
    except RuntimeError as exc:
        # Missing API key etc. — server misconfiguration.
        logger.error("Server misconfiguration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured for extraction.",
        ) from exc
    except GoogleAPIError as exc:
        # Never leak raw upstream errors to the client.
        logger.exception("Gemini extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream vision service is currently unavailable.",
        ) from exc
    except Exception as exc:  # pragma: no cover — defensive catch-all
        logger.exception("Unexpected extraction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error during extraction.",
        ) from exc


async def _enrich_result(result: ExtractionResult) -> ExtractionResult:
    """Apply Deliveroo V2 postcode/mileage enrichment, never failing the request.

    Enrichment is a best-effort, value-add step (postcodes + estimated miles).
    If anything goes wrong — a bug, an unexpected error — we log it and return
    the original result untouched. The core extraction must never be lost to an
    enrichment failure.
    """
    try:
        return await enrich_deliveroo_postcodes(result)
    except Exception as exc:  # pragma: no cover — defensive catch-all
        logger.warning("Postcode enrichment failed; returning base result: %s", exc)
        return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_model=RootResponse, tags=["meta"])
async def root() -> RootResponse:
    """Return basic API metadata and endpoint index."""
    return RootResponse(
        name="rideup-ocr-backend",
        version=API_VERSION,
        description=(
            "OCR backend for UK gig-economy delivery screenshots, powered "
            "by Google Gemini Vision."
        ),
        endpoints={
            "GET  /": "API metadata",
            "GET  /health": "Health check",
            "POST /extract": "Extract from multipart image upload",
            "POST /extract/base64": "Extract from base64 JSON payload",
            "GET  /docs": "Interactive Swagger UI",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness/health probe."""
    return HealthResponse(status="ok", model=settings.model_name, version=API_VERSION)


@app.post(
    "/extract",
    response_model=ExtractionResult,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    tags=["extraction"],
)
async def extract(
    image: UploadFile = File(..., description="JPG/PNG/WEBP image, max 10 MB."),
) -> JSONResponse:
    """Extract structured data from a delivery-app screenshot upload.

    Platform is auto-detected from the image — there is no ``hint`` field.
    Earlier revisions exposed a caller-supplied hint, but it caused real
    pain in Swagger UI (the literal string "string" being passed through
    as a hint) and added no signal the model couldn't recover from the
    image colour signature + its own visual classification.
    """
    _validate_upload(image)
    image_bytes = await image.read()
    _validate_size(len(image_bytes))

    logger.info(
        "extract upload filename=%s content_type=%s bytes=%d",
        image.filename,
        image.content_type,
        len(image_bytes),
    )

    result = _run_extraction(image_bytes)
    result = await _enrich_result(result)
    return JSONResponse(
        content=result.model_dump(),
        headers={"X-Confidence": result.confidence},
    )


@app.post(
    "/extract/base64",
    response_model=ExtractionResult,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    tags=["extraction"],
)
async def extract_base64(payload: Base64ImageRequest) -> JSONResponse:
    """Extract structured data from a base64-encoded image payload.

    Like ``/extract``, platform is auto-detected — there is no ``hint``
    field. Older clients that still send ``hint`` in the JSON body will
    have the field silently ignored (Pydantic's ``extra="ignore"``).
    """
    try:
        image_bytes = base64.b64decode(payload.image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data.",
        ) from exc

    _validate_size(len(image_bytes))

    logger.info("extract/base64 bytes=%d", len(image_bytes))

    result = _run_extraction(image_bytes)
    result = await _enrich_result(result)
    return JSONResponse(
        content=result.model_dump(),
        headers={"X-Confidence": result.confidence},
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )
