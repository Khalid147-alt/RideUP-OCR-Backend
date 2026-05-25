"""End-to-end and unit tests for the RideUP OCR backend.

All Gemini calls are mocked. No live network access is required.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import extractor
from extractor import calculate_confidence
from main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_png(width: int = 600, height: int = 600, colour: str = "black") -> bytes:
    """Return PNG bytes of the given size, large enough to pass quality checks.

    A solid-colour PNG compresses to a few hundred bytes, which would trip the
    minimum-file-size quality check. We sprinkle random pixels so the encoded
    file is comfortably above 10 KB.
    """
    import os

    img = Image.new("RGB", (width, height), color=colour)
    # Inject random bytes as raw pixel noise so the PNG is large after compression.
    noise = os.urandom(width * height * 3)
    noise_img = Image.frombytes("RGB", (width, height), noise)
    buf = io.BytesIO()
    noise_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def good_png() -> bytes:
    return _make_png()


@pytest.fixture
def good_png_b64(good_png: bytes) -> str:
    return base64.b64encode(good_png).decode("ascii")


@pytest.fixture
def fake_extraction_payload() -> dict[str, Any]:
    return {
        "pay": 8.42,
        "currency": "GBP",
        "miles": 3.1,
        "minutes": 24,
        "orders": 2,
        "platform": "uber_eats",
        "confidence": "high",
        "notes": "Two-stop batch.",
        "raw_text": "£8.42 · 24 min (3.1 mi) total · 2 deliveries",
    }


@pytest.fixture
def patched_vision(fake_extraction_payload: dict[str, Any]):
    """Patch the internal ``_call_vision_model`` to return canned JSON."""
    json_text = json.dumps(fake_extraction_payload)
    with patch.object(extractor, "_call_vision_model", return_value=json_text) as m:
        yield m


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health and meta
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """``GET /health`` returns 200 with the expected shape."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert body["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """``GET /`` returns the API index."""
    response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "rideup-ocr-backend"
    assert "endpoints" in body
    assert any("extract" in k for k in body["endpoints"])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_file_type_returns_400(client: AsyncClient) -> None:
    """Uploading a PDF is rejected with 400."""
    files = {"image": ("trip.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    response = await client.post("/extract", files=files)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_request"
    assert "JPG" in body["detail"] or "PNG" in body["detail"]


@pytest.mark.asyncio
async def test_file_too_large_returns_400(client: AsyncClient) -> None:
    """Uploads above ``MAX_FILE_SIZE_MB`` are rejected with 400."""
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
    files = {"image": ("big.png", oversized, "image/png")}
    response = await client.post("/extract", files=files)
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_empty_payload_returns_400(client: AsyncClient) -> None:
    """An empty image upload is rejected."""
    files = {"image": ("empty.png", b"", "image/png")}
    response = await client.post("/extract", files=files)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Happy path — mocked Gemini
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_image_upload(
    client: AsyncClient,
    good_png: bytes,
    patched_vision,
    fake_extraction_payload: dict[str, Any],
) -> None:
    """A well-formed PNG returns an ``ExtractionResult`` with all fields."""
    files = {"image": ("trip.png", good_png, "image/png")}
    response = await client.post("/extract", files=files)
    assert response.status_code == 200, response.text

    body = response.json()
    for key in (
        "pay",
        "currency",
        "miles",
        "minutes",
        "orders",
        "platform",
        "confidence",
        "notes",
        "raw_text",
    ):
        assert key in body, f"Missing key {key} in {body}"

    assert body["pay"] == fake_extraction_payload["pay"]
    assert body["platform"] == "uber_eats"
    assert body["confidence"] == "high"
    assert response.headers.get("x-confidence") == "high"


@pytest.mark.asyncio
async def test_base64_endpoint(
    client: AsyncClient,
    good_png_b64: str,
    patched_vision,
) -> None:
    """``POST /extract/base64`` accepts a base64 payload and returns a result."""
    response = await client.post(
        "/extract/base64",
        json={"image": good_png_b64, "hint": "uber_eats"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["platform"] == "uber_eats"
    assert body["confidence"] == "high"


@pytest.mark.asyncio
async def test_base64_endpoint_strips_data_url_prefix(
    client: AsyncClient,
    good_png_b64: str,
    patched_vision,
) -> None:
    """Data-URL prefixes on base64 strings are tolerated."""
    payload = {"image": f"data:image/png;base64,{good_png_b64}"}
    response = await client.post("/extract/base64", json=payload)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_base64_endpoint_rejects_invalid_b64(client: AsyncClient) -> None:
    """Malformed base64 produces a 400."""
    response = await client.post(
        "/extract/base64", json={"image": "!!!not-base64!!!"}
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Retry path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_parse_retry_then_success(
    client: AsyncClient,
    good_png: bytes,
    fake_extraction_payload: dict[str, Any],
) -> None:
    """A first non-JSON response triggers exactly one retry."""
    responses = iter([
        "Sure thing! ```json\nnot really json\n```",
        json.dumps(fake_extraction_payload),
    ])

    def _fake_call(*_args, **_kwargs) -> str:
        return next(responses)

    with patch.object(extractor, "_call_vision_model", side_effect=_fake_call):
        files = {"image": ("trip.png", good_png, "image/png")}
        response = await client.post("/extract", files=files)

    assert response.status_code == 200
    assert response.json()["pay"] == fake_extraction_payload["pay"]


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------


def test_confidence_high_when_all_core_fields_present() -> None:
    result = {"pay": 5.0, "miles": 1.2, "minutes": 10, "orders": 1}
    assert calculate_confidence(result) == "high"


def test_confidence_medium_when_three_of_four_present() -> None:
    result = {"pay": 5.0, "miles": 1.2, "minutes": 10, "orders": None}
    assert calculate_confidence(result) == "medium"


def test_confidence_low_when_fewer_than_three() -> None:
    result = {"pay": 5.0, "miles": None, "minutes": None, "orders": None}
    assert calculate_confidence(result) == "low"


def test_confidence_respects_model_low_self_report() -> None:
    """All four present but model self-reported low → medium (downgrade)."""
    result = {
        "pay": 5.0,
        "miles": 1.2,
        "minutes": 10,
        "orders": 1,
        "confidence": "low",
    }
    assert calculate_confidence(result) == "medium"


# ---------------------------------------------------------------------------
# Image quality
# ---------------------------------------------------------------------------


def test_detect_image_quality_good(good_png: bytes) -> None:
    assert extractor.detect_image_quality(good_png) == "good"


def test_detect_image_quality_poor_when_tiny_bytes() -> None:
    assert extractor.detect_image_quality(b"\x89PNG\r\n\x1a\n") == "poor"


def test_detect_image_quality_poor_when_low_resolution() -> None:
    tiny = _make_png(width=100, height=100)
    assert extractor.detect_image_quality(tiny) == "poor"
