"""Core extraction logic: image preprocessing, Gemini Vision calls, validation.

This module is deliberately framework-agnostic — it knows nothing about HTTP
status codes or FastAPI. It exposes a small, testable surface:

- ``encode_image``           — bytes → base64 string
- ``detect_image_quality``   — quick sanity check before calling the API
- ``calculate_confidence``   — server-side confidence override
- ``extract_from_image``     — full pipeline, returns ``ExtractionResult``
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Literal

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from PIL import Image, UnidentifiedImageError

from config import settings
from models import ExtractionResult
from prompts import RETRY_STRICT_PROMPT, build_extraction_prompt

logger = logging.getLogger(__name__)

# Core fields used to compute confidence. Must remain non-empty.
_CORE_FIELDS: tuple[str, ...] = ("pay", "miles", "minutes", "orders")

# Image quality thresholds.
_MIN_DIMENSION_PX = 300
_MIN_FILE_SIZE_BYTES = 10 * 1024  # 10 KB

# Gemini call parameters.
_MAX_OUTPUT_TOKENS = 500
_TEMPERATURE = 0.0


ImageQuality = Literal["good", "poor"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def encode_image(image_bytes: bytes) -> str:
    """Return the base64-encoded representation of the given image bytes.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Base64-encoded ASCII string (no ``data:`` URL prefix).
    """
    return base64.b64encode(image_bytes).decode("ascii")


def detect_image_quality(image_bytes: bytes) -> ImageQuality:
    """Heuristically classify image quality as ``"good"`` or ``"poor"``.

    The check is intentionally cheap — it rejects obviously unusable inputs
    (too small in bytes, or too low in pixel dimensions) before we spend an
    API call on them.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        ``"good"`` if the image passes minimum size and resolution checks,
        otherwise ``"poor"``.
    """
    if len(image_bytes) < _MIN_FILE_SIZE_BYTES:
        return "poor"
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError):
        return "poor"
    if width < _MIN_DIMENSION_PX or height < _MIN_DIMENSION_PX:
        return "poor"
    return "good"


def _detect_mime_type(image_bytes: bytes) -> str:
    """Sniff the MIME type from the image header. Defaults to ``image/png``."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = (img.format or "PNG").lower()
    except (UnidentifiedImageError, OSError, ValueError):
        return "image/png"
    if fmt in {"jpeg", "jpg"}:
        return "image/jpeg"
    if fmt == "webp":
        return "image/webp"
    return "image/png"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def calculate_confidence(result: dict[str, Any]) -> Literal["high", "medium", "low"]:
    """Compute a confidence label from the parsed extraction dictionary.

    The rules are deliberately stricter than whatever the model self-reports:

    - **high**   — all 4 core fields are non-null AND the model did not flag
      approximation in its own ``confidence`` claim.
    - **medium** — 3 of 4 core fields are non-null, OR all 4 are present but
      the model reported ``"medium"`` or ``"low"``.
    - **low**    — fewer than 3 core fields, or any image-quality concern.

    Args:
        result: Parsed JSON response from the vision model.

    Returns:
        One of ``"high"``, ``"medium"``, ``"low"``.
    """
    found = sum(1 for field in _CORE_FIELDS if result.get(field) is not None)
    model_claim = str(result.get("confidence", "")).lower()

    if found == len(_CORE_FIELDS):
        if model_claim in {"low", "medium"}:
            return "medium"
        return "high"
    if found == 3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

_client_configured = False


def _ensure_client_configured() -> None:
    """Configure the Gemini client lazily, exactly once.

    Raises:
        RuntimeError: if ``GEMINI_API_KEY`` is not set.
    """
    global _client_configured
    if _client_configured:
        return
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Set it in the environment "
            "before calling the extraction pipeline."
        )
    genai.configure(api_key=settings.gemini_api_key)
    _client_configured = True


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model emitted any."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (``` or ```json) and trailing fence.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _call_vision_model(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
) -> str:
    """Send the prompt + image to Gemini and return the raw text response.

    Args:
        prompt: The full user-side prompt.
        image_bytes: Raw image bytes (Gemini consumes bytes + mime type
            natively — no base64 needed for the SDK).
        mime_type: Detected image MIME type.

    Returns:
        The raw textual response from Gemini.
    """
    _ensure_client_configured()
    model = genai.GenerativeModel(
        model_name=settings.model_name,
        generation_config={
            "temperature": _TEMPERATURE,
            "max_output_tokens": _MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
        },
    )
    response = model.generate_content(
        [
            {"mime_type": mime_type, "data": image_bytes},
            prompt,
        ]
    )
    # ``response.text`` aggregates all candidate text parts. If the model
    # returned no text (e.g. safety block), surface an empty string and let
    # the JSON parser raise downstream.
    try:
        return response.text or ""
    except (ValueError, AttributeError):
        # ``response.text`` raises ValueError if there are no parts.
        return ""


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON object from the model output, raising ``ValueError`` on failure."""
    cleaned = _strip_json_fences(raw)
    if not cleaned:
        raise ValueError("Empty model response.")
    # Some models occasionally emit leading text — try to locate the first {.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _coerce_result(parsed: dict[str, Any], raw_text_fallback: str) -> ExtractionResult:
    """Build an ``ExtractionResult`` from a parsed dict, applying defaults."""
    platform = str(parsed.get("platform") or "unknown").lower().replace(" ", "_")
    if platform not in {
        "uber_eats",
        "deliveroo",
        "stuart",
        "just_eat",
        "rideup",
        "unknown",
    }:
        platform = "unknown"

    currency_raw = str(parsed.get("currency") or "unknown")
    currency = currency_raw.upper() if currency_raw.lower() != "unknown" else "unknown"
    if currency not in {"GBP", "USD", "EUR", "unknown"}:
        currency = "unknown"

    def _maybe_float(v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _maybe_int(v: Any) -> int | None:
        if v is None or v == "":
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    server_confidence = calculate_confidence(parsed)

    return ExtractionResult(
        pay=_maybe_float(parsed.get("pay")),
        currency=currency,  # type: ignore[arg-type]
        miles=_maybe_float(parsed.get("miles")),
        minutes=_maybe_int(parsed.get("minutes")),
        orders=_maybe_int(parsed.get("orders")),
        platform=platform,  # type: ignore[arg-type]
        confidence=server_confidence,
        notes=str(parsed.get("notes") or "").strip(),
        raw_text=str(parsed.get("raw_text") or raw_text_fallback).strip(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_from_image(
    image_bytes: bytes,
    hint: str | None = None,
) -> ExtractionResult:
    """Run the full extraction pipeline against the given image bytes.

    The pipeline:

    1. Detects image quality. Poor-quality images still go to the model but
       the result is flagged in ``notes`` and confidence is capped at "low".
    2. Calls Gemini Vision with the master prompt.
    3. Parses the JSON response. On failure, retries once with a stricter
       prompt.
    4. Coerces the parsed dict into an ``ExtractionResult`` and re-computes
       confidence server-side.

    Args:
        image_bytes: Raw image bytes (already validated for size/type by the
            HTTP layer).
        hint: Optional platform hint from the caller.

    Returns:
        A fully-populated ``ExtractionResult``.

    Raises:
        RuntimeError: when the Gemini API key is missing.
        GoogleAPIError: when the Gemini API call fails irrecoverably.
    """
    quality = detect_image_quality(image_bytes)
    mime_type = _detect_mime_type(image_bytes)
    prompt = build_extraction_prompt(hint)

    raw_response: str = ""
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None

    try:
        raw_response = _call_vision_model(prompt, image_bytes, mime_type)
    except GoogleAPIError as exc:
        logger.exception("Gemini API error during initial call: %s", exc)
        raise

    try:
        parsed = _parse_json_response(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        parse_error = str(exc)
        logger.warning(
            "Initial JSON parse failed (%s); retrying with strict prompt.", exc
        )

    if parsed is None:
        # One retry with a stricter prompt — covers the rare case where the
        # model wrapped JSON in prose despite the instructions.
        try:
            raw_response = _call_vision_model(
                RETRY_STRICT_PROMPT, image_bytes, mime_type
            )
            parsed = _parse_json_response(raw_response)
        except GoogleAPIError as exc:
            logger.exception("Gemini API error during retry: %s", exc)
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Retry JSON parse also failed: %s", exc)
            # Build a low-confidence empty result instead of raising.
            notes_bits = [
                "Model response could not be parsed as JSON after one retry.",
            ]
            if parse_error:
                notes_bits.append(f"First error: {parse_error}.")
            notes_bits.append(f"Retry error: {exc}.")
            return ExtractionResult(
                pay=None,
                currency="unknown",
                miles=None,
                minutes=None,
                orders=None,
                platform="unknown",
                confidence="low",
                notes=" ".join(notes_bits),
                raw_text=raw_response[:1000],
            )

    result = _coerce_result(parsed, raw_text_fallback=raw_response[:1000])

    if quality == "poor":
        quality_note = (
            "Image quality flagged as poor (low resolution or very small "
            "file); confidence reduced."
        )
        result = result.model_copy(
            update={
                "confidence": "low",
                "notes": (result.notes + " " + quality_note).strip(),
            }
        )

    return result
