"""Core extraction logic: image preprocessing, Gemini Vision calls, validation.

This module is deliberately framework-agnostic — it knows nothing about HTTP
status codes or FastAPI. It exposes a small, testable surface:

- ``encode_image``           — bytes → base64 string
- ``detect_image_quality``   — quick sanity check before calling the API
- ``calculate_confidence``   — server-side confidence override
- ``sanitize_model_output``  — strip fences/whitespace, isolate JSON body
- ``extract_from_image``     — full pipeline, returns ``ExtractionResult``

Resilience:

- API calls are wrapped in exponential backoff (1s → 2s → 4s, max 3 retries)
  that targets transient upstream failures (429 / 502 / 503 / timeouts).
- Model output is aggressively sanitized before parsing. If JSON parsing
  still fails after the configured number of attempts, a low-confidence
  result is returned with an explanatory ``notes`` entry rather than raising.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import re
import time
from typing import Any, Callable, Literal

import google.generativeai as genai
from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPIError,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
)
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

# Backoff: max attempts and base delays (seconds). The Nth retry sleeps
# ``_BACKOFF_DELAYS[N-1]`` plus a small jitter.
_MAX_API_ATTEMPTS = 3
_BACKOFF_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)
_BACKOFF_JITTER_SECONDS = 0.25

# Transient errors worth retrying. ``GoogleAPIError`` subclasses cover the
# 429 / 502 / 503 / deadline-exceeded surface.
_RETRYABLE_API_ERRORS: tuple[type[Exception], ...] = (
    TooManyRequests,        # 429
    ResourceExhausted,      # 429 quota
    InternalServerError,    # 500
    ServiceUnavailable,     # 503
    DeadlineExceeded,       # 504 / timeouts
)


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
# JSON sanitization
# ---------------------------------------------------------------------------

# Matches an opening ```...``` fence, optionally with a language tag, anywhere
# at the start of the text. We strip the whole fenced wrapper in one pass.
_FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_+\-]*\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?\s*```\s*$")


def _strip_json_fences(text: str) -> str:
    """Remove ``` and ```json fences (open + close) from a text block."""
    stripped = text.strip()
    stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
    stripped = _FENCE_CLOSE_RE.sub("", stripped, count=1)
    return stripped.strip()


def sanitize_model_output(raw: str) -> str:
    """Aggressively clean a model response so it parses as JSON.

    Steps applied in order:

    1. Strip leading/trailing whitespace.
    2. Strip ```/```json markdown fences (both open and close).
    3. Strip any text before the first ``{`` and after the last ``}``.
    4. If the JSON is missing a final ``}`` (truncated output), append one.

    Args:
        raw: Raw text from the vision model.

    Returns:
        A best-effort JSON-looking string. The caller still has to call
        ``json.loads`` and handle failure.
    """
    if not raw:
        return ""

    cleaned = _strip_json_fences(raw)
    if not cleaned:
        return ""

    # Trim anything before the first '{' / after the last '}'. This handles
    # leading prose like "Sure! Here's the data:" that some models emit.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1:
        return ""
    if end == -1 or end < start:
        # Open brace but no close — probably truncated. Take from start to end
        # of string and append a closing brace.
        cleaned = cleaned[start:].rstrip() + "}"
    else:
        cleaned = cleaned[start : end + 1]

    return cleaned.strip()


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse the model output as JSON, raising ``ValueError`` on failure.

    Sanitization is applied unconditionally. The exception message preserves
    the underlying ``json.JSONDecodeError`` detail for log analysis.
    """
    cleaned = sanitize_model_output(raw)
    if not cleaned:
        raise ValueError("Empty model response.")
    if not cleaned.startswith("{"):
        raise ValueError("No JSON object found in model response.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode failed: {exc.msg} at pos {exc.pos}") from exc


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


def _sleep_backoff(attempt_index: int) -> None:
    """Sleep with the backoff delay for the given (1-based) retry attempt.

    The first retry waits ~1s, the second ~2s, the third ~4s. A small random
    jitter is added so concurrent callers do not all hammer the API at the
    same instant.
    """
    base = _BACKOFF_DELAYS[min(attempt_index - 1, len(_BACKOFF_DELAYS) - 1)]
    jitter = random.uniform(0.0, _BACKOFF_JITTER_SECONDS)
    time.sleep(base + jitter)


def _with_api_retries(
    operation_name: str,
    func: Callable[[], str],
) -> str:
    """Run ``func`` with exponential backoff on transient Gemini failures.

    Args:
        operation_name: Short label for logging (e.g. "extract", "retry").
        func: Zero-arg callable that performs a single Gemini API call and
            returns its text response.

    Returns:
        The successful response text.

    Raises:
        GoogleAPIError: re-raised if the final attempt also fails.
    """
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_API_ATTEMPTS + 1):
        try:
            return func()
        except _RETRYABLE_API_ERRORS as exc:
            last_exc = exc
            if attempt >= _MAX_API_ATTEMPTS:
                logger.error(
                    "Gemini API call '%s' failed after %d attempts: %s",
                    operation_name,
                    attempt,
                    exc,
                )
                raise
            delay = _BACKOFF_DELAYS[min(attempt - 1, len(_BACKOFF_DELAYS) - 1)]
            logger.warning(
                "Gemini API call '%s' attempt %d/%d failed with %s; "
                "retrying in ~%.1fs",
                operation_name,
                attempt,
                _MAX_API_ATTEMPTS,
                type(exc).__name__,
                delay,
            )
            _sleep_backoff(attempt)
        except GoogleAPIError as exc:
            # Non-retryable Google API error (4xx other than 429) — bail out.
            logger.error(
                "Gemini API call '%s' failed with non-retryable error: %s",
                operation_name,
                exc,
            )
            raise

    # Defensive — should be unreachable.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unreachable: retry loop exited without result.")


def _call_vision_model(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
) -> str:
    """Send the prompt + image to Gemini and return the raw text response.

    This is the *single-attempt* call. Retries are layered on by
    ``_with_api_retries``.

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
    try:
        return response.text or ""
    except (ValueError, AttributeError):
        # ``response.text`` raises ValueError if there are no parts.
        return ""


# ---------------------------------------------------------------------------
# Platform routing
# ---------------------------------------------------------------------------


def _select_prompt(image_bytes: bytes, hint: str | None) -> str:
    """Choose the best prompt variant for this image.

    Routing priority:

    1. An explicit ``deliveroo`` hint always wins — the caller knows best.
    2. If the image looks like Deliveroo by colour signature (teal-dominant),
       use the Deliveroo prompt.
    3. Otherwise, use the master prompt (optionally augmented with the hint).
    """
    hint_clean = (hint or "").strip().lower()
    if hint_clean == "deliveroo":
        logger.info("Routing to Deliveroo prompt (explicit hint).")
        return build_extraction_prompt("deliveroo")

    if _looks_like_deliveroo(image_bytes):
        logger.info("Routing to Deliveroo prompt (teal colour signature detected).")
        return build_extraction_prompt("deliveroo")

    return build_extraction_prompt(hint_clean or None)


def _looks_like_deliveroo(image_bytes: bytes) -> bool:
    """Cheap colour-signature check for Deliveroo's teal/cyan UI.

    Returns True if a meaningful proportion of pixels fall in the teal/cyan
    hue range. The check is intentionally conservative — false positives are
    harmless (the Deliveroo prompt also self-classifies platform), but false
    negatives just fall back to the master prompt.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            small = img.convert("RGB").resize((64, 64))
    except (UnidentifiedImageError, OSError, ValueError):
        return False

    teal_pixels = 0
    total = 0
    for r, g, b in small.getdata():
        total += 1
        # Teal/cyan: green and blue both substantially higher than red.
        if g > 100 and b > 100 and g + b > 2 * r + 60 and abs(g - b) < 80:
            teal_pixels += 1

    if total == 0:
        return False
    ratio = teal_pixels / total
    if ratio >= 0.06:  # ~6% of the image — generous but specific to teal UIs
        logger.debug("Teal pixel ratio: %.3f (Deliveroo signature)", ratio)
        return True
    return False


# ---------------------------------------------------------------------------
# Result coercion
# ---------------------------------------------------------------------------


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


def _low_confidence_failure(
    raw_response: str,
    errors: list[str],
) -> ExtractionResult:
    """Return a placeholder result describing why parsing failed."""
    notes = "Model response could not be parsed as JSON after all retries. "
    notes += " ".join(errors)
    return ExtractionResult(
        pay=None,
        currency="unknown",
        miles=None,
        minutes=None,
        orders=None,
        platform="unknown",
        confidence="low",
        notes=notes.strip(),
        raw_text=raw_response[:1000],
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
    2. Selects the best prompt variant (master vs Deliveroo-specific).
    3. Calls Gemini Vision with exponential backoff retries on 429/502/503.
    4. Sanitizes the response and parses it as JSON. On failure, retries
       once with a stricter prompt (also with backoff).
    5. If JSON parsing still fails, returns a low-confidence placeholder
       rather than raising.

    Args:
        image_bytes: Raw image bytes (already validated for size/type by the
            HTTP layer).
        hint: Optional platform hint from the caller.

    Returns:
        A fully-populated ``ExtractionResult``.

    Raises:
        RuntimeError: when the Gemini API key is missing.
        GoogleAPIError: when the Gemini API call fails irrecoverably (after
            exhausting retries on transient errors).
    """
    quality = detect_image_quality(image_bytes)
    mime_type = _detect_mime_type(image_bytes)
    prompt = _select_prompt(image_bytes, hint)

    parse_errors: list[str] = []
    raw_response: str = ""
    parsed: dict[str, Any] | None = None

    # --- First attempt: master/Deliveroo prompt, with backoff retries ----
    raw_response = _with_api_retries(
        "extract",
        lambda: _call_vision_model(prompt, image_bytes, mime_type),
    )
    try:
        parsed = _parse_json_response(raw_response)
    except ValueError as exc:
        parse_errors.append(f"Initial parse: {exc}.")
        logger.warning(
            "Initial JSON parse failed (%s); retrying with strict prompt.", exc
        )

    # --- Retry attempt: stricter prompt, with backoff retries ----
    if parsed is None:
        try:
            raw_response = _with_api_retries(
                "extract-retry",
                lambda: _call_vision_model(RETRY_STRICT_PROMPT, image_bytes, mime_type),
            )
        except GoogleAPIError as exc:
            logger.exception("Gemini API error during strict-prompt retry: %s", exc)
            raise

        try:
            parsed = _parse_json_response(raw_response)
        except ValueError as exc:
            parse_errors.append(f"Retry parse: {exc}.")
            logger.error(
                "Retry JSON parse also failed (%s); returning low-confidence result.",
                exc,
            )
            return _low_confidence_failure(raw_response, parse_errors)

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
