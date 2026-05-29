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

    Deliveroo V2 carve-out
    ----------------------
    The Deliveroo V2 offer-card layout does not display distance or estimated
    time at all — they are genuinely absent from the screen, not unreadable.
    Penalising a Deliveroo extraction for missing miles/minutes would punish
    a correct read of an incomplete layout. So: when ``platform`` is
    ``deliveroo`` AND both ``miles`` and ``minutes`` are null, we score
    confidence on ``pay`` + ``orders`` alone:

      - pay and orders both present   → "high"
      - only pay present              → "medium"
      - neither present               → "low"

    Args:
        result: Parsed JSON response from the vision model.

    Returns:
        One of ``"high"``, ``"medium"``, ``"low"``.
    """
    platform = str(result.get("platform") or "").lower()
    miles = result.get("miles")
    minutes = result.get("minutes")
    model_claim = str(result.get("confidence", "")).lower()

    # Deliveroo V2: distance/time are layout-absent, not unreadable.
    if platform == "deliveroo" and miles is None and minutes is None:
        has_pay = result.get("pay") is not None
        has_orders = result.get("orders") is not None
        if has_pay and has_orders:
            return "medium" if model_claim == "low" else "high"
        if has_pay:
            return "medium"
        return "low"

    found = sum(1 for field in _CORE_FIELDS if result.get(field) is not None)

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

# Matches a bare "json" / "JSON" prefix that some models emit when they have
# been told not to use fences but still want to "label" the output. Examples:
#   json\n{...}
#   JSON: {...}
#   Json -\n{...}
_JSON_LABEL_PREFIX_RE = re.compile(r"^\s*json\s*[:\-]?\s*\n?", re.IGNORECASE)

# Matches numeric literals embedded in a string, e.g. " £6.50 " → "6.50",
# " 2.1 mi" → "2.1", "18 min" → "18". Capture is the bare number.
_FIRST_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_json_fences(text: str) -> str:
    """Remove ``` and ```json fences (open + close) from a text block."""
    stripped = text.strip()
    stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
    stripped = _FENCE_CLOSE_RE.sub("", stripped, count=1)
    return stripped.strip()


def _strip_json_label_prefix(text: str) -> str:
    """Remove a leading 'json' / 'JSON:' label some models add as a header."""
    stripped = text.lstrip()
    # Only strip the label if what follows clearly starts a JSON object —
    # otherwise we might eat real content.
    candidate = _JSON_LABEL_PREFIX_RE.sub("", stripped, count=1)
    if candidate is not stripped and candidate.lstrip().startswith("{"):
        return candidate.lstrip()
    return text


def sanitize_model_output(raw: str) -> str:
    """Aggressively clean a model response so it parses as JSON.

    Steps applied in order:

    1. Strip leading/trailing whitespace.
    2. Strip ```/```json markdown fences (both open and close).
    3. Strip a bare ``json`` / ``JSON:`` label prefix some models emit when
       fences are disabled (e.g. ``"json\\n{...}"``).
    4. Strip any text before the first ``{`` and after the last ``}``.
    5. If the JSON is missing a final ``}`` (truncated output), append one.

    Args:
        raw: Raw text from the vision model.

    Returns:
        A best-effort JSON-looking string. The caller still has to call
        ``json.loads`` and handle failure.
    """
    if not raw:
        return ""

    cleaned = _strip_json_fences(raw)
    cleaned = _strip_json_label_prefix(cleaned)
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
# raw_text defensive validation
# ---------------------------------------------------------------------------
#
# Real stress-test responses revealed Gemini occasionally echoing its own
# JSON output back into the ``raw_text`` field — values like ``{"pay": 11``
# or ``{\n  "pay": 7.22``. This is a fundamentally wrong read: ``raw_text``
# is meant to hold the text the model literally saw in the image, not the
# JSON it is about to emit. We catch and clear it here so contaminated
# values never reach the client.

# A response that opens with one of these tokens is structurally JSON, not
# image text. Whitespace is stripped first, so leading newlines/spaces do
# not save a contaminated value.
_RAW_TEXT_FORBIDDEN_PREFIXES: tuple[str, ...] = ("{", "[")

# A JSON-escaped quoted key — the smoking gun for "model echoed its own
# output". We check both ``\"pay\"`` and the unescaped variant because the
# value might or might not have already been JSON-unescaped at this point.
_RAW_TEXT_JSON_SIGNATURES: tuple[str, ...] = (
    '\\"pay\\"',
    '"pay":',
    '\\"currency\\"',
    '"currency":',
)

# Placeholder written into ``raw_text`` after a contaminated value is
# cleared. Mirrors the brief: never return raw_text starting with { or [,
# and never return an empty string either.
_RAW_TEXT_PLACEHOLDER = "extracted from image"


def validate_raw_text(parsed: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON-shaped contamination from ``parsed["raw_text"]``.

    If the model echoed its own JSON output back into the ``raw_text`` field
    (a documented failure mode), the contaminated value is replaced with a
    neutral placeholder and a warning is logged. The function mutates and
    returns the same dict for chaining.

    A value is considered contaminated if it:
      - is a string, and
      - after stripping whitespace, starts with ``{`` or ``[``, OR
      - contains a JSON-escaped key signature like ``\\"pay\\"``.

    Args:
        parsed: The decoded JSON dict from the model.

    Returns:
        The same dict with a cleaned ``raw_text`` value.
    """
    raw = parsed.get("raw_text", "")
    if not isinstance(raw, str):
        return parsed

    stripped = raw.strip()
    starts_with_json = any(
        stripped.startswith(prefix) for prefix in _RAW_TEXT_FORBIDDEN_PREFIXES
    )
    has_json_signature = any(sig in raw for sig in _RAW_TEXT_JSON_SIGNATURES)

    if starts_with_json or has_json_signature:
        logger.warning(
            "raw_text contained JSON (first 80 chars: %r); "
            "cleared to prevent confusion.",
            raw[:80],
        )
        parsed["raw_text"] = _RAW_TEXT_PLACEHOLDER
        return parsed

    if not stripped:
        # Empty raw_text is also unhelpful — keep the placeholder rule
        # consistent: every response carries something meaningful here.
        parsed["raw_text"] = _RAW_TEXT_PLACEHOLDER

    return parsed


# ---------------------------------------------------------------------------
# Type coercion — handles model output that ignored "emit numbers as numbers"
# ---------------------------------------------------------------------------


def force_numeric(value: Any, *, as_int: bool = False) -> float | int | None:
    """Coerce a possibly-stringified, possibly-unit-suffixed value to number.

    Handles the messy shapes vision models emit when they ignore "emit numbers
    as numbers" instructions:

    - Already a number (int/float)  → cast to the requested type.
    - ``None`` / empty string       → return ``None``.
    - ``"£6.50"``                   → ``6.50``  (currency symbol stripped).
    - ``"6.50"``                    → ``6.50``  (plain string number).
    - ``"2.1 mi"`` / ``"3.4 km"``   → ``2.1`` / ``3.4`` (unit stripped — note,
      km→mi conversion is NOT done here; that is the prompt's job).
    - ``"18 min"`` / ``"18m"``      → ``18``.
    - ``"2 stacked"`` / ``"2 orders"`` → ``2``.
    - ``{"gross": 6.50, "tip": 1.0}`` (nested object) → ``6.50`` (uses the
      first numeric value found among common keys: ``gross``, ``total``,
      ``amount``, ``value``).
    - Anything else                  → ``None``.

    Args:
        value: The raw value from the parsed JSON.
        as_int: When True, return ``int`` instead of ``float``.

    Returns:
        The coerced number, or ``None`` if no number could be recovered.
    """
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int`` — guard against True/False sneaking
        # through as 1/0.
        return None

    if isinstance(value, (int, float)):
        return int(value) if as_int else float(value)

    if isinstance(value, dict):
        # Nested object — pull the most likely number out by key preference.
        for key in ("gross", "total", "amount", "value", "headline", "number"):
            if key in value:
                recovered = force_numeric(value[key], as_int=as_int)
                if recovered is not None:
                    return recovered
        # Fallback: first numeric value in dict.
        for v in value.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v) if as_int else float(v)
        return None

    if isinstance(value, list):
        # First numeric element of a list.
        for item in value:
            recovered = force_numeric(item, as_int=as_int)
            if recovered is not None:
                return recovered
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = _FIRST_NUMBER_RE.search(text)
        if not match:
            return None
        try:
            number = float(match.group(0))
        except (TypeError, ValueError):
            return None
        return int(number) if as_int else number

    return None


def _normalize_extraction_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    """Post-parse cleanup: coerce known-numeric fields and log corrections.

    The Deliveroo prompt insists on raw numbers, but the model occasionally
    still emits strings (``"£6.50"``) or nested objects (``{"gross": 6.5}``)
    for ``pay`` and unit-suffixed strings (``"2.1 mi"``) for the metric
    fields. This function rewrites them in place so downstream coercion sees
    clean Python numbers.

    Corrections are logged at INFO level so we can spot prompt drift in
    production.
    """
    corrections: list[str] = []

    def _coerce(field: str, *, as_int: bool) -> None:
        if field not in parsed:
            return
        original = parsed[field]
        if original is None:
            return
        if isinstance(original, bool):
            parsed[field] = None
            corrections.append(f"{field}: bool→null")
            return
        if isinstance(original, (int, float)):
            if as_int and isinstance(original, float):
                parsed[field] = int(original)
                corrections.append(f"{field}: float→int ({original}→{parsed[field]})")
            return
        coerced = force_numeric(original, as_int=as_int)
        if coerced is None:
            parsed[field] = None
            corrections.append(f"{field}: unrecoverable ({original!r}→null)")
        else:
            parsed[field] = coerced
            corrections.append(f"{field}: {original!r}→{coerced}")

    _coerce("pay", as_int=False)
    _coerce("miles", as_int=False)
    _coerce("minutes", as_int=True)
    _coerce("orders", as_int=True)

    # Currency — sometimes returned with the £ symbol embedded.
    if "currency" in parsed and parsed["currency"] is not None:
        cur = str(parsed["currency"]).strip().upper()
        if "£" in cur or "GBP" in cur:
            new = "GBP"
        elif "$" in cur or "USD" in cur:
            new = "USD"
        elif "€" in cur or "EUR" in cur:
            new = "EUR"
        elif cur in {"GBP", "USD", "EUR", "UNKNOWN"}:
            new = cur if cur != "UNKNOWN" else "unknown"
        else:
            new = "unknown"
        if new != parsed["currency"]:
            corrections.append(f"currency: {parsed['currency']!r}→{new!r}")
            parsed["currency"] = new

    if corrections:
        logger.info("Post-parse coercions applied: %s", "; ".join(corrections))

    return parsed


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

    Routing priority (mirrors the brief from the client stress test):

    1. An explicit ``deliveroo`` hint always wins — the caller knows best
       and the Deliveroo V2 prompt is now the default for that hint.
    2. If the image looks like Deliveroo by colour signature (teal pixel
       ratio above the threshold), use the Deliveroo V2 prompt.
    3. Otherwise, use the master prompt (optionally augmented with the hint).

    Detection 2 originally included "Accept and go" text matching and
    suitcase-icon detection, but both require OCR — which is exactly what
    Gemini does in step 4 of the pipeline. They are folded into the
    Deliveroo V2 prompt itself: once teal triggers routing, the prompt
    handles V2 layout detection by content.
    """
    hint_clean = (hint or "").strip().lower()
    if hint_clean == "deliveroo":
        logger.info("Routing to Deliveroo V2 prompt (explicit hint).")
        return build_extraction_prompt("deliveroo")

    if _looks_like_deliveroo(image_bytes):
        logger.info("Routing to Deliveroo V2 prompt (teal colour signature detected).")
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
    # Threshold matches the brief from the client stress test: ≥5% teal
    # pixels is a reliable Deliveroo V2 signature (the teal "Accept and go"
    # button alone occupies roughly this much of a typical offer screen).
    if ratio >= 0.05:
        logger.debug("Teal pixel ratio: %.3f (Deliveroo signature)", ratio)
        return True
    return False


# ---------------------------------------------------------------------------
# Result coercion
# ---------------------------------------------------------------------------


def _coerce_result(parsed: dict[str, Any], raw_text_fallback: str) -> ExtractionResult:
    """Build an ``ExtractionResult`` from a parsed dict.

    ``parsed`` should already have been run through
    ``_normalize_extraction_payload`` so that numeric fields are real numbers.
    This function is the *last* line of defence — it still calls
    ``force_numeric`` on each field so that direct test callers (and any
    future entry point that skips normalization) cannot break the schema.
    """
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

    server_confidence = calculate_confidence(parsed)

    return ExtractionResult(
        pay=force_numeric(parsed.get("pay"), as_int=False),
        currency=currency,  # type: ignore[arg-type]
        miles=force_numeric(parsed.get("miles"), as_int=False),
        minutes=force_numeric(parsed.get("minutes"), as_int=True),
        orders=force_numeric(parsed.get("orders"), as_int=True),
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

    # Post-parse normalization: rewrite string/nested-object numeric fields
    # before the Pydantic coercion happens. Logged corrections give us
    # production visibility into prompt drift.
    parsed = _normalize_extraction_payload(parsed)

    # Defensive: scrub JSON contamination out of raw_text before it leaks
    # into the client response. See validate_raw_text() for the failure
    # mode this guards against.
    parsed = validate_raw_text(parsed)

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
