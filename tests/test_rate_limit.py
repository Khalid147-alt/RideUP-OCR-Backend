"""Tests covering the Gemini free-tier rate-limit handling.

Two-class backoff:

- Rate-limit class (429 / quota / 503) → 15s, 30s, 60s, then graceful
  degradation: a low-confidence ``ExtractionResult`` with ``confidence``
  set to ``"low"`` and ``retry_attempted=true`` in ``notes``. HTTP-200,
  not 502, so the client app stays responsive during quota incidents.
- Fast class (500 / DeadlineExceeded) → 1s, 2s, 4s, then the original
  exception propagates as a 502.

Every test mocks ``time.sleep`` so the suite runs in a few seconds even
when the schedule under test waits a minute.
"""

from __future__ import annotations

import io
import json
import os
from unittest.mock import patch

import pytest
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    InvalidArgument,
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
)
from httpx import ASGITransport, AsyncClient
from PIL import Image

import extractor
from extractor import (
    _RATE_LIMIT_BACKOFF_DELAYS,
    _RateLimitExhausted,
    _is_rate_limit_error,
    _with_api_retries,
    _with_api_retries_tracked,
)
from main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_png(width: int = 600, height: int = 600) -> bytes:
    noise = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), noise)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def good_png() -> bytes:
    return _make_png()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_rate_limit_class_includes_429_and_503() -> None:
    """All three rate-limit-class errors are classified together."""
    assert _is_rate_limit_error(TooManyRequests("429"))
    assert _is_rate_limit_error(ResourceExhausted("429 quota"))
    assert _is_rate_limit_error(ServiceUnavailable("503"))


def test_fast_class_is_not_rate_limit() -> None:
    """500 and DeadlineExceeded retry on the fast ladder, not the long one."""
    assert not _is_rate_limit_error(InternalServerError("500"))
    assert not _is_rate_limit_error(DeadlineExceeded("504"))


def test_non_retryable_is_not_rate_limit() -> None:
    """A 400 InvalidArgument is neither retryable nor in the rate-limit set."""
    assert not _is_rate_limit_error(InvalidArgument("400"))


# ---------------------------------------------------------------------------
# Backoff ladder values
# ---------------------------------------------------------------------------


def test_rate_limit_ladder_is_15_30_60() -> None:
    """The exact schedule the brief specifies."""
    assert _RATE_LIMIT_BACKOFF_DELAYS == (15.0, 30.0, 60.0)


def test_rate_limit_backoff_sleeps_15_then_30() -> None:
    """First 429 → wait ~15s, second 429 → wait ~30s, third call succeeds."""
    calls = {"n": 0}
    delays: list[float] = []

    def _fake_call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TooManyRequests("429")
        return '{"pay": 1.0}'

    with patch.object(extractor.time, "sleep", side_effect=delays.append):
        result = _with_api_retries("test-rate-limit", _fake_call)

    assert result == '{"pay": 1.0}'
    assert len(delays) == 2
    # First retry: base 15.0 + jitter [0, 0.25]
    assert 15.0 <= delays[0] <= 15.30
    # Second retry: base 30.0 + jitter
    assert 30.0 <= delays[1] <= 30.30


def test_fast_class_still_sleeps_1_then_2() -> None:
    """500 errors use the fast ladder; the long one is reserved for 429/503."""
    calls = {"n": 0}
    delays: list[float] = []

    def _fake_call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise InternalServerError("500")
        return "{}"

    with patch.object(extractor.time, "sleep", side_effect=delays.append):
        _with_api_retries("test-fast", _fake_call)

    assert len(delays) == 2
    assert 1.0 <= delays[0] <= 1.30
    assert 2.0 <= delays[1] <= 2.30


def test_rate_limit_log_wording_mentions_retry_n_of_3(caplog) -> None:
    """Log messages match the brief: 'Rate limit hit, waiting Xs before retry N/3...'."""
    import logging

    calls = {"n": 0}

    def _fake_call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResourceExhausted("429 quota")
        return "{}"

    with caplog.at_level(logging.WARNING, logger="extractor"):
        with patch.object(extractor.time, "sleep"):
            _with_api_retries("test-log", _fake_call)

    messages = [r.getMessage() for r in caplog.records]
    rate_limit_warnings = [m for m in messages if "Rate limit hit" in m]
    assert len(rate_limit_warnings) == 2
    assert "retry 1/3" in rate_limit_warnings[0]
    assert "waiting 15" in rate_limit_warnings[0]
    assert "retry 2/3" in rate_limit_warnings[1]
    assert "waiting 30" in rate_limit_warnings[1]


# ---------------------------------------------------------------------------
# Exhaustion behaviour
# ---------------------------------------------------------------------------


def test_rate_limit_exhaustion_raises_sentinel() -> None:
    """After 3 rate-limit failures, _with_api_retries raises _RateLimitExhausted."""
    calls = {"n": 0}

    def _always_429() -> str:
        calls["n"] += 1
        raise TooManyRequests("429 every time")

    with patch.object(extractor.time, "sleep"):
        with pytest.raises(_RateLimitExhausted) as excinfo:
            _with_api_retries("test-exhaust", _always_429)

    assert calls["n"] == 3
    assert isinstance(excinfo.value.original, TooManyRequests)


def test_fast_class_exhaustion_propagates_original_exception() -> None:
    """Fast-class exhaustion re-raises the underlying exception (not a sentinel)."""
    calls = {"n": 0}

    def _always_500() -> str:
        calls["n"] += 1
        raise InternalServerError("500 every time")

    with patch.object(extractor.time, "sleep"):
        with pytest.raises(InternalServerError):
            _with_api_retries("test-exhaust-fast", _always_500)

    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Pipeline-level graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_returns_low_confidence_on_persistent_rate_limit(
    client: AsyncClient, good_png: bytes
) -> None:
    """A persistent 429 yields HTTP 200 with confidence=low, not a 502."""

    def _always_429(*_args, **_kwargs) -> str:
        raise TooManyRequests("429 quota exceeded")

    with patch.object(extractor.time, "sleep"):
        with patch.object(extractor, "_call_vision_model", side_effect=_always_429):
            files = {"image": ("trip.png", good_png, "image/png")}
            response = await client.post("/extract", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["confidence"] == "low"
    assert body["pay"] is None
    assert "rate limit" in body["notes"].lower()
    assert "retry_attempted=true" in body["notes"]


@pytest.mark.asyncio
async def test_extract_still_returns_502_on_persistent_500(
    client: AsyncClient, good_png: bytes
) -> None:
    """A persistent 500 is a real incident — still surfaces as 502 to the client."""

    def _always_500(*_args, **_kwargs) -> str:
        raise InternalServerError("500 every time")

    with patch.object(extractor.time, "sleep"):
        with patch.object(extractor, "_call_vision_model", side_effect=_always_500):
            files = {"image": ("trip.png", good_png, "image/png")}
            response = await client.post("/extract", files=files)

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "bad_gateway"


@pytest.mark.asyncio
async def test_extract_success_marks_retry_attempted_false(
    client: AsyncClient, good_png: bytes
) -> None:
    """Happy-path single-call success annotates notes with retry_attempted=false."""
    payload = {
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
    with patch.object(extractor, "_call_vision_model", return_value=json.dumps(payload)):
        files = {"image": ("trip.png", good_png, "image/png")}
        response = await client.post("/extract", files=files)

    assert response.status_code == 200
    body = response.json()
    assert "retry_attempted=false" in body["notes"]


@pytest.mark.asyncio
async def test_extract_marks_retry_attempted_true_after_transient_recovery(
    client: AsyncClient, good_png: bytes
) -> None:
    """One 429 then a clean response → retry_attempted=true in notes."""
    payload = {
        "pay": 8.42,
        "currency": "GBP",
        "miles": 3.1,
        "minutes": 24,
        "orders": 2,
        "platform": "uber_eats",
        "confidence": "high",
        "notes": "",
        "raw_text": "£8.42 · 24 min (3.1 mi) total · 2 deliveries",
    }
    responses = iter([TooManyRequests("429"), json.dumps(payload)])

    def _fake_call(*_args, **_kwargs) -> str:
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(extractor.time, "sleep"):
        with patch.object(extractor, "_call_vision_model", side_effect=_fake_call):
            files = {"image": ("trip.png", good_png, "image/png")}
            response = await client.post("/extract", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["pay"] == 8.42
    assert "retry_attempted=true" in body["notes"]


# ---------------------------------------------------------------------------
# _with_api_retries_tracked helper
# ---------------------------------------------------------------------------


def test_tracked_helper_reports_attempt_count() -> None:
    """The tracked wrapper returns the number of attempts that were made."""
    calls = {"n": 0}

    def _fake_call() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise TooManyRequests("429 once")
        return "ok"

    with patch.object(extractor.time, "sleep"):
        text, attempts = _with_api_retries_tracked("test-tracked", _fake_call)

    assert text == "ok"
    assert attempts == 2


def test_tracked_helper_reports_one_attempt_on_immediate_success() -> None:
    """A first-try success reports exactly 1 attempt."""
    with patch.object(extractor.time, "sleep"):
        text, attempts = _with_api_retries_tracked("test-tracked", lambda: "ok")

    assert text == "ok"
    assert attempts == 1
