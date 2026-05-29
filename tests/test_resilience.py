"""Tests for the three resilience features added in the latest pass:

1. Deliveroo-specific prompt routing
2. Exponential-backoff retries on 429 / 502 / 503
3. JSON sanitization (fences, prose, partial output)
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from google.api_core.exceptions import (
    InternalServerError,
    ServiceUnavailable,
    TooManyRequests,
)
from PIL import Image

import extractor
import prompts
from extractor import (
    _looks_like_deliveroo,
    _select_prompt,
    _with_api_retries,
    sanitize_model_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _teal_png(width: int = 600, height: int = 600) -> bytes:
    """Solid teal image — should trigger Deliveroo detection."""
    img = Image.new("RGB", (width, height), color=(0, 180, 170))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _black_png(width: int = 600, height: int = 600) -> bytes:
    """Solid dark image — should NOT trigger Deliveroo detection."""
    img = Image.new("RGB", (width, height), color=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Deliveroo routing
# ---------------------------------------------------------------------------


def test_deliveroo_hint_routes_to_deliveroo_prompt() -> None:
    """An explicit ``deliveroo`` hint always selects a Deliveroo prompt.

    Post-V2 cutover this is the V2 prompt — V2 is now the default Deliveroo
    template because the V2 layout is what the client stress test exposed as
    the primary failure mode.
    """
    chosen = _select_prompt(_black_png(), hint="deliveroo")
    assert chosen == prompts.DELIVEROO_V2_EXTRACTION_PROMPT


def test_teal_image_routes_to_deliveroo_prompt() -> None:
    """A teal-dominated image triggers the Deliveroo V2 prompt automatically."""
    chosen = _select_prompt(_teal_png(), hint=None)
    assert chosen == prompts.DELIVEROO_V2_EXTRACTION_PROMPT


def test_non_teal_image_uses_master_prompt() -> None:
    """A non-teal image with no hint falls through to the master prompt."""
    chosen = _select_prompt(_black_png(), hint=None)
    assert chosen == prompts.MASTER_EXTRACTION_PROMPT


def test_uber_hint_does_not_use_deliveroo_prompt() -> None:
    """A non-Deliveroo hint preserves the master prompt with the hint appended."""
    chosen = _select_prompt(_black_png(), hint="uber_eats")
    assert chosen != prompts.DELIVEROO_EXTRACTION_PROMPT
    assert "uber_eats" in chosen.lower()


def test_looks_like_deliveroo_positive() -> None:
    assert _looks_like_deliveroo(_teal_png()) is True


def test_looks_like_deliveroo_negative() -> None:
    assert _looks_like_deliveroo(_black_png()) is False


def test_deliveroo_prompt_mentions_teal_and_stacked() -> None:
    """Sanity check the Deliveroo prompt contains its platform-specific cues."""
    text = prompts.DELIVEROO_EXTRACTION_PROMPT.lower()
    assert "teal" in text
    assert "stacked" in text
    assert "deliveroo" in text


# ---------------------------------------------------------------------------
# 2. Exponential backoff
# ---------------------------------------------------------------------------


def test_backoff_succeeds_after_two_transient_failures() -> None:
    """A 503 then a 429 then success should still return the success body."""
    calls = {"n": 0}

    def _fake_call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServiceUnavailable("503 unavailable")
        if calls["n"] == 2:
            raise TooManyRequests("429 rate limit")
        return '{"pay": 1.23}'

    with patch.object(extractor.time, "sleep") as fake_sleep:
        result = _with_api_retries("test", _fake_call)

    assert result == '{"pay": 1.23}'
    assert calls["n"] == 3
    # Should have slept twice (after attempt 1 and attempt 2).
    assert fake_sleep.call_count == 2


def test_backoff_gives_up_after_three_attempts() -> None:
    """After the configured max attempts, the last exception propagates."""
    calls = {"n": 0}

    def _always_503() -> str:
        calls["n"] += 1
        raise ServiceUnavailable("503 always")

    with patch.object(extractor.time, "sleep"):
        with pytest.raises(ServiceUnavailable):
            _with_api_retries("test", _always_503)

    assert calls["n"] == 3  # _MAX_API_ATTEMPTS


def test_backoff_does_not_retry_non_transient_errors() -> None:
    """A non-retryable GoogleAPIError aborts immediately."""
    from google.api_core.exceptions import InvalidArgument

    calls = {"n": 0}

    def _bad_request() -> str:
        calls["n"] += 1
        raise InvalidArgument("400 bad request")

    with patch.object(extractor.time, "sleep"):
        with pytest.raises(InvalidArgument):
            _with_api_retries("test", _bad_request)

    assert calls["n"] == 1


def test_backoff_delays_follow_1_2_4_pattern() -> None:
    """The first sleep is ~1s, the second is ~2s (within jitter)."""
    calls = {"n": 0}
    delays: list[float] = []

    def _fake_call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise InternalServerError("500")
        return "{}"

    def _capture_sleep(seconds: float) -> None:
        delays.append(seconds)

    with patch.object(extractor.time, "sleep", side_effect=_capture_sleep):
        _with_api_retries("test", _fake_call)

    assert len(delays) == 2
    # First retry: base 1.0 + jitter [0, 0.25]
    assert 1.0 <= delays[0] <= 1.30
    # Second retry: base 2.0 + jitter
    assert 2.0 <= delays[1] <= 2.30


# ---------------------------------------------------------------------------
# 3. JSON sanitization
# ---------------------------------------------------------------------------


def test_sanitize_strips_json_fence() -> None:
    raw = '```json\n{"pay": 5}\n```'
    assert sanitize_model_output(raw) == '{"pay": 5}'


def test_sanitize_strips_plain_fence() -> None:
    raw = '```\n{"pay": 5}\n```'
    assert sanitize_model_output(raw) == '{"pay": 5}'


def test_sanitize_strips_leading_prose() -> None:
    raw = 'Sure! Here is the data:\n{"pay": 5, "currency": "GBP"}\nHope this helps!'
    cleaned = sanitize_model_output(raw)
    assert cleaned.startswith("{") and cleaned.endswith("}")
    assert json.loads(cleaned)["pay"] == 5


def test_sanitize_handles_truncated_json() -> None:
    """A response missing the closing brace gets one appended."""
    raw = '{"pay": 5, "currency": "GBP"'
    cleaned = sanitize_model_output(raw)
    assert cleaned.endswith("}")
    # It should now parse — pay still present.
    parsed = json.loads(cleaned)
    assert parsed["pay"] == 5


def test_sanitize_empty_input_returns_empty() -> None:
    assert sanitize_model_output("") == ""
    assert sanitize_model_output("   \n\n   ") == ""


def test_sanitize_handles_extra_whitespace() -> None:
    raw = '\n\n   ```json   \n  {"pay": 5}  \n   ```   \n'
    assert sanitize_model_output(raw) == '{"pay": 5}'


def test_sanitize_strips_bare_json_label_prefix() -> None:
    """A response that starts with ``json\\n{...}`` (no fences) parses cleanly."""
    raw = 'json\n{"pay": 5}'
    cleaned = sanitize_model_output(raw)
    assert cleaned == '{"pay": 5}'


def test_sanitize_strips_json_colon_label_prefix() -> None:
    raw = 'JSON: {"pay": 5}'
    cleaned = sanitize_model_output(raw)
    assert cleaned.startswith("{") and cleaned.endswith("}")


def test_sanitize_does_not_eat_word_inside_value() -> None:
    """``"json"`` only stripped as a prefix, never inside the JSON itself."""
    raw = '{"notes": "json was malformed"}'
    cleaned = sanitize_model_output(raw)
    assert "json was malformed" in cleaned


# ---------------------------------------------------------------------------
# 4. force_numeric helper — used to recover from messy model outputs
# ---------------------------------------------------------------------------


def test_force_numeric_passes_through_numbers() -> None:
    assert extractor.force_numeric(5.5) == 5.5
    assert extractor.force_numeric(5, as_int=True) == 5
    assert extractor.force_numeric(5.7, as_int=True) == 5


def test_force_numeric_handles_currency_string() -> None:
    assert extractor.force_numeric("£6.50") == 6.50
    assert extractor.force_numeric("$12.00") == 12.00
    assert extractor.force_numeric("€9.99") == 9.99


def test_force_numeric_handles_unit_suffixes() -> None:
    assert extractor.force_numeric("2.1 mi") == 2.1
    assert extractor.force_numeric("3.4 km") == 3.4
    assert extractor.force_numeric("18 min", as_int=True) == 18
    assert extractor.force_numeric("2 stacked", as_int=True) == 2


def test_force_numeric_handles_nested_dict_pay() -> None:
    assert extractor.force_numeric({"gross": 6.50, "tip": 1.00}) == 6.50
    assert extractor.force_numeric({"total": 9.99}) == 9.99
    # Fallback when known keys absent — first numeric value wins.
    assert extractor.force_numeric({"foo": "bar", "x": 4.2}) == 4.2


def test_force_numeric_handles_empty_and_invalid() -> None:
    assert extractor.force_numeric(None) is None
    assert extractor.force_numeric("") is None
    assert extractor.force_numeric("  ") is None
    assert extractor.force_numeric("no numbers here") is None
    assert extractor.force_numeric(True) is None  # booleans rejected
    assert extractor.force_numeric(False) is None


def test_force_numeric_handles_list() -> None:
    assert extractor.force_numeric([None, "2.1 mi", "3.4 mi"]) == 2.1


# ---------------------------------------------------------------------------
# 5. _normalize_extraction_payload — full post-parse cleanup
# ---------------------------------------------------------------------------


def test_normalize_payload_fixes_all_numeric_fields() -> None:
    parsed = {
        "pay": "£6.50",
        "miles": "2.1 mi",
        "minutes": "18 min",
        "orders": "2 stacked",
        "currency": "£",
        "platform": "deliveroo",
    }
    out = extractor._normalize_extraction_payload(parsed)
    assert out["pay"] == 6.50
    assert out["miles"] == 2.1
    assert out["minutes"] == 18
    assert out["orders"] == 2
    assert out["currency"] == "GBP"


def test_normalize_payload_handles_nested_pay() -> None:
    parsed = {"pay": {"gross": 7.20, "tip": 1.50}}
    out = extractor._normalize_extraction_payload(parsed)
    assert out["pay"] == 7.20


def test_normalize_payload_preserves_already_clean_data() -> None:
    parsed = {
        "pay": 6.50,
        "miles": 2.1,
        "minutes": 18,
        "orders": 1,
        "currency": "GBP",
    }
    out = extractor._normalize_extraction_payload(parsed)
    assert out == parsed  # unchanged


def test_normalize_payload_currency_symbol_to_iso() -> None:
    assert extractor._normalize_extraction_payload({"currency": "£"})["currency"] == "GBP"
    assert extractor._normalize_extraction_payload({"currency": "GBP £"})["currency"] == "GBP"
    assert extractor._normalize_extraction_payload({"currency": "USD"})["currency"] == "USD"


def test_extract_returns_low_confidence_when_parse_fails_twice() -> None:
    """If both calls return unparseable text, the pipeline degrades gracefully."""

    def _gibberish(*_args, **_kwargs) -> str:
        return "this is not json at all, no braces anywhere"

    png = _black_png()
    with patch.object(extractor, "_call_vision_model", side_effect=_gibberish):
        result = extractor.extract_from_image(png)

    assert result.confidence == "low"
    assert "could not be parsed" in result.notes.lower()
    assert result.pay is None
