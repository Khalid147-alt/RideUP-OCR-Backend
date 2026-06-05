"""Tests for the Deliveroo V2 postcode-mileage estimation feature.

Covers:

- ``extract_postcodes``        — regex extraction + normalisation (8+ cases)
- ``normalize_postcode``       — spacing/casing canonicalisation
- ``get_postcode_coordinates`` — async httpx geocode (success / 404 / network)
- ``calculate_distance_miles`` — Haversine against known London pairs
- ``estimate_deliveroo_mileage`` — end-to-end estimate with mocked geocoding
- ``enrich_deliveroo_postcodes`` — gating rules + result mutation

No live network access is required: the httpx client is driven by
``httpx.MockTransport`` and the higher-level geocoder is patched.
"""

from __future__ import annotations

import base64
import io
import json
import os
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import extractor
from extractor import enrich_deliveroo_postcodes, estimate_deliveroo_mileage
from main import app
from models import ExtractionResult
from utils.postcode_utils import (
    calculate_distance_miles,
    extract_postcodes,
    get_postcode_coordinates,
    normalize_postcode,
)


# ---------------------------------------------------------------------------
# normalize_postcode
# ---------------------------------------------------------------------------


def test_normalize_inserts_missing_space() -> None:
    assert normalize_postcode("NW100NX") == "NW10 0NX"


def test_normalize_uppercases_and_trims() -> None:
    assert normalize_postcode("  ha96ff ") == "HA9 6FF"


def test_normalize_collapses_existing_space() -> None:
    assert normalize_postcode("SW1X  7JW") == "SW1X 7JW"


# ---------------------------------------------------------------------------
# extract_postcodes — 8+ edge cases
# ---------------------------------------------------------------------------


def test_extract_two_postcodes_with_spaces() -> None:
    raw = "£4.23 · 1 order · 139 Road NW10 0NX · 1 Harrow Road HA9 6FF · Accept and go"
    assert extract_postcodes(raw) == ("NW10 0NX", "HA9 6FF")


def test_extract_two_postcodes_missing_spaces() -> None:
    """Run-together postcodes are normalised with a space inserted."""
    raw = "£4.23 · McDonald's · 139 NORTH CIRCULAR ROAD NW100NX · 1 Harrow Road HA96FF · Accept and go"
    assert extract_postcodes(raw) == ("NW10 0NX", "HA9 6FF")


def test_extract_single_postcode_returns_pickup_only() -> None:
    raw = "£5.00 · 1 order · Some Cafe · 12 High Street EC1A 1BB · Accept and go"
    assert extract_postcodes(raw) == ("EC1A 1BB", None)


def test_extract_no_postcodes_returns_none_none() -> None:
    raw = "£11.08 · 3 orders · Accept and go"
    assert extract_postcodes(raw) == (None, None)


def test_extract_empty_string() -> None:
    assert extract_postcodes("") == (None, None)


def test_extract_three_postcodes_takes_first_two() -> None:
    raw = "NW2 5DS then NW2 5SJ then NW6 1TN · Accept and go"
    assert extract_postcodes(raw) == ("NW2 5DS", "NW2 5SJ")


def test_extract_lowercase_postcodes_uppercased() -> None:
    raw = "pickup nw10 0nx dropoff ha9 6ff"
    assert extract_postcodes(raw) == ("NW10 0NX", "HA9 6FF")


def test_extract_short_outward_code() -> None:
    """Single-letter, single-digit outward code (e.g. W1A 0AX)."""
    raw = "BBC · W1A 0AX · customer N1 9GU · go"
    assert extract_postcodes(raw) == ("W1A 0AX", "N1 9GU")


def test_extract_mixed_spacing_normalises_both() -> None:
    raw = "from NW100NX to HA9 6FF"
    assert extract_postcodes(raw) == ("NW10 0NX", "HA9 6FF")


# ---------------------------------------------------------------------------
# calculate_distance_miles — known London pairs
# ---------------------------------------------------------------------------


def test_distance_zero_for_identical_points() -> None:
    assert calculate_distance_miles(51.5074, -0.1278, 51.5074, -0.1278) == 0.0


def test_distance_known_london_pair() -> None:
    """NW10 0NX (~51.541, -0.247) → HA9 6FF (~51.557, -0.281).

    Straight-line distance is roughly 1.8 miles. Allow a small tolerance for
    the rounding to 1 dp and coordinate precision.
    """
    miles = calculate_distance_miles(51.5410, -0.2470, 51.5573, -0.2817)
    assert 1.5 <= miles <= 2.2


def test_distance_rounds_to_one_decimal() -> None:
    miles = calculate_distance_miles(51.5074, -0.1278, 51.5174, -0.1378)
    # Result must be a float rounded to a single decimal place.
    assert isinstance(miles, float)
    assert round(miles, 1) == miles


def test_distance_central_london_to_heathrow() -> None:
    """Charing Cross (51.508, -0.128) → Heathrow (51.470, -0.454) ≈ 14-15 mi."""
    miles = calculate_distance_miles(51.5080, -0.1281, 51.4700, -0.4543)
    assert 13.0 <= miles <= 16.0


# ---------------------------------------------------------------------------
# get_postcode_coordinates — async httpx, mocked transport
# ---------------------------------------------------------------------------


def _mock_client_factory(handler):
    """Return a factory that yields an AsyncClient driven by ``handler``.

    Patches ``httpx.AsyncClient`` so the real network is never touched; the
    ``MockTransport`` routes every request through ``handler``. We capture the
    real class up front so the factory doesn't recurse into the patched name.
    """
    real_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        kwargs.pop("transport", None)
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    return _factory


@pytest.mark.asyncio
async def test_get_coordinates_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "NW10%200NX" in str(request.url)  # space URL-encoded
        return httpx.Response(
            200,
            json={"status": 200, "result": {"latitude": 51.541, "longitude": -0.247}},
        )

    with patch("httpx.AsyncClient", _mock_client_factory(handler)):
        coords = await get_postcode_coordinates("NW10 0NX")

    assert coords == (51.541, -0.247)


@pytest.mark.asyncio
async def test_get_coordinates_404_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"status": 404, "error": "Postcode not found"})

    with patch("httpx.AsyncClient", _mock_client_factory(handler)):
        coords = await get_postcode_coordinates("ZZ99 9ZZ")

    assert coords is None


@pytest.mark.asyncio
async def test_get_coordinates_network_error_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with patch("httpx.AsyncClient", _mock_client_factory(handler)):
        coords = await get_postcode_coordinates("NW10 0NX")

    assert coords is None


@pytest.mark.asyncio
async def test_get_coordinates_timeout_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with patch("httpx.AsyncClient", _mock_client_factory(handler)):
        coords = await get_postcode_coordinates("NW10 0NX")

    assert coords is None


@pytest.mark.asyncio
async def test_get_coordinates_malformed_body_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 200, "result": None})

    with patch("httpx.AsyncClient", _mock_client_factory(handler)):
        coords = await get_postcode_coordinates("NW10 0NX")

    assert coords is None


@pytest.mark.asyncio
async def test_get_coordinates_empty_postcode_returns_none() -> None:
    coords = await get_postcode_coordinates("   ")
    assert coords is None


# ---------------------------------------------------------------------------
# estimate_deliveroo_mileage — mocked geocoder
# ---------------------------------------------------------------------------

_RAW_TWO_POSTCODES = (
    "£4.23 · 1 order · McDonald's · 139 NORTH CIRCULAR ROAD NW100NX · "
    "1 Harrow Road HA96FF · Accept and go"
)


@pytest.mark.asyncio
async def test_estimate_mileage_success() -> None:
    async def fake_coords(postcode: str):
        return {"NW10 0NX": (51.541, -0.247), "HA9 6FF": (51.5573, -0.2817)}[postcode]

    # estimate_deliveroo_mileage lives in extractor and calls the geocoder via
    # the name imported into extractor's namespace — patch it there.
    with patch.object(extractor, "get_postcode_coordinates", fake_coords):
        miles, note = await estimate_deliveroo_mileage(_RAW_TWO_POSTCODES)

    assert miles is not None
    assert miles > 0
    assert "NW10 0NX → HA9 6FF" in note
    assert "postcodes.io" in note


@pytest.mark.asyncio
async def test_estimate_mileage_single_postcode_returns_none() -> None:
    raw = "£5.00 · 1 order · Cafe · 12 High Street EC1A 1BB · Accept and go"
    miles, note = await estimate_deliveroo_mileage(raw)
    assert miles is None
    assert note == ""


@pytest.mark.asyncio
async def test_estimate_mileage_geocode_miss_returns_none() -> None:
    async def fake_coords(postcode: str):
        return None  # both postcodes fail to geocode

    with patch.object(extractor, "get_postcode_coordinates", fake_coords):
        miles, note = await estimate_deliveroo_mileage(_RAW_TWO_POSTCODES)

    assert miles is None
    assert note == ""


@pytest.mark.asyncio
async def test_estimate_mileage_no_postcodes_returns_none() -> None:
    miles, note = await estimate_deliveroo_mileage("£11.08 · 3 orders · Accept and go")
    assert miles is None
    assert note == ""


# ---------------------------------------------------------------------------
# enrich_deliveroo_postcodes — gating + mutation rules
# ---------------------------------------------------------------------------


def _deliveroo_result(**overrides) -> ExtractionResult:
    base = dict(
        pay=4.23,
        currency="GBP",
        miles=None,
        minutes=None,
        orders=1,
        platform="deliveroo",
        confidence="high",
        notes="Deliveroo V2 layout. retry_attempted=false",
        raw_text=_RAW_TWO_POSTCODES,
    )
    base.update(overrides)
    return ExtractionResult(**base)


@pytest.mark.asyncio
async def test_enrich_populates_postcodes_and_miles() -> None:
    async def fake_coords(postcode: str):
        return {"NW10 0NX": (51.541, -0.247), "HA9 6FF": (51.5573, -0.2817)}[postcode]

    with patch.object(extractor, "get_postcode_coordinates", fake_coords):
        enriched = await enrich_deliveroo_postcodes(_deliveroo_result())

    assert enriched.pickup_postcode == "NW10 0NX"
    assert enriched.dropoff_postcode == "HA9 6FF"
    assert enriched.miles is not None and enriched.miles > 0
    assert "Miles estimated from postcodes NW10 0NX → HA9 6FF" in enriched.notes
    # Original retry marker preserved.
    assert "retry_attempted=false" in enriched.notes
    # Confidence not downgraded.
    assert enriched.confidence == "high"


@pytest.mark.asyncio
async def test_enrich_populates_postcodes_even_when_geocode_fails() -> None:
    async def fake_coords(postcode: str):
        return None

    with patch.object(extractor, "get_postcode_coordinates", fake_coords):
        enriched = await enrich_deliveroo_postcodes(_deliveroo_result())

    # Postcodes are still surfaced...
    assert enriched.pickup_postcode == "NW10 0NX"
    assert enriched.dropoff_postcode == "HA9 6FF"
    # ...but miles stays null and notes are unchanged.
    assert enriched.miles is None
    assert enriched.notes == "Deliveroo V2 layout. retry_attempted=false"


@pytest.mark.asyncio
async def test_enrich_skips_non_deliveroo() -> None:
    result = _deliveroo_result(platform="uber_eats")
    enriched = await enrich_deliveroo_postcodes(result)
    # Untouched — same object returned, no postcode fields set.
    assert enriched.pickup_postcode is None
    assert enriched.dropoff_postcode is None
    assert enriched.miles is None


@pytest.mark.asyncio
async def test_enrich_skips_when_miles_already_present() -> None:
    """Deliveroo V1 (native miles) must not be touched."""
    result = _deliveroo_result(miles=3.0)
    enriched = await enrich_deliveroo_postcodes(result)
    assert enriched.miles == 3.0
    assert enriched.pickup_postcode is None
    assert enriched.dropoff_postcode is None


@pytest.mark.asyncio
async def test_enrich_single_postcode_sets_pickup_only_no_miles() -> None:
    result = _deliveroo_result(
        raw_text="£5.00 · 1 order · Cafe · 12 High Street EC1A 1BB · Accept and go"
    )
    enriched = await enrich_deliveroo_postcodes(result)
    assert enriched.pickup_postcode == "EC1A 1BB"
    assert enriched.dropoff_postcode is None
    assert enriched.miles is None
    assert enriched.notes == "Deliveroo V2 layout. retry_attempted=false"


@pytest.mark.asyncio
async def test_enrich_empty_raw_text_is_noop() -> None:
    result = _deliveroo_result(raw_text="")
    enriched = await enrich_deliveroo_postcodes(result)
    assert enriched.pickup_postcode is None
    assert enriched.dropoff_postcode is None
    assert enriched.miles is None


# ---------------------------------------------------------------------------
# End-to-end: full /extract/base64 HTTP path with Gemini + geocoding mocked
# ---------------------------------------------------------------------------


def _png_base64() -> str:
    """A PNG large enough to pass the quality check, base64-encoded."""
    noise = os.urandom(600 * 600 * 3)
    img = Image.frombytes("RGB", (600, 600), noise)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_endpoint_deliveroo_v2_enriched_response() -> None:
    """The real /extract/base64 path returns postcodes + estimated miles."""
    gemini_payload = {
        "pay": 4.23,
        "currency": "GBP",
        "miles": None,
        "minutes": None,
        "orders": 1,
        "platform": "deliveroo",
        "confidence": "high",
        "notes": "Deliveroo V2 layout.",
        "raw_text": _RAW_TWO_POSTCODES,
    }

    async def fake_coords(postcode: str):
        return {"NW10 0NX": (51.541, -0.247), "HA9 6FF": (51.5573, -0.2817)}[postcode]

    transport = ASGITransport(app=app)
    with patch.object(
        extractor, "_call_vision_model", return_value=json.dumps(gemini_payload)
    ), patch.object(extractor, "get_postcode_coordinates", fake_coords):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/extract/base64", json={"image": _png_base64()}
            )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["platform"] == "deliveroo"
    assert body["pickup_postcode"] == "NW10 0NX"
    assert body["dropoff_postcode"] == "HA9 6FF"
    assert body["miles"] is not None and body["miles"] > 0
    assert "Miles estimated from postcodes NW10 0NX → HA9 6FF" in body["notes"]
    assert body["confidence"] == "high"  # not downgraded
    assert body["minutes"] is None  # still null — V2 never shows it
