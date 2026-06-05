"""UK postcode extraction, geocoding, and distance estimation.

This module is the backing logic for the Deliveroo V2 mileage-estimation
feature. Deliveroo's "Accept and go" offer cards do not display distance to
the driver, but they *do* print the pickup and drop-off addresses (including
postcodes) in the offer text. We recover those postcodes from ``raw_text``,
geocode them via the free `postcodes.io <https://postcodes.io>`_ API, and
estimate the great-circle distance with the Haversine formula.

Design constraints (from the brief):

- **Async, fail-silent geocoding.** ``get_postcode_coordinates`` uses an async
  ``httpx`` client with a 5-second timeout and returns ``None`` on *any*
  failure (404, network error, malformed response). The feature must be
  completely invisible when it fails — no exceptions ever propagate.
- **No external geo dependency.** ``calculate_distance_miles`` is pure Python.
- **No API key.** postcodes.io is free and key-less for UK postcodes.

The module is deliberately framework-agnostic — it knows nothing about
FastAPI, Gemini, or the ``ExtractionResult`` schema. Integration lives in the
extractor/endpoint layer.
"""

from __future__ import annotations

import logging
import math
import re
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# UK postcode pattern. The outward code is 1-2 letters followed by 1-2 digits
# (optionally with a trailing letter, e.g. ``NW1 0`` / ``EC1A``); the inward
# code is always 1 digit + 2 letters. We allow an optional space between the
# two halves so we catch both ``NW10 0NX`` and the run-together ``NW100NX``
# that the OCR frequently produces when the screenshot crops the gap.
_POSTCODE_RE = re.compile(
    r"[A-Z]{1,2}[0-9]{1,2}[A-Z]?\s?[0-9][A-Z]{2}",
    re.IGNORECASE,
)

# postcodes.io single-postcode lookup endpoint.
_POSTCODES_IO_URL = "https://api.postcodes.io/postcodes/{postcode}"

# Per-request timeout for the geocoding call. The brief specifies 5 seconds.
_GEOCODE_TIMEOUT_SECONDS = 5.0

# Earth radius in miles, per the brief's Haversine spec.
_EARTH_RADIUS_MILES = 3956.0


def normalize_postcode(postcode: str) -> str:
    """Normalise a raw UK postcode to canonical ``OUTWARD INWARD`` form.

    Uppercases, strips surrounding whitespace, and guarantees exactly one
    space separating the outward and inward codes. The inward code is always
    the final three characters (1 digit + 2 letters) of a valid UK postcode,
    so we split there regardless of whether the input already had a space.

    Examples:
        ``"nw100nx"``   → ``"NW10 0NX"``
        ``"NW10 0NX"``  → ``"NW10 0NX"``
        ``"  ha96ff "`` → ``"HA9 6FF"``

    Args:
        postcode: A raw postcode string (with or without a space).

    Returns:
        The normalised postcode. If the input is too short to carry an inward
        code it is returned uppercased/stripped without further surgery.
    """
    compact = re.sub(r"\s+", "", postcode).upper()
    if len(compact) < 5:
        return compact
    # Inward code is always the last 3 chars; outward is everything before it.
    return f"{compact[:-3]} {compact[-3:]}"


def extract_postcodes(raw_text: str) -> tuple[str | None, str | None]:
    """Extract pickup and drop-off postcodes from ``raw_text``.

    Scans the text left-to-right for UK postcodes. The first match is treated
    as the pickup, the second as the drop-off — this mirrors how Deliveroo
    offer text is ordered (restaurant/pickup address first, customer/drop-off
    second). All matches are normalised to canonical ``OUTWARD INWARD`` form.

    Args:
        raw_text: The OCR text from a screenshot.

    Returns:
        ``(pickup_postcode, dropoff_postcode)``:

        - two postcodes found → both populated
        - exactly one found   → ``(pickup, None)``
        - none found          → ``(None, None)``
    """
    if not raw_text:
        return None, None

    matches = _POSTCODE_RE.findall(raw_text)
    if not matches:
        return None, None

    normalized = [normalize_postcode(m) for m in matches]
    pickup = normalized[0]
    dropoff = normalized[1] if len(normalized) >= 2 else None
    return pickup, dropoff


async def get_postcode_coordinates(postcode: str) -> tuple[float, float] | None:
    """Geocode a UK postcode to ``(latitude, longitude)`` via postcodes.io.

    Uses an async ``httpx`` client with a 5-second timeout. Every failure mode
    — 404 (unknown postcode), network error, timeout, or a malformed/partial
    response body — resolves to ``None``. No exception ever escapes; the
    caller treats ``None`` as "could not geocode" and silently skips the
    estimation. No API key is required.

    Args:
        postcode: A UK postcode (any casing/spacing; URL-encoded internally).

    Returns:
        ``(latitude, longitude)`` as floats, or ``None`` if the postcode could
        not be resolved for any reason.
    """
    if not postcode or not postcode.strip():
        return None

    # URL-encode so a space (``NW10 0NX``) becomes ``%20`` rather than breaking
    # the path. ``quote`` with an empty ``safe`` encodes spaces and any other
    # reserved characters.
    encoded = quote(postcode.strip(), safe="")
    url = _POSTCODES_IO_URL.format(postcode=encoded)

    try:
        async with httpx.AsyncClient(timeout=_GEOCODE_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        # Connection errors, timeouts, DNS failures, etc. Fail silently.
        logger.warning("postcodes.io request failed for %r: %s", postcode, exc)
        return None

    if response.status_code != 200:
        # 404 (unknown postcode) and any other non-200 are non-fatal.
        logger.info(
            "postcodes.io returned %d for %r; skipping geocode.",
            response.status_code,
            postcode,
        )
        return None

    try:
        payload = response.json()
        result = payload["result"]
        latitude = float(result["latitude"])
        longitude = float(result["longitude"])
    except (ValueError, KeyError, TypeError) as exc:
        # Malformed JSON, missing keys, or null coordinates (postcodes.io
        # returns ``result: null`` for some terminated postcodes).
        logger.warning(
            "postcodes.io response for %r was unusable: %s", postcode, exc
        )
        return None

    return latitude, longitude


def calculate_distance_miles(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two lat/lng points, in miles.

    Implements the Haversine formula with an Earth radius of 3956 miles
    (per the brief). Pure Python — no external dependencies.

    Args:
        lat1: Latitude of the first point, in decimal degrees.
        lon1: Longitude of the first point, in decimal degrees.
        lat2: Latitude of the second point, in decimal degrees.
        lon2: Longitude of the second point, in decimal degrees.

    Returns:
        Distance in miles, rounded to 1 decimal place.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return round(_EARTH_RADIUS_MILES * c, 1)
