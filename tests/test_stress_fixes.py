"""Tests covering the post-stress-test fixes.

The real-client stress test surfaced five failure modes that needed
permanent fixes:

1. Deliveroo V2 layout: pay + orders visible, no distance/time on screen.
2. Uber Eats single order: ``"Delivery"`` (no number) ↔ orders == 1.
3. raw_text contamination: model echoing its own JSON into ``raw_text``.
4. Confidence scoring: don't penalise Deliveroo V2 for null miles/minutes.
5. Routing: ``hint="deliveroo"`` must reach the V2 prompt.

Each test below corresponds to one of the eight items in the brief. The
Gemini call is mocked throughout — these are unit tests for the
post-parse pipeline and the prompt-routing layer, not live calls.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import extractor
import prompts
from extractor import calculate_confidence, validate_raw_text
from main import app


# ---------------------------------------------------------------------------
# Local fixtures (kept separate from tests/test_extract.py to avoid coupling)
# ---------------------------------------------------------------------------


def _make_png(width: int = 600, height: int = 600) -> bytes:
    """PNG large enough to survive the quality check (random pixel noise)."""
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


def _deliveroo_v2_payload(**overrides: Any) -> dict[str, Any]:
    """Baseline payload for a Deliveroo V2 response. miles/minutes are null."""
    base = {
        "pay": 11.08,
        "currency": "GBP",
        "miles": None,
        "minutes": None,
        "orders": 3,
        "platform": "deliveroo",
        "confidence": "medium",
        "notes": (
            "Deliveroo V2 layout — distance and estimated time "
            "not displayed in order card."
        ),
        "raw_text": (
            "£11.08 · 3 orders · 2x The Poke Shack · "
            "255 West End Lane NW61XN · 1x Banana Tree · "
            "237-239 West End Lane NW61XN · Accept and go"
        ),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1–3. Deliveroo V2 layouts — three real client stress-test screenshots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deliveroo_v2_three_orders(
    client: AsyncClient, good_png: bytes
) -> None:
    """£11.08 · 3 orders Deliveroo V2 card — no distance, no minutes."""
    payload = _deliveroo_v2_payload(pay=11.08, orders=3)
    with patch.object(
        extractor, "_call_vision_model", return_value=json.dumps(payload)
    ):
        files = {"image": ("deliveroo.png", good_png, "image/png")}
        response = await client.post(
            "/extract", files=files, data={"hint": "deliveroo"}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pay"] == 11.08
    assert body["orders"] == 3
    assert body["miles"] is None
    assert body["minutes"] is None
    assert body["platform"] == "deliveroo"
    assert body["confidence"] == "high"


@pytest.mark.asyncio
async def test_deliveroo_v2_one_order(
    client: AsyncClient, good_png: bytes
) -> None:
    """£6.39 · 1 order Deliveroo V2 card — Gopuff pickup."""
    payload = _deliveroo_v2_payload(
        pay=6.39,
        orders=1,
        raw_text=(
            "£6.39 · 1 order · Gopuff · Unit 5 Wembley Trade Park NW100JF · "
            "237 Willesden Lane Flat 3 NW25RT · Accept and go"
        ),
    )
    with patch.object(
        extractor, "_call_vision_model", return_value=json.dumps(payload)
    ):
        files = {"image": ("deliveroo.png", good_png, "image/png")}
        response = await client.post(
            "/extract", files=files, data={"hint": "deliveroo"}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pay"] == 6.39
    assert body["orders"] == 1
    assert body["miles"] is None
    assert body["minutes"] is None
    assert body["platform"] == "deliveroo"


@pytest.mark.asyncio
async def test_deliveroo_v2_two_orders(
    client: AsyncClient, good_png: bytes
) -> None:
    """£7.22 · 2 orders Deliveroo V2 card — Flùr Flowers + Pizza Hut."""
    payload = _deliveroo_v2_payload(
        pay=7.22,
        orders=2,
        raw_text=(
            "£7.22 · 2 orders · 1x Flùr Flowers · 24B Windsor Road NW25DS · "
            "1x Pizza Hut Delivery · 7 Walm Lane NW25SJ · "
            "17 Glenbrook Road NW61TN · Accept and go"
        ),
    )
    with patch.object(
        extractor, "_call_vision_model", return_value=json.dumps(payload)
    ):
        files = {"image": ("deliveroo.png", good_png, "image/png")}
        response = await client.post(
            "/extract", files=files, data={"hint": "deliveroo"}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pay"] == 7.22
    assert body["orders"] == 2
    assert body["miles"] is None
    assert body["minutes"] is None
    assert body["platform"] == "deliveroo"


# ---------------------------------------------------------------------------
# 4. Uber Eats single order — the secondary stress-test failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uber_single_order(
    client: AsyncClient, good_png: bytes
) -> None:
    """£9.35 · 38 min (7.5 mi) total · "Delivery" → orders == 1."""
    payload = {
        "pay": 9.35,
        "currency": "GBP",
        "miles": 7.5,
        "minutes": 38,
        "orders": 1,
        "platform": "uber_eats",
        "confidence": "high",
        "notes": "Single Uber Eats delivery; green Confirm button, dark map.",
        "raw_text": (
            "£9.35 · 38 min (7.5 mi) total · Delivery · "
            "patisserie land ltd · London SW1X 7JW · Confirm"
        ),
    }
    with patch.object(
        extractor, "_call_vision_model", return_value=json.dumps(payload)
    ):
        files = {"image": ("uber.png", good_png, "image/png")}
        response = await client.post(
            "/extract", files=files, data={"hint": "uber_eats"}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pay"] == 9.35
    assert body["miles"] == 7.5
    assert body["minutes"] == 38
    assert body["orders"] == 1
    assert body["platform"] == "uber_eats"
    assert body["confidence"] == "high"


# ---------------------------------------------------------------------------
# 5–6. raw_text contamination defence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_text_json_contamination(
    client: AsyncClient, good_png: bytes
) -> None:
    """When Gemini echoes its own JSON into raw_text, it must be cleared."""
    payload = {
        "pay": 11.08,
        "currency": "GBP",
        "miles": None,
        "minutes": None,
        "orders": 3,
        "platform": "deliveroo",
        "confidence": "medium",
        "notes": "",
        # The exact contamination pattern observed in production.
        "raw_text": '{"pay": 11',
    }
    with patch.object(
        extractor, "_call_vision_model", return_value=json.dumps(payload)
    ):
        files = {"image": ("deliveroo.png", good_png, "image/png")}
        response = await client.post(
            "/extract", files=files, data={"hint": "deliveroo"}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["raw_text"].startswith("{")
    assert '"pay"' not in body["raw_text"]
    # Should not be the contaminated value — but should still be non-empty.
    assert body["raw_text"] != '{"pay": 11'
    assert body["raw_text"].strip() != ""


def test_raw_text_never_starts_with_brace() -> None:
    """validate_raw_text rejects any string opening with { or [."""
    for bad in (
        '{"pay": 11}',
        '{\n  "pay": 7.22',
        '   {"pay": 6.39}',  # leading whitespace doesn't save it
        '[{"pay": 1}]',
        '\n\n[1, 2, 3]',
    ):
        parsed = {"raw_text": bad}
        cleaned = validate_raw_text(parsed)
        assert not cleaned["raw_text"].lstrip().startswith("{")
        assert not cleaned["raw_text"].lstrip().startswith("[")

    # Sanity check: a legitimate raw_text passes through untouched.
    good = {"raw_text": "£11.08 · 3 orders · The Poke Shack"}
    assert validate_raw_text(good)["raw_text"] == "£11.08 · 3 orders · The Poke Shack"


# ---------------------------------------------------------------------------
# 7. Deliveroo confidence carve-out
# ---------------------------------------------------------------------------


def test_deliveroo_confidence_without_distance() -> None:
    """Deliveroo with pay + orders but null miles/minutes → "high"."""
    result = {
        "pay": 11.08,
        "currency": "GBP",
        "miles": None,
        "minutes": None,
        "orders": 3,
        "platform": "deliveroo",
    }
    assert calculate_confidence(result) == "high"

    # Only pay present → medium.
    result_pay_only = {
        "pay": 11.08,
        "miles": None,
        "minutes": None,
        "orders": None,
        "platform": "deliveroo",
    }
    assert calculate_confidence(result_pay_only) == "medium"

    # Non-Deliveroo with the same null-distance shape gets the strict rule:
    # 2 of 4 core fields → "low". Regression guard: the carve-out must NOT
    # leak to other platforms.
    result_uber = {
        "pay": 11.08,
        "miles": None,
        "minutes": None,
        "orders": 3,
        "platform": "uber_eats",
    }
    assert calculate_confidence(result_uber) == "low"


# ---------------------------------------------------------------------------
# 8. Routing: hint="deliveroo" reaches the V2 prompt
# ---------------------------------------------------------------------------


def test_hint_deliveroo_routes_to_v2_prompt() -> None:
    """The ``deliveroo`` hint reaches the V2 prompt regardless of image."""
    # Black image — no teal signature. Routing should be hint-driven.
    black = _make_png()  # random RGB noise, no teal dominance
    chosen = extractor._select_prompt(black, hint="deliveroo")
    assert chosen == prompts.DELIVEROO_V2_EXTRACTION_PROMPT

    # build_extraction_prompt with the same hint returns the same prompt.
    assert prompts.build_extraction_prompt("deliveroo") == prompts.DELIVEROO_V2_EXTRACTION_PROMPT
