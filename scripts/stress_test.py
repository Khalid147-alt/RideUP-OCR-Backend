"""Live stress test runner — exercises every edge case end-to-end.

Run with the dev server running on 127.0.0.1:8001:
    .\.venv\Scripts\python.exe scripts\stress_test.py

This is an integration test against a *real* server (not the mocked pytest
suite). It validates:

- /health and / metadata endpoints
- Rejection paths (PDF, oversize, empty, bad base64)
- Both /extract and /extract/base64 routes
- Real Gemini extraction on a synthetic image (no Gemini key → 502 expected)
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont

BASE_URL = os.environ.get("RIDEUP_TEST_URL", "http://127.0.0.1:8001")

# ANSI colours
G = "\033[92m"  # green
R = "\033[91m"  # red
Y = "\033[93m"  # yellow
B = "\033[94m"  # blue
D = "\033[2m"   # dim
N = "\033[0m"   # reset


_passed = 0
_failed = 0
_skipped = 0


def case(name: str) -> None:
    print(f"\n{B}-- {name}{N}")


def ok(msg: str) -> None:
    global _passed
    _passed += 1
    print(f"  {G}[OK]{N} {msg}")


def fail(msg: str) -> None:
    global _failed
    _failed += 1
    print(f"  {R}[FAIL]{N} {msg}")


def skip(msg: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  {Y}[SKIP]{N} {msg}")


def dim(msg: str) -> None:
    print(f"  {D}{msg}{N}")


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------


def _png_with_text(text_lines: list[str], width: int = 800, height: int = 600) -> bytes:
    """Create a dark-mode PNG with the given lines of white text — mimics a
    delivery-app screenshot well enough for the model to extract from."""
    img = Image.new("RGB", (width, height), color=(15, 15, 15))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    y = 60
    for line in text_lines:
        draw.text((50, y), line, fill=(245, 245, 245), font=font)
        y += 60
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid_png(width: int = 50, height: int = 50) -> bytes:
    """Return a tiny solid-colour PNG that should fail quality checks."""
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def run_health(client: httpx.Client) -> dict[str, Any]:
    case("GET /health")
    r = client.get("/health")
    if r.status_code == 200 and r.json().get("status") == "ok":
        ok(f"status=200 body={r.json()}")
    else:
        fail(f"unexpected: {r.status_code} {r.text}")
    return r.json() if r.status_code == 200 else {}


def run_root(client: httpx.Client) -> None:
    case("GET /")
    r = client.get("/")
    if r.status_code == 200 and "endpoints" in r.json():
        ok(f"metadata returned, {len(r.json()['endpoints'])} endpoints")
    else:
        fail(f"unexpected: {r.status_code}")


def run_rejection_pdf(client: httpx.Client) -> None:
    case("POST /extract — rejects PDF upload (expect 400)")
    files = {"image": ("trip.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = client.post("/extract", files=files)
    if r.status_code == 400 and r.json().get("error") == "bad_request":
        ok(f"correctly rejected: {r.json()['detail']}")
    else:
        fail(f"expected 400 bad_request, got {r.status_code}: {r.text}")


def run_rejection_oversize(client: httpx.Client) -> None:
    case("POST /extract — rejects 11 MB upload (expect 400)")
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
    files = {"image": ("big.png", oversized, "image/png")}
    r = client.post("/extract", files=files)
    if r.status_code == 400 and "too large" in r.json().get("detail", "").lower():
        ok(f"correctly rejected: {r.json()['detail']}")
    else:
        fail(f"expected 400, got {r.status_code}: {r.text[:200]}")


def run_rejection_empty(client: httpx.Client) -> None:
    case("POST /extract — rejects empty file (expect 400)")
    files = {"image": ("empty.png", b"", "image/png")}
    r = client.post("/extract", files=files)
    if r.status_code == 400:
        ok(f"correctly rejected: {r.json().get('detail')}")
    else:
        fail(f"expected 400, got {r.status_code}: {r.text}")


def run_rejection_bad_b64(client: httpx.Client) -> None:
    case("POST /extract/base64 — rejects malformed base64 (expect 400)")
    r = client.post("/extract/base64", json={"image": "!!!not-base64!!!"})
    if r.status_code == 400:
        ok(f"correctly rejected: {r.json().get('detail')}")
    else:
        fail(f"expected 400, got {r.status_code}: {r.text}")


def run_synthetic_uber(client: httpx.Client, has_real_key: bool) -> None:
    case("POST /extract — synthetic Uber Eats offer card")
    png = _png_with_text([
        "Uber Eats",
        "£12.04",
        "49 min (7.2 mi) total",
        "Delivery (2)",
    ])
    files = {"image": ("offer.png", png, "image/png")}
    t0 = time.perf_counter()
    r = client.post("/extract", files=files, params={"hint": "uber_eats"})
    elapsed = (time.perf_counter() - t0) * 1000
    if not has_real_key:
        if r.status_code == 502:
            ok(f"502 returned as expected (no live Gemini key) in {elapsed:.0f} ms")
        else:
            fail(f"expected 502 without key, got {r.status_code}: {r.text[:200]}")
        return
    if r.status_code == 200:
        body = r.json()
        ok(f"200 OK in {elapsed:.0f} ms")
        dim(f"pay={body.get('pay')} miles={body.get('miles')} "
            f"minutes={body.get('minutes')} orders={body.get('orders')} "
            f"platform={body.get('platform')} confidence={body.get('confidence')}")
        if body.get("confidence") in {"high", "medium"}:
            ok(f"confidence={body['confidence']} (acceptable)")
        else:
            fail(f"confidence={body['confidence']} (synthetic should extract clean fields)")
    else:
        fail(f"expected 200, got {r.status_code}: {r.text[:200]}")


def run_base64_endpoint(client: httpx.Client, has_real_key: bool) -> None:
    case("POST /extract/base64 — synthetic image via base64")
    png = _png_with_text([
        "Deliveroo",
        "£6.50",
        "18 min (2.1 mi)",
        "1 delivery",
    ])
    b64 = base64.b64encode(png).decode("ascii")
    r = client.post(
        "/extract/base64",
        json={"image": f"data:image/png;base64,{b64}", "hint": "deliveroo"},
    )
    if not has_real_key:
        if r.status_code == 502:
            ok("502 expected without live Gemini key (data-URL prefix accepted)")
        else:
            fail(f"expected 502, got {r.status_code}: {r.text[:200]}")
        return
    if r.status_code == 200:
        body = r.json()
        ok(f"200 OK pay={body.get('pay')} platform={body.get('platform')}")
        dim(str(body))
    else:
        fail(f"expected 200, got {r.status_code}: {r.text[:200]}")


def run_poor_quality_image(client: httpx.Client, has_real_key: bool) -> None:
    case("POST /extract — tiny low-res image (should flag low confidence)")
    tiny = _solid_png(50, 50)
    files = {"image": ("tiny.png", tiny, "image/png")}
    r = client.post("/extract", files=files)
    if not has_real_key:
        if r.status_code == 502:
            ok("502 (no live key) — quality cap would apply downstream")
        else:
            fail(f"got {r.status_code}: {r.text[:200]}")
        return
    if r.status_code == 200:
        body = r.json()
        if body["confidence"] == "low":
            ok(f"correctly capped to low confidence; notes={body['notes']!r}")
        else:
            fail(f"expected confidence=low for tiny image, got {body['confidence']}")
    else:
        fail(f"got {r.status_code}: {r.text[:200]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"{B}=================================================={N}")
    print(f"{B}  RideUP OCR Backend - live stress test{N}")
    print(f"{B}=================================================={N}")
    print(f"Target: {BASE_URL}")

    try:
        with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
            health = run_health(client)
            model = health.get("model", "?")
            print(f"  {D}server reports model={model}{N}")

            # Detect whether a real Gemini key is configured by attempting
            # a tiny call and checking the response code pattern.
            has_real_key = False
            probe = client.post(
                "/extract",
                files={"image": ("probe.png", _png_with_text(["test"]), "image/png")},
            )
            if probe.status_code == 200:
                has_real_key = True
                dim("Live Gemini key detected — running full integration suite")
            elif probe.status_code == 502:
                dim("No live Gemini key — running validation-only suite")
            else:
                dim(f"Probe returned {probe.status_code}; treating as no-key")

            run_root(client)
            run_rejection_pdf(client)
            run_rejection_oversize(client)
            run_rejection_empty(client)
            run_rejection_bad_b64(client)
            run_synthetic_uber(client, has_real_key)
            run_base64_endpoint(client, has_real_key)
            run_poor_quality_image(client, has_real_key)

    except httpx.ConnectError:
        print(f"\n{R}Could not connect to {BASE_URL}.{N}")
        print(f"{Y}Make sure the dev server is running:{N}")
        print(f"  uvicorn main:app --host 127.0.0.1 --port 8001")
        return 2

    total = _passed + _failed + _skipped
    print(f"\n{B}--- Summary ---{N}")
    print(f"  {G}passed:  {_passed}{N}")
    print(f"  {R}failed:  {_failed}{N}")
    print(f"  {Y}skipped: {_skipped}{N}")
    print(f"  total:   {total}\n")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
