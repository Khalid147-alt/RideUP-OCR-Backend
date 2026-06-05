"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the project root importable from inside ``tests/``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure the extractor never tries to instantiate a real client during tests
# that don't already monkeypatch it. Any test that needs a real key must set
# it explicitly.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("ENVIRONMENT", "development")


@pytest.fixture(autouse=True)
def _no_live_geocoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic — no live postcodes.io calls.

    The Deliveroo V2 postcode-mileage feature geocodes via the live
    postcodes.io API. The test suite is designed to require **no network
    access**, so by default we stub the geocoder to return ``None`` (the
    "could not resolve" path). Endpoint tests that don't care about mileage
    estimation therefore behave exactly as before — ``miles`` stays ``None``.

    Tests that specifically exercise estimation override this by patching
    ``get_postcode_coordinates`` (or a higher-level function) themselves; an
    explicit ``patch`` inside the test takes precedence over this autouse
    stub.
    """

    async def _stub(_postcode: str):
        return None

    # ``extractor`` imported the symbol by name, so patch it where it is used.
    monkeypatch.setattr("extractor.get_postcode_coordinates", _stub)
