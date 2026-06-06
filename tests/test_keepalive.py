"""Tests for the keepalive self-ping scheduler.

The keepalive system pings the Space's own /health endpoint every 25 minutes
so HuggingFace doesn't mark the free-tier Space inactive. These tests cover:

1. ``GET /keepalive/status`` returns 200 with a ``scheduler_running`` field.
2. ``ping_self()`` swallows network errors — a failed ping never raises.

No live network access is required: the HTTP client in ``ping_self`` is
mocked to raise, and the status endpoint reports in-process scheduler state.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import app, ping_self


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_keepalive_status_returns_200_with_scheduler_running(
    client: AsyncClient,
) -> None:
    """GET /keepalive/status returns 200 and a boolean scheduler_running flag."""
    response = await client.get("/keepalive/status")

    assert response.status_code == 200
    body = response.json()
    assert "scheduler_running" in body
    assert isinstance(body["scheduler_running"], bool)
    # ``jobs`` is always present (possibly empty) and is a list.
    assert "jobs" in body
    assert isinstance(body["jobs"], list)


@pytest.mark.asyncio
async def test_ping_self_handles_network_error_gracefully() -> None:
    """ping_self() must swallow any httpx error and never raise."""

    class _BoomClient:
        """Stand-in AsyncClient whose context manager / get always fails."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("simulated network failure")

    with patch.object(main.httpx, "AsyncClient", _BoomClient):
        # Must complete without propagating the exception.
        result = await ping_self()

    assert result is None
