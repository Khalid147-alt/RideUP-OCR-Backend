"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable from inside ``tests/``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure the extractor never tries to instantiate a real client during tests
# that don't already monkeypatch it. Any test that needs a real key must set
# it explicitly.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("ENVIRONMENT", "development")
