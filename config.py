"""Application configuration loaded from environment variables.

All settings are loaded once at import time from environment variables (or a
local ``.env`` file if present). The ``settings`` singleton is imported by the
rest of the application — do not read ``os.environ`` directly elsewhere.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env early so BaseSettings picks values up consistently.
load_dotenv()


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # ``model_name`` clashes with Pydantic's default "model_" protected
        # namespace; rename the guard so we keep the natural field name.
        protected_namespaces=("settings_",),
    )

    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Google Gemini API key — required for live extraction.",
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "gemini-2.5-flash"),
        description="Gemini vision model identifier.",
    )
    max_file_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "10")),
        description="Maximum upload size in megabytes.",
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "7860")),
        description="HTTP port — defaults to 7860 for HuggingFace Spaces.",
    )
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "production"),
        description="Deployment environment: production | staging | development.",
    )

    @property
    def max_file_size_bytes(self) -> int:
        """Return the maximum upload size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


settings = get_settings()
