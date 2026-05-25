"""Pydantic v2 schemas for request and response payloads."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Platform = Literal[
    "uber_eats",
    "deliveroo",
    "stuart",
    "just_eat",
    "rideup",
    "unknown",
]

Currency = Literal["GBP", "USD", "EUR", "unknown"]

Confidence = Literal["high", "medium", "low"]


class Base64ImageRequest(BaseModel):
    """JSON payload for the ``/extract/base64`` endpoint."""

    image: str = Field(
        ...,
        description=(
            "Base64-encoded image bytes. May include the standard "
            "``data:image/...;base64,`` prefix — it will be stripped."
        ),
        min_length=16,
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional caller hint, e.g. 'uber_eats' or 'deliveroo'.",
        max_length=64,
    )

    @field_validator("image")
    @classmethod
    def _strip_data_url_prefix(cls, value: str) -> str:
        """Strip the ``data:image/...;base64,`` prefix when present."""
        if "," in value and value.lstrip().lower().startswith("data:"):
            return value.split(",", 1)[1].strip()
        return value.strip()


class ExtractionResult(BaseModel):
    """Structured data extracted from a delivery-app screenshot."""

    pay: Optional[float] = Field(
        default=None, description="Monetary payment amount, e.g. 12.50."
    )
    currency: Currency = Field(
        default="unknown",
        description="ISO currency code detected from the screenshot.",
    )
    miles: Optional[float] = Field(
        default=None, description="Total distance in miles."
    )
    minutes: Optional[int] = Field(
        default=None, description="Estimated total trip time in minutes."
    )
    orders: Optional[int] = Field(
        default=None, description="Number of orders or stops in the batch."
    )
    platform: Platform = Field(
        default="unknown",
        description="Delivery platform detected from UI cues.",
    )
    confidence: Confidence = Field(
        default="low",
        description="Overall confidence in the extraction.",
    )
    notes: str = Field(
        default="",
        description="Free-text notes about edge cases or ambiguities.",
    )
    raw_text: str = Field(
        default="",
        description="Raw text detected by the vision model in the image.",
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["ok"] = "ok"
    model: str
    version: str


class RootResponse(BaseModel):
    """Response body for ``GET /``."""

    name: str
    version: str
    description: str
    endpoints: dict[str, str]


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for non-2xx responses."""

    error: str = Field(..., description="Short error identifier.")
    detail: str = Field(..., description="Human-readable explanation.")
    status_code: int = Field(..., description="HTTP status code.")
