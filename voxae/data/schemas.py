"""Pydantic schemas shared across the pipeline.

Coordinate conventions
----------------------
- The VLM is asked for coordinates in a normalized 0-1000 integer space
  (robust across model input resizing); ``to_pixels`` maps them to the
  actual image size.
- Pixel-space types use ``float`` x/y with origin at the top-left.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

NORM_MAX = 1000


class QueryFamily(StrEnum):
    referring = "referring"
    affordance = "affordance"
    metric = "metric"


class BBoxNorm(BaseModel):
    """Bounding box in normalized 0-1000 space: [x1, y1, x2, y2]."""

    x1: int = Field(ge=0, le=NORM_MAX)
    y1: int = Field(ge=0, le=NORM_MAX)
    x2: int = Field(ge=0, le=NORM_MAX)
    y2: int = Field(ge=0, le=NORM_MAX)

    @field_validator("x2")
    @classmethod
    def _x_ordered(cls, v: int, info):
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y_ordered(cls, v: int, info):
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError("y2 must be > y1")
        return v

    def to_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        sx, sy = width / NORM_MAX, height / NORM_MAX
        return (self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)


class PointNorm(BaseModel):
    """A point in normalized 0-1000 space."""

    x: int = Field(ge=0, le=NORM_MAX)
    y: int = Field(ge=0, le=NORM_MAX)

    def to_pixels(self, width: int, height: int) -> tuple[float, float]:
        return (self.x * width / NORM_MAX, self.y * height / NORM_MAX)


class GroundingResult(BaseModel):
    """Structured output the VLM must produce for a grounding query."""

    bbox: BBoxNorm
    points: list[PointNorm] = Field(min_length=1, max_length=4)
    rationale: str = ""

    @classmethod
    def json_schema_prompt(cls) -> str:
        """Compact schema description embedded in the VLM prompt."""
        return (
            '{"bbox": {"x1": int, "y1": int, "x2": int, "y2": int}, '
            '"points": [{"x": int, "y": int}, ...], '
            '"rationale": "one short sentence"}'
        )


class ZeroShotTrace(BaseModel):
    """Auditable record of one zero-shot pipeline run."""

    query: str
    model: str
    grounding: GroundingResult
    image_width: int
    image_height: int
    vlm_latency_ms: float
    sam_latency_ms: float
    total_latency_ms: float
