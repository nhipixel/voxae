"""Pydantic schemas shared across the pipeline.

Coordinate conventions
----------------------
- The VLM is asked for coordinates in a normalized 0-1000 integer space
  (robust across model input resizing); ``to_pixels`` maps them to the
  actual image size.
- Pixel-space types use ``float`` x/y with origin at the top-left.

Dataset schemas
---------------
Query generation is split between language and geometry: the LLM authors a
``QuerySpec`` (text + symbolic ``TargetSpec``), and the target mask is always
materialized programmatically from dataset masks. The LLM never draws pixels,
so every ground-truth mask is exactly reproducible from its spec.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

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


# --- dataset schemas ---


class MaskRLE(BaseModel):
    """COCO-style compressed run-length encoding of a binary mask."""

    size: list[int] = Field(min_length=2, max_length=2)  # [height, width]
    counts: str


class ComponentRecord(BaseModel):
    """One connected component of a class mask, with spatial descriptors.

    ``comp_id`` is the component's rank under deterministic labeling
    (sorted by area descending, ties broken by bbox position), so specs can
    reference components without storing per-component masks.
    """

    comp_id: int = Field(ge=0)
    cls: str
    category: str
    area_px: int = Field(gt=0)
    area_pct: float = Field(gt=0, le=100)
    bbox_px: list[float] = Field(min_length=4, max_length=4)  # x1, y1, x2, y2
    centroid: list[float] = Field(min_length=2, max_length=2)  # x, y
    grid_cell: str  # 3x3 position: "top-left" .. "bottom-right"
    width_m: float | None = None
    height_m: float | None = None
    area_m2: float | None = None


class ImageRecord(BaseModel):
    """Unified per-image record built from a source dataset's semantic mask."""

    image_id: str
    dataset: str
    rel_path: str
    mask_rel_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    gsd_m_per_px: float | None = None
    gsd_confidence: str = "low"
    gsd_source: str = ""
    class_pixel_pct: dict[str, float] = Field(default_factory=dict)
    unknown_pixel_pct: float = 0.0
    components: list[ComponentRecord] = Field(default_factory=list)
    class_rles: dict[str, MaskRLE] = Field(default_factory=dict)


class NearExclusion(BaseModel):
    """Remove target pixels within ``radius_px`` of another class (via dilation)."""

    cls: str
    radius_px: int = Field(gt=0, le=512)


class ClassUnionTarget(BaseModel):
    """Union of one or more class masks, with optional spatial constraints."""

    type: Literal["class_union"] = "class_union"
    classes: list[str] = Field(min_length=1, max_length=6)
    exclude_near: NearExclusion | None = None
    min_component_area_pct: float | None = Field(default=None, ge=0, le=100)


class ComponentsTarget(BaseModel):
    """Specific connected components of one class, by deterministic comp_id."""

    type: Literal["components"] = "components"
    cls: str
    comp_ids: list[int] = Field(min_length=1, max_length=16)


class MetricFilterTarget(BaseModel):
    """Components of a class whose real-world dimension satisfies a predicate.

    Requires the image to have a GSD estimate; ground truth is computed, not
    authored, so metric answers are exactly verifiable.
    """

    type: Literal["metric_filter"] = "metric_filter"
    cls: str
    attr: Literal["width_m", "height_m", "area_m2"]
    op: Literal[">=", "<="]
    value: float = Field(gt=0)


TargetSpec = Annotated[
    ClassUnionTarget | ComponentsTarget | MetricFilterTarget,
    Field(discriminator="type"),
]


class QuerySpec(BaseModel):
    """LLM-authored query: natural language plus a symbolic target."""

    family: QueryFamily
    text: str = Field(min_length=8, max_length=300)
    target: TargetSpec


class GenMeta(BaseModel):
    """Provenance of a generated sample (model, prompt version, seed, cache)."""

    model: str
    prompt_version: str
    seed: int
    cached: bool = False


class QuerySample(BaseModel):
    """One dataset sample: query text, spec, and the materialized mask."""

    sample_id: str
    dataset: str
    image_id: str
    rel_path: str
    family: QueryFamily
    text: str
    target: TargetSpec
    rle: MaskRLE
    area_pct: float = Field(ge=0, le=100)
    gsd_m_per_px: float | None = None
    gen: GenMeta
    split: str | None = None
