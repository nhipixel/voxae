"""Ground-sample-distance (GSD) math — the metric engine.

GSD is the real-world size of one pixel on the ground; it turns a
segmentation mask into meters.

For a nadir (straight-down) image with a pinhole camera:

    gsd [m/px] = (altitude_m * sensor_width_m) / (focal_length_m * image_width_px)

Confidence tiers (metric queries are only generated at HIGH confidence):
- HIGH:   nadir capture, altitude + camera intrinsics known (e.g., EXIF).
- MEDIUM: altitude known, intrinsics assumed from camera model defaults.
- LOW:    everything else — metric queries are NOT generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GSDConfidence(StrEnum):
    high = "high"
    medium = "medium"
    low = "low"


@dataclass(frozen=True)
class GSDEstimate:
    meters_per_px: float
    confidence: GSDConfidence
    source: str  # human-readable provenance, e.g. "EXIF altitude + camera spec"


def gsd_nadir(
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float:
    """GSD in meters/pixel for a nadir shot with a pinhole camera model."""
    if min(altitude_m, focal_length_mm, sensor_width_mm) <= 0 or image_width_px <= 0:
        raise ValueError("all camera parameters must be positive")
    return (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)


def object_size_m(pixel_extent: float, gsd_m_per_px: float) -> float:
    """Real-world size of an image extent (e.g., mask width in px) in meters."""
    if pixel_extent < 0 or gsd_m_per_px <= 0:
        raise ValueError("pixel_extent must be >= 0 and gsd positive")
    return pixel_extent * gsd_m_per_px


def fits_clearance(gap_px: float, gsd_m_per_px: float, required_m: float) -> bool:
    """Does a pixel gap correspond to at least ``required_m`` meters of clearance?"""
    return object_size_m(gap_px, gsd_m_per_px) >= required_m
