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


def gsd_from_35mm_equiv(altitude_m: float, focal_35mm: float, image_width_px: int) -> float:
    """GSD using the 35mm-equivalent focal length (36 mm sensor width)."""
    return gsd_nadir(altitude_m, focal_35mm, 36.0, image_width_px)


# EXIF tag ids (JEITA standard): focal length and its 35mm equivalent.
_TAG_FOCAL_LENGTH = 37386
_TAG_FOCAL_35MM = 41989


def _xmp_relative_altitude(path) -> float | None:
    """Extract the drone's above-ground altitude from XMP metadata, if present.

    Consumer drones (e.g. DJI) store AGL altitude as ``RelativeAltitude`` in
    XMP, not EXIF; EXIF GPSAltitude is above sea level and unusable for GSD.
    A bounded byte scan is enough — XMP sits in the file header.
    """
    import re

    try:
        head = path.open("rb").read(131072)
    except OSError:
        return None
    m = re.search(rb'RelativeAltitude\s*=?\s*"?\s*([+-]?\d+(?:\.\d+)?)', head)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if 0 < value < 1000 else None


def estimate_gsd(image_path, altitude_m: float | None = None) -> GSDEstimate | None:
    """Estimate nadir GSD for an image from EXIF/XMP metadata.

    Altitude precedence: explicit ``altitude_m`` argument, then XMP
    RelativeAltitude (AGL). Focal length comes from EXIF's 35mm-equivalent
    tag. Returns None when either quantity is unavailable — callers must not
    generate metric queries without an estimate.
    """
    from pathlib import Path

    from PIL import Image

    path = Path(image_path)
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            width_px = img.size[0]
    except OSError:
        return None

    focal_35 = exif.get(_TAG_FOCAL_35MM)
    if not focal_35:
        # Plain focal length without sensor width cannot give scale.
        return None
    focal_35 = float(focal_35)

    if altitude_m is not None:
        alt, alt_source = float(altitude_m), "explicit altitude"
        confidence = GSDConfidence.high
    else:
        xmp_alt = _xmp_relative_altitude(path)
        if xmp_alt is None:
            return None
        alt, alt_source = xmp_alt, "XMP RelativeAltitude"
        confidence = GSDConfidence.high

    try:
        gsd = gsd_from_35mm_equiv(alt, focal_35, width_px)
    except ValueError:
        return None
    return GSDEstimate(
        meters_per_px=gsd,
        confidence=confidence,
        source=f"{alt_source} + EXIF 35mm-equivalent focal",
    )
