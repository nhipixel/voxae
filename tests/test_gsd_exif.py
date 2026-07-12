"""EXIF/XMP-based GSD estimation on synthetic files."""

import pytest
from PIL import Image

from voxae.data.gsd import GSDConfidence, estimate_gsd, gsd_from_35mm_equiv

# EXIF tag id for the 35mm-equivalent focal length.
TAG_FOCAL_35MM = 41989


def _write_jpeg(path, focal_35mm=None, width=1000):
    img = Image.new("RGB", (width, 100))
    exif = Image.Exif()
    if focal_35mm is not None:
        exif[TAG_FOCAL_35MM] = focal_35mm
    img.save(path, format="JPEG", exif=exif)


def test_gsd_from_35mm_math():
    # 100 m altitude, 36 mm-equivalent focal, 1000 px wide: 100*36/(36*1000) = 0.1
    assert gsd_from_35mm_equiv(100, 36, 1000) == pytest.approx(0.1)


def test_explicit_altitude_gives_high_confidence(tmp_path):
    p = tmp_path / "a.jpg"
    _write_jpeg(p, focal_35mm=36)
    est = estimate_gsd(p, altitude_m=100)
    assert est is not None
    assert est.confidence == GSDConfidence.high
    assert est.meters_per_px == pytest.approx(0.1)
    assert "explicit altitude" in est.source


def test_no_focal_returns_none(tmp_path):
    p = tmp_path / "b.jpg"
    _write_jpeg(p, focal_35mm=None)
    assert estimate_gsd(p, altitude_m=100) is None


def test_no_altitude_returns_none(tmp_path):
    p = tmp_path / "c.jpg"
    _write_jpeg(p, focal_35mm=36)
    assert estimate_gsd(p) is None


def test_xmp_relative_altitude_is_used(tmp_path):
    p = tmp_path / "d.jpg"
    _write_jpeg(p, focal_35mm=36)
    # Emulate DJI-style XMP by appending it into the file header region read
    # by the scanner (JPEGs tolerate trailing bytes).
    data = p.read_bytes() + b'<x:xmpmeta drone:RelativeAltitude="+50.0"/>'
    p.write_bytes(data)
    est = estimate_gsd(p)
    assert est is not None
    assert est.meters_per_px == pytest.approx(0.05)
    assert "XMP" in est.source
