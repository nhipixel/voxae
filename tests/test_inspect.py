"""Decode-inspection rendering (offline, synthetic masks)."""

import numpy as np
from PIL import Image

from voxae.data.inspect import class_color, render_overlay


def test_class_color_is_deterministic_and_in_palette():
    assert class_color("road") == class_color("road")
    c = class_color("building")
    assert len(c) == 3 and all(0 <= v <= 255 for v in c)


def test_render_overlay_matches_downscaled_size_and_colors_classes():
    image = Image.new("RGB", (200, 100), (10, 10, 10))
    road = np.zeros((100, 200), dtype=bool)
    road[:, :100] = True
    known = road
    overlay = render_overlay(image, {"road": road}, ~known, max_px=200)

    assert overlay.size == (200, 100)
    arr = np.asarray(overlay)
    # Left half (road) blends toward the class color; right half (unknown)
    # blends toward magenta — the two halves must differ.
    assert not np.array_equal(arr[:, :100], arr[:, 100:])


def test_render_overlay_handles_no_unknown():
    image = Image.new("RGB", (60, 60), (0, 0, 0))
    full = np.ones((60, 60), dtype=bool)
    overlay = render_overlay(image, {"grass": full}, np.zeros((60, 60), dtype=bool))
    assert overlay.size == (60, 60)
