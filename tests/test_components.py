"""Component labeling determinism and descriptors (requires scipy)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from voxae.data.prep_masks import extract_components, grid_cell, label_components


def _two_blob_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:40] = True  # 900 px — the larger blob
    mask[70:90, 70:90] = True  # 400 px
    return mask


def test_labeling_is_area_desc_and_deterministic():
    comps = label_components(_two_blob_mask())
    assert len(comps) == 2
    assert comps[0].sum() == 900
    assert comps[1].sum() == 400
    again = label_components(_two_blob_mask())
    assert all(np.array_equal(a, b) for a, b in zip(comps, again, strict=True))


def test_extract_components_descriptors():
    records = extract_components("building", "building", _two_blob_mask(), gsd_m_per_px=0.5)
    assert [r.comp_id for r in records] == [0, 1]
    big = records[0]
    assert big.bbox_px == [10.0, 10.0, 40.0, 40.0]
    assert big.grid_cell == "top-left"
    # 30 px wide at 0.5 m/px = 15 m; area 900 px * 0.25 m2 = 225 m2
    assert big.width_m == pytest.approx(15.0)
    assert big.area_m2 == pytest.approx(225.0)


def test_extract_components_skips_specks():
    mask = np.zeros((100, 100), dtype=bool)
    mask[0, 0] = True  # 0.01% — below the area floor
    assert extract_components("road", "road", mask, None) == []


def test_grid_cell_names():
    assert grid_cell(5, 5, 90, 90) == "top-left"
    assert grid_cell(45, 45, 90, 90) == "center"
    assert grid_cell(89, 89, 90, 90) == "bottom-right"
