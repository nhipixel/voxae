"""Mask decoding against the dataset registry (synthetic arrays, no downloads)."""

import numpy as np

from voxae.data.registry import REGISTRY, decode_mask


def test_registry_has_expected_datasets():
    assert set(REGISTRY) == {"uavid", "sdd", "vdd"}
    for spec in REGISTRY.values():
        assert set(spec.categories) == {cls for cls in spec.class_table.values()}, (
            f"{spec.name}: every class needs a category"
        )


def test_rgb_decode_splits_classes_and_buckets_unknown():
    spec = REGISTRY["uavid"]
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0, :] = (128, 0, 0)  # building
    mask[1, :] = (128, 64, 128)  # road
    mask[2, :] = (1, 2, 3)  # not in palette -> unknown
    # row 3 stays (0,0,0) = clutter

    classes, unknown_pct = decode_mask(spec, mask)
    assert classes["building"].sum() == 4
    assert classes["road"].sum() == 4
    assert classes["clutter"].sum() == 4
    assert unknown_pct == 25.0


def test_index_decode():
    spec = REGISTRY["vdd"]
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, :] = 2  # road
    mask[1, :] = 6  # water
    classes, unknown_pct = decode_mask(spec, mask)
    assert classes["road"].sum() == 3
    assert classes["water"].sum() == 3
    assert unknown_pct == 0.0


def test_index_decode_reports_unknown_ids():
    spec = REGISTRY["vdd"]
    mask = np.full((2, 2), 250, dtype=np.uint8)
    classes, unknown_pct = decode_mask(spec, mask)
    assert classes == {}
    assert unknown_pct == 100.0
