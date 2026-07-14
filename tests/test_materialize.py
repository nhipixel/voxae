"""TargetSpec materialization on synthetic class masks (requires scipy)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from voxae.data.materialize import MaterializeError, dilate, materialize
from voxae.data.schemas import (
    ClassUnionTarget,
    ComponentsTarget,
    MetricFilterTarget,
    NearExclusion,
)


@pytest.fixture()
def class_masks():
    road = np.zeros((100, 100), dtype=bool)
    road[40:60, :] = True
    building = np.zeros((100, 100), dtype=bool)
    building[0:20, 0:30] = True  # comp 0 (600 px)
    building[80:95, 80:95] = True  # comp 1 (225 px)
    tree = np.zeros((100, 100), dtype=bool)
    tree[35:40, 0:100] = True  # strip adjacent to the road
    return {"road": road, "building": building, "tree": tree}


def test_class_union(class_masks):
    mask = materialize(ClassUnionTarget(classes=["road", "building"]), class_masks)
    assert mask.sum() == class_masks["road"].sum() + class_masks["building"].sum()


def test_class_union_missing_class_raises(class_masks):
    with pytest.raises(MaterializeError, match="not present"):
        materialize(ClassUnionTarget(classes=["water"]), class_masks)


def test_exclude_near_removes_boundary(class_masks):
    plain = materialize(ClassUnionTarget(classes=["road"]), class_masks)
    excluded = materialize(
        ClassUnionTarget(classes=["road"], exclude_near=NearExclusion(cls="tree", radius_px=5)),
        class_masks,
    )
    assert excluded.sum() < plain.sum()
    assert not (excluded & dilate(class_masks["tree"], 5)).any()


def test_components_target_selects_by_rank(class_masks):
    mask = materialize(ComponentsTarget(cls="building", comp_ids=[1]), class_masks)
    assert mask.sum() == 225
    ys, xs = np.nonzero(mask)
    assert ys.min() >= 80 and xs.min() >= 80


def test_components_target_out_of_range(class_masks):
    with pytest.raises(MaterializeError, match="out of range"):
        materialize(ComponentsTarget(cls="building", comp_ids=[5]), class_masks)


def test_metric_filter_width(class_masks):
    # gsd 0.5 m/px: comp0 width 30px -> 15 m, comp1 width 15px -> 7.5 m
    mask = materialize(
        MetricFilterTarget(cls="building", attr="width_m", op=">=", value=10.0),
        class_masks,
        gsd_m_per_px=0.5,
    )
    assert mask.sum() == 600


def test_metric_filter_requires_gsd(class_masks):
    with pytest.raises(MaterializeError, match="GSD"):
        materialize(
            MetricFilterTarget(cls="building", attr="width_m", op=">=", value=10.0),
            class_masks,
            gsd_m_per_px=None,
        )


def test_metric_filter_no_match_raises(class_masks):
    with pytest.raises(MaterializeError, match="predicate"):
        materialize(
            MetricFilterTarget(cls="building", attr="width_m", op=">=", value=1000.0),
            class_masks,
            gsd_m_per_px=0.5,
        )


def test_dilate_zero_radius_is_noop():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 5] = True
    assert np.array_equal(dilate(mask, 0), mask)


def test_dilate_grows_by_exactly_radius():
    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 20] = True  # single center pixel
    grown = dilate(mask, radius_px=3)
    # a square of side (2*3+1)=7 centered on (20,20)
    expected = np.zeros((41, 41), dtype=bool)
    expected[17:24, 17:24] = True
    assert np.array_equal(grown, expected)


def test_dilate_large_radius_on_large_image_is_fast():
    # Regression test for the pathological case that made a single call take
    # CPU-minutes: a large radius (near the schema's 512px cap) on a
    # full-resolution aerial frame. scipy's separable maximum_filter must
    # stay well under a second regardless of kernel size; a naive per-pixel
    # filter (e.g. PIL's MaxFilter) would take minutes here.
    import time

    mask = np.zeros((2160, 3840), dtype=bool)
    mask[1000:1100, 1800:1900] = True
    t0 = time.perf_counter()
    result = dilate(mask, radius_px=400)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, (
        f"dilate took {elapsed:.2f}s — expected O(1)-per-kernel-size, not O(kernel_area)"
    )
    assert result.sum() > mask.sum()
