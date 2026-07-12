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
