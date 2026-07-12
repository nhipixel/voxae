"""gIoU / cIoU vs hand-computed values."""

import numpy as np
import pytest

from voxae.eval.metrics import ciou, giou, iou


def _mask(h, w, ys, xs):
    m = np.zeros((h, w), dtype=bool)
    m[ys, xs] = True
    return m


def test_iou_hand_computed():
    # pred: left half; gt: top half of a 2x2 -> inter=1, union=3
    pred = np.array([[1, 0], [1, 0]], dtype=bool)
    gt = np.array([[1, 1], [0, 0]], dtype=bool)
    assert iou(pred, gt) == pytest.approx(1 / 3)


def test_iou_identical_is_one():
    m = _mask(4, 4, slice(0, 2), slice(0, 2))
    assert iou(m, m) == 1.0


def test_iou_disjoint_is_zero():
    a = _mask(4, 4, slice(0, 2), slice(0, 2))
    b = _mask(4, 4, slice(2, 4), slice(2, 4))
    assert iou(a, b) == 0.0


def test_iou_both_empty_is_one():
    a = np.zeros((4, 4), dtype=bool)
    assert iou(a, a) == 1.0


def test_iou_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        iou(np.zeros((2, 2), dtype=bool), np.zeros((3, 3), dtype=bool))


def test_giou_is_mean_of_ious():
    a = _mask(4, 4, slice(0, 2), slice(0, 2))
    b = _mask(4, 4, slice(2, 4), slice(2, 4))
    # pair1 identical (IoU 1.0), pair2 disjoint (IoU 0.0) -> mean 0.5
    assert giou([a, a], [a, a]) == 1.0
    assert giou([a, b], [a, a]) == pytest.approx(0.5)


def test_ciou_weights_by_area():
    big_pred = _mask(10, 10, slice(0, 10), slice(0, 10))  # 100 px, matches gt
    small_pred = _mask(10, 10, slice(0, 1), slice(0, 1))  # 1 px, gt disjoint 1 px
    small_gt = _mask(10, 10, slice(9, 10), slice(9, 10))
    # cumulative: inter=100, union=100+2 -> 100/102
    assert ciou([big_pred, small_pred], [big_pred, small_gt]) == pytest.approx(100 / 102)


def test_giou_empty_lists_raise():
    with pytest.raises(ValueError):
        giou([], [])
