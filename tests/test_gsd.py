"""Metric engine (GSD) math vs hand-computed values."""

import pytest

from voxae.data.gsd import fits_clearance, gsd_nadir, object_size_m


def test_gsd_nadir_hand_computed():
    # 100 m altitude, 10 mm focal, 10 mm sensor width, 1000 px wide image:
    # gsd = (100 * 10) / (10 * 1000) = 0.1 m/px
    assert gsd_nadir(100, 10, 10, 1000) == pytest.approx(0.1)


def test_gsd_scales_linearly_with_altitude():
    g1 = gsd_nadir(50, 8.8, 13.2, 4000)
    g2 = gsd_nadir(100, 8.8, 13.2, 4000)
    assert g2 == pytest.approx(2 * g1)


def test_gsd_rejects_nonpositive():
    with pytest.raises(ValueError):
        gsd_nadir(0, 10, 10, 1000)
    with pytest.raises(ValueError):
        gsd_nadir(100, 10, 10, 0)


def test_object_size():
    # 50 px at 0.1 m/px = 5 m
    assert object_size_m(50, 0.1) == pytest.approx(5.0)


def test_fits_clearance_boundary():
    # 25 px gap at 0.1 m/px = 2.5 m — exactly a 2.5 m fire-truck width
    assert fits_clearance(25, 0.1, 2.5) is True
    assert fits_clearance(24, 0.1, 2.5) is False
