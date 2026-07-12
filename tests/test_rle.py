"""RLE roundtrip (requires the data extra)."""

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from voxae.data import rle


def test_roundtrip_preserves_mask():
    mask = np.zeros((16, 12), dtype=bool)
    mask[3:9, 2:7] = True
    encoded = rle.encode(mask)
    assert encoded.size == [16, 12]
    assert np.array_equal(rle.decode(encoded), mask)


def test_area_pct():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True
    assert rle.area_pct(rle.encode(mask)) == pytest.approx(50.0)


def test_encode_rejects_non_2d():
    with pytest.raises(rle.RLEError, match="HxW"):
        rle.encode(np.zeros((2, 2, 3), dtype=bool))
