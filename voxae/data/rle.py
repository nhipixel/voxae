"""COCO-compatible RLE encode/decode for binary masks.

pycocotools is imported lazily (part of the ``data`` extra) so importing this
module never pulls compiled dependencies.
"""

from __future__ import annotations

import numpy as np

from voxae.data.schemas import MaskRLE


class RLEError(RuntimeError):
    """Raised when RLE support is unavailable or inputs are invalid."""


def _mask_utils():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as e:  # pragma: no cover - environment-specific
        raise RLEError(
            f"RLE support requires the 'data' extra: uv sync --extra data (import failed: {e})"
        ) from e
    return mask_utils


def encode(mask: np.ndarray) -> MaskRLE:
    """Encode an HxW boolean mask to compressed COCO RLE."""
    if mask.ndim != 2:
        raise RLEError(f"expected HxW mask, got shape {mask.shape}")
    m = _mask_utils()
    rle = m.encode(np.asfortranarray(mask.astype(np.uint8)))
    return MaskRLE(size=[int(s) for s in rle["size"]], counts=rle["counts"].decode("ascii"))


def decode(rle: MaskRLE) -> np.ndarray:
    """Decode COCO RLE back to an HxW boolean mask."""
    m = _mask_utils()
    raw = {"size": list(rle.size), "counts": rle.counts.encode("ascii")}
    return m.decode(raw).astype(bool)


def area_pct(rle: MaskRLE) -> float:
    """Mask area as a percentage of the image."""
    m = _mask_utils()
    raw = {"size": list(rle.size), "counts": rle.counts.encode("ascii")}
    h, w = rle.size
    return float(m.area(raw)) * 100.0 / float(h * w)
