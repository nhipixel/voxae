"""Mask overlay rendering (PIL-only — no OpenCV dependency in the core)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

HIGHLIGHT = (46, 204, 113)  # green
BOX = (241, 196, 15)  # amber


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    bbox_px: tuple[float, float, float, float] | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a binary mask onto the image; optionally draw the prompt bbox."""
    img = image.convert("RGB")
    w, h = img.size
    if mask.shape != (h, w):
        raise ValueError(f"mask shape {mask.shape} != image (h,w) {(h, w)}")

    color_layer = Image.new("RGB", (w, h), HIGHLIGHT)
    mask_img = Image.fromarray(mask.astype(np.uint8) * int(alpha * 255))
    out = img.copy()
    out.paste(color_layer, (0, 0), mask_img)

    if bbox_px is not None:
        draw = ImageDraw.Draw(out)
        draw.rectangle(bbox_px, outline=BOX, width=3)
    return out
