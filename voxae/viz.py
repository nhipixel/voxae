"""Mask overlay rendering (PIL-only — no OpenCV dependency in the core)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

HIGHLIGHT = (46, 204, 113)  # green
BOX = (241, 196, 15)  # amber


def overlay_heatmap(
    image: Image.Image,
    probs: np.ndarray,
    threshold: float = 0.5,
    max_alpha: float = 0.7,
) -> Image.Image:
    """Blend a continuous confidence field, opacity following probability.

    A binary mask hides that the model outputs a field and a threshold was
    chosen for it. Confident pixels read solid, uncertain ones stay faint, and
    the region below the threshold is dimmed rather than discarded.
    """
    img = image.convert("RGB")
    h, w = probs.shape
    if (w, h) != img.size:
        raise ValueError(f"probs shape {probs.shape} != image (h,w) {(img.height, img.width)}")

    confident = np.clip(probs, 0.0, 1.0) ** 2
    weight = np.where(probs >= threshold, np.maximum(confident, 0.45), confident * 0.45)
    alpha = Image.fromarray((weight * max_alpha * 255).astype(np.uint8))

    out = img.copy()
    out.paste(Image.new("RGB", (w, h), HIGHLIGHT), (0, 0), alpha)
    return out


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
