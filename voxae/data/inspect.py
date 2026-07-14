"""Render decoded class masks to an HTML page for visual verification.

Before generating queries, this confirms that a dataset's palette/index table
decodes to the right classes: each class is drawn in a fixed distinct color
and unmapped pixels are drawn in magenta, so a wrong mapping (a class colored
over the wrong region) or an incomplete palette (large magenta areas) is
obvious at a glance. Runs offline with no API calls.
"""

from __future__ import annotations

import base64
import html
import io
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from voxae.data.gsd import estimate_gsd
from voxae.data.registry import REGISTRY, decode_mask, iter_samples

UNKNOWN_COLOR = (255, 0, 255)
PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (170, 110, 40),
    (128, 0, 0),
    (128, 128, 0),
    (0, 0, 128),
    (255, 215, 0),
    (128, 128, 128),
]


def class_color(cls: str) -> tuple[int, int, int]:
    """Deterministic display color for a class name (stable across runs)."""
    return PALETTE[zlib.crc32(cls.encode()) % len(PALETTE)]


def _downscale_bool(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.resize(size, Image.NEAREST)) > 0


def render_overlay(
    image: Image.Image,
    class_masks: dict[str, np.ndarray],
    unknown_mask: np.ndarray | None,
    max_px: int = 700,
) -> Image.Image:
    """Blend per-class colors (and magenta for unknown) onto a downscaled image."""
    img = image.convert("RGB").copy()
    img.thumbnail((max_px, max_px))
    w, h = img.size
    base = np.asarray(img)
    overlay = base.copy()
    for cls, m in sorted(class_masks.items()):
        overlay[_downscale_bool(m, (w, h))] = class_color(cls)
    if unknown_mask is not None and unknown_mask.any():
        overlay[_downscale_bool(unknown_mask, (w, h))] = UNKNOWN_COLOR
    blended = (0.5 * base + 0.5 * overlay).astype(np.uint8)
    return Image.fromarray(blended)


def _thumb_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_inspect(dataset: str, data_root: Path, limit: int = 8) -> Path:
    """Render the first ``limit`` decoded images of a dataset to inspect.html."""
    spec = REGISTRY[dataset]
    raw = data_root / "raw" / dataset

    cards: list[str] = []
    unknown_pcts: list[float] = []
    n_with_gsd = 0
    for i, (image_path, mask_path) in enumerate(iter_samples(spec, raw)):
        if i >= limit:
            break
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        mask_arr = np.asarray(Image.open(mask_path))
        class_masks, unknown_pct = decode_mask(spec, mask_arr)
        unknown_pcts.append(unknown_pct)

        known = np.zeros(mask_arr.shape[:2], dtype=bool)
        for m in class_masks.values():
            known |= m
        overlay = render_overlay(image, class_masks, ~known)

        est = estimate_gsd(image_path)
        if est:
            n_with_gsd += 1
        total = mask_arr.shape[0] * mask_arr.shape[1]
        legend = " ".join(
            f'<span style="border-left:12px solid rgb{class_color(c)};padding-left:4px">'
            f"{html.escape(c)} {m.sum() * 100.0 / total:.1f}%</span>"
            for c, m in sorted(class_masks.items())
        )
        cards.append(
            "<div class='card'>"
            f"<div class='meta'><code>{html.escape(image_path.name)}</code> "
            f"unknown={unknown_pct:.1f}% "
            f"gsd={f'{est.meters_per_px:.3f} m/px' if est else 'none'}</div>"
            f"<img src='data:image/jpeg;base64,{_thumb_b64(overlay)}'>"
            f"<div class='legend'>{legend}</div></div>"
        )

    mean_unknown = sum(unknown_pcts) / len(unknown_pcts) if unknown_pcts else 0.0
    out_dir = data_root / "processed" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "inspect.html"
    out.write_text(
        "<!doctype html><meta charset='utf-8'><title>voxae inspect</title>"
        "<style>body{font-family:sans-serif;max-width:780px;margin:2rem auto}"
        ".card{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}"
        "img{max-width:100%;border-radius:4px}.legend{font-size:.8em;margin-top:.5rem;"
        "display:flex;flex-wrap:wrap;gap:8px}.meta{font-size:.85em;color:#333}"
        "code{background:#eee;padding:1px 4px}</style>"
        f"<h1>{dataset}: decoded classes ({len(cards)} images)</h1>"
        f"<p>Mean unknown pixels: <b>{mean_unknown:.1f}%</b> "
        f"(high means the palette/index table is wrong or incomplete). "
        f"Images with a GSD estimate: {n_with_gsd}/{len(cards)}. "
        f"Magenta = unmapped pixels.</p>" + "\n".join(cards),
        encoding="utf-8",
    )
    return out
