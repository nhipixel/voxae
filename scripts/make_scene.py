"""Per-frame relief assets for the 3D view, computed offline.

The relief keeps the photograph's own rectangular sheet, lifts every pixel by
a ground-removed surface model, and corrects the oblique by relief
displacement: a point of height H appears shifted toward the horizon by
H / tan(depression), so the mesh shifts it back. Walls come out vertical to
first order and nothing is resampled, so streets, cars, and texture survive
untouched.

Outputs per frame, in web/public/scene:
  <stem>-height.png   L    normalized surface model, ground at 0
  <stem>.json              the shear factor for relief-displacement correction

    uv run python scripts/make_scene.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
FRAMES = ROOT / "web" / "public" / "examples"
OUT = ROOT / "web" / "public" / "scene"
FOV_DEG = 55.0
DEPTH_W = 640


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = vt[-1]
    return n, float(n @ centroid)


def main() -> None:
    from PIL import Image
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1
    )
    OUT.mkdir(parents=True, exist_ok=True)

    for frame in sorted(FRAMES.glob("*.png")):
        image = Image.open(frame).convert("RGB")
        pred = estimator(image)["predicted_depth"].squeeze().float().numpy()
        inv = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)

        grey = Image.fromarray((inv * 255).astype(np.uint8), mode="L")
        if grey.size != image.size:
            grey = grey.resize(image.size, Image.BILINEAR)
        grey.thumbnail((DEPTH_W, DEPTH_W), Image.BILINEAR)
        w, h = grey.size
        inv_s = np.asarray(grey, dtype=np.float64) / 255.0

        # Camera depression angle from the fitted ground plane, using the same
        # unprojection the plane was fitted in.
        depth = 1.0 / (0.18 + inv_s)
        f = w / (2 * np.tan(np.radians(FOV_DEG) / 2))
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        dirs = np.stack(
            [(xs - w / 2) / f, (ys - h / 2) / f, np.ones_like(xs, dtype=np.float64)], axis=-1
        )
        points = dirs * depth[..., None]
        row_ground = np.percentile(inv_s, 35, axis=1)
        ground = inv_s <= (row_ground[:, None] + 0.04)
        n, _c = fit_plane(points[ground].reshape(-1, 3)[::5])
        n = n / np.linalg.norm(n)
        depression = float(np.arcsin(min(0.95, abs(n[2]))))
        shear = 1.0 / max(np.tan(depression), 0.2)

        # Normalized surface model: subtract the per-row ground trend so roads
        # sit at zero and structures stand proud of it.
        gcurve = np.percentile(inv_s, 30, axis=1)
        kernel = np.ones(31) / 31
        gcurve = np.convolve(np.pad(gcurve, 15, mode="edge"), kernel, mode="valid")
        ndsm = np.clip(inv_s - gcurve[:, None], 0.0, None)

        def nb(img, fn, r=1):
            layers = [
                np.roll(np.roll(img, dy, 0), dx, 1)
                for dy in range(-r, r + 1)
                for dx in range(-r, r + 1)
            ]
            return fn(np.stack(layers), axis=0)

        # Morphological opening kills thin spikes, the median calms speckle,
        # and a floor drops the low crumple that morphs distant blocks. Roof
        # domes from monocular depth stay smooth on purpose: every levelling
        # scheme tried against them traded smooth walls for striping, and the
        # back-slope card shading in the renderer already absorbs the domes.
        ndsm = nb(nb(ndsm, np.min), np.max)
        ndsm = nb(ndsm, np.median)
        top = float(np.percentile(ndsm, 99.5))
        norm = np.clip(ndsm / max(top, 1e-6), 0, 1)
        norm[norm < 0.1] = 0.0
        norm = norm**1.35

        Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(
            OUT / f"{frame.stem}-height.png", optimize=True
        )
        meta = {"shear": round(shear, 4), "depressionDeg": round(np.degrees(depression), 2)}
        (OUT / f"{frame.stem}.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"{frame.stem}: depression {meta['depressionDeg']} deg, shear {meta['shear']}")


if __name__ == "__main__":
    main()
