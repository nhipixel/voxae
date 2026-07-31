"""Estimate a relief map for each worked-example photograph, once, offline.

The relief view drapes the photo over real scene structure so buildings stand
up; confidence is painted on top of that geometry rather than pretending to be
it. Monocular depth is estimated with Depth-Anything V2 small on CPU and saved
as an 8 bit greyscale PNG, bright meaning near the camera, which for oblique
aerial photography is a serviceable stand-in for tall.

    uv run python scripts/make_depth.py

Rerun only when the example photographs change.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent
FRAMES = ROOT / "web" / "public" / "examples"
OUT = ROOT / "web" / "public" / "depth"
MAX_W = 768


def main() -> None:
    import numpy as np
    from PIL import Image
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1
    )
    OUT.mkdir(parents=True, exist_ok=True)

    for frame in sorted(FRAMES.glob("*.png")):
        image = Image.open(frame).convert("RGB")
        pred = estimator(image)["predicted_depth"]
        arr = pred.squeeze().float().numpy()
        # Relative inverse depth: normalize to the frame, near is bright.
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)

        # An oblique frame's inverse depth is mostly the ground getting closer
        # toward the bottom edge. Subtracting a per-row ground estimate leaves
        # what surveyors call a normalized surface model: roads near zero,
        # buildings and trees standing proud of it.
        ground = np.percentile(arr, 30, axis=1)
        kernel = np.ones(31) / 31
        ground = np.convolve(np.pad(ground, 15, mode="edge"), kernel, mode="valid")
        arr = np.clip(arr - ground[:, None], 0.0, None)
        arr = arr / max(float(arr.max()), 1e-6)
        # Gamma lifts mid structures without letting one tower own the range.
        arr = arr**0.75
        grey = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        if grey.width != image.width:
            grey = grey.resize(image.size, Image.BILINEAR)
        grey.thumbnail((MAX_W, MAX_W), Image.BILINEAR)
        dest = OUT / f"{frame.stem}.png"
        grey.save(dest, optimize=True)
        print(f"{frame.name}: {grey.size[0]}x{grey.size[1]} -> {dest.name} "
              f"({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
