"""Evaluate mask predictors on the Voxae-Reason val/test split.

Reports gIoU and cIoU overall and per query family, for any predictor that
maps (image, query) -> boolean mask — the trained pipeline and the zero-shot
baseline share the same interface, so the comparison table is one command:

    uv run python -m voxae.eval.run_eval --split val --predictor zero-shot
    uv run python -m voxae.eval.run_eval --split val --predictor trained \
        --checkpoint outputs/full_2b/latest --backbone Qwen/Qwen2-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from voxae.data import rle
from voxae.data.schemas import QuerySample
from voxae.eval.metrics import ciou, giou


def _to_gt_resolution(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a prediction onto the ground-truth grid.

    A predictor works at whatever resolution the image file happens to be, and
    a packaged dataset may ship downscaled copies, while the RLE ground truth
    stays at source resolution. Scoring resamples the prediction rather than
    the target so the reference is never degraded to make a number agree.
    """
    if pred.shape == shape:
        return pred
    resized = Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (shape[1], shape[0]), Image.NEAREST
    )
    return np.asarray(resized) > 127


def evaluate(
    predictor,
    samples: list[QuerySample],
    data_root: Path,
) -> dict:
    """Run a predictor over samples; aggregate gIoU/cIoU overall + per family."""
    preds: dict[str, list[np.ndarray]] = defaultdict(list)
    gts: dict[str, list[np.ndarray]] = defaultdict(list)
    latencies: list[float] = []
    failures = 0

    for s in samples:
        image = Image.open(data_root / s.rel_path)
        t0 = time.perf_counter()
        try:
            pred = predictor.predict(image, s.text)
        except Exception:
            failures += 1
            pred = np.zeros((image.height, image.width), dtype=bool)
        latencies.append(time.perf_counter() - t0)
        gt = rle.decode(s.rle)
        pred = _to_gt_resolution(pred, gt.shape)
        for key in ("all", str(s.family)):
            preds[key].append(pred)
            gts[key].append(gt)

    report: dict = {
        "n": len(samples),
        "failures": failures,
        "latency_s_mean": round(float(np.mean(latencies)), 3) if latencies else None,
    }
    for key in sorted(preds):
        report[f"giou_{key}"] = round(giou(preds[key], gts[key]), 4)
        report[f"ciou_{key}"] = round(ciou(preds[key], gts[key]), 4)
    return report


class _ZeroShotAdapter:
    """Wraps the zero-shot pipeline to the (image, query) -> mask interface."""

    def __init__(self):
        from voxae.eval.baselines.zero_shot import ZeroShotPipeline
        from voxae.model.grounder import QwenAPIGrounder
        from voxae.model.segmenter import Sam2Segmenter

        self.pipeline = ZeroShotPipeline(QwenAPIGrounder(), Sam2Segmenter())
        self.name = "zero-shot"

    def predict(self, image, query):
        return self.pipeline.run(image, query).mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--predictor", choices=["zero-shot", "trained"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--sam2", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", help="backbone dtype; defaults to bfloat16 on cuda, float32 on cpu"
    )
    args = parser.parse_args()

    from voxae.train.collate import load_samples

    samples = load_samples(
        args.data_root / "processed" / "annotations" / "voxae_reason.jsonl",
        split=args.split,
    )
    if args.limit:
        samples = samples[: args.limit]

    if args.predictor == "trained":
        from voxae.model.pipeline import VoxaeSegPipeline

        # Full precision doubles the backbone's footprint for no accuracy the
        # model ever had: it was trained in bfloat16.
        dtype = args.dtype or ("bfloat16" if args.device.startswith("cuda") else "float32")
        predictor = VoxaeSegPipeline.from_checkpoint(
            args.checkpoint, args.backbone, args.sam2, device=args.device, dtype=dtype
        )
    else:
        predictor = _ZeroShotAdapter()

    report = {
        "predictor": args.predictor,
        "split": args.split,
        **evaluate(predictor, samples, args.data_root),
    }
    print(json.dumps(report, indent=2))
    out = args.data_root / "processed" / "annotations" / f"eval_{args.predictor}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
