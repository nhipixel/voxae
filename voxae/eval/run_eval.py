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
from voxae.eval.metrics import aggregate, intersection_union


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


def _load_records(path: Path | None) -> dict[str, dict]:
    """Per-sample results already on disk, keyed by sample id."""
    if path is None or not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done[r["sample_id"]] = r
    return done


def evaluate(
    predictor,
    samples: list[QuerySample],
    data_root: Path,
    log_every: int = 25,
    records_path: Path | None = None,
) -> dict:
    """Run a predictor over samples; aggregate gIoU/cIoU overall + per family.

    Each sample is scored to scalars and appended to records_path as it
    completes. A hosted-API predictor takes tens of minutes and costs real
    money per call, so a run that dies partway must not lose the calls it
    already paid for, and silence for that long is indistinguishable from a
    hang. Scoring to scalars also keeps memory flat: full-resolution masks are
    never accumulated.
    """
    done = _load_records(records_path)
    if done:
        print(f"resuming: {len(done)} samples already scored", flush=True)
    records: list[dict] = []
    latencies: list[float] = []
    failures = 0

    for i, s in enumerate(samples, start=1):
        if s.sample_id in done:
            records.append(done[s.sample_id])
            continue

        image = Image.open(data_root / s.rel_path)
        t0 = time.perf_counter()
        failed = False
        try:
            pred = predictor.predict(image, s.text)
        except Exception as e:
            failed = True
            failures += 1
            print(f"  sample {s.sample_id} failed: {e}", flush=True)
            pred = np.zeros((image.height, image.width), dtype=bool)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        gt = rle.decode(s.rle)
        inter, union = intersection_union(_to_gt_resolution(pred, gt.shape), gt)
        record = {
            "sample_id": s.sample_id,
            "family": str(s.family),
            "iou": 1.0 if union == 0 else inter / union,
            "intersection": inter,
            "union": union,
            "latency_s": round(latency, 3),
            "failed": failed,
        }
        records.append(record)
        if records_path is not None:
            with records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if log_every and i % log_every == 0:
            rate = float(np.mean(latencies)) if latencies else 0.0
            eta_min = (len(samples) - i) * rate / 60
            print(
                f"{i}/{len(samples)}  {rate:.1f}s/sample  ~{eta_min:.0f} min left  "
                f"failures={failures}",
                flush=True,
            )

    report: dict = {
        "n": len(samples),
        "failures": sum(r["failed"] for r in records),
        "latency_s_mean": round(float(np.mean(latencies)), 3) if latencies else None,
    }
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_family["all"].append(r)
        by_family[r["family"]].append(r)
    for key in sorted(by_family):
        scores = aggregate(by_family[key])
        report[f"giou_{key}"] = round(scores["giou"], 4)
        report[f"ciou_{key}"] = round(scores["ciou"], 4)
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

    ann = args.data_root / "processed" / "annotations"
    records_path = ann / f"eval_{args.predictor}_{args.split}.records.jsonl"
    report = {
        "predictor": args.predictor,
        "split": args.split,
        **evaluate(predictor, samples, args.data_root, records_path=records_path),
    }
    print(json.dumps(report, indent=2))
    out = ann / f"eval_{args.predictor}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    print(f"per-sample records -> {records_path}")


if __name__ == "__main__":
    main()
