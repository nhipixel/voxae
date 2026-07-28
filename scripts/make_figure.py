"""Render the README comparison figure from evaluation records.

Examples are chosen by a stated rule rather than by eye: the affordance
samples where the trained bridge most outscores the baseline, among those the
trained model actually segments well. Print the rule alongside the figure and
the aggregate table, and a reader can judge the selection for themselves.

    uv run python scripts/make_figure.py \
        --data-root data --checkpoint outputs/full_2b/latest --out assets/comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

TILE = 320
PAD = 12
CAPTION_H = 54
COLUMNS = ("image", "ground truth", "trained bridge", "zero-shot baseline")


def _load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise typer.BadParameter(f"missing records file: {path}")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 has no size argument
        return ImageFont.load_default()


def _tile(img: Image.Image, empty: bool = False) -> Image.Image:
    """One cell of the grid.

    An empty prediction renders as the untouched image, which reads as a broken
    figure rather than a wrong answer, so it gets labelled.
    """
    out = img.convert("RGB").copy()
    out.thumbnail((TILE, TILE), Image.LANCZOS)
    canvas = Image.new("RGB", (TILE, TILE), (17, 20, 26))
    canvas.paste(out, ((TILE - out.width) // 2, (TILE - out.height) // 2))
    if empty:
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((0, TILE - 26, TILE, TILE), fill=(10, 12, 16, 220))
        draw.text((8, TILE - 20), "no region predicted", fill=(226, 138, 128), font=_font(13))
    return canvas


@app.command()
def make(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    checkpoint: Annotated[Path | None, typer.Option(help="Directory holding state.pt")] = None,
    backbone: Annotated[str, typer.Option()] = "Qwen/Qwen2-VL-2B-Instruct",
    sam2: Annotated[str, typer.Option()] = "facebook/sam2.1-hiera-small",
    device: Annotated[str, typer.Option()] = "cuda",
    split: Annotated[str, typer.Option()] = "test",
    family: Annotated[str, typer.Option()] = "affordance",
    n: Annotated[int, typer.Option(help="Rows in the figure")] = 3,
    min_trained_iou: Annotated[float, typer.Option(help="Trained mask must be good")] = 0.5,
    max_baseline_iou: Annotated[float, typer.Option(help="Baseline must clearly miss")] = 0.3,
    min_area_pct: Annotated[float, typer.Option(help="Target must be visible at tile size")] = 4.0,
    out: Annotated[Path, typer.Option()] = Path("assets/comparison.png"),
) -> None:
    """Pick examples by measured margin, then render both predictions."""
    from voxae.data import rle
    from voxae.eval.baselines.zero_shot import ZeroShotPipeline
    from voxae.eval.run_eval import _to_gt_resolution
    from voxae.model.grounder import QwenAPIGrounder
    from voxae.model.pipeline import VoxaeSegPipeline
    from voxae.model.segmenter import Sam2Segmenter
    from voxae.train.collate import load_samples
    from voxae.viz import overlay_mask

    ann = data_root / "processed" / "annotations"
    trained_rec = _load_records(ann / f"eval_trained_{split}.records.jsonl")
    baseline_rec = _load_records(ann / f"eval_zero-shot_{split}.records.jsonl")

    all_samples = {s.sample_id: s for s in load_samples(ann / "voxae_reason.jsonl", split=split)}
    ranked = sorted(
        (
            (t["iou"] - baseline_rec[sid]["iou"], sid)
            for sid, t in trained_rec.items()
            if sid in baseline_rec
            and sid in all_samples
            and t["family"] == family
            and t["iou"] >= min_trained_iou
            and baseline_rec[sid]["iou"] <= max_baseline_iou
            and all_samples[sid].area_pct >= min_area_pct
            and not baseline_rec[sid]["failed"]
        ),
        reverse=True,
    )
    if not ranked:
        raise typer.BadParameter(
            f"no {family} samples with trained IoU >= {min_trained_iou}, "
            f"baseline IoU <= {max_baseline_iou}, target area >= {min_area_pct}%"
        )
    chosen = {sid: margin for margin, sid in ranked[:n]}
    console.print(f"selected {len(chosen)} of {len(ranked)} eligible {family} samples")

    samples = {sid: all_samples[sid] for sid in chosen}

    trained = VoxaeSegPipeline.from_checkpoint(
        checkpoint,
        backbone,
        sam2,
        device=device,
        dtype="bfloat16" if "cuda" in device else "float32",
    )
    baseline = ZeroShotPipeline(QwenAPIGrounder(), Sam2Segmenter())

    rows = []
    for sid in chosen:
        s = samples[sid]
        image = Image.open(data_root / s.rel_path)
        gt = rle.decode(s.rle)
        t_mask = _to_gt_resolution(trained.predict(image, s.text), gt.shape)
        b_mask = _to_gt_resolution(baseline.run(image, s.text).mask, gt.shape)
        gt_img = image.resize(gt.shape[::-1]) if image.size != gt.shape[::-1] else image
        rows.append(
            (
                s.text,
                [
                    _tile(image),
                    _tile(overlay_mask(gt_img, gt)),
                    _tile(overlay_mask(gt_img, t_mask), empty=not t_mask.any()),
                    _tile(overlay_mask(gt_img, b_mask), empty=not b_mask.any()),
                ],
                trained_rec[sid]["iou"],
                baseline_rec[sid]["iou"],
            )
        )
        console.print(
            f"  {sid}: trained {trained_rec[sid]['iou']:.2f} vs {baseline_rec[sid]['iou']:.2f}"
        )

    width = len(COLUMNS) * TILE + (len(COLUMNS) + 1) * PAD
    height = len(rows) * (TILE + CAPTION_H) + PAD + 26
    canvas = Image.new("RGB", (width, height), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    header, caption = _font(15), _font(13)

    for col, name in enumerate(COLUMNS):
        draw.text((PAD + col * (TILE + PAD), 6), name, fill=(150, 160, 175), font=header)

    y = 26
    for text, tiles, t_iou, b_iou in rows:
        for col, tile in enumerate(tiles):
            canvas.paste(tile, (PAD + col * (TILE + PAD), y))
        draw.text((PAD, y + TILE + 6), f'"{text}"', fill=(228, 232, 238), font=caption)
        draw.text(
            (PAD, y + TILE + 26),
            f"IoU  trained {t_iou:.2f}   zero-shot {b_iou:.2f}",
            fill=(150, 160, 175),
            font=caption,
        )
        y += TILE + CAPTION_H

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    console.print(f"[green]wrote {out}[/green] ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()
