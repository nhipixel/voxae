"""Render the demo's worked examples so the weak ones can be replaced.

The gallery is openly-licensed imagery with no ground truth, so a pairing can
only be judged by looking at it. This renders every (image, query) pair the
demo offers, in one sheet, before a visitor is the one who discovers that
example three produces nothing.

    uv run python scripts/preview_examples.py --checkpoint outputs/full_2b/latest
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

COL_W = 420
PAD = 12
CAPTION_H = 34
HEADER_H = 26
COLUMNS = ("image", "trained bridge", "zero-shot baseline")


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 has no size argument
        return ImageFont.load_default()


def _tile(img: Image.Image, height: int, area_pct: float | None = None) -> Image.Image:
    out = img.convert("RGB").resize((COL_W, height), Image.LANCZOS)
    if area_pct is not None:
        draw = ImageDraw.Draw(out, "RGBA")
        label = "no region matched" if area_pct < 0.01 else f"{area_pct:.1f}% of image"
        colour = (226, 138, 128) if area_pct < 0.01 else (170, 220, 180)
        draw.rectangle((0, height - 24, COL_W, height), fill=(10, 12, 16, 220))
        draw.text((8, height - 19), label, fill=colour, font=_font(13))
    return out


@app.command()
def preview(
    checkpoint: Annotated[Path | None, typer.Option(help="Directory holding state.pt")] = None,
    backbone: Annotated[str, typer.Option()] = "Qwen/Qwen2-VL-2B-Instruct",
    sam2: Annotated[str, typer.Option()] = "facebook/sam2.1-hiera-small",
    device: Annotated[str, typer.Option()] = "cuda",
    out: Annotated[Path, typer.Option()] = Path("assets/examples_preview.png"),
) -> None:
    """Run both predictors over every demo example pair."""
    from voxae.demo.gradio_app import _example_pairs
    from voxae.eval.baselines.zero_shot import ZeroShotPipeline
    from voxae.model.grounder import QwenAPIGrounder
    from voxae.model.pipeline import VoxaeSegPipeline
    from voxae.model.segmenter import Sam2Segmenter
    from voxae.viz import overlay_mask

    pairs = [(p, q) for p, q in _example_pairs() if p]
    if not pairs:
        raise typer.BadParameter("no gallery images; run scripts/fetch_demo_images.py first")

    trained = VoxaeSegPipeline.from_checkpoint(
        checkpoint,
        backbone,
        sam2,
        device=device,
        dtype="bfloat16" if "cuda" in device else "float32",
    )
    baseline = ZeroShotPipeline(QwenAPIGrounder(), Sam2Segmenter())

    rows = []
    for path, query in pairs:
        image = Image.open(path)
        t_mask = trained.predict(image, query)
        b_mask = baseline.run(image, query).mask
        height = max(120, round(COL_W * image.height / image.width))
        t_area, b_area = float(t_mask.mean()) * 100, float(b_mask.mean()) * 100
        rows.append(
            (
                query,
                Path(path).name,
                [
                    _tile(image, height),
                    _tile(overlay_mask(image, t_mask), height, t_area),
                    _tile(overlay_mask(image, b_mask), height, b_area),
                ],
            )
        )
        console.print(
            f"  {Path(path).name[:40]:42} trained {t_area:5.1f}%  baseline {b_area:5.1f}%"
        )

    width = len(COLUMNS) * COL_W + (len(COLUMNS) + 1) * PAD
    height = HEADER_H + sum(r[2][0].height + CAPTION_H for r in rows)
    canvas = Image.new("RGB", (width, height), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)

    for col, name in enumerate(COLUMNS):
        draw.text((PAD + col * (COL_W + PAD), 6), name, fill=(150, 160, 175), font=_font(16))

    y = HEADER_H
    for query, name, tiles in rows:
        for col, tile in enumerate(tiles):
            canvas.paste(tile, (PAD + col * (COL_W + PAD), y))
        bottom = y + tiles[0].height
        draw.text((PAD, bottom + 8), f'"{query}"   [{name}]', fill=(228, 232, 238), font=_font(14))
        y = bottom + CAPTION_H

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    console.print(f"[green]wrote {out}[/green]")


if __name__ == "__main__":
    app()
