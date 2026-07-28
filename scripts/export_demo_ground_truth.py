"""Freeze ground-truth masks for the demo's dataset examples.

The demo can score itself only where a correct answer exists. Its first
examples are held-out evaluation samples, so their masks are exported here as
PNG and shipped with the app; every other input, including anything a visitor
uploads, has no ground truth and is reported as unscored.

Keyed by query text: the demo knows the query a visitor clicked, not which
sample it came from, and the texts are unique within the split.

    uv run python scripts/export_demo_ground_truth.py --data-root data/colab_bundle
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

OUT = Path(__file__).parent.parent / "voxae" / "demo" / "assets" / "ground_truth.json"


@app.command()
def export(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    split: Annotated[str, typer.Option()] = "test",
    out: Annotated[Path, typer.Option()] = OUT,
) -> None:
    """Write ground truth masks for every dataset-backed demo example."""
    from voxae.data import rle
    from voxae.demo.gradio_app import DATASET_EXAMPLES, EXAMPLE_QUERIES
    from voxae.train.collate import load_samples

    wanted = {query for query, _ in EXAMPLE_QUERIES[:DATASET_EXAMPLES]}
    samples = load_samples(
        data_root / "processed" / "annotations" / "voxae_reason.jsonl", split=split
    )

    by_text: dict[str, list] = {}
    for s in samples:
        if s.text in wanted:
            by_text.setdefault(s.text, []).append(s)

    missing = wanted - set(by_text)
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} demo example(s) not found in the {split} split; "
            f"first: {next(iter(missing))[:60]!r}"
        )
    ambiguous = {t for t, group in by_text.items() if len(group) > 1}
    if ambiguous:
        raise typer.BadParameter(
            f"{len(ambiguous)} query text(s) match several samples, so a mask "
            f"cannot be chosen unambiguously; first: {next(iter(ambiguous))[:60]!r}"
        )

    # Ground truth is only valid for the exact frame it was drawn on. Storing a
    # pixel digest lets the demo refuse to score a dataset query that a visitor
    # typed over some other image.
    gallery = OUT.parent / "gallery"
    payload = {}
    for text, group in by_text.items():
        s = group[0]
        stem = Path(s.rel_path).stem
        shipped = next(gallery.glob(f"*{stem}.*"), None)
        source = shipped or (data_root / s.rel_path)
        digest = hashlib.md5(Image.open(source).convert("RGB").tobytes()).hexdigest()
        # PNG rather than RLE so the demo needs no pycocotools.
        mask = rle.decode(s.rle)
        buf = io.BytesIO()
        Image.fromarray(mask).convert("1").save(buf, format="PNG", optimize=True)
        payload[text] = {
            "sample_id": s.sample_id,
            "image_stem": stem,
            "image_md5": digest,
            "family": str(s.family),
            "area_pct": round(s.area_pct, 3),
            "mask_png_b64": base64.b64encode(buf.getvalue()).decode(),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for text, entry in payload.items():
        console.print(f"  {entry['sample_id']:24} {entry['area_pct']:5.1f}%  {text[:52]}")
    console.print(f"[green]wrote {out}[/green] ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    app()
