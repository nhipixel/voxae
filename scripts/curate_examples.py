"""Choose demo examples from the evaluation records, restricted to UAVid.

The Wikimedia gallery is out of distribution: the model learned drone imagery
at particular altitudes and sensors, so arbitrary aerial photos produce weak
output and demo nothing. Real dataset frames with measured scores demo the
model as it is.

Only UAVid is eligible. It is CC BY-NC-SA 4.0, so non-commercial
redistribution with attribution is permitted; the other two source datasets
forbid redistribution and stay out of the gallery.

    uv run python scripts/curate_examples.py --data-root data --n 6
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

GALLERY = Path(__file__).parent.parent / "voxae" / "demo" / "assets" / "gallery"
ATTRIBUTION = (
    "\n## Dataset frames\n\n"
    "Demo examples prefixed `uavid-` are frames from the UAVid dataset\n"
    "(https://uavid.nl), licensed CC BY-NC-SA 4.0. Redistributed here\n"
    "unmodified for non-commercial demonstration under the same licence.\n"
)


def _load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise typer.BadParameter(f"missing records file: {path}")
    return {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


@app.command()
def curate(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    split: Annotated[str, typer.Option()] = "test",
    n: Annotated[int, typer.Option(help="Examples to copy")] = 6,
    min_trained_iou: Annotated[float, typer.Option()] = 0.6,
    min_area_pct: Annotated[float, typer.Option()] = 4.0,
) -> None:
    """Copy the best-scoring UAVid frames into the gallery and print the pairs."""
    from voxae.train.collate import load_samples

    ann = data_root / "processed" / "annotations"
    trained_rec = _load_records(ann / f"eval_trained_{split}.records.jsonl")
    samples = {s.sample_id: s for s in load_samples(ann / "voxae_reason.jsonl", split=split)}

    ranked = sorted(
        (
            (r["iou"], sid)
            for sid, r in trained_rec.items()
            if sid.startswith("uavid-")
            and sid in samples
            and r["iou"] >= min_trained_iou
            and samples[sid].area_pct >= min_area_pct
        ),
        reverse=True,
    )
    if not ranked:
        raise typer.BadParameter(
            f"no UAVid samples with trained IoU >= {min_trained_iou} and area >= {min_area_pct}%"
        )

    # One example per image: several queries over one frame reads as padding.
    chosen: list[tuple[float, str]] = []
    seen_images: set[str] = set()
    for score, sid in ranked:
        image_id = samples[sid].image_id
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        chosen.append((score, sid))
        if len(chosen) == n:
            break

    console.print(f"selected {len(chosen)} of {len(ranked)} eligible UAVid samples")
    GALLERY.mkdir(parents=True, exist_ok=True)
    lines = []
    for score, sid in chosen:
        s = samples[sid]
        dest = GALLERY / f"{sid}{Path(s.rel_path).suffix}"
        shutil.copy2(data_root / s.rel_path, dest)
        console.print(f"  {sid}  IoU {score:.2f}  -> {dest.name}")
        lines.append(f'    ("{s.text}", "{dest.stem}"),')

    attribution = GALLERY.parent / "ATTRIBUTION.md"
    if attribution.exists() and "uavid.nl" not in attribution.read_text(encoding="utf-8"):
        with attribution.open("a", encoding="utf-8") as f:
            f.write(ATTRIBUTION)
        console.print(f"appended UAVid attribution to {attribution.name}")

    console.print("\n[bold]Paste into gradio_app.EXAMPLE_QUERIES:[/bold]\n")
    console.print("\n".join(lines))


if __name__ == "__main__":
    app()
