"""Bundle the dataset for a remote training session.

Training never sees source resolution: the SAM head resizes to its own input
size and the VLM processor downscales internally, so shipping 6000x4000
source images to a GPU host wastes bandwidth and disk. Masks are stored as
RLE and resized independently of the image, so downscaling the images does
not desynchronize them.

The bundle mirrors the local layout (raw/... plus processed/annotations/...),
so it can be used as ``data_root`` directly with no config changes.

    uv run python scripts/package_for_colab.py --max-px 1536
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@app.command()
def package(
    jsonl: Annotated[Path, typer.Option(help="Dataset JSONL to bundle")] = Path(
        "data/processed/annotations/voxae_reason.jsonl"
    ),
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    out: Annotated[Path, typer.Option(help="Bundle directory")] = Path("data/colab_bundle"),
    max_px: Annotated[int, typer.Option(help="Longest edge after downscaling")] = 1536,
    quality: Annotated[int, typer.Option(help="JPEG quality")] = 92,
    splits: Annotated[
        str, typer.Option(help="Comma-separated splits to include")
    ] = "train,val,test",
    archive: Annotated[bool, typer.Option(help="Also write a .zip of the bundle")] = True,
) -> None:
    """Copy the samples' images at reduced resolution into a portable bundle."""
    if not jsonl.exists():
        console.print(f"[red]{jsonl} not found — run 'voxae dataset split' first.[/red]")
        raise typer.Exit(1)

    wanted = {s.strip() for s in splits.split(",") if s.strip()}
    rows = [
        json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    kept = [r for r in rows if not r.get("split") or r["split"] in wanted]
    rel_paths = sorted({r["rel_path"] for r in kept})
    console.print(f"samples: {len(kept)}/{len(rows)}  images: {len(rel_paths)}")

    if out.exists():
        shutil.rmtree(out)
    ann_dir = out / "processed" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    with (ann_dir / jsonl.name).open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    skipped = 0
    for i, rel in enumerate(rel_paths, 1):
        src = data_root / rel
        if not src.exists():
            skipped += 1
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px), Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality)
        if i % 100 == 0:
            console.print(f"  {i}/{len(rel_paths)}")

    if skipped:
        console.print(f"[yellow]{skipped} source images missing, skipped[/yellow]")

    size = _dir_size(out)
    console.print(f"[green]bundle -> {out}  ({size / 1e9:.2f} GB)[/green]")

    if archive:
        zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
        console.print(
            f"[green]archive -> {zip_path}  ({Path(zip_path).stat().st_size / 1e9:.2f} GB)[/green]"
        )

    console.print(
        "\nOn the GPU host, unzip and point the config at the bundle:\n"
        f"  data_root: <bundle>\n  train_jsonl: <bundle>/processed/annotations/{jsonl.name}"
    )


if __name__ == "__main__":
    app()
