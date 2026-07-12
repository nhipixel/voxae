"""Voxae CLI: `voxae <command>`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def segment(
    image: Annotated[Path, typer.Argument(exists=True, readable=True)],
    query: Annotated[str, typer.Argument()],
    out: Annotated[Path, typer.Option(help="Output overlay path")] = Path("overlay.png"),
    mock: Annotated[bool, typer.Option(help="Use mock backends (no API key / weights)")] = False,
) -> None:
    """Run the zero-shot pipeline on one image and save an overlay."""
    from PIL import Image as PILImage

    from voxae.eval.baselines.zero_shot import ZeroShotPipeline
    from voxae.viz import overlay_mask

    if mock:
        from voxae.model.grounder import MockGrounder
        from voxae.model.segmenter import MockSegmenter

        pipeline = ZeroShotPipeline(MockGrounder(), MockSegmenter())
    else:
        from voxae.model.grounder import QwenAPIGrounder
        from voxae.model.segmenter import Sam2Segmenter

        pipeline = ZeroShotPipeline(QwenAPIGrounder(), Sam2Segmenter())

    img = PILImage.open(image)
    result = pipeline.run(img, query)
    bbox_px = result.trace.grounding.bbox.to_pixels(*img.size)
    overlay_mask(img, result.mask, bbox_px).save(out)
    console.print_json(result.trace.model_dump_json())
    console.print(f"[green]overlay saved ->[/green] {out}")


@app.command()
def version() -> None:
    from voxae import __version__

    console.print(f"voxae {__version__}")


dataset_app = typer.Typer(no_args_is_help=True, help="Voxae-Reason dataset pipeline")
app.add_typer(dataset_app, name="dataset")


def _data_paths():
    from voxae.config import get_settings

    root = get_settings().data_root
    ann = root / "processed" / "annotations"
    return root, {
        "raw_jsonl": ann / "raw.jsonl",
        "qc_jsonl": ann / "qc_passed.jsonl",
        "qc_report": ann / "qc_report.json",
        "audit_dir": ann / "audit",
        "splits": ann / "splits.json",
        "final_jsonl": ann / "voxae_reason.jsonl",
        "cache_dir": root / "cache" / "qgen",
    }


@dataset_app.command()
def prepare(name: Annotated[str, typer.Argument(help="uavid | sdd | vdd")]) -> None:
    """Extract downloaded archives and validate image/mask pair discovery."""
    from voxae.data.prepare import PrepareError
    from voxae.data.prepare import prepare as run_prepare

    root, _ = _data_paths()
    try:
        result = run_prepare(name, root)
    except PrepareError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    for a in result.archives:
        console.print(f"archive: {a}")
    console.print(f"[green]{result.n_pairs} image/mask pairs ->[/green] {result.extracted_to}")


@dataset_app.command()
def inspect(
    name: Annotated[str, typer.Argument(help="uavid | sdd | vdd")],
    limit: Annotated[int, typer.Option(help="Render at most N images")] = 8,
) -> None:
    """Render decoded class masks to an HTML page for offline visual verification."""
    from voxae.data.inspect import build_inspect

    root, _ = _data_paths()
    out = build_inspect(name, root, limit=limit)
    console.print(f"[green]inspect page ->[/green] {out}")


@dataset_app.command("build-records")
def build_records_cmd(
    name: Annotated[str, typer.Argument(help="uavid | sdd | vdd")],
    limit: Annotated[int | None, typer.Option(help="Process at most N images")] = None,
    altitude_m: Annotated[float | None, typer.Option(help="Override flight altitude")] = None,
) -> None:
    """Decode masks into unified per-image records (components, RLEs, GSD)."""
    from voxae.data.prep_masks import build_records

    root, _ = _data_paths()
    out = build_records(name, root, limit=limit, altitude_m=altitude_m)
    console.print(f"[green]records ->[/green] {out}")


@dataset_app.command()
def generate(
    name: Annotated[str, typer.Argument(help="uavid | sdd | vdd")],
    limit: Annotated[int | None, typer.Option(help="Generate for at most N images")] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Generate query samples with the configured LLM (cached, resumable)."""
    from voxae.data.query_gen import generate as run_generate

    def on_image(i: int, total: int, image_id: str, elapsed_s: float, n_samples: int) -> None:
        console.print(f"[{i}/{total}] {image_id}: {n_samples} samples ({elapsed_s:.1f}s)")

    root, paths = _data_paths()
    stats = run_generate(
        root / "processed" / name / "records.parquet",
        paths["raw_jsonl"],
        paths["cache_dir"],
        limit=limit,
        seed=seed,
        on_image=on_image,
    )
    console.print(
        f"images={stats.images} specs={stats.specs_returned} valid={stats.specs_valid} "
        f"emitted={stats.samples_emitted} cache_hits={stats.cache_hits}"
    )
    console.print(f"[green]samples appended ->[/green] {paths['raw_jsonl']}")


@dataset_app.command()
def qc() -> None:
    """Run automatic quality checks; write the QC-passed JSONL and report."""
    from voxae.data.qc import run_qc

    _, paths = _data_paths()
    report = run_qc(paths["raw_jsonl"], paths["qc_jsonl"], paths["qc_report"])
    console.print_json(data=report.to_dict())
    console.print(f"[green]qc-passed ->[/green] {paths['qc_jsonl']}")


@dataset_app.command()
def audit(
    n: Annotated[int, typer.Option(help="Sample size to review")] = 100,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Build the human-review page (audit.html + audit_decisions.csv)."""
    from voxae.data.audit import build_audit

    root, paths = _data_paths()
    html_path, csv_path = build_audit(paths["qc_jsonl"], root, paths["audit_dir"], n=n, seed=seed)
    console.print(f"review: {html_path}\ndecisions: {csv_path}")


@dataset_app.command("audit-score")
def audit_score() -> None:
    """Score the filled decisions CSV against the precision gate."""
    from voxae.data.audit import PRECISION_THRESHOLD, score_audit

    _, paths = _data_paths()
    score = score_audit(paths["audit_dir"] / "audit_decisions.csv")
    console.print(
        f"reviewed={score.reviewed} accepted={score.accepted} precision={score.precision:.3f}"
    )
    if score.passed:
        console.print(f"[green]PASS[/green] (threshold {PRECISION_THRESHOLD})")
    else:
        console.print(
            f"[red]FAIL[/red] (threshold {PRECISION_THRESHOLD}) — fix prompts/QC before training"
        )
        raise typer.Exit(1)


@dataset_app.command()
def split(seed: Annotated[int, typer.Option()] = 0) -> None:
    """Assign by-image stratified train/val/test splits; write the final JSONL."""
    from voxae.data.splits import apply_splits

    _, paths = _data_paths()
    counts = apply_splits(paths["qc_jsonl"], paths["final_jsonl"], paths["splits"], seed=seed)
    console.print_json(data=counts)
    console.print(f"[green]dataset ->[/green] {paths['final_jsonl']}")


if __name__ == "__main__":
    app()
