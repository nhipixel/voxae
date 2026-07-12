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


if __name__ == "__main__":
    app()
