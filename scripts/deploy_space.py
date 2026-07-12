"""Create/update the public Hugging Face Space for the Voxae demo.

Requires HF_TOKEN in .env or env:
    uv run python scripts/deploy_space.py --space-id <username>/voxae

Uploads: app.py, the voxae package, demo assets, Space README
(with Gradio SDK front-matter), and a Space-specific requirements.txt
(CPU torch wheels).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

ROOT = Path(__file__).parent.parent

SPACE_README = """---
title: Voxae
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Voxae — language-grounded, metric 3D scene understanding for physical AI

Zero-shot baseline demo: hosted Qwen2.5-VL grounding + SAM 2.1 masks.
Code: https://github.com/nhipixel/voxae
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.4
transformers>=4.53
gradio>=6.20
numpy>=1.26
pillow>=10.3
pydantic>=2.7
pydantic-settings>=2.3
httpx>=0.27
pyyaml>=6.0
typer>=0.12
rich>=13.7
"""


@app.command()
def deploy(
    space_id: str = typer.Option(..., help="e.g. nhipixel/voxae"),
    private: bool = typer.Option(False),
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        console.print("[red]pip install huggingface_hub first (uv add huggingface_hub --dev)[/red]")
        raise typer.Exit(1) from None

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        console.print("[red]HF_TOKEN not set (env or .env)[/red]")
        raise typer.Exit(1)

    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        (staging / "README.md").write_text(SPACE_README, encoding="utf-8")
        (staging / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
        (staging / "app.py").write_text(
            (ROOT / "app.py").read_text(encoding="utf-8"), encoding="utf-8"
        )
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(staging))

    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(ROOT / "voxae"),
        path_in_repo="voxae",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    console.print(
        f"[bold green]Space deployed:[/bold green] https://huggingface.co/spaces/{space_id}"
    )
    console.print("Add the VOXAE_VLM_API_KEY secret in Space settings for live mode.")


if __name__ == "__main__":
    app()
