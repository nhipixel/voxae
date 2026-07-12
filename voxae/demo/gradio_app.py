"""Voxae public demo (Gradio) — runs on Hugging Face Spaces free CPU tier.

Backend selection:
- VOXAE_VLM_API_KEY set -> hosted Qwen2.5-VL grounding + SAM 2.1 (CPU) segmentation.
- No key                -> mock mode with a visible banner (demo still clickable).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import gradio as gr
from PIL import Image

from voxae.config import get_settings
from voxae.eval.baselines.zero_shot import ZeroShotPipeline
from voxae.viz import overlay_mask

GALLERY_DIR = Path(__file__).parent / "assets" / "gallery"

EXAMPLE_QUERIES = [
    "highlight the buildings with direct road access",
    "where could a small drone land safely?",
    "segment all vehicles on paved surfaces",
    "which open area is large enough for a helicopter?",
    "highlight vegetation close to the structures",
]

_request_times: deque[float] = deque(maxlen=64)


def _rate_limited(limit_per_min: int) -> bool:
    now = time.time()
    while _request_times and now - _request_times[0] > 60:
        _request_times.popleft()
    if len(_request_times) >= limit_per_min:
        return True
    _request_times.append(now)
    return False


def build_pipeline() -> tuple[ZeroShotPipeline, bool]:
    """Returns (pipeline, is_live). Falls back to mocks when unconfigured."""
    settings = get_settings()
    if settings.vlm_api_key:
        try:
            from voxae.model.grounder import QwenAPIGrounder
            from voxae.model.segmenter import Sam2Segmenter

            return ZeroShotPipeline(QwenAPIGrounder(settings), Sam2Segmenter(settings)), True
        except Exception:  # degrade gracefully in the public demo
            pass
    from voxae.model.grounder import MockGrounder
    from voxae.model.segmenter import MockSegmenter

    return ZeroShotPipeline(MockGrounder(), MockSegmenter()), False


def _gallery_paths() -> list[str]:
    if not GALLERY_DIR.exists():
        return []
    return sorted(
        str(p) for p in GALLERY_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def run_query(image: Image.Image | None, query: str):
    settings = get_settings()
    if image is None:
        raise gr.Error("Please provide an image (upload or pick an example).")
    if not query or not query.strip():
        raise gr.Error("Please type a query, e.g. 'highlight the safest landing zone'.")
    if _rate_limited(settings.demo_rate_limit_per_min):
        raise gr.Error("Rate limit reached — please wait a minute and try again.")
    if max(image.size) > settings.demo_max_image_px:
        image = image.copy()
        image.thumbnail((settings.demo_max_image_px, settings.demo_max_image_px))

    pipeline, is_live = PIPELINE, IS_LIVE
    try:
        result = pipeline.run(image, query)
    except Exception as e:  # user-facing error, not a stack trace
        raise gr.Error(f"Pipeline error: {e}") from e

    bbox_px = result.trace.grounding.bbox.to_pixels(*image.size)
    overlay = overlay_mask(image, result.mask, bbox_px)
    trace = json.loads(result.trace.model_dump_json())
    if not is_live:
        trace["note"] = "MOCK MODE — set VOXAE_VLM_API_KEY for live grounding"
    return overlay, trace


PIPELINE, IS_LIVE = build_pipeline()

ABOUT = """
## Voxae
**Language-grounded, metric 3D scene understanding for physical AI.**

Ask a scene *what's where, how big, and how far*. This Space runs the
**zero-shot baseline**: a hosted VLM proposes a region (bbox + points), SAM 2.1
turns it into a mask — no fine-tuning yet. Future work trains a `<SEG>` bridge
and lifts masks into a metric 3D Gaussian-splat scene.

Pipeline: `image + query → Qwen2.5-VL (grounding JSON) → SAM 2.1 → mask`
"""

ROADMAP = """
| Stage | Artifact | Status |
|---|---|---|
| Zero-shot demo | This live demo | done |
| Dataset | Voxae-Reason (annotations + datasheet) | planned |
| Trained model | `<SEG>` bridge (Qwen2.5-VL→SAM2) + ablations | planned |
| Metric 3D | Scene you can query in the browser | planned |

Built in public — code: [github.com/nhipixel/voxae](https://github.com/nhipixel/voxae)
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Voxae") as demo:
        gr.Markdown(ABOUT)
        if not IS_LIVE:
            gr.Markdown("**Mock mode** — live VLM key not configured; masks are placeholders.")
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Aerial / outdoor image")
                    query_in = gr.Textbox(
                        label="Query",
                        placeholder="e.g. where could a small drone land safely?",
                    )
                    gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[query_in])
                    run_btn = gr.Button("Ground it", variant="primary")
                with gr.Column():
                    image_out = gr.Image(label="Mask overlay")
                    trace_out = gr.JSON(label="Reasoning trace")
            run_btn.click(run_query, [image_in, query_in], [image_out, trace_out])
        with gr.Tab("Example images"):
            gr.Gallery(value=_gallery_paths(), columns=4, label="Openly-licensed aerial imagery")
            gr.Markdown("Images are CC0/CC-BY from Wikimedia Commons — see ATTRIBUTION.md.")
        with gr.Tab("Roadmap"):
            gr.Markdown(ROADMAP)
    return demo


def main() -> None:
    build_demo().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
