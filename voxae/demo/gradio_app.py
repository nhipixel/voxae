"""Voxae public demo (Gradio) — runs on Hugging Face Spaces free CPU tier.

The demo runs two predictors side by side on the same query:
- the trained <SEG> bridge (loaded from VOXAE_CHECKPOINT_DIR when present),
- the zero-shot compose baseline (hosted VLM grounding + SAM 2.1).

Either side degrades independently: without a checkpoint the trained column
is hidden, and without an API key the baseline falls back to mock output, so
the Space stays clickable in every configuration.
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

# Implicit, reasoning-style queries: the contrast between a compose baseline
# and a model trained on them is most visible when nothing is named directly.
EXAMPLE_QUERIES = [
    "what would block a fire truck reaching the center?",
    "where could a small drone land safely?",
    "which surfaces could a heavy vehicle drive on?",
    "what vegetation is close enough to the buildings to be a risk?",
    "where is there open space clear of overhead obstacles?",
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


def build_baseline() -> tuple[ZeroShotPipeline, bool]:
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


def build_trained():
    """Load the trained bridge if a checkpoint is configured, else None."""
    settings = get_settings()
    if not settings.checkpoint_dir:
        return None
    try:
        from voxae.model.pipeline import VoxaeSegPipeline

        return VoxaeSegPipeline.from_checkpoint(
            settings.checkpoint_dir,
            backbone_id=settings.trained_backbone_id,
            sam2_id=settings.sam2_model,
            device=settings.device,
        )
    except Exception as e:  # a broken checkpoint must not take the Space down
        print(f"trained model unavailable: {e}")
        return None


def _gallery_paths() -> list[str]:
    if not GALLERY_DIR.exists():
        return []
    return sorted(
        str(p) for p in GALLERY_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _prepare(image: Image.Image | None, query: str) -> Image.Image:
    settings = get_settings()
    if image is None:
        raise gr.Error("Please provide an image (upload or pick an example).")
    if not query or not query.strip():
        raise gr.Error("Please type a query, e.g. 'where could a drone land safely?'.")
    if _rate_limited(settings.demo_rate_limit_per_min):
        raise gr.Error("Rate limit reached — please wait a minute and try again.")
    if max(image.size) > settings.demo_max_image_px:
        image = image.copy()
        image.thumbnail((settings.demo_max_image_px, settings.demo_max_image_px))
    return image


def run_comparison(image: Image.Image | None, query: str):
    """Run both predictors; returns (trained overlay, baseline overlay, trace)."""
    image = _prepare(image, query)
    trace: dict = {"query": query.strip()}

    trained_overlay = None
    if TRAINED is not None:
        t0 = time.perf_counter()
        try:
            mask = TRAINED.predict(image, query)
            trained_overlay = overlay_mask(image, mask)
            trace["trained"] = {
                "model": TRAINED.name,
                "latency_s": round(time.perf_counter() - t0, 2),
                "mask_area_pct": round(float(mask.mean()) * 100, 2),
            }
        except Exception as e:
            trace["trained"] = {"error": str(e)}

    try:
        result = BASELINE.run(image, query)
        bbox_px = result.trace.grounding.bbox.to_pixels(*image.size)
        baseline_overlay = overlay_mask(image, result.mask, bbox_px)
        trace["baseline"] = json.loads(result.trace.model_dump_json())
        if not BASELINE_LIVE:
            trace["baseline"]["note"] = "MOCK MODE — set VOXAE_VLM_API_KEY for live grounding"
    except Exception as e:
        raise gr.Error(f"Baseline error: {e}") from e

    return trained_overlay, baseline_overlay, trace


BASELINE, BASELINE_LIVE = build_baseline()
TRAINED = build_trained()

ABOUT = """
## Voxae
**Language-grounded, metric scene understanding for physical AI.**

Ask an aerial scene a question in plain language and get the region that
answers it. Two systems run on the same query:

- **Trained `<SEG>` bridge** — a vision-language model whose hidden state at a
  `<SEG>` token is projected directly into SAM 2.1's prompt space and trained
  end to end on Voxae-Reason (referring + affordance queries over drone imagery).
- **Zero-shot baseline** — a hosted VLM emits a box and points as text, SAM 2.1
  turns them into a mask. No training.

The gap is widest on implicit queries: naming an object is easy, reasoning
about what a query *implies* is what the bridge is trained for.
"""

ROADMAP = """
| Stage | Artifact | Status |
|---|---|---|
| Zero-shot baseline | Compose pipeline, live here | done |
| Voxae-Reason dataset | Query/mask annotations over public drone imagery | done |
| Trained `<SEG>` bridge | VLM hidden state to SAM 2.1 prompt space | done |
| Metric 3D | Query a reconstructed scene in the browser | next |

Built in public — code: [github.com/nhipixel/voxae](https://github.com/nhipixel/voxae)
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Voxae") as demo:
        gr.Markdown(ABOUT)
        if TRAINED is None:
            gr.Markdown(
                "**Baseline only** — no trained checkpoint configured "
                "(set VOXAE_CHECKPOINT_DIR to enable the comparison)."
            )
        if not BASELINE_LIVE:
            gr.Markdown("**Mock baseline** — VOXAE_VLM_API_KEY not set; masks are placeholders.")
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Aerial / outdoor image")
                    query_in = gr.Textbox(
                        label="Query",
                        placeholder="e.g. what would block a fire truck reaching the center?",
                    )
                    gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[query_in])
                    run_btn = gr.Button("Segment", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        trained_out = gr.Image(label="Trained <SEG> bridge")
                        baseline_out = gr.Image(label="Zero-shot baseline")
                    trace_out = gr.JSON(label="Run details")
            run_btn.click(
                run_comparison, [image_in, query_in], [trained_out, baseline_out, trace_out]
            )
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
