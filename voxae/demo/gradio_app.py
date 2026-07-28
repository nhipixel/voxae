"""Voxae public demo (Gradio).

The demo runs two predictors side by side on the same query:
- the trained <SEG> bridge (from VOXAE_CHECKPOINT_REPO or VOXAE_CHECKPOINT_DIR),
- the zero-shot compose baseline (hosted VLM grounding + SAM 2.1).

Either side degrades independently: without a checkpoint the trained column
is hidden, and without an API key the baseline falls back to mock output, so
the Space stays clickable in every configuration.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image

from voxae.config import get_settings
from voxae.eval.baselines.zero_shot import ZeroShotPipeline
from voxae.viz import overlay_mask

GALLERY_DIR = Path(__file__).parent / "assets" / "gallery"


def _gpu(duration: int):
    """ZeroGPU allocates a GPU only to decorated functions; a no-op elsewhere."""
    try:
        import spaces
    except ImportError:
        return lambda fn: fn
    return spaces.GPU(duration=duration)


# Implicit, reasoning-style queries: the contrast between a compose baseline
# and a model trained on them is most visible when nothing is named directly.
# Each is paired with a scene that actually contains what it asks about, since
# an affordance query over an image with no such affordance tests nothing.
EXAMPLE_QUERIES: list[tuple[str, str]] = [
    ("what vegetation is close enough to the buildings to be a risk?", "Facultad"),
    ("what would block a fire truck reaching the center?", "Recoleta"),
    ("which surfaces could a heavy vehicle drive on?", "La_Boca"),
    ("where could a small drone land safely?", "Agronom"),
    ("where is there open space clear of overhead obstacles?", "farmland"),
]

_request_times: deque[float] = deque(maxlen=64)
# Gradio injects the real tracker per request; this is only the default binding.
_PROGRESS = gr.Progress()


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
    if not (settings.checkpoint_repo or settings.checkpoint_dir):
        return None
    try:
        from voxae.model.pipeline import VoxaeSegPipeline

        checkpoint_dir = settings.checkpoint_dir
        if settings.checkpoint_repo:
            from huggingface_hub import hf_hub_download

            checkpoint_dir = str(Path(hf_hub_download(settings.checkpoint_repo, "state.pt")).parent)
        return VoxaeSegPipeline.from_checkpoint(
            checkpoint_dir,
            backbone_id=settings.trained_backbone_id,
            sam2_id=settings.sam2_model,
            device=settings.device,
            # Match the training dtype on GPU; CPU has no bfloat16 kernels worth using.
            dtype="bfloat16" if settings.device.startswith("cuda") else "float32",
        )
    except Exception as e:  # a broken checkpoint must not take the Space down
        print(f"trained model unavailable: {e}")
        return None


@lru_cache(maxsize=1)
def _gallery_paths() -> tuple[str, ...]:
    """Sample images, deduplicated by content.

    The fetch script can land the same photo twice under different filename
    encodings, which reads as carelessness in a gallery.
    """
    if not GALLERY_DIR.exists():
        return ()
    seen: set[str] = set()
    paths: list[str] = []
    for p in sorted(GALLERY_DIR.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()  # dedupe, not security
        if digest in seen:
            continue
        seen.add(digest)
        paths.append(str(p))
    return tuple(paths)


def _example_pairs() -> list[list[str | None]]:
    """Image and query together, so one click produces a real result."""
    paths = _gallery_paths()
    pairs: list[list[str | None]] = []
    for i, (query, subject) in enumerate(EXAMPLE_QUERIES):
        match = next((p for p in paths if subject.lower() in Path(p).name.lower()), None)
        if match is None and paths:  # gallery contents drift; still offer the query
            match = paths[i % len(paths)]
        pairs.append([match, query])
    return pairs


def _summary_line(label: str, latency: float, area_pct: float) -> str:
    """One human-readable line per predictor; empty masks say so explicitly."""
    if area_pct < 0.01:
        return f"**{label}** &middot; {latency:.1f}s &middot; no region matched"
    return f"**{label}** &middot; {latency:.1f}s &middot; {area_pct:.1f}% of the image"


BENCHMARK_NOTE = (
    "Held-out test split, 306 samples: **gIoU 0.42 vs 0.29 on affordance queries** "
    "(trained vs zero-shot), 0.40 vs 0.54 on referring. Uploaded images have no "
    "ground truth, so nothing here is scored; the numbers above are latency and coverage."
)


def _agreement_line(trained_mask, baseline_mask) -> str:
    """How far apart the two answers are, for a query with no ground truth.

    Not a quality score. It separates 'both found the same region' from 'these
    two systems disagree about what the question means'.
    """
    if trained_mask is None:
        return ""
    from voxae.eval.metrics import iou

    return f"**Agreement between the two masks**: IoU {iou(trained_mask, baseline_mask):.2f}"


def _prepare(image: Image.Image | None, query: str) -> Image.Image:
    settings = get_settings()
    if image is None:
        raise gr.Error("Pick an example below or upload an image to get started.")
    if not query or not query.strip():
        raise gr.Error("Type a query, for example 'where could a drone land safely?'.")
    if _rate_limited(settings.demo_rate_limit_per_min):
        raise gr.Error("Rate limit reached. Wait a minute and try again.")
    if max(image.size) > settings.demo_max_image_px:
        image = image.copy()
        image.thumbnail((settings.demo_max_image_px, settings.demo_max_image_px))
    return image


@_gpu(duration=120)
def run_comparison(image: Image.Image | None, query: str, progress=_PROGRESS):
    """Run both predictors; returns (trained, baseline, summary, trace)."""
    image = _prepare(image, query)
    trace: dict = {"query": query.strip()}
    lines: list[str] = []
    trained_mask = None

    trained_overlay = None
    if TRAINED is not None:
        progress(0.1, desc="Trained bridge: one forward pass")
        t0 = time.perf_counter()
        try:
            trained_mask = TRAINED.predict(image, query)
            latency = time.perf_counter() - t0
            area = float(trained_mask.mean()) * 100
            trained_overlay = overlay_mask(image, trained_mask)
            lines.append(_summary_line("Trained bridge", latency, area))
            trace["trained"] = {
                "model": TRAINED.name,
                "latency_s": round(latency, 2),
                "mask_area_pct": round(area, 2),
            }
        except Exception as e:
            lines.append(f"**Trained bridge** &middot; failed: {e}")
            trace["trained"] = {"error": str(e)}

    progress(0.45, desc="Baseline: asking a hosted VLM for a box")
    t0 = time.perf_counter()
    try:
        result = BASELINE.run(image, query)
        latency = time.perf_counter() - t0
        area = float(result.mask.mean()) * 100
        bbox_px = result.trace.grounding.bbox.to_pixels(*image.size)
        baseline_overlay = overlay_mask(image, result.mask, bbox_px)
        lines.append(_summary_line("Zero-shot baseline", latency, area))
        trace["baseline"] = json.loads(result.trace.model_dump_json())
        if not BASELINE_LIVE:
            trace["baseline"]["note"] = "MOCK MODE - set VOXAE_VLM_API_KEY for live grounding"
    except Exception as e:
        raise gr.Error(f"Baseline error: {e}") from e

    progress(0.95, desc="Comparing")
    lines.append(_agreement_line(trained_mask, result.mask))
    if result.trace.grounding.rationale:
        lines.append(f"> Baseline reasoning: *{result.trace.grounding.rationale}*")
    lines.append(BENCHMARK_NOTE)

    return trained_overlay, baseline_overlay, "\n\n".join(lines), trace


BASELINE, BASELINE_LIVE = build_baseline()
TRAINED = build_trained()

ABOUT = """
## Voxae
**Language-grounded, metric scene understanding for physical AI.**

Ask an aerial scene a question in plain language and get the region that
answers it. Two systems run on the same query:

- **Trained `<SEG>` bridge**: a vision-language model whose hidden state at a
  `<SEG>` token is projected directly into SAM 2.1's prompt space and trained
  end to end on Voxae-Reason (referring + affordance queries over drone imagery).
- **Zero-shot baseline**: a hosted VLM emits a box and points as text, SAM 2.1
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

Built in public. Code: [github.com/nhipixel/voxae](https://github.com/nhipixel/voxae)
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Voxae") as demo:
        gr.Markdown(ABOUT)
        if TRAINED is None:
            gr.Markdown(
                "**Baseline only**: no trained checkpoint configured "
                "(set VOXAE_CHECKPOINT_REPO to a Hub model repo, or "
                "VOXAE_CHECKPOINT_DIR to a local path, to enable the comparison)."
            )
        if not BASELINE_LIVE:
            gr.Markdown("**Mock baseline**: VOXAE_VLM_API_KEY not set; masks are placeholders.")
        with gr.Tabs() as tabs:
            with gr.Tab("Demo", id="demo"), gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Aerial / outdoor image")
                    query_in = gr.Textbox(
                        label="Query",
                        placeholder="e.g. what would block a fire truck reaching the center?",
                    )
                    run_btn = gr.Button("Segment", variant="primary")
                    gr.Markdown("Or start from a worked example:")
                    gr.Examples(examples=_example_pairs(), inputs=[image_in, query_in])
                with gr.Column(scale=2):
                    with gr.Row():
                        # Hidden without a checkpoint: an empty pane beside a
                        # filled one reads as broken rather than unavailable.
                        trained_out = gr.Image(
                            label="Trained <SEG> bridge", visible=TRAINED is not None
                        )
                        baseline_out = gr.Image(label="Zero-shot baseline")
                    summary_out = gr.Markdown()
                    with gr.Accordion("Run details", open=False):
                        trace_out = gr.JSON()
            with gr.Tab("Example images", id="gallery"):
                gallery = gr.Gallery(
                    value=list(_gallery_paths()),
                    columns=4,
                    label="Click an image to load it into the demo",
                )
                gr.Markdown("Images are CC0/CC-BY from Wikimedia Commons. See ATTRIBUTION.md.")
            with gr.Tab("Roadmap", id="roadmap"):
                gr.Markdown(ROADMAP)

        outputs = [trained_out, baseline_out, summary_out, trace_out]
        run_btn.click(run_comparison, [image_in, query_in], outputs)
        query_in.submit(run_comparison, [image_in, query_in], outputs)

        def _load_from_gallery(evt: gr.SelectData):
            """Selecting a sample image drops it into the demo tab, ready to run."""
            paths = _gallery_paths()
            chosen = paths[evt.index] if 0 <= evt.index < len(paths) else None
            return chosen, gr.Tabs(selected="demo")

        gallery.select(_load_from_gallery, None, [image_in, tabs])
    return demo


def main() -> None:
    """Serve the demo. VOXAE_SHARE=1 requests a temporary public tunnel."""
    import os

    share = os.environ.get("VOXAE_SHARE", "").lower() in {"1", "true", "yes"}
    # Inference is serial on one accelerator, so queue rather than run concurrently.
    build_demo().queue(default_concurrency_limit=1).launch(theme=gr.themes.Soft(), share=share)


if __name__ == "__main__":
    main()
