"""Demo wiring: graceful degradation and comparison output shape."""

import pytest

pytest.importorskip("gradio")

from PIL import Image

from voxae.demo import gradio_app


@pytest.fixture(autouse=True)
def _clear_rate_limit():
    """The limiter is module-level state shared across tests."""
    gradio_app._request_times.clear()
    yield
    gradio_app._request_times.clear()


def test_build_trained_returns_none_without_checkpoint(monkeypatch):
    monkeypatch.setenv("VOXAE_CHECKPOINT_DIR", "")
    assert gradio_app.build_trained() is None


def test_build_trained_survives_bad_checkpoint(monkeypatch, tmp_path):
    # A missing/corrupt checkpoint must not raise — the Space stays up with
    # the baseline column only.
    monkeypatch.setenv("VOXAE_CHECKPOINT_DIR", str(tmp_path / "nope"))
    assert gradio_app.build_trained() is None


def test_run_comparison_returns_three_outputs(monkeypatch):
    # Force mock backends: the module-level BASELINE may have been built with
    # a real API key from .env, and tests must never hit the network.
    from voxae.eval.baselines.zero_shot import ZeroShotPipeline
    from voxae.model.grounder import MockGrounder
    from voxae.model.segmenter import MockSegmenter

    monkeypatch.setattr(gradio_app, "TRAINED", None)
    monkeypatch.setattr(gradio_app, "BASELINE", ZeroShotPipeline(MockGrounder(), MockSegmenter()))
    monkeypatch.setattr(gradio_app, "BASELINE_LIVE", False)

    image = Image.new("RGB", (64, 48), (30, 90, 30))
    trained, baseline, trace = gradio_app.run_comparison(image, "where can a drone land?")
    assert trained is None  # no checkpoint configured
    assert baseline.size == image.size
    assert trace["query"] == "where can a drone land?"
    assert "MOCK MODE" in trace["baseline"]["note"]


def test_run_comparison_rejects_empty_query():
    import gradio as gr

    with pytest.raises(gr.Error):
        gradio_app.run_comparison(Image.new("RGB", (32, 32)), "   ")


def test_build_demo_constructs():
    assert gradio_app.build_demo() is not None


def test_rate_limit_blocks_after_budget():
    """The limiter must trip before any backend runs (no network in tests)."""
    import gradio as gr

    settings = gradio_app.get_settings()
    image = Image.new("RGB", (32, 32))
    for _ in range(settings.demo_rate_limit_per_min):
        gradio_app._rate_limited(settings.demo_rate_limit_per_min)
    with pytest.raises(gr.Error, match="Rate limit"):
        gradio_app.run_comparison(image, "anything")
