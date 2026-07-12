"""End-to-end zero-shot pipeline with mock backends (no network, no weights)."""

import numpy as np
import pytest
from PIL import Image

from voxae.eval.baselines.zero_shot import ZeroShotPipeline
from voxae.model.grounder import MockGrounder
from voxae.model.segmenter import MockSegmenter
from voxae.viz import overlay_mask


@pytest.fixture()
def pipeline():
    return ZeroShotPipeline(MockGrounder(), MockSegmenter())


@pytest.fixture()
def image():
    return Image.new("RGB", (320, 240), (120, 140, 90))


def test_pipeline_returns_mask_of_image_size(pipeline, image):
    result = pipeline.run(image, "highlight the open area")
    assert result.mask.shape == (240, 320)
    assert result.mask.dtype == bool
    assert result.mask.any(), "mock pipeline should produce a non-empty mask"


def test_pipeline_trace_fields(pipeline, image):
    result = pipeline.run(image, "  where can a drone land?  ")
    t = result.trace
    assert t.query == "where can a drone land?"  # stripped
    assert t.image_width == 320 and t.image_height == 240
    assert t.total_latency_ms >= t.vlm_latency_ms
    assert "mock" in t.model


def test_pipeline_rejects_empty_query(pipeline, image):
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.run(image, "   ")


def test_mock_mask_matches_bbox_region(pipeline, image):
    result = pipeline.run(image, "anything")
    # MockGrounder returns the centered half-size box; mask must be inside it.
    ys, xs = np.nonzero(result.mask)
    assert xs.min() >= 320 * 0.25 - 1 and xs.max() <= 320 * 0.75 + 1
    assert ys.min() >= 240 * 0.25 - 1 and ys.max() <= 240 * 0.75 + 1


def test_overlay_returns_same_size_image(pipeline, image):
    result = pipeline.run(image, "anything")
    bbox_px = result.trace.grounding.bbox.to_pixels(*image.size)
    out = overlay_mask(image, result.mask, bbox_px)
    assert out.size == image.size


def test_overlay_rejects_mismatched_mask(image):
    with pytest.raises(ValueError, match="mask shape"):
        overlay_mask(image, np.zeros((10, 10), dtype=bool))
