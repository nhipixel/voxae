"""Schema validation + coordinate conversion tests."""

import pytest
from pydantic import ValidationError

from voxae.data.schemas import BBoxNorm, GroundingResult, PointNorm


def test_bbox_to_pixels_scales_correctly():
    bbox = BBoxNorm(x1=0, y1=0, x2=500, y2=1000)
    assert bbox.to_pixels(width=200, height=100) == (0.0, 0.0, 100.0, 100.0)


def test_bbox_rejects_reversed_coords():
    with pytest.raises(ValidationError):
        BBoxNorm(x1=600, y1=0, x2=500, y2=1000)
    with pytest.raises(ValidationError):
        BBoxNorm(x1=0, y1=900, x2=500, y2=800)


def test_bbox_rejects_out_of_range():
    with pytest.raises(ValidationError):
        BBoxNorm(x1=-1, y1=0, x2=500, y2=1000)
    with pytest.raises(ValidationError):
        BBoxNorm(x1=0, y1=0, x2=1500, y2=1000)


def test_point_to_pixels():
    p = PointNorm(x=250, y=750)
    assert p.to_pixels(width=400, height=400) == (100.0, 300.0)


def test_grounding_result_parses_full_payload():
    payload = {
        "bbox": {"x1": 10, "y1": 20, "x2": 400, "y2": 600},
        "points": [{"x": 100, "y": 200}],
        "rationale": "the paved area next to the building",
    }
    result = GroundingResult.model_validate(payload)
    assert result.bbox.x2 == 400
    assert len(result.points) == 1


def test_grounding_result_falls_back_to_the_box_centre():
    """A box with no points still localizes the target; SAM takes box-only prompts."""
    payload = {"bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}, "points": []}
    result = GroundingResult.model_validate(payload)
    assert [(p.x, p.y) for p in result.points] == [(5, 10)]


def test_grounding_accepts_array_boxes_and_parallel_array_points():
    """Hosted models emit these shapes regardless of the requested schema."""
    from voxae.data.schemas import GroundingResult

    r = GroundingResult.model_validate(
        {"bbox": [292, 0, 393, 162], "points": [{"x": [333, 60], "y": [60, 120]}]}
    )
    assert (r.bbox.x1, r.bbox.y2) == (292, 162)
    assert [(p.x, p.y) for p in r.points] == [(333, 60), (60, 120)]
