"""JSON extraction from noisy VLM output."""

import pytest

from voxae.model.grounder import GrounderError, extract_json


def test_extracts_bare_json():
    text = '{"bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "points": []}'
    assert extract_json(text)["bbox"]["x1"] == 1


def test_extracts_fenced_json():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert extract_json(text) == {"a": 1}


def test_extracts_json_with_surrounding_prose():
    text = 'The region is the road. {"a": {"b": 2}} That is my answer.'
    assert extract_json(text) == {"a": {"b": 2}}


def test_raises_on_no_json():
    with pytest.raises(GrounderError, match="no JSON object"):
        extract_json("I cannot find the region, sorry.")


def test_raises_on_malformed_json():
    with pytest.raises(GrounderError, match="malformed JSON"):
        extract_json('{"a": unquoted}')


def test_takes_the_first_object_when_the_model_returns_several():
    """Asked for one region, models sometimes list many; a greedy match spans them all."""
    text = (
        '{"bbox": [292, 0, 393, 162], "points": [{"x": 3, "y": 6}], "rationale": "a"},\n'
        '{"bbox": [415, 211, 479, 368], "points": [{"x": 4, "y": 5}]}'
    )
    assert extract_json(text)["bbox"] == [292, 0, 393, 162]


def test_braces_inside_strings_do_not_end_the_object():
    text = '{"rationale": "a road }", "bbox": [1, 2, 3, 4]}'
    assert extract_json(text)["bbox"] == [1, 2, 3, 4]
