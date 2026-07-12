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
