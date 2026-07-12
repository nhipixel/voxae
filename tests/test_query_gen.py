"""Query generation with a mocked chat function (no network; needs data extra)."""

import json

import numpy as np
import pytest

pytest.importorskip("pycocotools")
pytest.importorskip("scipy")

from voxae.data import rle
from voxae.data.query_gen import (
    GenStats,
    QueryGenError,
    build_facts,
    extract_json_array,
    generate_for_record,
    load_prompt,
)
from voxae.data.schemas import ImageRecord


def _record() -> ImageRecord:
    road = np.zeros((50, 50), dtype=bool)
    road[20:30, :] = True
    return ImageRecord(
        image_id="test-img1",
        dataset="test",
        rel_path="raw/test/img1.png",
        mask_rel_path="raw/test/mask1.png",
        width=50,
        height=50,
        gsd_m_per_px=0.5,
        class_pixel_pct={"road": 20.0},
        components=[],
        class_rles={"road": rle.encode(road)},
    )


VALID_SPEC = {
    "family": "referring",
    "text": "highlight the road crossing the frame",
    "target": {"type": "class_union", "classes": ["road"]},
}
INVALID_SPEC = {
    "family": "referring",
    "text": "highlight the missing water",
    "target": {"type": "class_union", "classes": ["water"]},
}


def test_generate_validates_materializes_and_caches(tmp_path):
    calls = []

    def chat(system, user):
        calls.append(user)
        return json.dumps([VALID_SPEC, INVALID_SPEC])

    stats = GenStats()
    samples = generate_for_record(
        _record(), chat, model="m", cache_dir=tmp_path, n_queries=2, stats=stats
    )
    assert len(samples) == 1  # invalid spec dropped
    assert samples[0].area_pct == pytest.approx(20.0)
    assert samples[0].gen.cached is False
    assert stats.specs_returned == 2 and stats.specs_valid == 1

    # Second run must hit the cache: no new chat calls, samples marked cached.
    again = generate_for_record(_record(), chat, model="m", cache_dir=tmp_path, n_queries=2)
    assert len(calls) == 1
    assert again[0].gen.cached is True


def test_facts_include_gsd_and_classes():
    facts = build_facts(_record())
    assert facts["gsd_m_per_px"] == 0.5
    assert "road" in facts["classes"]


def test_prompt_loads_and_has_placeholders():
    system, user = load_prompt()
    assert "JSON array" in system
    assert "{facts_json}" in user and "{n_queries}" in user


def test_extract_json_array_variants():
    assert extract_json_array('[{"a": 1}]') == [{"a": 1}]
    assert extract_json_array("Sure!\n```json\n[1, 2]\n```") == [1, 2]
    with pytest.raises(QueryGenError, match="no JSON array"):
        extract_json_array("cannot help")
    with pytest.raises(QueryGenError, match="no JSON array"):
        extract_json_array('{"a": 1}')  # bare object is not an array
    with pytest.raises(QueryGenError, match="malformed JSON"):
        extract_json_array("[1, 2,]")
