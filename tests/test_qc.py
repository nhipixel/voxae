"""QC rules and dedupe (needs data extra for RLE)."""

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from voxae.data import rle
from voxae.data.qc import check_sample, run_qc
from voxae.data.schemas import GenMeta, QuerySample


def _sample(text="highlight the main road", area_rows=(10, 40), image_id="img1", **kw):
    mask = np.zeros((100, 100), dtype=bool)
    mask[area_rows[0] : area_rows[1], :] = True
    encoded = rle.encode(mask)
    return QuerySample(
        sample_id=kw.get("sample_id", f"{image_id}-{text[:8]}"),
        dataset="test",
        image_id=image_id,
        rel_path="raw/test/img.png",
        family=kw.get("family", "referring"),
        text=text,
        target={"type": "class_union", "classes": ["road"]},
        rle=encoded,
        area_pct=rle.area_pct(encoded),
        gsd_m_per_px=kw.get("gsd_m_per_px", 0.5),
        gen=GenMeta(model="m", prompt_version="qgen_v1", seed=0),
    )


def test_valid_sample_passes():
    assert check_sample(_sample()) is None


def test_short_text_fails():
    assert check_sample(_sample(text="the road")) == "text_too_short"


def test_area_bounds():
    assert check_sample(_sample(area_rows=(0, 90))) == "area_out_of_bounds"  # 90%
    tiny = _sample()
    tiny.area_pct = 0.01
    assert check_sample(tiny) == "area_out_of_bounds"


def test_metric_without_gsd_fails():
    s = _sample(gsd_m_per_px=None)
    s.target = {"type": "metric_filter", "cls": "road", "attr": "width_m", "op": ">=", "value": 2.5}
    s = QuerySample.model_validate(s.model_dump())
    assert check_sample(s) == "metric_without_gsd"


def test_singular_reference_too_large_fails():
    # comp_ids=[0] (a "specific object" reference) but 30% of the frame —
    # the merged-cluster signature on semantic (non-instance) segmentation.
    s = _sample(area_rows=(0, 30))  # 30%
    s.target = {"type": "components", "cls": "building", "comp_ids": [0]}
    s = QuerySample.model_validate(s.model_dump())
    assert check_sample(s) == "singular_reference_too_large"


def test_components_target_small_area_passes():
    s = _sample(area_rows=(0, 5))  # 5%, under the 12% cap
    s.target = {"type": "components", "cls": "building", "comp_ids": [0]}
    s = QuerySample.model_validate(s.model_dump())
    assert check_sample(s) is None


def test_components_target_many_ids_large_area_passes():
    # Explicitly plural ("these 5 cars") — large combined area is expected
    # and not a merged-single-object mismatch.
    s = _sample(area_rows=(0, 30))  # 30%
    s.target = {"type": "components", "cls": "static_car", "comp_ids": [0, 1, 2, 3, 4]}
    s = QuerySample.model_validate(s.model_dump())
    assert check_sample(s) is None


def test_run_qc_dedupes_and_reports(tmp_path):
    raw = tmp_path / "raw.jsonl"
    lines = [
        _sample().model_dump_json(),
        _sample(sample_id="dup").model_dump_json(),  # same image + same text
        _sample(text="the road", sample_id="short").model_dump_json(),
    ]
    raw.write_text("\n".join(lines), encoding="utf-8")

    out = tmp_path / "qc.jsonl"
    report = run_qc(raw, out, tmp_path / "report.json")
    assert report.total == 3
    assert report.passed == 1
    assert report.failures["duplicate_text"] == 1
    assert report.failures["text_too_short"] == 1
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1
