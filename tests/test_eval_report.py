"""The eval report has to carry its own per-family counts.

Without them, anyone publishing a per-family score has to supply the
denominator from somewhere else, and that is how the site ended up stating a
test split it did not have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pycocotools")

from voxae.data import rle
from voxae.data.schemas import GenMeta, QuerySample
from voxae.eval.run_eval import evaluate


def _sample(sample_id: str, family: str, rows: tuple[int, int]) -> QuerySample:
    mask = np.zeros((20, 20), dtype=bool)
    mask[rows[0] : rows[1], :] = True
    encoded = rle.encode(mask)
    return QuerySample(
        sample_id=sample_id,
        dataset="test",
        image_id="img1",
        rel_path="img.png",
        family=family,
        text="highlight the main road",
        target={"type": "class_union", "classes": ["road"]},
        rle=encoded,
        area_pct=rle.area_pct(encoded),
        gen=GenMeta(model="m", prompt_version="qgen_v1", seed=0),
    )


class _Half:
    """Predicts the top half of the frame, whatever it is asked."""

    def predict(self, image, query):
        out = np.zeros((image.height, image.width), dtype=bool)
        out[: image.height // 2] = True
        return out


class _Broken:
    def predict(self, image, query):
        raise RuntimeError("no")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    Image.new("RGB", (20, 20)).save(tmp_path / "img.png")
    return tmp_path


@pytest.fixture
def samples() -> list[QuerySample]:
    return [
        _sample("a1", "affordance", (0, 10)),
        _sample("a2", "affordance", (0, 10)),
        _sample("r1", "referring", (10, 20)),
    ]


def test_report_counts_each_family(data_root: Path, samples):
    report = evaluate(_Half(), samples, data_root, log_every=0)
    assert report["n"] == 3
    assert report["n_all"] == 3
    assert report["n_affordance"] == 2
    assert report["n_referring"] == 1


def test_family_counts_sum_to_the_total(data_root: Path, samples):
    report = evaluate(_Half(), samples, data_root, log_every=0)
    families = [k for k in report if k.startswith("n_") and k != "n_all"]
    assert sum(report[k] for k in families) == report["n_all"]


def test_scores_are_reported_per_family(data_root: Path, samples):
    """The predictor matches affordance exactly and misses referring entirely."""
    report = evaluate(_Half(), samples, data_root, log_every=0)
    assert report["giou_affordance"] == pytest.approx(1.0)
    assert report["giou_referring"] == pytest.approx(0.0)
    # Reports round to four places, so the tolerance has to allow for it.
    assert report["giou_all"] == pytest.approx(2 / 3, abs=1e-4)


def test_a_failing_predictor_is_counted_not_raised(data_root: Path, samples):
    report = evaluate(_Broken(), samples, data_root, log_every=0)
    assert report["failures"] == 3
    assert report["n_all"] == 3


def test_records_resume_without_recalling_the_predictor(data_root: Path, samples, tmp_path: Path):
    """A hosted-API run costs money per call, so a rerun must reuse its records."""
    records = tmp_path / "records.jsonl"
    first = evaluate(_Half(), samples, data_root, log_every=0, records_path=records)
    assert records.exists()

    second = evaluate(_Broken(), samples, data_root, log_every=0, records_path=records)
    assert second["failures"] == 0
    assert second["giou_all"] == pytest.approx(first["giou_all"])
    assert second["n_affordance"] == 2


def test_latency_covers_resumed_samples_too(data_root: Path, samples, tmp_path: Path):
    """A resumed run must report the mean per query, not the mean of the tail.

    The published zero-shot latency came from a resumed run, so it was the mean
    over whatever the final session happened to recompute.
    """
    records = tmp_path / "records.jsonl"
    first = evaluate(_Half(), samples, data_root, log_every=0, records_path=records)

    # Everything resumes, so nothing is timed this pass.
    second = evaluate(_Half(), samples, data_root, log_every=0, records_path=records)
    assert second["latency_s_mean"] is not None
    assert second["latency_s_mean"] == pytest.approx(first["latency_s_mean"], abs=1e-3)
