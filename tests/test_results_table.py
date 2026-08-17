"""The published tables come from reports, and the count check has to bite.

The site carried a wrong per-family n for months because the report did not
record one, so it was estimated. These cover both halves of the fix: the report
now carries n, and the table script flags a report that disagrees with the
dataset.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "results_table.py"

spec = importlib.util.spec_from_file_location("results_table", SCRIPT)
results_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results_table)


def report(predictor: str, **over) -> dict:
    base = {
        "predictor": predictor,
        "split": "test",
        "tag": None,
        "n": 306,
        "failures": 0,
        "latency_s_mean": 0.25,
        "giou_all": 0.414,
        "ciou_all": 0.502,
        "n_all": 306,
        "giou_affordance": 0.421,
        "ciou_affordance": 0.511,
        "n_affordance": 206,
        "giou_referring": 0.399,
        "ciou_referring": 0.393,
        "n_referring": 100,
    }
    return {**base, **over}


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    trained = report("trained")
    zero = report(
        "zero-shot",
        latency_s_mean=20.8,
        failures=1,
        giou_all=0.371,
        ciou_all=0.307,
        giou_affordance=0.290,
        ciou_affordance=0.290,
        giou_referring=0.538,
        ciou_referring=0.494,
    )
    (tmp_path / "eval_trained_test.json").write_text(json.dumps(trained), encoding="utf-8")
    (tmp_path / "eval_zero-shot_test.json").write_text(json.dumps(zero), encoding="utf-8")
    return tmp_path


def test_load_skips_unparseable_reports(results_dir: Path):
    (results_dir / "eval_broken_test.json").write_text("{not json", encoding="utf-8")
    assert len(results_table.load(results_dir)) == 2


def test_pick_ignores_tagged_reports(results_dir: Path):
    (results_dir / "eval_trained_test__base.json").write_text(
        json.dumps(report("trained", split="val", tag="base")), encoding="utf-8"
    )
    reports = results_table.load(results_dir)
    picked = results_table.pick(reports, "trained", "test")
    assert picked is not None and picked["tag"] is None


def test_headline_carries_the_measured_counts(results_dir: Path):
    reports = results_table.load(results_dir)
    text = "\n".join(
        results_table.headline(
            results_table.pick(reports, "trained", "test"),
            results_table.pick(reports, "zero-shot", "test"),
            "test",
        )
    )
    assert "| 206 |" in text
    assert "| 100 |" in text
    assert "Latency ratio: 83.2x." in text
    assert "cIoU lift, all: +64%." in text


def test_headline_survives_a_missing_predictor(results_dir: Path):
    (results_dir / "eval_zero-shot_test.json").unlink()
    reports = results_table.load(results_dir)
    text = "\n".join(
        results_table.headline(results_table.pick(reports, "trained", "test"), None, "test")
    )
    assert "zero-shot report is missing" in text
    assert "0.4140" in text


def test_record_literal_matches_the_site_shape(results_dir: Path):
    reports = results_table.load(results_dir)
    text = "\n".join(
        results_table.record_literal(
            results_table.pick(reports, "trained", "test"),
            results_table.pick(reports, "zero-shot", "test"),
        )
    )
    assert '{ split: "Affordance questions", trained: "0.421", zero: "0.290", n: "206" },' in text
    assert '{ split: "Referring questions", trained: "0.399", zero: "0.538", n: "100" },' in text


def _dataset(path: Path, affordance: int, referring: int) -> Path:
    rows = [{"split": "test", "family": "affordance"}] * affordance
    rows += [{"split": "test", "family": "referring"}] * referring
    rows += [{"split": "train", "family": "affordance"}] * 5
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_count_check_passes_when_the_report_is_right(results_dir: Path, tmp_path: Path):
    jsonl = _dataset(tmp_path / "d.jsonl", 206, 100)
    text = "\n".join(
        results_table.check_counts(results_table.load(results_dir), jsonl, "test")
    )
    assert "MISMATCH" not in text
    assert "affordance 206" in text


def test_count_check_catches_the_bug_it_exists_for(results_dir: Path, tmp_path: Path):
    """A report claiming 222/84 against a 206/100 dataset has to be flagged."""
    (results_dir / "eval_trained_test.json").write_text(
        json.dumps(report("trained", n_affordance=222, n_referring=84)), encoding="utf-8"
    )
    jsonl = _dataset(tmp_path / "d.jsonl", 206, 100)
    text = "\n".join(
        results_table.check_counts(results_table.load(results_dir), jsonl, "test")
    )
    assert "MISMATCH trained affordance: report says 222, dataset has 206" in text
    assert "MISMATCH trained referring: report says 84, dataset has 100" in text


def test_ablation_table_deltas_against_the_control(tmp_path: Path):
    for tag, giou in (("base", 0.30), ("no_ce", 0.28), ("dice_light", 0.33)):
        (tmp_path / f"eval_trained_val__{tag}.json").write_text(
            json.dumps(
                report("trained", split="val", tag=tag, giou_all=giou, ciou_all=giou + 0.05)
            ),
            encoding="utf-8",
        )
    text = "\n".join(results_table.ablations(results_table.load(tmp_path), "base"))
    assert "| base | 0.3000 | reference |" in text
    assert "-0.0200" in text  # no_ce is worse
    assert "+0.0300" in text  # dice_light is better


def test_ablation_table_is_empty_without_tagged_reports(results_dir: Path):
    assert results_table.ablations(results_table.load(results_dir), "base") == []
