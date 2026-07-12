"""Automatic quality checks over generated query samples.

Every sample passes through a fixed rule set; failures are recorded per rule
so the QC report shows exactly where yield is lost. Deduplication is by
normalized text within an image (same wording pointing at anything is a
duplicate — retrieval-style ambiguity, not signal).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from voxae.data import rle
from voxae.data.schemas import ComponentsTarget, MetricFilterTarget, QuerySample

AREA_MIN_PCT = 0.1
AREA_MAX_PCT = 60.0
TEXT_MIN_WORDS = 3

# A `components` target with few comp_ids reads as "a specific object" (e.g.
# "the building on the left"). On semantic (non-instance) segmentation,
# touching instances of a class merge into one connected component, so
# comp_id 0 can legitimately be a fused cluster of several objects rather
# than one. A real single object (a car, a building that doesn't touch its
# neighbors) is a small fraction of the frame; observed genuine cases run
# under 1%. A large area at a low comp_id count is the merged-cluster
# signature, not a materialization bug — flag it rather than ship a mask
# that contradicts its own "the X" singular phrasing.
SINGULAR_REF_MAX_COMP_IDS = 2
SINGULAR_REF_MAX_AREA_PCT = 12.0


@dataclass
class QCReport:
    total: int = 0
    passed: int = 0
    failures: Counter = field(default_factory=Counter)
    by_family: Counter = field(default_factory=Counter)
    by_dataset: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.passed / self.total, 4) if self.total else 0.0,
            "failures": dict(self.failures),
            "passed_by_family": dict(self.by_family),
            "passed_by_dataset": dict(self.by_dataset),
        }


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def check_sample(sample: QuerySample) -> str | None:
    """Return the name of the first failed rule, or None if all pass."""
    if len(sample.text.split()) < TEXT_MIN_WORDS:
        return "text_too_short"
    if not (AREA_MIN_PCT <= sample.area_pct <= AREA_MAX_PCT):
        return "area_out_of_bounds"
    if isinstance(sample.target, MetricFilterTarget) and sample.gsd_m_per_px is None:
        return "metric_without_gsd"
    if (
        isinstance(sample.target, ComponentsTarget)
        and len(sample.target.comp_ids) <= SINGULAR_REF_MAX_COMP_IDS
        and sample.area_pct > SINGULAR_REF_MAX_AREA_PCT
    ):
        return "singular_reference_too_large"
    try:
        mask = rle.decode(sample.rle)
    except Exception:
        return "rle_undecodable"
    if not mask.any():
        return "mask_empty"
    return None


def run_qc(in_jsonl: Path, out_jsonl: Path, report_path: Path) -> QCReport:
    """Filter a raw samples JSONL into a QC-passed JSONL plus a report."""
    report = QCReport()
    seen: set[tuple[str, str]] = set()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with in_jsonl.open(encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            report.total += 1
            sample = QuerySample.model_validate_json(line)

            failure = check_sample(sample)
            if failure is None:
                key = (sample.image_id, _norm_text(sample.text))
                if key in seen:
                    failure = "duplicate_text"
                else:
                    seen.add(key)
            if failure is not None:
                report.failures[failure] += 1
                continue

            report.passed += 1
            report.by_family[sample.family] += 1
            report.by_dataset[sample.dataset] += 1
            fout.write(sample.model_dump_json() + "\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
