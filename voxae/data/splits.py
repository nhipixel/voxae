"""Deterministic train/val/test splits, grouped by image.

Splitting is by image (never by sample) so no image's queries leak across
splits, and stratified by source dataset so each split keeps the same
dataset mix. The assignment is a pure function of (image ids, ratios, seed).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from voxae.data.schemas import QuerySample

DEFAULT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    image_ids_by_dataset: dict[str, list[str]],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 0,
) -> dict[str, str]:
    """Map image_id -> split, stratified per dataset."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    assignment: dict[str, str] = {}
    for dataset in sorted(image_ids_by_dataset):
        ids = sorted(set(image_ids_by_dataset[dataset]))
        random.Random(f"{seed}:{dataset}").shuffle(ids)
        n = len(ids)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, image_id in enumerate(ids):
            if i < n_train:
                assignment[image_id] = "train"
            elif i < n_train + n_val:
                assignment[image_id] = "val"
            else:
                assignment[image_id] = "test"
    return assignment


def apply_splits(
    qc_jsonl: Path,
    out_jsonl: Path,
    splits_path: Path,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 0,
) -> dict[str, int]:
    """Assign splits to QC-passed samples and write the final dataset JSONL."""
    by_dataset: dict[str, list[str]] = defaultdict(list)
    samples: list[QuerySample] = []
    with qc_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = QuerySample.model_validate_json(line)
            samples.append(s)
            by_dataset[s.dataset].append(s.image_id)

    assignment = make_splits(by_dataset, ratios=ratios, seed=seed)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(assignment, indent=1, sort_keys=True), encoding="utf-8")

    counts: dict[str, int] = dict.fromkeys(SPLIT_NAMES, 0)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for s in samples:
            s.split = assignment[s.image_id]
            counts[s.split] += 1
            f.write(s.model_dump_json() + "\n")
    return counts
