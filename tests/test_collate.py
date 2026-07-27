"""Collator pure helpers (requires ml + data extras)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pycocotools")

import numpy as np
import torch
from PIL import Image

from voxae.data import rle
from voxae.data.schemas import GenMeta, QuerySample
from voxae.train.collate import VoxaeCollator, load_samples, mask_ce_labels, vlm_image


def _sample(split=None):
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, :] = True
    encoded = rle.encode(mask)
    return QuerySample(
        sample_id="s1",
        dataset="test",
        image_id="img1",
        rel_path="raw/test/img.png",
        family="referring",
        text="highlight the band",
        target={"type": "class_union", "classes": ["road"]},
        rle=encoded,
        area_pct=rle.area_pct(encoded),
        gen=GenMeta(model="m", prompt_version="qgen_v2", seed=0),
        split=split,
    )


def test_load_samples_filters_by_split(tmp_path):
    p = tmp_path / "d.jsonl"
    lines = [_sample("train").model_dump_json(), _sample("val").model_dump_json()]
    p.write_text("\n".join(lines), encoding="utf-8")
    assert len(load_samples(p)) == 2
    assert len(load_samples(p, split="train")) == 1


def test_mask_ce_labels_ignores_prompt():
    ids = torch.tensor([1, 2, 3, 4, 5])
    labels = mask_ce_labels(ids, assistant_start=3)
    assert labels[:3].tolist() == [-100, -100, -100]
    assert labels[3:].tolist() == [4, 5]


def test_vlm_image_bounds_size_and_keeps_aspect():
    out = vlm_image(Image.new("RGB", (4000, 2000)), max_px=448)
    assert max(out.size) == 448
    assert out.size == (448, 224)


def test_resize_mask_preserves_binary_values():
    mask = torch.zeros(32, 32)
    mask[8:24, :] = 1.0
    out = VoxaeCollator._resize_mask(mask, 64)
    assert out.shape == (64, 64)
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert out.mean().item() == pytest.approx(0.5, abs=0.05)
