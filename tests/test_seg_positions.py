"""<SEG> hidden-state extraction (requires the ml extra)."""

import pytest

pytest.importorskip("torch")

import torch

from voxae.model.qwen_backbone import seg_hidden_states

SEG = 7


def test_picks_last_seg_position_per_sample():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    ids = torch.tensor([[1, SEG, 2, SEG, 3], [SEG, 1, 2, 3, 4]])
    out = seg_hidden_states(hidden, ids, SEG)
    assert torch.equal(out[0], hidden[0, 3])  # last SEG at position 3
    assert torch.equal(out[1], hidden[1, 0])


def test_missing_seg_raises():
    hidden = torch.zeros(1, 4, 3)
    ids = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="<SEG>"):
        seg_hidden_states(hidden, ids, SEG)
