"""Mask loss components (requires the ml extra)."""

import pytest

pytest.importorskip("torch")

import torch

from voxae.model.losses import dice_loss, mask_loss


def test_dice_perfect_prediction_near_zero():
    target = torch.zeros(2, 8, 8)
    target[:, :4] = 1.0
    logits = torch.where(
        target.bool(), torch.full_like(target, 20.0), torch.full_like(target, -20.0)
    )
    assert dice_loss(logits, target).item() < 1e-3


def test_dice_worst_prediction_near_one():
    target = torch.zeros(1, 8, 8)
    target[:, :4] = 1.0
    logits = torch.where(
        target.bool(), torch.full_like(target, -20.0), torch.full_like(target, 20.0)
    )
    assert dice_loss(logits, target).item() > 0.95


def test_mask_loss_weighting():
    target = torch.rand(2, 8, 8).round()
    logits = torch.randn(2, 8, 8)
    parts = mask_loss(logits, target, bce_weight=2.0, dice_weight=0.5)
    expected = 2.0 * parts["loss_bce"] + 0.5 * parts["loss_dice"]
    assert torch.allclose(parts["loss_mask"], expected)
