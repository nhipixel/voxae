"""Segmentation losses for bridge training (LISA-style weighting)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss over per-sample masks. logits/targets: (B, H, W)."""
    probs = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()


def mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 2.0,
    dice_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Combined BCE + Dice mask loss; returns components for logging."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return {
        "loss_bce": bce,
        "loss_dice": dice,
        "loss_mask": bce_weight * bce + dice_weight * dice,
    }
