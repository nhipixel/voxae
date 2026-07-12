"""Segmentation metrics: per-sample gIoU and dataset-level cIoU.

Definitions (following the referring/reasoning-segmentation literature, e.g.
LISA / RRSIS-D):

- IoU per sample:  |pred AND gt| / |pred OR gt|
- gIoU: mean of per-sample IoUs (every sample weighs equally).
- cIoU: cumulative intersection over cumulative union across the dataset
  (large objects weigh more).
"""

from __future__ import annotations

import numpy as np


def _validate(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    return pred.astype(bool), gt.astype(bool)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU of two binary masks. Convention: both empty -> 1.0 (correct 'nothing')."""
    pred, gt = _validate(pred, gt)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def giou(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    """Mean per-sample IoU."""
    if len(preds) != len(gts) or not preds:
        raise ValueError("preds and gts must be equal-length, non-empty lists")
    return float(np.mean([iou(p, g) for p, g in zip(preds, gts, strict=True)]))


def ciou(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    """Cumulative-intersection / cumulative-union across all samples."""
    if len(preds) != len(gts) or not preds:
        raise ValueError("preds and gts must be equal-length, non-empty lists")
    total_i = 0
    total_u = 0
    for p, g in zip(preds, gts, strict=True):
        p, g = _validate(p, g)
        total_i += int(np.logical_and(p, g).sum())
        total_u += int(np.logical_or(p, g).sum())
    if total_u == 0:
        return 1.0
    return float(total_i / total_u)
