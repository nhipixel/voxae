"""Resolve a symbolic TargetSpec into a concrete binary mask.

This is the geometry half of query generation: specs authored by the LLM are
turned into masks purely from dataset class masks, so ground truth is exactly
reproducible and never depends on model output beyond the spec itself.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from voxae.data.prep_masks import MIN_COMPONENT_AREA_PCT, label_components
from voxae.data.schemas import (
    ClassUnionTarget,
    ComponentsTarget,
    MetricFilterTarget,
    TargetSpec,
)


class MaterializeError(ValueError):
    """Raised when a spec cannot be resolved against an image's masks."""


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation with a square kernel (PIL MaxFilter; size must be odd)."""
    if radius_px <= 0:
        return mask
    size = 2 * radius_px + 1
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MaxFilter(size))) > 0


def _component_attr(m: np.ndarray, attr: str, gsd: float) -> float:
    ys, xs = np.nonzero(m)
    if attr == "width_m":
        return float(xs.max() + 1 - xs.min()) * gsd
    if attr == "height_m":
        return float(ys.max() + 1 - ys.min()) * gsd
    if attr == "area_m2":
        return float(m.sum()) * gsd**2
    raise MaterializeError(f"unknown metric attribute: {attr}")


def materialize(
    target: TargetSpec,
    class_masks: dict[str, np.ndarray],
    gsd_m_per_px: float | None = None,
) -> np.ndarray:
    """Compute the ground-truth mask for a spec from per-class masks."""
    if isinstance(target, ClassUnionTarget):
        missing = [c for c in target.classes if c not in class_masks]
        if missing:
            raise MaterializeError(f"classes not present in image: {missing}")
        out = np.zeros_like(next(iter(class_masks.values())), dtype=bool)
        for c in target.classes:
            out |= class_masks[c]
        if target.exclude_near is not None:
            near_cls = target.exclude_near.cls
            if near_cls not in class_masks:
                raise MaterializeError(f"exclude_near class not present: {near_cls}")
            out &= ~dilate(class_masks[near_cls], target.exclude_near.radius_px)
        if target.min_component_area_pct:
            total = out.size
            kept = np.zeros_like(out)
            for m in label_components(out):
                if m.sum() * 100.0 / total >= target.min_component_area_pct:
                    kept |= m
            out = kept
        if not out.any():
            raise MaterializeError("constraints removed all target pixels")
        return out

    if isinstance(target, ComponentsTarget):
        if target.cls not in class_masks:
            raise MaterializeError(f"class not present in image: {target.cls}")
        comps = _eligible_components(class_masks[target.cls])
        out = np.zeros_like(class_masks[target.cls], dtype=bool)
        for comp_id in target.comp_ids:
            if comp_id >= len(comps):
                raise MaterializeError(
                    f"comp_id {comp_id} out of range ({len(comps)} eligible components)"
                )
            out |= comps[comp_id]
        return out

    if isinstance(target, MetricFilterTarget):
        if gsd_m_per_px is None:
            raise MaterializeError("metric target requires a GSD estimate")
        if target.cls not in class_masks:
            raise MaterializeError(f"class not present in image: {target.cls}")
        out = np.zeros_like(class_masks[target.cls], dtype=bool)
        for m in _eligible_components(class_masks[target.cls]):
            value = _component_attr(m, target.attr, gsd_m_per_px)
            keep = value >= target.value if target.op == ">=" else value <= target.value
            if keep:
                out |= m
        if not out.any():
            raise MaterializeError("no components satisfy the metric predicate")
        return out

    raise MaterializeError(f"unsupported target type: {type(target).__name__}")


def _eligible_components(mask: np.ndarray) -> list[np.ndarray]:
    """Components in the same deterministic order used by prep_masks.

    The same area floor is applied so comp_id ranks always line up with the
    ComponentRecords the LLM saw in its prompt facts.
    """
    total = mask.size
    return [m for m in label_components(mask) if m.sum() * 100.0 / total >= MIN_COMPONENT_AREA_PCT]
