"""Build unified per-image records from raw dataset masks.

For every image/mask pair: decode the mask into per-class boolean arrays,
extract connected components with spatial descriptors, estimate GSD, and
serialize everything (class RLEs included) into an ``ImageRecord``. Records
are written to parquet, one row per image, RLEs as JSON strings.

Component labeling is deterministic (area-descending, bbox tie-break), which
lets query specs reference components by rank without storing per-component
masks — the materializer reproduces the same labeling at resolution time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from voxae.data import rle
from voxae.data.gsd import estimate_gsd
from voxae.data.registry import REGISTRY, DatasetSpec, decode_mask, iter_samples
from voxae.data.schemas import ComponentRecord, ImageRecord

# Guardrails for component extraction: skip specks, cap the count so prompt
# facts stay small.
MIN_COMPONENT_AREA_PCT = 0.05
MAX_COMPONENTS_PER_CLASS = 8

GRID_NAMES = [
    ["top-left", "top", "top-right"],
    ["left", "center", "right"],
    ["bottom-left", "bottom", "bottom-right"],
]


def grid_cell(cx: float, cy: float, width: int, height: int) -> str:
    """Name of the 3x3 grid cell containing a point."""
    col = min(2, int(3 * cx / width))
    row = min(2, int(3 * cy / height))
    return GRID_NAMES[row][col]


def label_components(mask: np.ndarray) -> list[np.ndarray]:
    """Split a boolean mask into per-component masks in deterministic order.

    Order: area descending, then bbox top-left (y, then x). scipy's labeling
    is itself deterministic; the explicit sort makes rank stable across runs.
    """
    try:
        from scipy import ndimage
    except ImportError as e:  # pragma: no cover - environment-specific
        raise RuntimeError(
            f"component labeling requires the 'data' extra: uv sync --extra data ({e})"
        ) from e

    labeled, n = ndimage.label(mask)
    comps: list[tuple[int, int, int, np.ndarray]] = []
    for i in range(1, n + 1):
        m = labeled == i
        ys, xs = np.nonzero(m)
        comps.append((int(m.sum()), int(ys.min()), int(xs.min()), m))
    comps.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [m for _, _, _, m in comps]


def extract_components(
    cls: str,
    category: str,
    mask: np.ndarray,
    gsd_m_per_px: float | None,
) -> list[ComponentRecord]:
    """Connected components of one class mask, with spatial/metric descriptors."""
    h, w = mask.shape
    total = h * w
    records: list[ComponentRecord] = []
    for comp_id, m in enumerate(label_components(mask)):
        area_px = int(m.sum())
        area_pct = area_px * 100.0 / total
        if area_pct < MIN_COMPONENT_AREA_PCT:
            break  # components are area-sorted; the rest are smaller
        if comp_id >= MAX_COMPONENTS_PER_CLASS:
            break
        ys, xs = np.nonzero(m)
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        cx, cy = float(xs.mean()), float(ys.mean())
        width_m = height_m = area_m2 = None
        if gsd_m_per_px is not None:
            width_m = round((x2 - x1) * gsd_m_per_px, 2)
            height_m = round((y2 - y1) * gsd_m_per_px, 2)
            area_m2 = round(area_px * gsd_m_per_px**2, 2)
        records.append(
            ComponentRecord(
                comp_id=comp_id,
                cls=cls,
                category=category,
                area_px=area_px,
                area_pct=round(area_pct, 4),
                bbox_px=[x1, y1, x2, y2],
                centroid=[round(cx, 1), round(cy, 1)],
                grid_cell=grid_cell(cx, cy, w, h),
                width_m=width_m,
                height_m=height_m,
                area_m2=area_m2,
            )
        )
    return records


def build_record(
    spec: DatasetSpec,
    image_path: Path,
    mask_path: Path,
    data_root: Path,
    altitude_m: float | None = None,
) -> ImageRecord:
    """Build one ImageRecord from an image/mask pair."""
    with Image.open(image_path) as img:
        width, height = img.size
    mask_arr = np.asarray(Image.open(mask_path))
    class_masks, unknown_pct = decode_mask(spec, mask_arr)

    est = estimate_gsd(image_path, altitude_m=altitude_m)
    gsd = est.meters_per_px if est else None

    components: list[ComponentRecord] = []
    class_pct: dict[str, float] = {}
    class_rles = {}
    total = mask_arr.shape[0] * mask_arr.shape[1]
    for cls, m in class_masks.items():
        class_pct[cls] = round(float(m.sum()) * 100.0 / total, 4)
        class_rles[cls] = rle.encode(m)
        components.extend(extract_components(cls, str(spec.categories.get(cls, "other")), m, gsd))

    return ImageRecord(
        image_id=f"{spec.name}-{image_path.stem}",
        dataset=spec.name,
        rel_path=str(image_path.relative_to(data_root)).replace("\\", "/"),
        mask_rel_path=str(mask_path.relative_to(data_root)).replace("\\", "/"),
        width=width,
        height=height,
        gsd_m_per_px=gsd,
        gsd_confidence=est.confidence if est else "low",
        gsd_source=est.source if est else "",
        class_pixel_pct=class_pct,
        unknown_pixel_pct=round(unknown_pct, 4),
        components=components,
        class_rles=class_rles,
    )


def build_records(
    dataset: str,
    data_root: Path,
    limit: int | None = None,
    altitude_m: float | None = None,
) -> Path:
    """Process a prepared dataset into ``data/processed/<dataset>/records.parquet``."""
    import pandas as pd

    spec = REGISTRY[dataset]
    raw = data_root / "raw" / dataset
    rows = []
    for i, (image_path, mask_path) in enumerate(iter_samples(spec, raw)):
        if limit is not None and i >= limit:
            break
        record = build_record(spec, image_path, mask_path, data_root, altitude_m=altitude_m)
        rows.append(record.model_dump_json())

    if not rows:
        raise RuntimeError(f"no samples found for {dataset} under {raw}")

    out_dir = data_root / "processed" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "records.parquet"
    pd.DataFrame({"record": rows}).to_parquet(out, index=False)
    return out


def load_records(parquet_path: Path) -> list[ImageRecord]:
    """Load ImageRecords back from a records parquet file."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    return [ImageRecord.model_validate_json(s) for s in df["record"]]
