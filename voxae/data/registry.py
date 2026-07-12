"""Source-dataset registry: mask decoding, class tables, and file layout.

Each supported dataset ships semantic masks in its own format (RGB palette or
single-channel class indices). The registry normalizes them into a shared
vocabulary: dataset-native class names plus a small unified category set used
for cross-dataset queries.

Pixels whose color/index is not in the class table are bucketed as
``unknown`` and reported per image, so palette mismatches surface as data
instead of silently corrupting masks.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

UNKNOWN = "unknown"


class Category(StrEnum):
    building = "building"
    road = "road"
    vehicle = "vehicle"
    person = "person"
    tree = "tree"
    low_vegetation = "low_vegetation"
    water = "water"
    ground = "ground"
    obstacle = "obstacle"
    other = "other"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    license: str
    homepage: str
    mask_format: str  # "rgb" (color palette) or "index" (single-channel ids)
    class_table: dict[tuple[int, int, int] | int, str]  # color or index -> class name
    categories: dict[str, Category]  # class name -> unified category
    image_dir_hint: str  # relative layout hint used by iter_samples
    mask_dir_hint: str
    # "mirrored": image/mask dirs are siblings at the same depth, paired by
    #   their shared parent path (e.g. UAVid's seq*/Images vs seq*/Labels;
    #   VDD's train/src vs train/gt) — required when filename stems repeat
    #   across subfolders, since path context disambiguates them.
    # "stem": image/mask dirs sit at different depths (e.g. Kaggle's SDD
    #   mirror: training_set/images vs training_set/gt/semantic/label_images)
    #   — pairing is by filename stem alone, which is only safe when stems
    #   are unique dataset-wide (checked at iteration time).
    pairing: str = "mirrored"


def _uavid_spec() -> DatasetSpec:
    return DatasetSpec(
        name="uavid",
        license="CC BY-NC-SA 4.0",
        homepage="https://uavid.nl/",
        mask_format="rgb",
        class_table={
            (0, 0, 0): "clutter",
            (128, 0, 0): "building",
            (128, 64, 128): "road",
            (0, 128, 0): "tree",
            (128, 128, 0): "low_vegetation",
            (64, 0, 128): "moving_car",
            (192, 0, 192): "static_car",
            (64, 64, 0): "human",
        },
        categories={
            "clutter": Category.other,
            "building": Category.building,
            "road": Category.road,
            "tree": Category.tree,
            "low_vegetation": Category.low_vegetation,
            "moving_car": Category.vehicle,
            "static_car": Category.vehicle,
            "human": Category.person,
        },
        image_dir_hint="Images",
        mask_dir_hint="Labels",
    )


def _sdd_spec() -> DatasetSpec:
    # Palette from the dataset's class_dict_seg.csv; unmapped colors are
    # bucketed as unknown and reported, so any palette drift is visible.
    table: dict[tuple[int, int, int] | int, str] = {
        (0, 0, 0): "unlabeled",
        (128, 64, 128): "paved_area",
        (130, 76, 0): "dirt",
        (0, 102, 0): "grass",
        (112, 103, 87): "gravel",
        (28, 42, 168): "water",
        (48, 41, 30): "rocks",
        (0, 50, 89): "pool",
        (107, 142, 35): "vegetation",
        (70, 70, 70): "roof",
        (102, 102, 156): "wall",
        (254, 228, 12): "window",
        (254, 148, 12): "door",
        (190, 153, 153): "fence",
        (153, 153, 153): "fence_pole",
        (255, 22, 96): "person",
        (102, 51, 0): "dog",
        (9, 143, 150): "car",
        (119, 11, 32): "bicycle",
        (51, 51, 0): "tree",
        (190, 250, 190): "bald_tree",
        (112, 150, 146): "ar_marker",
        (2, 135, 115): "obstacle",
        (255, 0, 0): "conflicting",
    }
    cats = {
        "unlabeled": Category.other,
        "paved_area": Category.road,
        "dirt": Category.ground,
        "grass": Category.low_vegetation,
        "gravel": Category.ground,
        "water": Category.water,
        "rocks": Category.obstacle,
        "pool": Category.water,
        "vegetation": Category.low_vegetation,
        "roof": Category.building,
        "wall": Category.building,
        "window": Category.building,
        "door": Category.building,
        "fence": Category.obstacle,
        "fence_pole": Category.obstacle,
        "person": Category.person,
        "dog": Category.other,
        "car": Category.vehicle,
        "bicycle": Category.vehicle,
        "tree": Category.tree,
        "bald_tree": Category.tree,
        "ar_marker": Category.other,
        "obstacle": Category.obstacle,
        "conflicting": Category.other,
    }
    return DatasetSpec(
        name="sdd",
        license="non-commercial (TU Graz Semantic Drone Dataset terms)",
        homepage="https://ivc.tugraz.at/research-project/semantic-drone-dataset/",
        mask_format="rgb",
        class_table=table,
        categories=cats,
        image_dir_hint="images",
        mask_dir_hint="label_images",
        pairing="stem",
    )


def _vdd_spec() -> DatasetSpec:
    # Index ids follow the class list in the VDD repository; verify against
    # the downloaded release — unknown ids are bucketed and reported.
    return DatasetSpec(
        name="vdd",
        license="non-commercial research (VDD terms)",
        homepage="https://github.com/RussRobin/VDD",
        mask_format="index",
        class_table={
            0: "other",
            1: "wall",
            2: "road",
            3: "vegetation",
            4: "vehicle",
            5: "roof",
            6: "water",
        },
        categories={
            "other": Category.other,
            "wall": Category.building,
            "road": Category.road,
            "vegetation": Category.low_vegetation,
            "vehicle": Category.vehicle,
            "roof": Category.building,
            "water": Category.water,
        },
        image_dir_hint="src",
        mask_dir_hint="gt",
    )


REGISTRY: dict[str, DatasetSpec] = {s.name: s for s in (_uavid_spec(), _sdd_spec(), _vdd_spec())}


def decode_mask(spec: DatasetSpec, mask: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """Split a raw mask array into per-class boolean masks.

    Returns (class_name -> HxW bool mask, unknown_pixel_pct). Only classes
    present in the image appear in the dict.
    """
    if spec.mask_format == "rgb":
        if mask.ndim != 3 or mask.shape[2] < 3:
            raise ValueError(f"{spec.name}: expected HxWx3 RGB mask, got {mask.shape}")
        rgb = mask[:, :, :3].astype(np.uint32)
        packed = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
        known = np.zeros(mask.shape[:2], dtype=bool)
        out: dict[str, np.ndarray] = {}
        for color, cls in spec.class_table.items():
            r, g, b = color  # type: ignore[misc]
            key = (r << 16) | (g << 8) | b
            m = packed == key
            if m.any():
                out[cls] = m
                known |= m
    elif spec.mask_format == "index":
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        known = np.zeros(mask.shape, dtype=bool)
        out = {}
        for idx, cls in spec.class_table.items():
            m = mask == idx
            if m.any():
                out[cls] = m
                known |= m
    else:
        raise ValueError(f"unsupported mask format: {spec.mask_format}")

    unknown_pct = float((~known).mean()) * 100.0
    return out, unknown_pct


class PairingError(RuntimeError):
    """Raised when image/mask pairing is ambiguous (e.g. duplicate stems)."""


def _files_under_hint(dataset_root: Path, hint: str, image_exts: set[str]) -> Iterator[Path]:
    hint_l = hint.lower()
    for p in sorted(dataset_root.rglob("*")):
        if p.suffix.lower() in image_exts and hint_l in (part.lower() for part in p.parts):
            yield p


def _mirrored_key(p: Path, dataset_root: Path, hint: str) -> str:
    """Relative path with the hint directory itself removed — the shared prefix."""
    rel = p.relative_to(dataset_root)
    hint_l = hint.lower()
    parts = tuple(part for part in rel.parent.parts if part.lower() != hint_l)
    return "/".join(parts)


def iter_samples(spec: DatasetSpec, dataset_root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (image_path, mask_path) pairs for a dataset's directory layout.

    Two pairing strategies (``spec.pairing``):
    - "mirrored": image/mask trees are structurally symmetric (same relative
      path once the hint folder is stripped) — required whenever filename
      stems repeat across subfolders, since only path context disambiguates
      them (e.g. UAVid's per-sequence frame numbering).
    - "stem": image/mask dirs sit at different depths and aren't symmetric —
      pairing is by filename stem alone. Raises if a stem is not unique
      dataset-wide, since silent mispairing would corrupt ground truth.
    """
    image_exts = {".png", ".jpg", ".jpeg"}

    if spec.pairing == "stem":
        masks_by_stem: dict[str, Path] = {}
        for p in _files_under_hint(dataset_root, spec.mask_dir_hint, image_exts):
            if p.stem in masks_by_stem:
                raise PairingError(
                    f"{spec.name}: duplicate mask stem '{p.stem}' "
                    f"({masks_by_stem[p.stem]} and {p}) — stem-based pairing requires "
                    f"unique filenames; use pairing='mirrored' instead"
                )
            masks_by_stem[p.stem] = p
        for p in _files_under_hint(dataset_root, spec.image_dir_hint, image_exts):
            mask = masks_by_stem.get(p.stem)
            if mask is not None:
                yield p, mask
        return

    masks: dict[tuple[str, str], Path] = {}
    for p in _files_under_hint(dataset_root, spec.mask_dir_hint, image_exts):
        masks[(_mirrored_key(p, dataset_root, spec.mask_dir_hint), p.stem)] = p

    for p in _files_under_hint(dataset_root, spec.image_dir_hint, image_exts):
        mask = masks.get((_mirrored_key(p, dataset_root, spec.image_dir_hint), p.stem))
        if mask is not None:
            yield p, mask
