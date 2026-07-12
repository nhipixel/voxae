"""Verify and extract manually-downloaded dataset archives.

All supported source datasets sit behind registration or license-acceptance
walls, so downloads are manual: archives go into ``data/downloads/<dataset>/``
and this module extracts them into ``data/raw/<dataset>/`` and validates that
image/mask pairs are discoverable. SHA256 hashes are printed for provenance
(recorded in the datasheet rather than pinned, since releases vary).
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from voxae.data.registry import REGISTRY, iter_samples


@dataclass
class PrepareResult:
    dataset: str
    archives: list[str]
    extracted_to: str
    n_pairs: int


class PrepareError(RuntimeError):
    """Raised when archives are missing or extraction yields no usable pairs."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def prepare(dataset: str, data_root: Path) -> PrepareResult:
    """Extract downloaded archives for ``dataset`` and validate pair discovery."""
    spec = REGISTRY.get(dataset)
    if spec is None:
        raise PrepareError(f"unknown dataset '{dataset}' (known: {sorted(REGISTRY)})")

    downloads = data_root / "downloads" / dataset
    raw = data_root / "raw" / dataset
    archives = sorted(downloads.glob("*.zip")) if downloads.exists() else []
    if not archives and not raw.exists():
        raise PrepareError(
            f"no archives found in {downloads}. Download {dataset} from {spec.homepage} "
            f"and place the zip there."
        )

    raw.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for archive in archives:
        names.append(f"{archive.name} sha256={_sha256(archive)}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)

    n_pairs = sum(1 for _ in iter_samples(spec, raw))
    if n_pairs == 0:
        raise PrepareError(
            f"extracted {dataset} but found no image/mask pairs under {raw} — "
            f"expected '{spec.image_dir_hint}' and '{spec.mask_dir_hint}' directories"
        )
    return PrepareResult(dataset=dataset, archives=names, extracted_to=str(raw), n_pairs=n_pairs)
