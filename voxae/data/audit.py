"""Human audit of generated samples: review page, decisions file, precision gate.

A seeded sample of QC-passed queries is rendered into a single self-contained
HTML page (mask overlays inlined as base64 thumbnails) next to a CSV where
each sample is marked accept/reject. Precision on the audited sample gates
training: below the threshold, the generation prompt or QC rules need work,
not the model.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from voxae.data import rle
from voxae.data.schemas import QuerySample
from voxae.viz import overlay_mask

PRECISION_THRESHOLD = 0.90
THUMB_PX = 560


@dataclass
class AuditScore:
    reviewed: int
    accepted: int

    @property
    def precision(self) -> float:
        return self.accepted / self.reviewed if self.reviewed else 0.0

    @property
    def passed(self) -> bool:
        return self.reviewed > 0 and self.precision >= PRECISION_THRESHOLD


def _load_samples(jsonl: Path) -> list[QuerySample]:
    with jsonl.open(encoding="utf-8") as f:
        return [QuerySample.model_validate_json(line) for line in f if line.strip()]


def _thumb_b64(image_path: Path, sample: QuerySample) -> str:
    img = Image.open(image_path).convert("RGB")
    mask = rle.decode(sample.rle)
    over = overlay_mask(img, mask)
    over.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    over.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_audit(
    qc_jsonl: Path,
    data_root: Path,
    out_dir: Path,
    n: int = 100,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write audit.html + audit_decisions.csv for a seeded sample."""
    samples = _load_samples(qc_jsonl)
    rng = random.Random(seed)
    chosen = samples if len(samples) <= n else rng.sample(samples, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_html: list[str] = []
    csv_path = out_dir / "audit_decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "family", "verdict (accept/reject)", "note"])
        for s in chosen:
            writer.writerow([s.sample_id, s.family, "", ""])
            img_path = data_root / s.rel_path
            thumb = _thumb_b64(img_path, s) if img_path.exists() else ""
            img_tag = (
                f'<img src="data:image/jpeg;base64,{thumb}">'
                if thumb
                else "<em>image not found locally</em>"
            )
            rows_html.append(
                "<div class='card'>"
                f"<div class='meta'><code>{html.escape(s.sample_id)}</code>"
                f" <span class='fam'>{s.family}</span></div>"
                f"<p>{html.escape(s.text)}</p>{img_tag}</div>"
            )

    html_path = out_dir / "audit.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Voxae audit</title>"
        "<style>body{font-family:sans-serif;max-width:640px;margin:2rem auto}"
        ".card{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}"
        ".fam{background:#eee;border-radius:4px;padding:2px 6px;font-size:.8em}"
        "img{max-width:100%;border-radius:4px}</style>"
        f"<h1>Sample audit ({len(chosen)} of {len(samples)})</h1>"
        "<p>Mark each sample accept/reject in audit_decisions.csv, then run: "
        "<code>voxae dataset audit-score</code></p>" + "\n".join(rows_html),
        encoding="utf-8",
    )
    return html_path, csv_path


def score_audit(csv_path: Path) -> AuditScore:
    """Compute precision from a filled decisions CSV (blank rows are skipped)."""
    reviewed = accepted = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verdict = (row.get("verdict (accept/reject)") or "").strip().lower()
            if verdict in {"accept", "a", "yes", "y", "1"}:
                reviewed += 1
                accepted += 1
            elif verdict in {"reject", "r", "no", "n", "0"}:
                reviewed += 1
    return AuditScore(reviewed=reviewed, accepted=accepted)
