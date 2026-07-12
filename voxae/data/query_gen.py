"""LLM-driven query generation over prepared image records.

Division of labor: the LLM authors language and picks symbolic targets from
the facts it is shown; all geometry (masks, metric ground truth) is computed
by the materializer. Responses are cached to disk keyed by
(prompt version, model, image, seed), so regeneration is free and the exact
generation inputs are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voxae.config import Settings, get_settings
from voxae.data import rle
from voxae.data.materialize import MaterializeError, materialize
from voxae.data.prep_masks import load_records
from voxae.data.schemas import GenMeta, ImageRecord, QuerySample, QuerySpec

PROMPT_VERSION = "qgen_v1"
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# chat_fn(system, user) -> raw model text; injectable for tests and provider swaps.
ChatFn = Callable[[str, str], str]


class QueryGenError(RuntimeError):
    """Raised when generation fails for an image after retries."""


@dataclass
class GenStats:
    images: int = 0
    specs_returned: int = 0
    specs_valid: int = 0
    samples_emitted: int = 0
    cache_hits: int = 0


def load_prompt(version: str = PROMPT_VERSION) -> tuple[str, str]:
    """Load (system, user) templates from the versioned prompt file."""
    text = (_PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")
    parts = re.split(r"^## (system|user)\s*$", text, flags=re.MULTILINE)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return sections["system"], sections["user"]


def build_facts(record: ImageRecord) -> dict:
    """Compact, prompt-ready summary of one image's classes and components."""
    return {
        "image_size": [record.width, record.height],
        "gsd_m_per_px": record.gsd_m_per_px,
        "classes": {cls: {"pixel_pct": pct} for cls, pct in sorted(record.class_pixel_pct.items())},
        "components": [
            {
                "cls": c.cls,
                "comp_id": c.comp_id,
                "grid_cell": c.grid_cell,
                "area_pct": c.area_pct,
                **(
                    {"width_m": c.width_m, "height_m": c.height_m, "area_m2": c.area_m2}
                    if c.width_m is not None
                    else {}
                ),
            }
            for c in record.components
        ],
    }


def extract_json_array(text: str) -> list:
    """Pull the first JSON array out of possibly-noisy model output."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        bracket = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket is None:
            raise QueryGenError(f"no JSON array in model output: {text[:200]!r}")
        candidate = bracket.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise QueryGenError(f"malformed JSON from model: {e}") from e
    if not isinstance(parsed, list):
        raise QueryGenError("model output is not a JSON array")
    return parsed


def default_chat_fn(settings: Settings) -> ChatFn:
    """Chat completion against the configured OpenAI-compatible endpoint."""
    import httpx

    if not settings.vlm_api_key:
        raise QueryGenError("VOXAE_VLM_API_KEY is not set — required for generation")

    def chat(system: str, user: str) -> str:
        resp = httpx.post(
            f"{settings.vlm_base_url.rstrip('/')}/chat/completions",
            json={
                "model": settings.qgen_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": settings.qgen_temperature,
            },
            headers={"Authorization": f"Bearer {settings.vlm_api_key}"},
            timeout=settings.qgen_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return chat


def _cache_key(image_id: str, model: str, seed: int) -> str:
    raw = f"{PROMPT_VERSION}|{model}|{image_id}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def generate_for_record(
    record: ImageRecord,
    chat_fn: ChatFn,
    model: str,
    cache_dir: Path,
    n_queries: int = 6,
    seed: int = 0,
    stats: GenStats | None = None,
) -> list[QuerySample]:
    """Generate, validate, and materialize query samples for one image."""
    stats = stats if stats is not None else GenStats()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(record.image_id, model, seed)}.json"

    if cache_file.exists():
        raw_specs = json.loads(cache_file.read_text(encoding="utf-8"))
        cached = True
        stats.cache_hits += 1
    else:
        system, user_tpl = load_prompt()
        n_metric = n_queries // 3 if record.gsd_m_per_px else 0
        n_affordance = n_queries // 3 + (n_queries // 3 - n_metric)
        n_referring = n_queries - n_affordance - n_metric
        user = user_tpl.format(
            facts_json=json.dumps(build_facts(record), indent=1),
            n_queries=n_queries,
            n_referring=n_referring,
            n_affordance=n_affordance,
            n_metric=n_metric,
        )
        raw_specs = extract_json_array(chat_fn(system, user))
        cache_file.write_text(json.dumps(raw_specs), encoding="utf-8")
        cached = False

    stats.images += 1
    stats.specs_returned += len(raw_specs)

    class_masks = {cls: rle.decode(r) for cls, r in record.class_rles.items()}
    samples: list[QuerySample] = []
    for i, raw in enumerate(raw_specs):
        try:
            spec = QuerySpec.model_validate(raw)
            mask = materialize(spec.target, class_masks, record.gsd_m_per_px)
        except (ValueError, MaterializeError):
            continue  # invalid specs are dropped; QC reports overall yield
        stats.specs_valid += 1
        encoded = rle.encode(mask)
        samples.append(
            QuerySample(
                sample_id=f"{record.image_id}-q{i:02d}-s{seed}",
                dataset=record.dataset,
                image_id=record.image_id,
                rel_path=record.rel_path,
                family=spec.family,
                text=spec.text.strip(),
                target=spec.target,
                rle=encoded,
                area_pct=round(rle.area_pct(encoded), 4),
                gsd_m_per_px=record.gsd_m_per_px,
                gen=GenMeta(model=model, prompt_version=PROMPT_VERSION, seed=seed, cached=cached),
            )
        )
    stats.samples_emitted += len(samples)
    return samples


# on_image(index, total, image_id, elapsed_seconds, n_samples) -> None; lets
# callers (e.g. the CLI) show live progress. generate_for_record() has no
# internal progress signal of its own — a slow-but-healthy run is otherwise
# indistinguishable from a hung one until the whole batch finishes.
OnImageFn = Callable[[int, int, str, float, int], None]


def generate(
    records_parquet: Path,
    out_jsonl: Path,
    cache_dir: Path,
    chat_fn: ChatFn | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
    seed: int = 0,
    on_image: OnImageFn | None = None,
) -> GenStats:
    """Run generation over a records parquet, appending samples to JSONL."""
    settings = settings or get_settings()
    chat = chat_fn or default_chat_fn(settings)
    stats = GenStats()

    records = load_records(records_parquet)
    if limit is not None:
        records = records[:limit]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        for i, record in enumerate(records):
            t0 = time.perf_counter()
            samples = generate_for_record(
                record,
                chat,
                model=settings.qgen_model,
                cache_dir=cache_dir,
                n_queries=settings.qgen_queries_per_image,
                seed=seed,
                stats=stats,
            )
            for s in samples:
                f.write(s.model_dump_json() + "\n")
            f.flush()  # each image's results are safe on disk even if a later one fails
            if on_image is not None:
                on_image(
                    i + 1, len(records), record.image_id, time.perf_counter() - t0, len(samples)
                )
    return stats
