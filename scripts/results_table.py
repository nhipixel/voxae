"""Turn eval reports into the tables that get published.

Every figure on the site and in the README should be printed from a report
rather than typed from one. The per-family counts on the site were wrong for
exactly that reason: the report did not carry them, so they were estimated, and
an estimate that reads as measured survives review.

So this reads whatever reports exist, prints the tables in their published
shape, and recounts the split against the dataset to check them.

    uv run python scripts/results_table.py --results-dir docs/results
    uv run python scripts/results_table.py --results-dir docs/results --check data/processed/annotations/voxae_reason.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FAMILIES = ("all", "affordance", "referring")
# The site labels these; keep the wording in one place so it cannot drift.
LABELS = {
    "all": "All",
    "affordance": "Affordance questions",
    "referring": "Referring questions",
}


def from_records(report: dict, path: Path) -> dict:
    """Fill in what the per-sample records know and the aggregate does not.

    Reports written before the per-family counts existed carry none, and the
    latency mean in those reports covers only the samples the final session
    recomputed, so a resumed run understated or overstated it. The records hold
    every sample either way, so they are the better source for both.
    """
    records_path = path.with_suffix("").with_suffix(".records.jsonl")
    if not records_path.exists():
        return report
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return report

    counts: Counter = Counter(all=len(records))
    counts.update(r["family"] for r in records)
    for fam, n in counts.items():
        report.setdefault(f"n_{fam}", n)
    report["latency_s_mean"] = round(sum(r["latency_s"] for r in records) / len(records), 3)
    report["_latency_from_records"] = True
    return report


def load(results_dir: Path) -> list[dict]:
    """Every report in a directory, newest filename last."""
    out = []
    for path in sorted(results_dir.glob("eval_*.json")):
        if path.name.endswith(".records.json"):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! {path.name} is not valid JSON; skipped")
            continue
        report["_path"] = path
        out.append(from_records(report, path))
    return out


def pick(reports: list[dict], predictor: str, split: str) -> dict | None:
    """The untagged report for one predictor and split, if it ran."""
    for r in reports:
        if r.get("predictor") == predictor and r.get("split") == split and not r.get("tag"):
            return r
    return None


def cell(value: float | int | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def headline(trained: dict | None, zero: dict | None, split: str) -> list[str]:
    """The comparison table, in the README's shape."""
    lines = [f"### Held-out {split} split, both predictors on identical inputs", ""]
    if trained is None and zero is None:
        return [*lines, f"No {split}-split reports found.", ""]
    if trained is None or zero is None:
        missing = "trained" if trained is None else "zero-shot"
        lines.append(f"Only one predictor has run; the {missing} report is missing.")
        lines.append("")

    lines += [
        "| | gIoU trained | gIoU zero-shot | cIoU trained | cIoU zero-shot | n |",
        "|---|---|---|---|---|---|",
    ]
    def score(report: dict | None, metric: str, fam: str) -> float | None:
        return (report or {}).get(f"{metric}_{fam}")

    for fam in FAMILIES:
        if score(trained, "giou", fam) is None and score(zero, "giou", fam) is None:
            continue
        n = (trained or zero or {}).get(f"n_{fam}")
        cells = " | ".join(
            cell(score(r, metric, fam)) for metric in ("giou", "ciou") for r in (trained, zero)
        )
        lines.append(f"| {LABELS.get(fam, fam)} | {cells} | {n or '-'} |")
    lines.append("")

    lines += ["| | trained | zero-shot |", "|---|---|---|"]
    lat_t = (trained or {}).get("latency_s_mean")
    lat_z = (zero or {}).get("latency_s_mean")
    lines.append(f"| latency / query | {cell(lat_t, 2)} s | {cell(lat_z, 2)} s |")
    fail_t = (trained or {}).get("failures")
    fail_z = (zero or {}).get("failures")
    total = (trained or zero or {}).get("n", "?")
    lines.append(
        f"| unparseable responses | {fail_t if fail_t is not None else '-'} / {total} "
        f"| {fail_z if fail_z is not None else '-'} / {total} |"
    )
    if (trained or {}).get("_latency_from_records") or (zero or {}).get("_latency_from_records"):
        lines += [
            "",
            "Latency is the mean over every per-sample record. The aggregate field in a "
            "report written before this fix covered only the samples its final session "
            "recomputed, which understates or overstates a resumed run.",
        ]
    if lat_t and lat_z:
        lines += ["", f"Latency ratio: {lat_z / lat_t:.1f}x."]
    for metric in ("giou", "ciou"):
        a = (trained or {}).get(f"{metric}_all")
        b = (zero or {}).get(f"{metric}_all")
        if a and b:
            lines.append(f"{metric.replace('iou', 'IoU')} lift, all: {(a / b - 1) * 100:+.0f}%.")
    lines.append("")
    return lines


def record_literal(trained: dict | None, zero: dict | None) -> list[str]:
    """The RECORD array from web/app/page.tsx, printed rather than transcribed."""
    if trained is None or zero is None:
        return ["Both reports are needed to print the site's RECORD array.", ""]
    lines = ["### web/app/page.tsx RECORD", "", "```ts", "const RECORD = ["]
    for fam in ("affordance", "referring", "all"):
        t = trained.get(f"giou_{fam}")
        z = zero.get(f"giou_{fam}")
        n = trained.get(f"n_{fam}")
        if t is None or z is None:
            continue
        lines.append(
            f'  {{ split: "{LABELS[fam]}", trained: "{t:.3f}", '
            f'zero: "{z:.3f}", n: "{n}" }},'
        )
    lines += ["];", "```", ""]
    return lines


def ablations(reports: list[dict], control: str) -> list[str]:
    """One row per arm, with the delta against the control arm."""
    tagged = sorted((r for r in reports if r.get("tag")), key=lambda r: str(r["tag"]))
    if not tagged:
        return []
    base = next((r for r in tagged if r["tag"] == control), None)
    lines = [
        f"### Ablations (control arm: {control})",
        "",
        "| arm | gIoU all | d gIoU | cIoU all | d cIoU | gIoU afford. | gIoU refer. | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in tagged:
        g = r.get("giou_all")
        c = r.get("ciou_all")
        dg = (
            f"{g - base['giou_all']:+.4f}"
            if base and g is not None and base.get("giou_all") is not None
            else "-"
        )
        dc = (
            f"{c - base['ciou_all']:+.4f}"
            if base and c is not None and base.get("ciou_all") is not None
            else "-"
        )
        if r is base:
            dg = dc = "reference"
        lines.append(
            f"| {r['tag']} | {cell(g)} | {dg} | {cell(c)} | {dc} | "
            f"{cell(r.get('giou_affordance'))} | {cell(r.get('giou_referring'))} | "
            f"{r.get('n_all', '-')} |"
        )
    lines += [
        "",
        "Arms share a step budget, a seed, and an eval split, so they compare to "
        "each other. They do not compare to the full run, which trains longer.",
        "",
    ]
    return lines


def check_counts(reports: list[dict], jsonl: Path, split: str) -> list[str]:
    """Recount the split from the dataset and compare to what the reports claim."""
    counts: Counter = Counter()
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            counts["all"] += 1
            counts[row["family"]] += 1

    lines = [f"### Count check against {jsonl.name}", ""]
    disagreed = False
    for r in reports:
        if r.get("split") != split:
            continue
        name = f"{r.get('predictor')}{'/' + str(r['tag']) if r.get('tag') else ''}"
        for fam in FAMILIES:
            claimed = r.get(f"n_{fam}")
            actual = counts.get(fam)
            if claimed is None or actual is None:
                continue
            if claimed != actual:
                disagreed = True
                lines.append(f"- MISMATCH {name} {fam}: report says {claimed}, dataset has {actual}")
    if not disagreed:
        shown = ", ".join(f"{fam} {counts.get(fam, 0)}" for fam in FAMILIES)
        lines.append(f"- every report agrees with the dataset ({shown})")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--split", default="test", help="split for the headline table")
    parser.add_argument("--control", default="base", help="ablation arm used as the reference")
    parser.add_argument("--check", type=Path, help="dataset JSONL to recount the split from")
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(
            f"{args.results_dir} does not exist. Run an evaluation with "
            f"--out-dir {args.results_dir}, or point --results-dir at the reports."
        )

    reports = load(args.results_dir)
    if not reports:
        raise SystemExit(f"no eval_*.json under {args.results_dir}")

    lines = [f"# Results ({len(reports)} report(s) in {args.results_dir})", ""]
    trained = pick(reports, "trained", args.split)
    zero = pick(reports, "zero-shot", args.split)
    lines += headline(trained, zero, args.split)
    lines += record_literal(trained, zero)
    lines += ablations(reports, args.control)
    if args.check:
        lines += check_counts(reports, args.check, args.split)

    for r in reports:
        if r.get("checkpoint"):
            lines.append(f"- {r['_path'].name}: checkpoint {r['checkpoint']}")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
