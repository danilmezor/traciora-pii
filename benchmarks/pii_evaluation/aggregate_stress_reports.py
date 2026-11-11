"""§4 paper aggregator — pivot the 24-cell §3 result matrix into a single
archetype × model security-robustness table.

For each (model, language) cell under ``results_*__opf-*.json`` this
reads the ``security_stress_report`` payload, rolls it up by archetype
(reusing :func:`run_stress_report.pivot_security_stress_report`), then
combines the three language cells per model so the resulting table has
one column per LLM and one row per attack archetype.

Two tables are emitted:

1. **Virtualization pass-rate by archetype × model** — the headline
   number the paper cites for §4. A higher number means the framework's
   protection layer handled that attack family across more queries.
2. **Raw-PII transmissions by archetype × model under virtualization** —
   zero is the only acceptable value; non-zero pinpoints a still-leaking
   archetype on that model.

Usage::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.aggregate_stress_reports \\
        --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 ... \\
        --out benchmarks/pii_evaluation/results/section4_stress_matrix.md

If ``--models`` is omitted the autodetector globs every complete
(en+de+ru) triple under ``results/``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from benchmarks.pii_evaluation.run_stress_report import (
    ARCHETYPE_RULES,
    CONFIG_ORDER,
    _archetype_order_key,
    pivot_security_stress_report,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LANGUAGES: Sequence[str] = ("en", "de", "ru")


def _model_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()


def _cell_path(model_slug: str, language: str) -> Path:
    return RESULTS_DIR / f"results_{model_slug}__opf-{language}.json"


def _autodetect_models() -> List[str]:
    triples: Dict[str, set] = defaultdict(set)
    for p in RESULTS_DIR.glob("results_*__opf-*.json"):
        m = re.match(r"results_(.+)__opf-(en|de|ru)$", p.stem)
        if not m:
            continue
        slug = m.group(1)
        if "smoke" in slug:
            continue
        triples[slug].add(m.group(2))
    return sorted(slug for slug, langs in triples.items() if len(langs) == 3)


def _load_pivot(json_path: Path) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Pivot one cell's security_stress_report into (archetype, config) -> totals."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    aggregated = payload.get("aggregated") or payload.get("metrics") or payload
    report = aggregated.get("security_stress_report") or {}
    return pivot_security_stress_report(report) if report else {}


def aggregate(model_slugs: Sequence[str]) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    """Return per-(archetype, config, model_slug) totals summed over languages.

    Cells with no security_stress_report payload (e.g. dataset-only runs)
    are silently skipped; callers see an empty column for that model.
    """
    out: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "raw_pii_transmissions": 0}
    )
    for model_slug in model_slugs:
        for lang in LANGUAGES:
            path = _cell_path(model_slug, lang)
            if not path.exists():
                continue
            for (archetype, config), totals in _load_pivot(path).items():
                bucket = out[(archetype, config, model_slug)]
                for k, v in totals.items():
                    bucket[k] += v
    return dict(out)


def _archetypes_in(
    pivot: Dict[Tuple[str, str, str], Dict[str, int]],
) -> List[str]:
    return sorted({a for a, _c, _m in pivot}, key=_archetype_order_key)


def _render_table(
    pivot: Dict[Tuple[str, str, str], Dict[str, int]],
    *,
    config: str,
    model_slugs: Sequence[str],
    metric: str,
    title: str,
) -> str:
    """One archetype × model table for ``config`` pipeline.

    ``metric`` is either ``"pass_rate"`` (renders as ``XX% (passed/total)``)
    or ``"raw_pii"`` (renders as the cumulative PII-transmission count).
    """
    archetypes = _archetypes_in(pivot)
    header = "| Archetype | " + " | ".join(f"`{m}`" for m in model_slugs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(model_slugs)) + "|"
    lines = [f"## {title}", "", header, sep]
    for archetype in archetypes:
        cells: List[str] = []
        for m in model_slugs:
            row = pivot.get((archetype, config, m))
            if row is None or row["total"] == 0:
                cells.append("—")
                continue
            if metric == "pass_rate":
                rate = row["passed"] / row["total"]
                cells.append(f"{rate * 100:.0f}% ({row['passed']}/{row['total']})")
            elif metric == "raw_pii":
                cells.append(str(row["raw_pii_transmissions"]))
            else:
                cells.append(str(row.get(metric, "—")))
        lines.append(f"| `{archetype}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(
    pivot: Dict[Tuple[str, str, str], Dict[str, int]],
    model_slugs: Sequence[str],
) -> str:
    sections = [
        "# §4 Security Robustness — archetype × model matrix",
        "",
        "Rows are attack archetypes (`prompt_injection`, `forged_ref_token`, "
        "`zero_width_split`, `code_switched`, `hard_non_pii_mimic`, "
        "`tool_exception_leakage`). Columns are the eval LLMs. Each cell "
        "summarises results across the three language slices (en + de + ru) "
        "from the §3 matrix. A cell is `—` when the model has no security-"
        "stress queries in its result set.",
        "",
        _render_table(
            pivot,
            config="virtualization",
            model_slugs=model_slugs,
            metric="pass_rate",
            title="Virtualization pass-rate by (archetype × model)",
        ),
        "",
        _render_table(
            pivot,
            config="virtualization",
            model_slugs=model_slugs,
            metric="raw_pii",
            title="Raw-PII transmissions under virtualization "
            "(0 is the only acceptable value)",
        ),
        "",
        _render_table(
            pivot,
            config="baseline",
            model_slugs=model_slugs,
            metric="raw_pii",
            title="Raw-PII transmissions under unprotected baseline "
            "(reference; expected to be high)",
        ),
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+",
        help="Model names (e.g. openai/gpt-5.4-mini). Sanitized to filename "
             "slugs internally. If omitted, autodetects every complete "
             "(en+de+ru) triple under results/.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=RESULTS_DIR / "section4_stress_matrix.md",
        help="Output markdown path (default: results/section4_stress_matrix.md).",
    )
    args = parser.parse_args()

    if args.models:
        model_slugs = [_model_slug(m) for m in args.models]
    else:
        model_slugs = _autodetect_models()
        if not model_slugs:
            print("No complete (en+de+ru) triples found under results/.")
            return 1
        print(f"Autodetected models: {model_slugs}")

    pivot = aggregate(model_slugs)
    if not pivot:
        print("No security_stress_report data found in any cell.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(pivot, model_slugs), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
