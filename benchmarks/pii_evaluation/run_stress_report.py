"""Aggregate per-(config, workflow) stress scoring into an archetype × config table.

Reads ``results_<slug>.json`` produced by ``run_experiment`` (its
``security_stress_report`` field is already keyed by ``"<config>|<workflow>"``)
and pivots it into the table reviewers want:

  rows    = archetypes (prompt_injection, forged_ref_token, ...)
  columns = configs (baseline, destructive, destructive_presidio, virtualization)
  cells   = pass-rate (passed / total)

Archetypes are workflow prefixes — multiple workflows can roll up to the
same archetype (e.g. ``prompt_injection_for_raw_pii`` and
``prompt_injection_for_raw_pii_v2`` both fall under ``prompt_injection``).

Usage::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_stress_report \\
        --results benchmarks/pii_evaluation/results/results_ru-v2.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Workflow prefix → display archetype. Order here drives row order in the
# rendered table. Anything not matched falls back to the workflow itself.
ARCHETYPE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^prompt_injection"), "prompt_injection"),
    (re.compile(r"^forged_ref_token"), "forged_ref_token"),
    (re.compile(r"^zero_width_split"), "zero_width_split"),
    (re.compile(r"^code_switched"), "code_switched"),
    (re.compile(r"^hard_non_pii_mimic"), "hard_non_pii_mimic"),
    (re.compile(r"^tool_exception_leakage"), "tool_exception_leakage"),
]

# Display order for configs.
CONFIG_ORDER = ("baseline", "destructive", "destructive_presidio", "virtualization")


def _archetype_for(workflow: str) -> str:
    for pattern, name in ARCHETYPE_RULES:
        if pattern.match(workflow):
            return name
    return workflow


def pivot_security_stress_report(
    report: Dict[str, Dict],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Roll per-``(config, workflow)`` rows up to per-``(archetype, config)``.

    Returns a dict keyed by ``(archetype, config)`` with totals summed
    across constituent workflows. Pass-rate is computed at render time
    from the summed totals — averaging pre-computed pass-rates would
    silently weight workflows equally regardless of instance count.
    """
    pivot: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "raw_pii_transmissions": 0}
    )
    for key, payload in report.items():
        if "|" not in key:
            continue
        config, workflow = key.split("|", 1)
        archetype = _archetype_for(workflow)
        bucket = pivot[(archetype, config)]
        bucket["total"] += int(payload.get("total", 0) or 0)
        bucket["passed"] += int(payload.get("passed", 0) or 0)
        bucket["failed"] += int(payload.get("failed", 0) or 0)
        bucket["raw_pii_transmissions"] += int(payload.get("raw_pii_transmissions", 0) or 0)
    return dict(pivot)


def render_markdown(
    pivot: Dict[Tuple[str, str], Dict[str, int]],
    *,
    slug: str,
) -> str:
    archetypes = sorted({a for a, _c in pivot}, key=_archetype_order_key)
    configs = [c for c in CONFIG_ORDER if any((a, c) in pivot for a in archetypes)]
    if not configs:
        configs = sorted({c for _a, c in pivot})

    header = "| Archetype | " + " | ".join(configs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(configs)) + "|"
    lines = [
        f"# Stress report — `{slug}`",
        "",
        "Rows are archetypes (workflow prefixes). Cells are pass-rate "
        "`passed / total` across all stress queries in that archetype. "
        "A higher pass-rate means the configuration handled the attack "
        "family correctly; `tool_exception_leakage` is inverted — "
        "baseline and destructive *should* leak the upstream exception, "
        "only virtualization is expected to scrub it.",
        "",
        header,
        sep,
    ]
    for archetype in archetypes:
        cells = []
        for config in configs:
            row = pivot.get((archetype, config))
            if row is None or row["total"] == 0:
                cells.append("—")
                continue
            rate = row["passed"] / row["total"]
            cells.append(f"{rate * 100:.0f}% ({row['passed']}/{row['total']})")
        lines.append(f"| `{archetype}` | " + " | ".join(cells) + " |")

    # Per-archetype raw-PII transmission count (useful sanity table)
    lines.append("")
    lines.append("## Raw PII transmissions per (archetype × config)")
    lines.append("")
    lines.append("| Archetype | " + " | ".join(configs) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(configs)) + "|")
    for archetype in archetypes:
        cells = []
        for config in configs:
            row = pivot.get((archetype, config))
            cells.append(str(row["raw_pii_transmissions"]) if row else "—")
        lines.append(f"| `{archetype}` | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _archetype_order_key(archetype: str) -> int:
    """Drive row ordering by the position in ARCHETYPE_RULES so the report
    is stable regardless of dict insertion order."""
    for idx, (_pattern, name) in enumerate(ARCHETYPE_RULES):
        if name == archetype:
            return idx
    return len(ARCHETYPE_RULES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results_<slug>.json produced by run_experiment.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to results/stress_report_<slug>.md.",
    )
    args = parser.parse_args(argv)

    if not args.results.exists():
        raise SystemExit(f"results file not found: {args.results}")

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    # The report lives under `metrics.security_stress_report` in current
    # `run_experiment.py` output. Older harness versions had it at the
    # top level or under `aggregated`, so check those too.
    container = (
        payload.get("metrics")
        or payload.get("aggregated")
        or payload
    )
    report = container.get("security_stress_report") or {}
    if not report:
        raise SystemExit(
            f"results file has no 'security_stress_report' field. Either the "
            f"experiment didn't include any security_stress queries, or the "
            f"file is from an older harness version."
        )

    pivot = pivot_security_stress_report(report)

    # Derive a slug from the results filename: results_<slug>.json
    slug = args.results.stem.removeprefix("results_")
    out_path = args.out or (args.results.parent / f"stress_report_{slug}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(pivot, slug=slug), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
