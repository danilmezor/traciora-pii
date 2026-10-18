"""§2 paper aggregator — fold the per-invocation `detector_ablation_*.json`
files into one comparison table.

`run_detector_ablation` writes one JSON+MD pair per invocation. The §2
paper run produces:

  * one file for the non-LLM lineup (regex_only / regex_opf_base /
    regex_opf_routed / regex_presidio)
  * one file *per paper-reported LLM* for the regex_llm second-pass family

This aggregator reads every `detector_ablation_*.json` under
``results/``, picks the unique row from each (regex_llm runs have a
single row; the non-LLM run has four), and emits the §2 paper table:

  rows    = detector configuration (regex_only, regex_opf_base,
            regex_opf_routed, regex_presidio, regex_llm__<llm-1>, …)
  columns = macro P / micro P / macro R / micro R / status / per-query
            seconds

The `regex_llm` rows are annotated with the LLM model name. The paper
matrix intentionally excludes stale pre-submission cells such as the
llama endpoint that was replaced by gemma in the cross-source study.

Usage::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.aggregate_detector_ablation \\
        --out benchmarks/pii_evaluation/results/section2_ablation_matrix.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Render order for detector configurations.
CONFIG_ORDER: List[str] = [
    "regex_only",
    "regex_opf_base",
    "regex_opf_routed",
    "regex_presidio",
    "regex_llm",
]

SECTION2_LLM_ALLOWLIST = {
    "mistralai-ministral-3b-2512",
    "nvidia-nemotron-nano-9b-v2",
    "nvidia-nemotron-nano-9b-v2-free",
    "qwen-qwen3.5-9b",
}


def _discover_ablation_files() -> List[Path]:
    """Every detector_ablation_*.json under results/, skipping smoke runs."""
    paths = []
    for p in sorted(RESULTS_DIR.glob("detector_ablation_*.json")):
        if "smoke" in p.stem:
            continue
        paths.append(p)
    return paths


def _extract_llm_tag(slug: str) -> str:
    """Pull the LLM-tag off a slug like 'ablation__llm-nvidia-nemotron-…'."""
    m = re.search(r"__llm-(.+)$", slug)
    return m.group(1) if m else ""


def _per_row_display_name(row: Dict[str, Any], llm_tag: str) -> str:
    name = row.get("name", "?")
    if name == "regex_llm" and llm_tag:
        return f"regex_llm__{llm_tag}"
    return name


def _config_sort_key(name: str) -> Tuple[int, str]:
    for idx, base in enumerate(CONFIG_ORDER):
        if name == base or name.startswith(f"{base}__"):
            return (idx, name)
    return (len(CONFIG_ORDER), name)


def _row_to_record(row: Dict[str, Any], llm_tag: str) -> Dict[str, Any]:
    """Pick the fields the §2 table needs from a raw row."""
    return {
        "name": _per_row_display_name(row, llm_tag),
        "macro_P": row.get("macro_precision"),
        "macro_R": row.get("macro_recall"),
        "micro_P": row.get("micro_precision"),
        "micro_R": row.get("micro_recall"),
        "tp": row.get("tp"),
        "fp": row.get("fp"),
        "fn": row.get("fn"),
        "support": row.get("support"),
        "n_queries": row.get("n_queries"),
        "total_s": row.get("total_s"),
        "s_per_query": (
            row["total_s"] / row["n_queries"]
            if row.get("total_s") and row.get("n_queries")
            else None
        ),
        "status": row.get("status", "ok"),
        "opf_model": row.get("opf_model", ""),
    }


def collect_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    """Return one record per (config, llm) across every detector_ablation_*.json."""
    seen: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        slug = path.stem.removeprefix("detector_ablation_")
        llm_tag = _extract_llm_tag(slug)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            record = _row_to_record(row, llm_tag)
            if record["name"] == "regex_llm" and not llm_tag:
                continue
            if record["name"].startswith("regex_llm__") and llm_tag not in SECTION2_LLM_ALLOWLIST:
                continue
            # Deduplicate: keep the first occurrence of each display name.
            seen.setdefault(record["name"], record)
    return sorted(seen.values(), key=lambda r: _config_sort_key(r["name"]))


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:5.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_s(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown(records: List[Dict[str, Any]], dataset_size: int) -> str:
    lines = [
        "# §2 Detector Ablation — paper matrix",
        "",
        f"Dataset size: {dataset_size} queries. All P / R cells are point estimates "
        "summed across (en + de + ru). For 95 % bootstrap CIs see the per-row "
        "`detector_ablation_<slug>.md` source files.",
        "",
        "| Configuration | macro P | micro P | macro R | micro R | TP / FP / FN | s/query | status |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in records:
        cells = [
            f"`{r['name']}`",
            _fmt_pct(r["macro_P"]),
            _fmt_pct(r["micro_P"]),
            _fmt_pct(r["macro_R"]),
            _fmt_pct(r["micro_R"]),
            f"{r['tp'] or 0} / {r['fp'] or 0} / {r['fn'] or 0}",
            _fmt_s(r["s_per_query"]),
            r["status"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=RESULTS_DIR / "section2_ablation_matrix.md",
        help="Output markdown path (default: results/section2_ablation_matrix.md).",
    )
    args = parser.parse_args()

    paths = _discover_ablation_files()
    if not paths:
        print("No detector_ablation_*.json files found under results/.")
        return 1

    # Pull dataset_size from any file (they should all match).
    first_payload = json.loads(paths[0].read_text(encoding="utf-8"))
    dataset_size = first_payload.get("dataset_size", 0)

    records = collect_rows(paths)
    if not records:
        print("Discovered files but no rows to render.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(records, dataset_size), encoding="utf-8")
    print(f"Wrote {args.out} ({len(records)} configurations from {len(paths)} ablation files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
