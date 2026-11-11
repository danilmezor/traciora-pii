"""Aggregate §3 full-framework results across (model, language) cells.

The §3 matrix runs each (model, language) as its own invocation so the
OPF second pass sees its in-distribution language. Each invocation writes
``results_<model-slug>__opf-<lang>.{json,md}`` independently. This script
stitches the 9 cells back into a single per-model overview table plus a
per-(model, language) detail table for the paper.

Usage::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.aggregate_section3 \
        --models deepseek/deepseek-v4-flash openrouter/owl-alpha z-ai/glm-5 \
        --out results/section3_matrix.md

If ``--models`` is omitted, autodetects every ``results_*__opf-{en,de,ru}.json``
triple under ``results/`` and groups by the shared model slug.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LANGUAGES: Sequence[str] = ("en", "de", "ru")
PIPELINES: Sequence[str] = ("baseline", "destructive", "virtualization")


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-").lower()


def _cell_path(model_slug: str, language: str) -> Path:
    return RESULTS_DIR / f"results_{model_slug}__opf-{language}.json"


def _load_cell(path: Path) -> Dict:
    return json.loads(path.read_text())


def _weighted_rate(numerators: List[int], denominators: List[int]) -> float:
    n = sum(numerators)
    d = sum(denominators)
    return (n / d) if d else 0.0


def _per_cell_row(model_slug: str, language: str, pipeline: str, cell: Dict) -> Dict:
    metrics = cell["metrics"]
    n = cell["dataset_size"]
    tsr = metrics["tool_call_success_rate"].get(pipeline)
    aem_block = metrics["argument_exact_match"].get(pipeline, {})
    aem_exact = aem_block.get("exact")
    aem_n = aem_block.get("considered")
    pii_exposure = metrics["model_pii_exposure_count"].get(pipeline)
    leak_rate = metrics.get("leak_guard_trigger_rate", {}).get(pipeline)
    return {
        "model": model_slug,
        "language": language,
        "pipeline": pipeline,
        "n_queries": n,
        "tsr": tsr,
        "aem_exact": aem_exact,
        "aem_considered": aem_n,
        "pii_exposure_count": pii_exposure,
        "leak_guard_rate": leak_rate,
    }


def aggregate(model_names: Sequence[str]) -> List[Dict]:
    """Return one row per (model, language, pipeline). Skips missing cells."""
    rows: List[Dict] = []
    for model_name in model_names:
        model_slug = _model_slug(model_name)
        for language in LANGUAGES:
            path = _cell_path(model_slug, language)
            if not path.exists():
                print(f"  [skip] {model_slug} / {language} — {path.name} not found")
                continue
            cell = _load_cell(path)
            for pipeline in PIPELINES:
                rows.append(_per_cell_row(model_slug, language, pipeline, cell))
    return rows


def _fmt_pct(x):
    return f"{x*100:5.1f}%" if x is not None else "  n/a"


def _fmt_int(x):
    return f"{x:>5d}" if x is not None else "  n/a"


def render_per_model_summary(rows: List[Dict]) -> str:
    """One row per (model, pipeline), aggregated across all 3 languages."""
    by_model_pipe: Dict[tuple, Dict] = defaultdict(lambda: {
        "tsr_num": 0.0, "tsr_den": 0,
        "aem_num": 0.0, "aem_den": 0,
        "pii": 0,
        "leak_num": 0.0, "leak_den": 0,
    })
    for r in rows:
        if r["tsr"] is None:
            continue
        key = (r["model"], r["pipeline"])
        agg = by_model_pipe[key]
        agg["tsr_num"] += r["tsr"] * r["n_queries"]
        agg["tsr_den"] += r["n_queries"]
        if r["aem_exact"] is not None and r["aem_considered"]:
            agg["aem_num"] += r["aem_exact"] * r["aem_considered"]
            agg["aem_den"] += r["aem_considered"]
        if r["pii_exposure_count"] is not None:
            agg["pii"] += r["pii_exposure_count"]
        if r["leak_guard_rate"] is not None:
            agg["leak_num"] += r["leak_guard_rate"] * r["n_queries"]
            agg["leak_den"] += r["n_queries"]

    models_ordered = []
    seen = set()
    for r in rows:
        if r["model"] not in seen:
            seen.add(r["model"])
            models_ordered.append(r["model"])

    lines = [
        "## §3 Full Framework — per-model summary (all languages)",
        "",
        "| Model | Pipeline | N | TSR | Arg-EM | PII→model | Leak-guard |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in models_ordered:
        for pipe in PIPELINES:
            agg = by_model_pipe.get((m, pipe))
            if not agg or agg["tsr_den"] == 0:
                continue
            tsr = agg["tsr_num"] / agg["tsr_den"]
            aem = (agg["aem_num"] / agg["aem_den"]) if agg["aem_den"] else None
            leak = (agg["leak_num"] / agg["leak_den"]) if agg["leak_den"] else None
            lines.append(
                f"| `{m}` | {pipe} | {agg['tsr_den']} | "
                f"{_fmt_pct(tsr)} | {_fmt_pct(aem)} | "
                f"{_fmt_int(agg['pii'])} | {_fmt_pct(leak)} |"
            )
    return "\n".join(lines)


def render_per_cell_detail(rows: List[Dict]) -> str:
    lines = [
        "## §3 Full Framework — per-(model, language) detail",
        "",
        "| Model | Lang | Pipeline | N | TSR | Arg-EM | PII→model | Leak-guard |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["tsr"] is None:
            continue
        lines.append(
            f"| `{r['model']}` | {r['language']} | {r['pipeline']} | "
            f"{r['n_queries']} | {_fmt_pct(r['tsr'])} | "
            f"{_fmt_pct(r['aem_exact'])} | "
            f"{_fmt_int(r['pii_exposure_count'])} | "
            f"{_fmt_pct(r['leak_guard_rate'])} |"
        )
    return "\n".join(lines)


def render_virtualization_delta(rows: List[Dict]) -> str:
    """Headline: virtualization TSR vs baseline TSR, per (model, lang)."""
    by_key = {(r["model"], r["language"], r["pipeline"]): r for r in rows}
    models_ordered = []
    seen = set()
    for r in rows:
        if r["model"] not in seen:
            seen.add(r["model"])
            models_ordered.append(r["model"])

    lines = [
        "## §3 Headline — virtualization framework vs unprotected baseline",
        "",
        "| Model | Lang | Baseline TSR | Virt TSR | Δ TSR | Destr TSR | Virt vs Destr |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in models_ordered:
        for lang in LANGUAGES:
            b = by_key.get((m, lang, "baseline"))
            v = by_key.get((m, lang, "virtualization"))
            d = by_key.get((m, lang, "destructive"))
            if not (b and v and d) or b["tsr"] is None:
                continue
            delta = v["tsr"] - b["tsr"]
            delta_destr = v["tsr"] - d["tsr"]
            lines.append(
                f"| `{m}` | {lang} | {_fmt_pct(b['tsr'])} | "
                f"{_fmt_pct(v['tsr'])} | {delta*100:+5.1f}pp | "
                f"{_fmt_pct(d['tsr'])} | {delta_destr*100:+5.1f}pp |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+",
        help="Model names in OpenRouter form (e.g. anthropic/claude-sonnet-4-6). "
             "If omitted, autodetects every triple under results/.",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "section3_matrix.md",
        help="Output markdown path (default: results/section3_matrix.md)",
    )
    args = parser.parse_args()

    if args.models:
        models = list(args.models)
    else:
        triples: Dict[str, set] = defaultdict(set)
        for p in RESULTS_DIR.glob("results_*__opf-*.json"):
            stem = p.stem  # results_<slug>__opf-<lang>
            m = re.match(r"results_(.+)__opf-(en|de|ru)$", stem)
            if m:
                triples[m.group(1)].add(m.group(2))
        models = [slug for slug, langs in triples.items() if len(langs) == 3]
        if not models:
            print("No complete (en+de+ru) triples found in results/. "
                  "Pass --models explicitly to render partial results.")
            return 1
        models.sort()
        print(f"Autodetected models: {models}")

    rows = aggregate(models)
    if not rows:
        print("No result rows produced — nothing to render.")
        return 1

    sections = [
        render_per_model_summary(rows),
        "",
        render_virtualization_delta(rows),
        "",
        render_per_cell_detail(rows),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
