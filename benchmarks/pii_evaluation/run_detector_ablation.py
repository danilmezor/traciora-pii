"""Consolidated detector ablation driver.

Loops the dataset through five detector configurations and emits one
markdown + JSON table comparing macro / micro precision and recall plus
end-to-end seconds.

Configurations
--------------

  1. ``regex_only``       — first-pass regex detectors only (production lower bound).
  2. ``regex_opf_base``   — regex + base OPF token classifier.
  3. ``regex_opf_routed`` — regex + per-language fine-tuned OPF; languages
                            without a registered fine-tune fall back to ``base``.
                            Matches production routing semantics.
  4. ``regex_llm``        — regex + OpenAI-compatible LLM via OpenRouter
                            (or any local endpoint). Model name must be
                            supplied via ``--detector-llm-model``.
  5. ``regex_presidio``   — regex + Microsoft Presidio (third-party baseline).

Configurations whose dependencies are unavailable (extras not installed,
LLM unreachable, --detector-llm-model not set) are reported with
``status="skipped: <reason>"`` rather than dropped silently.

Usage
-----

::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_detector_ablation
    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_detector_ablation \\
        --language en --bucket single_turn

Outputs land in ``results/detector_ablation_<slug>.{json,md}``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional benchmark helper
    def load_dotenv(*args, **kwargs):
        return False

from traciora_pii._latency import LatencyRecorder, install_recorder, reset_recorder
from traciora_pii.config import PIIConfig, load_config
from traciora_pii.pii_virtualizer import build_second_pass_detector

from benchmarks.pii_evaluation.bootstrap import CIResult, bootstrap_ci, bootstrap_pr_f1
from benchmarks.pii_evaluation.cli_common import (
    add_detector_llm_flags,
    add_language_bucket_flags,
    detector_llm_overrides,
    model_slug,
)
from benchmarks.pii_evaluation.dataset import EvalQuery, load_dataset
from benchmarks.pii_evaluation import metrics


RESULTS_DIR = Path(__file__).resolve().parent / "results"

ROUTED_OPF = "ROUTED"
LANGUAGES = ("en", "de", "ru")


@dataclass
class AblationSpec:
    name: str
    second_pass: str
    opf_model: Optional[str] = None  # ROUTED, "base", or None when N/A


@dataclass
class AblationRow:
    name: str
    requested_mode: str
    opf_model: str
    pr: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_queries: int = 0
    total_s: float = 0.0
    detector_total_s: float = 0.0
    status: str = "ok"
    routing_decisions: Dict[str, str] = field(default_factory=dict)
    # Per-query (tp, fp, fn) records keyed by placeholder. Populated by
    # _run_one so the rendering layer can bootstrap CIs over P/R/F1 and
    # micro/macro totals.
    per_query_records: List[Dict[str, Tuple[int, int, int]]] = field(default_factory=list)


ABLATION_CONFIGS: List[AblationSpec] = [
    AblationSpec("regex_only",       second_pass="none"),
    AblationSpec("regex_opf_base",   second_pass="opf",      opf_model="base"),
    AblationSpec("regex_opf_routed", second_pass="opf",      opf_model=ROUTED_OPF),
    AblationSpec("regex_llm",   second_pass="llm"),
    AblationSpec("regex_presidio",   second_pass="presidio"),
]


def _config_for_spec(
    spec: AblationSpec, base: PIIConfig, language: str,
) -> PIIConfig:
    overrides: Dict[str, Any] = {"second_pass": spec.second_pass}
    if spec.second_pass == "presidio" and language in {"en", "de", "ru"}:
        overrides["presidio"] = {"language": language}
    return base.with_overrides({"detector": overrides})


def _resolve_routed_opf(
    config: PIIConfig, language: str,
) -> str:
    """Pick the OPF entry to use for *language* under the routed config.

    Prefers a registered per-language fine-tune (``en``/``de``/``ru``)
    and falls back to ``base`` when the language has no entry.
    """
    if language in config.detector.opf.models:
        return language
    return config.detector.opf.default_model or "base"


def _aggregate_pr(pr_runs: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Merge per-language P/R dicts (which include __macro__/__micro__) into one.

    Sums TP/FP/FN per placeholder across runs and re-derives
    macro/micro on the merged counts. The per-language ``__macro__``/
    ``__micro__`` rows of the input are ignored — they would double-count
    if summed.
    """
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    for pr in pr_runs:
        for ph, row in pr.items():
            if ph.startswith("__"):
                continue
            tp[ph] += row.get("tp") or 0
            fp[ph] += row.get("fp") or 0
            fn[ph] += row.get("fn") or 0
    return metrics._summarize_pr(tp, fp, fn)


def _run_one(
    spec: AblationSpec,
    *,
    base_config: PIIConfig,
    dataset: List[EvalQuery],
    language_filter: str,
    recorder: LatencyRecorder,
) -> AblationRow:
    row = AblationRow(
        name=spec.name,
        requested_mode=spec.second_pass,
        opf_model=spec.opf_model or "",
    )

    if spec.second_pass == "none":
        # Direct call, no per-language partitioning needed.
        recorder.reset()
        t0 = time.perf_counter()
        row.pr, row.per_query_records = metrics.detection_precision_recall_with_records(dataset)
        row.total_s = time.perf_counter() - t0
        snap = recorder.snapshot()
        row.detector_total_s = sum(s["total_s"] for s in snap.values())
        row.n_queries = len(dataset)
        return row

    # OPF (base or routed) / llm / presidio
    if spec.second_pass == "opf" and spec.opf_model == ROUTED_OPF:
        # Partition the dataset by language; build a detector per slice.
        recorder.reset()
        t0 = time.perf_counter()
        per_slice_pr: List[Dict[str, Any]] = []
        slice_languages = (
            [language_filter]
            if language_filter in LANGUAGES
            else sorted({q.language for q in dataset if q.language})
        )
        for lang in slice_languages:
            subset = [q for q in dataset if q.language == lang]
            if not subset:
                continue
            opf_choice = _resolve_routed_opf(base_config, lang)
            row.routing_decisions[lang] = opf_choice
            cfg = _config_for_spec(spec, base_config, lang)
            detector = build_second_pass_detector(cfg, opf_model_name=opf_choice)
            if detector is None:
                row.status = "skipped: routed OPF unavailable for at least one language"
                row.total_s = time.perf_counter() - t0
                return row
            try:
                pr, recs = metrics.detection_precision_recall_with_records(
                    subset, second_pass_detector=detector,
                )
                per_slice_pr.append(pr)
                row.per_query_records.extend(recs)
            finally:
                close = getattr(detector, "close", None)
                if callable(close):
                    close()
        row.pr = _aggregate_pr(per_slice_pr)
        row.total_s = time.perf_counter() - t0
        snap = recorder.snapshot()
        row.detector_total_s = sum(s["total_s"] for s in snap.values())
        row.n_queries = len(dataset)
        return row

    # opf-base / llm / presidio: single detector across the dataset.
    cfg = _config_for_spec(spec, base_config, language_filter)
    opf_choice = spec.opf_model if spec.second_pass == "opf" else None
    recorder.reset()
    t0 = time.perf_counter()
    detector = build_second_pass_detector(cfg, opf_model_name=opf_choice)
    if detector is None:
        row.status = f"skipped: {spec.second_pass} backend unavailable (extra not installed or unreachable)"
        row.total_s = time.perf_counter() - t0
        return row
    try:
        row.pr, row.per_query_records = metrics.detection_precision_recall_with_records(
            dataset, second_pass_detector=detector,
        )
    except Exception as exc:  # noqa: BLE001
        row.status = f"skipped: {type(exc).__name__}: {exc}"
        row.total_s = time.perf_counter() - t0
        return row
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    row.total_s = time.perf_counter() - t0
    snap = recorder.snapshot()
    row.detector_total_s = sum(s["total_s"] for s in snap.values())
    row.n_queries = len(dataset)
    return row


def _row_to_dict(row: AblationRow) -> Dict[str, Any]:
    macro = row.pr.get("__macro__", {})
    micro = row.pr.get("__micro__", {})
    return {
        "name": row.name,
        "requested_mode": row.requested_mode,
        "opf_model": row.opf_model,
        "n_queries": row.n_queries,
        "total_s": row.total_s,
        "detector_total_s": row.detector_total_s,
        "status": row.status,
        "routing_decisions": row.routing_decisions,
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "micro_precision": micro.get("precision"),
        "micro_recall": micro.get("recall"),
        "tp": micro.get("tp"),
        "fp": micro.get("fp"),
        "fn": micro.get("fn"),
        "support": micro.get("support"),
        "per_placeholder": {
            ph: row.pr[ph] for ph in row.pr if not ph.startswith("__")
        },
    }


def _macro_p_from_records(records: Sequence[Dict[str, Tuple[int, int, int]]]) -> float:
    placeholders = sorted({ph for r in records for ph in r})
    if not placeholders:
        return 0.0
    acc = 0.0
    for ph in placeholders:
        tp = sum(r.get(ph, (0, 0, 0))[0] for r in records)
        fp = sum(r.get(ph, (0, 0, 0))[1] for r in records)
        acc += tp / (tp + fp) if (tp + fp) else 0.0
    return acc / len(placeholders)


def _macro_r_from_records(records: Sequence[Dict[str, Tuple[int, int, int]]]) -> float:
    placeholders = sorted({ph for r in records for ph in r})
    if not placeholders:
        return 0.0
    acc = 0.0
    for ph in placeholders:
        tp = sum(r.get(ph, (0, 0, 0))[0] for r in records)
        fn = sum(r.get(ph, (0, 0, 0))[2] for r in records)
        acc += tp / (tp + fn) if (tp + fn) else 0.0
    return acc / len(placeholders)


def _micro_pr_from_records(
    records: Sequence[Dict[str, Tuple[int, int, int]]],
) -> Tuple[float, float]:
    tp = sum(v[0] for r in records for v in r.values())
    fp = sum(v[1] for r in records for v in r.values())
    fn = sum(v[2] for r in records for v in r.values())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return p, rec


def _format_ci_cell(ci: CIResult) -> str:
    return ci.format_md()


def _bootstrap_macro_micro_for_row(
    row: AblationRow,
    *,
    n_resamples: int,
) -> Optional[Dict[str, CIResult]]:
    """Return CIs for macro P, macro R, micro P, micro R. Returns None when
    per-query records aren't available (e.g. a skipped row)."""
    if not row.per_query_records:
        return None
    macro_p_ci = bootstrap_ci(row.per_query_records, _macro_p_from_records, n_resamples=n_resamples)
    macro_r_ci = bootstrap_ci(row.per_query_records, _macro_r_from_records, n_resamples=n_resamples)
    micro_p_ci = bootstrap_ci(
        row.per_query_records, lambda recs: _micro_pr_from_records(recs)[0], n_resamples=n_resamples,
    )
    micro_r_ci = bootstrap_ci(
        row.per_query_records, lambda recs: _micro_pr_from_records(recs)[1], n_resamples=n_resamples,
    )
    return {
        "macro_p": macro_p_ci,
        "macro_r": macro_r_ci,
        "micro_p": micro_p_ci,
        "micro_r": micro_r_ci,
    }


def _write_markdown(
    path: Path,
    rows: List[AblationRow],
    *,
    dataset_size: int,
    n_resamples: int = 1000,
) -> None:
    lines: List[str] = []
    lines.append("# Detector ablation\n")
    lines.append(f"Dataset size: {dataset_size} queries\n")
    lines.append(
        "\n_Total seconds is wall-clock time inside "
        "``metrics.detection_precision_recall_with_records``. "
        "Skipped rows mean the optional dependency was missing or the backend "
        "was unreachable — see ``status``. P/R cells carry 95% bootstrap CIs "
        f"({n_resamples} resamples, seed=42) when per-query records are "
        "available._\n"
    )
    lines.append(
        "\n| Configuration | macro P (95% CI) | micro P (95% CI) | "
        "macro R (95% CI) | micro R (95% CI) | total (s) | s/query | status |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---|")
    for row in rows:
        macro = row.pr.get("__macro__", {})
        micro = row.pr.get("__micro__", {})
        n = row.n_queries or 1
        ci_block = _bootstrap_macro_micro_for_row(row, n_resamples=n_resamples)
        if ci_block is not None:
            mp = _format_ci_cell(ci_block["macro_p"])
            ip = _format_ci_cell(ci_block["micro_p"])
            mr = _format_ci_cell(ci_block["macro_r"])
            ir = _format_ci_cell(ci_block["micro_r"])
        else:
            mp = f"{macro['precision']:.3f}" if "precision" in macro else "—"
            ip = f"{micro['precision']:.3f}" if "precision" in micro else "—"
            mr = f"{macro['recall']:.3f}" if "recall" in macro else "—"
            ir = f"{micro['recall']:.3f}" if "recall" in micro else "—"
        lines.append(
            "| `{name}` | {mp} | {ip} | {mr} | {ir} | {tot} | {sp} | {st} |".format(
                name=row.name,
                mp=mp,
                ip=ip,
                mr=mr,
                ir=ir,
                tot=f"{row.total_s:.2f}",
                sp=f"{(row.total_s / n):.3f}",
                st=row.status if row.status != "ok" else "ok",
            )
        )
    routing_rows = [r for r in rows if r.routing_decisions]
    if routing_rows:
        lines.append("\n## Routed OPF — per-language model decisions\n")
        lines.append("| Configuration | Language | OPF model |")
        lines.append("|---|---|---|")
        for row in routing_rows:
            for lang in sorted(row.routing_decisions):
                lines.append(
                    f"| `{row.name}` | {lang} | `{row.routing_decisions[lang]}` |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_bucket_flags(parser)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of configuration names to run (default: all).",
    )
    add_detector_llm_flags(parser)
    args = parser.parse_args()

    load_dotenv()

    base_config = load_config(Path(args.config) if args.config else None)
    det_llm = detector_llm_overrides(args)
    if det_llm:
        base_config = base_config.with_overrides({"detector": {"llm": det_llm}})

    dataset = list(load_dataset())
    if args.language != "all":
        dataset = [q for q in dataset if q.language == args.language]
    if args.bucket:
        dataset = [q for q in dataset if q.bucket == args.bucket]
    if not dataset:
        print("Empty dataset after filters; aborting.", file=sys.stderr)
        return 1

    selected = ABLATION_CONFIGS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [s for s in ABLATION_CONFIGS if s.name in wanted]
        if not selected:
            print(f"--only={args.only!r} matched none of {[s.name for s in ABLATION_CONFIGS]}", file=sys.stderr)
            return 1

    recorder = LatencyRecorder()
    token = install_recorder(recorder)
    try:
        rows: List[AblationRow] = []
        for spec in selected:
            print(f"[{spec.name}] running ({spec.second_pass}, opf_model={spec.opf_model})...")
            row = _run_one(
                spec,
                base_config=base_config,
                dataset=dataset,
                language_filter=args.language,
                recorder=recorder,
            )
            rows.append(row)
            macro = row.pr.get("__macro__", {})
            micro = row.pr.get("__micro__", {})
            print(
                f"  -> status={row.status} total_s={row.total_s:.2f} "
                f"macro_P={macro.get('precision', 0):.3f} micro_P={micro.get('precision', 0):.3f} "
                f"macro_R={macro.get('recall', 0):.3f} micro_R={micro.get('recall', 0):.3f}"
            )
    finally:
        reset_recorder(token)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_slug(fallback="ablation")
    # When the run includes the LLM second-pass, encode the LLM model name
    # into the filename so successive invocations with different
    # --detector-llm-model values don't overwrite each other.
    llm_in_lineup = any(s.second_pass == "llm" for s in selected)
    if llm_in_lineup and args.detector_llm_model:
        llm_tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.detector_llm_model).strip("-").lower()
        slug = f"{slug}__llm-{llm_tag}"
    json_path = RESULTS_DIR / f"detector_ablation_{slug}.json"
    md_path = RESULTS_DIR / f"detector_ablation_{slug}.md"
    json_path.write_text(
        json.dumps(
            {
                "dataset_size": len(dataset),
                "language_filter": args.language,
                "bucket_filter": args.bucket,
                "rows": [_row_to_dict(r) for r in rows],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_markdown(md_path, rows, dataset_size=len(dataset))
    print(f"\nResults written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
