"""Entry point for the PII virtualization evaluation experiment.

Usage:

  # Dry run — only computes detection precision/recall, no LLM calls
  python -m benchmarks.pii_evaluation.run_experiment --dataset-only

  # Full run — requires PII_EVAL_LLM_ENABLED=1 plus an LLM provider key
  PII_EVAL_LLM_ENABLED=1 python -m benchmarks.pii_evaluation.run_experiment

The script writes artifacts under benchmarks/pii_evaluation/results/:

  results.json   — full per-query transcripts and metric breakdowns
  results.md     — markdown tables shaped for direct paste into Section 6
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from cryptography.fernet import Fernet
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional benchmark helper
    def load_dotenv(*args, **kwargs):
        return False

from traciora_pii._latency import LatencyRecorder, install_recorder, reset_recorder
from traciora_pii.config import PIIConfig, load_config
from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer, build_second_pass_detector
from traciora_pii.redaction import PIIRedactor


# Detector ablation modes supported by --second-pass / detector.second_pass.
# After the routed-OPF cleanup, OPF model selection happens via the
# orthogonal --opf-model flag (or detector.opf.default_model in the YAML),
# not via this enum.
SECOND_PASS_MODES = ("none", "opf", "llm", "presidio")


from benchmarks.pii_evaluation.bootstrap import bootstrap_ci
from benchmarks.pii_evaluation.cli_common import (
    add_detector_llm_flags,
    add_language_bucket_flags,
    detector_llm_overrides,
    model_slug,
)
from benchmarks.pii_evaluation.dataset import EvalQuery, load_dataset
from benchmarks.pii_evaluation import metrics
from benchmarks.pii_evaluation import metrics_argfidelity
from benchmarks.pii_evaluation.pipelines import (
    RunResult,
    run_baseline,
    run_destructive,
    run_destructive_presidio,
    run_virtualization,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------
# LLM client builder
# ----------------------------------------------------------------------

def _build_eval_llm(config: PIIConfig) -> Any:
    """Build a standalone benchmark LLM client from ``config.eval.llm``.

    The default ``base_url`` is OpenRouter
    (``https://openrouter.ai/api/v1``), which is OpenAI-compatible.
    Model is selected by prefix-routed name like
    ``anthropic/claude-sonnet-4-6``. ``config.eval.llm.model`` has no
    default and must be set via ``--llm-model`` on the CLI (or by
    editing ``traciora.local.yaml``).
    """
    from langchain_openai import ChatOpenAI

    llm_cfg = config.eval.llm
    if not llm_cfg.model:
        raise SystemExit(
            "eval.llm.model is unset. Pass --llm-model <openrouter-name> on the CLI, "
            "e.g. --llm-model anthropic/claude-sonnet-4-6. "
            "See https://openrouter.ai/docs#models for the model catalog."
        )
    api_key = config.secret(llm_cfg.api_key_env, required=True)
    # Reasoning suppression. For reasoning-mode models (glm-4.x, glm-5.1,
    # qwen3.x, deepseek-r1) we set `enabled: false` so the model emits the
    # answer directly in `content` instead of burning tokens on hidden CoT
    # and leaving `content=None`. But `openai/gpt-oss-*` rejects that with
    # HTTP 400 ("Reasoning is mandatory for this endpoint and cannot be
    # disabled") — for that family we keep only `exclude: true` so the
    # reasoning still happens server-side but isn't echoed back.
    reasoning = {"exclude": True}
    if not llm_cfg.model.startswith("openai/gpt-oss"):
        reasoning["enabled"] = False
    return ChatOpenAI(
        model=llm_cfg.model,
        api_key=api_key,
        base_url=llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        extra_body={"reasoning": reasoning},
    )


# ----------------------------------------------------------------------
# Result aggregation + serialization
# ----------------------------------------------------------------------

def _aggregate(
    results: List[RunResult],
    dataset: List[EvalQuery],
    *,
    detector_configuration: Dict[str, Any],
    latency_contribution: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    # Records-emitting variant feeds Table 2 bootstrap CIs. Point-estimate
    # summary lives at "detection_precision_recall" (backward compat); per-query
    # records at "detection_precision_recall_records".
    pr_summary, pr_records = metrics.detection_precision_recall_with_records(dataset)
    return {
        "detector_configuration": detector_configuration,
        "tool_call_success_rate": metrics.tool_call_success_rate(results, dataset),
        "argument_exact_match": metrics_argfidelity.argument_exact_match(results, dataset),
        "model_pii_exposure_count": metrics.model_pii_exposure_count(results, dataset),
        "model_pii_exposure_examples": metrics.model_pii_exposure_examples(results, dataset),
        "latency_stats": metrics.latency_stats(results),
        "latency_contribution": latency_contribution or {},
        "leak_guard_trigger_rate": metrics.leak_guard_trigger_rate(results),
        "cross_turn_token_stability": metrics_argfidelity.cross_turn_token_stability(results, dataset),
        "detection_precision_recall": pr_summary,
        "detection_precision_recall_records": pr_records,
        "detection_precision_recall_per_language": metrics.detection_precision_recall_per_language(dataset),
        "detection_precision_recall_by_length_bucket": metrics.detection_precision_recall_by_length_bucket(dataset),
        "security_stress_report": metrics.security_stress_report(results, dataset),
        "per_bucket_per_language": _serialize_bucket_report(
            metrics_argfidelity.per_bucket_per_language_report(results, dataset),
        ),
        # Per-config per-query atoms for bootstrap CIs on M1/M2/M4/M5.
        # Stored as raw lists so the JSON output preserves them; the
        # markdown renderer derives CIs on the fly.
        "per_config_atoms": {
            cfg: {
                "tool_success": list(atoms.tool_success),
                "exposure_count": list(atoms.exposure_count),
                "elapsed_seconds": list(atoms.elapsed_seconds),
                "leak_guard_triggered": list(atoms.leak_guard_triggered),
            }
            for cfg, atoms in metrics.per_config_atoms(results, dataset).items()
        },
    }


def _detector_configuration(
    second_pass_detector: Any,
    *,
    config: PIIConfig,
    opf_model: str,
) -> Dict[str, Any]:
    """Snapshot the live detector wiring for inclusion in the results JSON.

    ``config.detector.second_pass`` is the user-requested mode
    (``none`` / ``opf`` / ``llm`` / ``presidio``). ``effective_mode``
    reports what the harness ended up with — they differ only when a
    request fell back to regex-only because the optional dependency was
    missing or construction failed.
    """
    detector_class = type(second_pass_detector).__name__ if second_pass_detector is not None else ""
    requested_mode = config.detector.second_pass
    if second_pass_detector is None:
        effective_mode = "none"
    elif detector_class == "OPFPIIDetector":
        effective_mode = "opf"
    elif detector_class == "LLMPIIDetector":
        effective_mode = "llm"
    else:
        effective_mode = detector_class.lower() or "unknown"
    fallback = requested_mode != "none" and second_pass_detector is None
    opf_entry = config.detector.opf.models.get(opf_model)
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "second_pass_enabled": second_pass_detector is not None,
        "fallback_to_regex_only": fallback,
        "second_pass_class": detector_class,
        "opf_model_name": opf_model if requested_mode == "opf" else "",
        "opf_checkpoint": opf_entry.checkpoint if (requested_mode == "opf" and opf_entry) else "",
        "opf_policy": opf_entry.policy if (requested_mode == "opf" and opf_entry) else "",
        "opf_device": config.detector.opf.device if requested_mode == "opf" else "",
        "opf_prefilter": config.detector.opf.prefilter if requested_mode == "opf" else None,
        "llm_base_url": config.detector.llm.base_url if requested_mode == "llm" else "",
        "llm_model": config.detector.llm.model if requested_mode == "llm" else "",
        "regex_policy": config.detector.regex_policy,
    }


def _serialize_bucket_report(
    report: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Convert the (config, bucket, language) keyed report into JSON keys.

    Tuples are JSON-unfriendly; we join with ``|`` so the artifact is
    parseable downstream without losing information.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for (config, bucket, language), payload in report.items():
        out[f"{config}|{bucket}|{language}"] = dict(payload)
    return out


def _result_to_dict(result: RunResult) -> Dict[str, Any]:
    payload = asdict(result)
    # tool_invocations are dataclasses; asdict already handled them
    return payload


def _write_json(path: Path, results: List[RunResult], dataset: List[EvalQuery], aggregated: Dict[str, Any]) -> None:
    payload = {
        "dataset_size": len(dataset),
        "results": [_result_to_dict(r) for r in results],
        "metrics": aggregated,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


_TABLE_CONFIG_ORDER = ("baseline", "destructive", "destructive_presidio", "virtualization")

_DETECTOR_CONFIG_KEYS = (
    "requested_mode",
    "effective_mode",
    "second_pass_enabled",
    "fallback_to_regex_only",
    "second_pass_class",
    "opf_model_name",
    "opf_checkpoint",
    "opf_policy",
    "opf_device",
    "llm_base_url",
    "llm_model",
)


def _p95(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    pos = 0.95 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def _md_header(dataset: List[EvalQuery], detector_config: Dict[str, Any]) -> List[str]:
    lines: List[str] = [
        "# PII Evaluation Results\n",
        f"Dataset size: {len(dataset)} queries\n",
    ]
    if detector_config:
        lines.append("\n## Detector configuration\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for key in _DETECTOR_CONFIG_KEYS:
            value = detector_config.get(key, "")
            if value == "" or value is None:
                continue
            lines.append(f"| {key} | `{value}` |")
    return lines


def _md_table1_end_to_end(aggregated: Dict[str, Any]) -> List[str]:
    success = aggregated.get("tool_call_success_rate", {})
    exposure = aggregated.get("model_pii_exposure_count", {})
    latency = aggregated.get("latency_stats", {})
    triggers = aggregated.get("leak_guard_trigger_rate", {})
    atoms_by_config = aggregated.get("per_config_atoms", {})

    lines: List[str] = ["\n## Table 1 — End-to-end metrics\n"]
    lines.append(
        "_Note: 'Raw PII transmissions to model' is a cumulative count. "
        "It directly scans for every query's boundary PII literals, including "
        "PII introduced by tool-result fixtures, in any message handed to the "
        "LLM. Multi-turn queries replay history on every iteration, so a "
        "single value can be counted multiple times. "
        "The intended interpretation is 'how often did raw PII cross the "
        "trust boundary?', not 'how many distinct values leaked?'. The scan "
        "is detector-independent, so detector misses still count as leaks. "
        "P/R/rate cells carry 95% bootstrap CIs (1000 resamples, seed=42) "
        "when per-query atoms are available._\n"
    )
    lines.append("| Configuration | Tool-call success | Raw PII transmissions to model | Median latency (s) | p95 latency (s) | Leak-guard triggers |")
    lines.append("|---|---|---|---|---|---|")

    order = [
        c for c in _TABLE_CONFIG_ORDER
        if c in atoms_by_config or c in success or c in latency
    ]
    for config in order:
        atoms = atoms_by_config.get(config)
        if atoms and atoms.get("tool_success") is not None and len(atoms["tool_success"]) > 0:
            success_ci = bootstrap_ci(
                [1.0 if b else 0.0 for b in atoms["tool_success"]],
                lambda xs: sum(xs) / len(xs),
            )
            s_cell = f"{success_ci.point*100:.1f}% [{success_ci.lower*100:.1f}%, {success_ci.upper*100:.1f}%]"
        else:
            s = success.get(config)
            s_cell = f"{s*100:.1f}%" if s is not None else "—"

        if atoms and atoms.get("exposure_count"):
            exposure_ci = bootstrap_ci(atoms["exposure_count"], lambda xs: sum(xs))
            e_cell = f"{int(exposure_ci.point)} [{int(exposure_ci.lower)}, {int(exposure_ci.upper)}]"
        else:
            e_cell = str(exposure.get(config, 0))

        if atoms and atoms.get("elapsed_seconds"):
            median_ci = bootstrap_ci(
                atoms["elapsed_seconds"], lambda xs: statistics.median(xs),
            )
            p95_ci = bootstrap_ci(atoms["elapsed_seconds"], _p95)
            med_cell = f"{median_ci.point:.2f} [{median_ci.lower:.2f}, {median_ci.upper:.2f}]"
            p95_cell = f"{p95_ci.point:.2f} [{p95_ci.lower:.2f}, {p95_ci.upper:.2f}]"
        else:
            lat = latency.get(config, {})
            med = lat.get("median")
            p95 = lat.get("p95")
            med_cell = f"{med:.2f}" if med is not None else "—"
            p95_cell = f"{p95:.2f}" if p95 is not None else "—"

        if atoms and atoms.get("leak_guard_triggered") is not None and len(atoms["leak_guard_triggered"]) > 0:
            trig_ci = bootstrap_ci(
                [1.0 if b else 0.0 for b in atoms["leak_guard_triggered"]],
                lambda xs: sum(xs) / len(xs),
            )
            trig_cell = f"{trig_ci.point*100:.1f}% [{trig_ci.lower*100:.1f}%, {trig_ci.upper*100:.1f}%]"
        else:
            trig = triggers.get(config)
            trig_cell = f"{trig*100:.1f}%" if trig is not None else "—"

        lines.append(
            f"| {config} | {s_cell} | {e_cell} | {med_cell} | {p95_cell} | {trig_cell} |"
        )
    return lines


def _md_table2_detection_sanity(aggregated: Dict[str, Any]) -> List[str]:
    pr = aggregated.get("detection_precision_recall", {})
    pr_records = aggregated.get("detection_precision_recall_records")
    lines: List[str] = ["\n## Table 2 — Detection sanity check\n"]
    lines.append(
        "_Per-detector P/R against the full-framework task dataset. "
        "**Macro** is the unweighted mean of the per-tag P/R rows — "
        "useful for comparing detector classes; sensitive to small-support "
        "tags (e.g. the security-stress code-switched bucket contributes "
        "1-sample tags that drag the mean). **Micro** is support-weighted "
        "(``sum(tp) / sum(tp+fp)``), so each detected span contributes "
        "equally regardless of which tag it falls under — preferred when "
        "comparing two detectors on the same dataset. Cells carry 95% "
        "bootstrap CIs (1000 resamples, seed=42) when per-query records "
        "are available._\n"
    )
    lines.extend(metrics.render_detection_pr_table(pr, per_query_records=pr_records))
    return lines


def _md_table2a_pr_per_language(aggregated: Dict[str, Any]) -> List[str]:
    pr_per_lang = aggregated.get("detection_precision_recall_per_language", {})
    if not pr_per_lang:
        return []
    lines: List[str] = ["\n## Table 2a — Detection P/R per (language × placeholder)\n"]
    for lang in sorted(pr_per_lang.keys()):
        payload = pr_per_lang[lang]
        lines.append(f"\n### Language: `{lang}`\n")
        lines.append("| Detector | Precision | Recall | TP | FP | FN | Support |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        tag_keys = sorted(k for k in payload.keys() if not k.startswith("__"))
        for det in tag_keys:
            d = payload[det]
            lines.append(
                f"| `{det}` | {d['precision']:.3f} | {d['recall']:.3f} | "
                f"{d['tp']} | {d['fp']} | {d['fn']} | {d['support']} |"
            )
        if "__macro__" in payload:
            m = payload["__macro__"]
            lines.append(
                f"| **macro avg** | **{m['precision']:.3f}** | **{m['recall']:.3f}** "
                f"| — | — | — | — |"
            )
        if "__micro__" in payload:
            m = payload["__micro__"]
            lines.append(
                f"| **micro avg** | **{m['precision']:.3f}** | **{m['recall']:.3f}** "
                f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['support']} |"
            )
    return lines


def _md_table2b_pr_by_length(aggregated: Dict[str, Any]) -> List[str]:
    pr_by_length = aggregated.get("detection_precision_recall_by_length_bucket", {})
    if not pr_by_length:
        return []
    lines: List[str] = [
        "\n## Table 2b — Detection P/R by length bucket × language × placeholder\n",
        "_Length buckets: ``sentence`` (single short line), ``paragraph`` "
        "(multi-line, no blank-line separator), ``multi_paragraph`` (has "
        "blank-line separator), ``structured`` (markdown table pipes, "
        "tab-aligned, or ≥3 colon-and-space lines — log/key-value text)._\n",
        "| Length bucket | Lang | Detector | P | R | TP | FP | FN | Support |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in sorted(pr_by_length.keys()):
        for lang in sorted(pr_by_length[bucket].keys()):
            payload = pr_by_length[bucket][lang]
            tag_keys = sorted(k for k in payload.keys() if not k.startswith("__"))
            for det in tag_keys:
                d = payload[det]
                lines.append(
                    f"| {bucket} | {lang} | `{det}` | {d['precision']:.3f} | "
                    f"{d['recall']:.3f} | {d['tp']} | {d['fp']} | {d['fn']} | "
                    f"{d['support']} |"
                )
            if "__micro__" in payload:
                m = payload["__micro__"]
                lines.append(
                    f"| {bucket} | {lang} | **micro** | **{m['precision']:.3f}** | "
                    f"**{m['recall']:.3f}** | {m['tp']} | {m['fp']} | {m['fn']} | "
                    f"{m['support']} |"
                )
    return lines


def _md_table3_arg_fidelity(aggregated: Dict[str, Any]) -> List[str]:
    arg_match = aggregated.get("argument_exact_match", {})
    stability = aggregated.get("cross_turn_token_stability", {})
    lines: List[str] = ["\n## Table 3 — Argument fidelity and cross-turn stability\n"]
    lines.append("| Configuration | Argument exact match | Argument partial match | Cross-turn token stability |")
    lines.append("|---|---|---|---|")
    order = [c for c in _TABLE_CONFIG_ORDER if c in arg_match or c in stability]
    for config in order:
        am = arg_match.get(config, {})
        st = stability.get(config)
        lines.append(
            "| {cfg} | {ex} | {pa} | {st} |".format(
                cfg=config,
                ex=f"{am.get('exact', 0.0)*100:.1f}%" if am else "—",
                pa=f"{am.get('partial', 0.0)*100:.1f}%" if am else "—",
                st=f"{st*100:.1f}%" if st is not None else "—",
            )
        )
    return lines


def _md_table4_per_bucket_language(aggregated: Dict[str, Any]) -> List[str]:
    bucket_report = aggregated.get("per_bucket_per_language", {})
    if not bucket_report:
        return []
    lines: List[str] = [
        "\n## Table 4 — Per-(config × bucket × language)\n",
        "| Config | Bucket | Lang | n | Tool success | Raw PII | Median latency (s) | Leak triggers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(bucket_report.keys()):
        payload = bucket_report[key]
        cfg, bucket, lang = key.split("|", 2)
        lines.append(
            "| {cfg} | {b} | {l} | {n} | {s} | {e} | {m} | {t} |".format(
                cfg=cfg,
                b=bucket,
                l=lang,
                n=payload.get("n", 0),
                s=f"{payload.get('tool_call_success_rate', 0.0)*100:.1f}%",
                e=payload.get("raw_pii_transmissions", 0),
                m=f"{payload.get('median_latency_s', 0.0):.2f}",
                t=f"{payload.get('leak_guard_trigger_rate', 0.0)*100:.1f}%",
            )
        )
    return lines


def _md_table6_latency_contribution(aggregated: Dict[str, Any]) -> List[str]:
    latency_contribution = aggregated.get("latency_contribution", {})
    if not latency_contribution:
        return []
    lines: List[str] = [
        "\n## Table 6 — Per-detector latency contribution\n",
        "_Wall-clock seconds spent inside each detector across the run. "
        "``regex`` is the first-pass detector; ``second_pass_*`` rows are "
        "the optional NER / generative LM layers. Production hot paths are "
        "unaffected — the recorder is benchmark-only._\n",
        "| Detector | Calls | Total (s) | Mean (ms) | p95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(latency_contribution.keys()):
        row = latency_contribution[name]
        lines.append(
            f"| `{name}` | {int(row['calls'])} | {row['total_s']:.3f} | "
            f"{row['mean_ms']:.3f} | {row['p95_ms']:.3f} |"
        )
    return lines


def _md_table5_security_stress(aggregated: Dict[str, Any]) -> List[str]:
    security_report = aggregated.get("security_stress_report", {})
    if not security_report:
        return []
    lines: List[str] = [
        "\n## Table 5 — Security-stress report\n",
        "| Config | Workflow | Total | Passed | Failed | Pass rate | Raw PII |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(security_report.keys()):
        cfg, workflow = key.split("|", 1)
        row = security_report[key]
        lines.append(
            "| {cfg} | {workflow} | {total} | {passed} | {failed} | {rate} | {raw} |".format(
                cfg=cfg,
                workflow=workflow,
                total=row.get("total", 0),
                passed=row.get("passed", 0),
                failed=row.get("failed", 0),
                rate=f"{row.get('pass_rate', 0.0)*100:.1f}%",
                raw=row.get("raw_pii_transmissions", 0),
            )
        )
    return lines


def _write_markdown(path: Path, dataset: List[EvalQuery], aggregated: Dict[str, Any]) -> None:
    detector_config = aggregated.get("detector_configuration", {})
    lines: List[str] = []
    lines.extend(_md_header(dataset, detector_config))
    lines.extend(_md_table1_end_to_end(aggregated))
    lines.extend(_md_table2_detection_sanity(aggregated))
    lines.extend(_md_table2a_pr_per_language(aggregated))
    lines.extend(_md_table2b_pr_by_length(aggregated))
    lines.extend(_md_table3_arg_fidelity(aggregated))
    lines.extend(_md_table4_per_bucket_language(aggregated))
    lines.extend(_md_table6_latency_contribution(aggregated))
    lines.extend(_md_table5_security_stress(aggregated))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

_SPACY_MODEL_BY_LANG = {
    "en": "en_core_web_lg",
    "de": "de_core_news_lg",
    "ru": "ru_core_news_lg",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Skip LLM calls; only compute and print detection precision/recall.",
    )
    add_language_bucket_flags(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the run at N queries (after --language / --bucket filters). "
             "Useful for smoke tests against expensive LLMs.",
    )
    parser.add_argument(
        "--second-pass",
        choices=SECOND_PASS_MODES,
        default=None,
        help=(
            "Second-pass detector for the virtualization pipeline. "
            "Overrides detector.second_pass from traciora.yaml. "
            "'none' = regex only. 'opf' = OPFPIIDetector with the model "
            "selected by --opf-model (or detector.opf.default_model). "
            "'llm' = LLMPIIDetector. "
            "'presidio' = not yet wired."
        ),
    )
    parser.add_argument(
        "--opf-model",
        default=None,
        help=(
            "Name of an OPF model registered under detector.opf.models in "
            "traciora.yaml (e.g. base | en | de | ru). Ignored unless "
            "--second-pass=opf. Defaults to detector.opf.default_model."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to traciora.yaml. Default: $TRACIORA_PII_CONFIG or repo-root.",
    )
    parser.add_argument(
        "--include-presidio-pipeline",
        action="store_true",
        help=(
            "Run the Presidio-as-framework baseline alongside the three "
            "standard pipelines (baseline / destructive / virtualization). "
            "Requires the '[presidio]' extra plus a spaCy model for each "
            "evaluated language. The new config appears as "
            "'destructive_presidio' in result tables."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Override eval.llm.model at runtime. Required when "
            "PII_EVAL_LLM_ENABLED=1 because the YAML has no default. "
            "Use OpenRouter prefix-routed names: "
            "'anthropic/claude-sonnet-4-6', 'openai/gpt-4o', "
            "'google/gemini-pro-1.5', etc. See https://openrouter.ai/docs#models."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "Override eval.llm.base_url. Default (from traciora.yaml) is "
            "https://openrouter.ai/api/v1. Use this only for one-off "
            "experiments against a non-OpenRouter endpoint."
        ),
    )
    add_detector_llm_flags(parser)
    return parser


def _apply_cli_overrides(config: PIIConfig, args: argparse.Namespace) -> PIIConfig:
    """Layer CLI overrides onto the loaded YAML config.

    Three blocks of overrides: ``detector`` (second_pass + presidio language
    auto-mirror), ``eval.llm`` (model / base_url), and ``detector.llm``
    (model / base_url for the regex_llm second pass).
    """
    detector_overrides: Dict[str, Any] = {}
    if args.second_pass is not None:
        detector_overrides["second_pass"] = args.second_pass
    # Auto-mirror --language into detector.presidio.language so trilingual
    # ablations don't run the EN spaCy model over DE/RU text. The user can
    # still override via PII_PRESIDIO_LANGUAGE or YAML if they want a
    # mismatch on purpose.
    effective_second_pass = args.second_pass or config.detector.second_pass
    if effective_second_pass == "presidio" and args.language in {"en", "de", "ru"}:
        detector_overrides["presidio"] = {"language": args.language}
    if detector_overrides:
        config = config.with_overrides({"detector": detector_overrides})

    # Eval-LLM: CLI flags supersede YAML defaults so model selection is
    # explicit per-run and visible in shell history.
    llm_overrides: Dict[str, Any] = {}
    if args.llm_model:
        llm_overrides["model"] = args.llm_model
    if args.llm_base_url:
        llm_overrides["base_url"] = args.llm_base_url
    if llm_overrides:
        config = config.with_overrides({"eval": {"llm": llm_overrides}})

    # Detector-LLM: routes to detector.llm (used when --second-pass llm).
    det_llm = detector_llm_overrides(args)
    if det_llm:
        config = config.with_overrides({"detector": {"llm": det_llm}})
    return config


def _filter_dataset(dataset: List[EvalQuery], args: argparse.Namespace) -> List[EvalQuery]:
    if args.language != "all":
        dataset = [q for q in dataset if q.language == args.language]
    if args.bucket:
        dataset = [q for q in dataset if q.bucket == args.bucket]
    if args.limit is not None and args.limit > 0:
        dataset = dataset[: args.limit]
    return dataset


def _print_dataset_only_report(
    dataset: List[EvalQuery],
    pr: Dict[str, Any],
    second_pass_detector: Any,
    detector_label: str,
    requested_mode: str,
    latency_recorder: LatencyRecorder,
) -> None:
    """Print the --dataset-only precision/recall + latency contribution."""
    print(f"Dataset: {len(dataset)} queries")
    print(f"Detector: requested={requested_mode} effective={detector_label}")
    heading = (
        "Regex-only detection sanity check:"
        if second_pass_detector is None
        else f"Detection sanity check (regex + {detector_label}):"
    )
    print(f"\n{heading}")
    tag_keys = sorted(k for k in pr.keys() if not k.startswith("__"))
    for det in tag_keys:
        d = pr[det]
        print(
            f"  {det:20s}  P={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"TP={d['tp']:3d} FP={d['fp']:3d} FN={d['fn']:3d}  Sup={d['support']:3d}"
        )
    if "__macro__" in pr:
        d = pr["__macro__"]
        print(
            f"  {'macro avg':20s}  P={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"(unweighted mean over {len(tag_keys)} tags)"
        )
    if "__micro__" in pr:
        d = pr["__micro__"]
        print(
            f"  {'micro avg':20s}  P={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"TP={d['tp']:3d} FP={d['fp']:3d} FN={d['fn']:3d}  Sup={d['support']:3d}"
        )
    snap = latency_recorder.snapshot()
    if snap:
        print("\nPer-detector latency contribution:")
        print(
            f"  {'detector':36s}  {'calls':>6}  "
            f"{'total_s':>10}  {'mean_ms':>10}  {'p95_ms':>10}"
        )
        for name, row in snap.items():
            print(
                f"  {name:36s}  {int(row['calls']):>6}  "
                f"{row['total_s']:>10.3f}  {row['mean_ms']:>10.3f}  {row['p95_ms']:>10.3f}"
            )


def _compute_output_slug(config: PIIConfig, opf_model: Optional[str]) -> str:
    """Build the slug used for results_<slug>.{json,md} and vault_<slug>.db.

    Suffixes the model name with the active second-pass mode so concurrent
    ablation runs in the same RESULTS_DIR don't overwrite each other.
    """
    base = model_slug()
    second_pass = config.detector.second_pass
    if second_pass == "opf":
        ablation_tag = f"opf-{opf_model}"
    else:
        ablation_tag = second_pass
    return f"{base}__{ablation_tag}" if ablation_tag != "none" else base


def _build_presidio_pipeline(dataset: List[EvalQuery]) -> Tuple[Dict[str, Any], Any]:
    """Build per-language Presidio AnalyzerEngines + a shared AnonymizerEngine.

    One AnalyzerEngine per language seen in the dataset, each backed by its
    own NlpEngineProvider with the matching spaCy model. A default
    AnalyzerEngine() loads only the English backbone and would silently
    degrade to regex-only matching on DE/RU rows — the source of unfair
    "Presidio looks weak" numbers. Mirrors
    ``src/traciora_pii/presidio_pii_detector.py:218-245``.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as exc:
        raise SystemExit(
            "--include-presidio-pipeline requires the [presidio] extra: "
            "pip install -e '.[presidio]' "
            f"(import error: {exc})"
        )
    languages_in_dataset = sorted({(q.language or "en") for q in dataset})
    print(f"Building Presidio analyzers for: {', '.join(languages_in_dataset)}")
    analyzers: Dict[str, Any] = {}
    for lang in languages_in_dataset:
        model_name = _SPACY_MODEL_BY_LANG.get(lang)
        if model_name is None:
            print(
                f"  ! no spaCy model registered for language={lang!r}; "
                f"skipping Presidio for this language",
                file=sys.stderr,
            )
            continue
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": model_name}],
        })
        try:
            nlp_engine = provider.create_engine()
        except Exception as exc:  # noqa: BLE001 — surfaces a SystemExit with install hint
            raise SystemExit(
                f"Failed to load spaCy model {model_name!r} for "
                f"language={lang!r}. Install with: python -m spacy download {model_name}\n"
                f"  (underlying error: {type(exc).__name__}: {exc})"
            )
        analyzers[lang] = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[lang],
        )
    if not analyzers:
        raise SystemExit(
            "--include-presidio-pipeline produced an empty analyzer registry "
            "(no languages mapped). Aborting."
        )
    return analyzers, AnonymizerEngine()


def _run_pipelines(
    dataset: List[EvalQuery],
    *,
    llm: Any,
    redactor: PIIRedactor,
    vault: PIIVaultStore,
    virtualizer: PIIVirtualizer,
    presidio_analyzers: Optional[Dict[str, Any]],
    presidio_anonymizer: Any,
) -> List[RunResult]:
    """Run each query through baseline / destructive / virtualization (and presidio when requested)."""
    results: List[RunResult] = []
    include_presidio = presidio_analyzers is not None
    n_configs = 4 if include_presidio else 3
    print(f"Running {len(dataset)} queries × {n_configs} configurations …")
    for idx, query in enumerate(dataset, start=1):
        print(f"  [{idx:>3}/{len(dataset)}] {query.id} ({query.bucket}) ", end="", flush=True)
        try:
            r_base = run_baseline(query=query, llm=llm)
            r_dest = run_destructive(query=query, llm=llm, redactor=redactor)
            r_virt = run_virtualization(query=query, llm=llm, vault=vault, virtualizer=virtualizer)
            r_presidio = None
            if include_presidio:
                r_presidio = run_destructive_presidio(
                    query=query,
                    llm=llm,
                    analyzers=presidio_analyzers,
                    anonymizer=presidio_anonymizer,
                    language=query.language or "en",
                )
        except Exception as exc:  # noqa: BLE001 — capture any per-query failure into the result record so the eval batch continues
            print(f"FAILED: {exc}")
            continue
        results.extend([r_base, r_dest, r_virt])
        if r_presidio is not None:
            results.append(r_presidio)
        msg = (
            f"baseline={'ok' if not r_base.error else 'ERR'} "
            f"destructive={'ok' if not r_dest.error else 'ERR'} "
            f"virtualization={'ok' if not r_virt.error else 'ERR'} "
            f"(triggers={r_virt.leak_guard_triggers})"
        )
        if r_presidio is not None:
            msg += f" presidio={'ok' if not r_presidio.error else 'ERR'}"
        print(msg)
    return results


def _print_summary(aggregated: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    print("\nMetrics summary:")
    success = aggregated["tool_call_success_rate"]
    exposure = aggregated["model_pii_exposure_count"]
    triggers = aggregated["leak_guard_trigger_rate"]
    summary_configs = [
        c for c in _TABLE_CONFIG_ORDER
        if c in success or c in exposure or c in triggers
    ]
    for config in summary_configs:
        s = success.get(config, 0.0)
        e = exposure.get(config, 0)
        t = triggers.get(config, 0.0)
        print(f"  {config:15s}  success={s*100:5.1f}%  pii_transmissions_to_model={e:3d}  leak_triggers={t*100:5.1f}%")
    print(f"\nResults written to {json_path} and {md_path}")


def main() -> int:
    args = _build_arg_parser().parse_args()
    load_dotenv()

    config = load_config(Path(args.config) if args.config else None)
    config = _apply_cli_overrides(config, args)
    opf_model = args.opf_model or config.detector.opf.default_model

    dataset = _filter_dataset(list(load_dataset()), args)

    # Build the second-pass detector before the dry-run branch so the
    # precision/recall sanity check actually reflects the requested
    # ablation (regex+presidio / regex+opf / regex+llm). Failures log a
    # warning and degrade to regex-only.
    second_pass_detector = build_second_pass_detector(config, opf_model_name=opf_model)
    detector_label = (
        type(second_pass_detector).__name__ if second_pass_detector is not None else "regex-only"
    )
    latency_recorder = LatencyRecorder()
    latency_token = install_recorder(latency_recorder)
    pr = metrics.detection_precision_recall(
        dataset, second_pass_detector=second_pass_detector,
    )

    if args.dataset_only:
        _print_dataset_only_report(
            dataset, pr, second_pass_detector, detector_label,
            requested_mode=config.detector.second_pass,
            latency_recorder=latency_recorder,
        )
        reset_recorder(latency_token)
        close_detector = getattr(second_pass_detector, "close", None)
        if callable(close_detector):
            close_detector()
        return 0

    if not config.eval.llm.enabled:
        print("eval.llm.enabled is false; skipping LLM run.")
        print(
            "Set eval.llm.enabled=true in traciora.yaml or PII_EVAL_LLM_ENABLED=1 "
            "to execute the full experiment, or pass --dataset-only."
        )
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    llm = _build_eval_llm(config)
    slug = _compute_output_slug(config, opf_model)
    print(
        f"Output slug: {slug} (second_pass={config.detector.second_pass}"
        + (f", opf_model={opf_model}" if config.detector.second_pass == "opf" else "")
        + f"; results -> results_{slug}.{{json,md}})"
    )

    # One ephemeral vault for the entire experiment. Each query uses its
    # own session inside the vault so cross-query state cannot leak.
    # Vault path is slugged so concurrent runs against different models
    # do not collide on the SQLite file.
    fernet_key = config.secret(config.vault.encryption_key_env) or Fernet.generate_key().decode()
    vault_path = RESULTS_DIR / f"vault_{slug}.db"
    if vault_path.exists():
        vault_path.unlink()
    vault = PIIVaultStore(db_path=vault_path, encryption_key=fernet_key)
    detector_configuration = _detector_configuration(
        second_pass_detector, config=config, opf_model=opf_model,
    )
    print(
        f"Detector wiring: requested={detector_configuration['requested_mode']} "
        f"effective={detector_configuration['effective_mode']} "
        f"class={detector_configuration['second_pass_class'] or '(regex-only)'}"
    )
    virtualizer = PIIVirtualizer(vault, second_pass_detector=second_pass_detector)
    redactor = PIIRedactor.from_config(config)

    presidio_analyzers: Optional[Dict[str, Any]] = None
    presidio_anonymizer = None
    if args.include_presidio_pipeline:
        presidio_analyzers, presidio_anonymizer = _build_presidio_pipeline(dataset)

    results = _run_pipelines(
        dataset,
        llm=llm,
        redactor=redactor,
        vault=vault,
        virtualizer=virtualizer,
        presidio_analyzers=presidio_analyzers,
        presidio_anonymizer=presidio_anonymizer,
    )

    aggregated = _aggregate(
        results,
        dataset,
        detector_configuration=detector_configuration,
        latency_contribution=latency_recorder.snapshot(),
    )
    reset_recorder(latency_token)

    json_path = RESULTS_DIR / f"results_{slug}.json"
    md_path = RESULTS_DIR / f"results_{slug}.md"
    _write_json(json_path, results, dataset, aggregated)
    _write_markdown(md_path, dataset, aggregated)

    _print_summary(aggregated, json_path, md_path)
    vault.close()
    close_detector = getattr(second_pass_detector, "close", None)
    if callable(close_detector):
        close_detector()
    return 0


if __name__ == "__main__":
    sys.exit(main())
