"""Metric computation for the PII evaluation experiment.

The five metrics map directly to Section 6 of the paper:

  M1  tool_call_success_rate          → §6.1
  M2  model_pii_exposure_count        → §6.2
  M3  detection_precision_recall      → §6.3
  M4  latency_stats                   → §6.4
  M5  leak_guard_trigger_rate         → discussion within §6.2

The detection P/R metric is computed independently of any pipeline run; it
only requires the dataset, so it can be reported even in dry-run mode.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from traciora_pii._latency import LatencyRecorder, time_detector
from traciora_pii.pii import detect_pii, redact_pii_text

logger = logging.getLogger(__name__)

from benchmarks.pii_evaluation.dataset import EvalQuery
from benchmarks.pii_evaluation.pipelines import RunResult


# ----------------------------------------------------------------------
# M1: tool-call success
# ----------------------------------------------------------------------

def tool_call_success_rate(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[str, float]:
    """Per-config tool-call success rate.

    A query is "successful" iff *every* expected tool call (as listed in
    ``query.effective_tool_calls``) has at least one matching captured
    invocation, where matching means same tool name AND same expected
    record id (or any record id when the expected one is empty, e.g. for
    cross-turn-taint update calls whose record id is dynamic).

    Queries with no expected tool calls (hard negatives, security/stress
    cases that exist only to score detection or leak-guard) are excluded
    from the denominator. For legacy single-call queries this reduces to
    the previous semantics.
    """
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        grouped[r.config].append(r)

    rates: Dict[str, float] = {}
    for config, runs in grouped.items():
        successes = 0
        considered = 0
        for r in runs:
            query = by_id.get(r.query_id)
            if query is None:
                continue
            expected_calls = query.effective_tool_calls
            if not expected_calls:
                continue
            considered += 1
            all_matched = True
            for expected in expected_calls:
                hit = False
                for inv in r.tool_invocations:
                    if inv.tool_name != expected.tool_name:
                        continue
                    if expected.expected_record_id and inv.record_id != expected.expected_record_id:
                        continue
                    hit = True
                    break
                if not hit:
                    all_matched = False
                    break
            if all_matched:
                successes += 1
        rates[config] = successes / considered if considered else 0.0
    return rates


# ----------------------------------------------------------------------
# M2: raw PII exposure to the model
# ----------------------------------------------------------------------

def _boundary_values_for_query(query: EvalQuery) -> Dict[str, Any]:
    """Raw PII literals that must not cross the model boundary for a query.

    Prefer the rendered ``boundary_pii_values`` surface because it includes
    tool-result fixture PII and injected cross-turn values. Fall back to
    user-turn spans for legacy datasets.
    """
    values: Dict[str, Any] = {}
    boundary = getattr(query, "boundary_pii_values", ()) or ()
    if boundary:
        for item in boundary:
            if item.value and item.value not in values:
                values[item.value] = item
        return values
    for span in query.pii_spans:
        if span.value and span.value not in values:
            values[span.value] = span
    return values


def _literal_exposure_count(result: RunResult, query: EvalQuery) -> int:
    count = 0
    boundary = _boundary_values_for_query(query)
    if not boundary:
        return 0
    for entry in result.sent_to_llm:
        content = entry.get("content")
        if not isinstance(content, str) or not content:
            continue
        for value in boundary:
            count += content.count(value)
    return count


def model_pii_exposure_count(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[str, int]:
    """Per-config cumulative count of raw PII transmissions to the LLM.

    For every message captured in ``r.sent_to_llm`` (the post-rewrite
    messages that actually crossed the trust boundary), count literal
    appearances of the query's ``boundary_pii_values``. This is deliberately
    detector-independent: a detector miss must still count as exposure if
    the raw value reached the model.

    The intended interpretation is "how often did raw PII cross the trust
    boundary?", not "how many distinct values leaked?". Zero is the only
    acceptable number under destructive or virtualization configs; any
    non-zero result is a wiring bug worth investigating.

    Critical: this metric scans `r.sent_to_llm` — the post-hook messages
    that actually crossed the trust boundary — *not* `r.transcript`. The
    transcript is the raw history retained for human inspection and would
    falsely report raw PII even under destructive and virtualization
    configs, since the pre-send hook returns rewritten copies without
    mutating the history.

    Values outside the query boundary set are intentionally excluded — the
    metric answers a single question: did values that were supposed to be
    protected actually reach the LLM?
    """
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[str, int] = defaultdict(int)

    for r in results:
        query = by_id.get(r.query_id)
        if query is None:
            continue
        grouped[r.config] += _literal_exposure_count(r, query)
    return dict(grouped)


def model_pii_exposure_examples(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
    *,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """First literal exposure examples for debugging benchmark failures."""
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    examples: List[Dict[str, Any]] = []
    for r in results:
        query = by_id.get(r.query_id)
        if query is None:
            continue
        boundary = _boundary_values_for_query(query)
        for entry in r.sent_to_llm:
            content = entry.get("content")
            if not isinstance(content, str) or not content:
                continue
            for value, item in boundary.items():
                if value not in content:
                    continue
                examples.append(
                    {
                        "query_id": r.query_id,
                        "config": r.config,
                        "value": value,
                        "placeholder": getattr(item, "placeholder", ""),
                        "source": getattr(item, "source", "user_turn"),
                        "record_id": getattr(item, "record_id", None),
                        "message_role": entry.get("role"),
                    }
                )
                if len(examples) >= limit:
                    return examples
    return examples


# ----------------------------------------------------------------------
# Security-stress report
# ----------------------------------------------------------------------

def security_stress_report(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[str, Dict[str, Any]]:
    """Pass/fail metrics for the security_stress bucket.

    The report is keyed by ``"<config>|<workflow>"`` and uses workflow-
    specific pass conditions from the benchmark plan.
    """
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "raw_pii_transmissions": 0}
    )

    for r in results:
        query = by_id.get(r.query_id)
        if query is None or query.bucket != "security_stress":
            continue
        key = (r.config, query.workflow)
        exposure = _literal_exposure_count(r, query)
        ok = _security_case_passed(r, query, exposure)
        grouped[key]["total"] += 1
        grouped[key]["raw_pii_transmissions"] += exposure
        if ok:
            grouped[key]["passed"] += 1
        else:
            grouped[key]["failed"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for (config, workflow), payload in sorted(grouped.items()):
        total = payload["total"]
        out[f"{config}|{workflow}"] = {
            **payload,
            "pass_rate": payload["passed"] / total if total else 0.0,
        }
    return out


def _security_case_passed(result: RunResult, query: EvalQuery, exposure: int) -> bool:
    if query.workflow.startswith("prompt_injection"):
        return exposure == 0
    if query.workflow.startswith("zero_width_split"):
        return exposure == 0
    if query.workflow.startswith("code_switched"):
        return exposure == 0
    if query.workflow.startswith("forged_ref_token"):
        return exposure == 0 and not any(inv.record_id for inv in result.tool_invocations)
    if query.workflow.startswith("hard_non_pii_mimic"):
        return exposure == 0 and not any(inv.record_id for inv in result.tool_invocations)
    if query.workflow.startswith("tool_exception_leakage"):
        # Inverted rule: baseline + destructive *should* expose the raw card
        # because the agent loop stringifies the upstream exception verbatim
        # into the next ToolMessage. A zero there would mean the harness
        # silently scrubbed the body — a wiring regression. Virtualization
        # must drive exposure back to zero via the leak guard.
        if result.config == "virtualization":
            return exposure == 0
        return exposure > 0
    return exposure == 0


# ----------------------------------------------------------------------
# M3: detection precision / recall (per detector + macro avg)
# ----------------------------------------------------------------------

def _gt_spans_for_query(query: EvalQuery) -> Dict[int, List[Tuple[str, int, int]]]:
    """Locate every ground-truth span's character range in its turn.

    Raises ``ValueError`` if the dataset claims a span value that does not
    appear in its turn — surfaces dataset corruption immediately.
    """
    gt_by_turn: Dict[int, List[Tuple[str, int, int]]] = defaultdict(list)
    for span in query.pii_spans:
        turn_text = query.turns[span.turn]
        start = turn_text.find(span.value)
        if start < 0:
            raise ValueError(
                f"Ground-truth value {span.value!r} not found in turn "
                f"{span.turn} of query {query.id!r}"
            )
        gt_by_turn[span.turn].append((span.placeholder, start, start + len(span.value)))
    return gt_by_turn


def _score_query_turn(
    turn_text: str,
    gt_spans: Sequence[Tuple[str, int, int]],
    second_pass_detector: Optional[Any],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Compute (tp, fp, fn) per placeholder for a single turn.

    Mirrors the production precedence: regex pass first, then optional
    second-pass detector with overlap-drop against any regex finding —
    matches ``PIIVirtualizer._apply_second_pass``.
    """
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    _redacted, regex_findings = redact_pii_text(turn_text)
    findings = list(regex_findings)
    if second_pass_detector is not None:
        try:
            with time_detector(f"second_pass_{type(second_pass_detector).__name__}"):
                sp_findings = second_pass_detector.detect(turn_text) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "second-pass detector failed on turn; skipping. error_type=%s",
                type(exc).__name__,
            )
            sp_findings = []
        for sf in sp_findings:
            if any(sf.start < rf.end and rf.start < sf.end for rf in regex_findings):
                continue
            findings.append(sf)

    matched_gt: set = set()
    for finding in findings:
        hit_idx = None
        for i, (gt_ph, gt_start, gt_end) in enumerate(gt_spans):
            if i in matched_gt:
                continue
            if finding.placeholder != gt_ph:
                continue
            if finding.start < gt_end and gt_start < finding.end:
                hit_idx = i
                break
        if hit_idx is not None:
            matched_gt.add(hit_idx)
            tp[finding.placeholder] += 1
        else:
            fp[finding.placeholder] += 1

    for i, (gt_ph, _s, _e) in enumerate(gt_spans):
        if i not in matched_gt:
            fn[gt_ph] += 1

    return tp, fp, fn


def _summarize_pr(
    tp: Dict[str, int],
    fp: Dict[str, int],
    fn: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """Build the canonical {placeholder: {p,r,tp,fp,fn,support}} + macro/micro shape."""
    detectors = set(tp) | set(fp) | set(fn)
    out: Dict[str, Dict[str, Any]] = {}
    for det in sorted(detectors):
        precision = tp[det] / (tp[det] + fp[det]) if (tp[det] + fp[det]) else 0.0
        recall = tp[det] / (tp[det] + fn[det]) if (tp[det] + fn[det]) else 0.0
        out[det] = {
            "precision": precision,
            "recall": recall,
            "tp": tp[det],
            "fp": fp[det],
            "fn": fn[det],
            "support": tp[det] + fn[det],
        }
    if out:
        out["__macro__"] = {
            "precision": sum(d["precision"] for d in out.values()) / len(out),
            "recall": sum(d["recall"] for d in out.values()) / len(out),
            "tp": None,
            "fp": None,
            "fn": None,
            "support": None,
        }
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        out["__micro__"] = {
            "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0,
            "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "support": total_tp + total_fn,
        }
    return out


def detection_precision_recall_with_records(
    dataset: Sequence[EvalQuery],
    *,
    second_pass_detector: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Tuple[int, int, int]]]]:
    """Same scoring as :func:`detection_precision_recall`, but also emits
    per-query ``{placeholder: (tp, fp, fn)}`` records for bootstrap CIs.

    Each query (not each turn) is one record; turn-level counts are
    summed inside the query so resampling the curated dataset preserves
    multi-turn coherence. Returns ``(summary_dict, per_query_records)``.
    """
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    per_query_records: List[Dict[str, Tuple[int, int, int]]] = []

    for query in dataset:
        gt_by_turn = _gt_spans_for_query(query)
        q_tp: Dict[str, int] = defaultdict(int)
        q_fp: Dict[str, int] = defaultdict(int)
        q_fn: Dict[str, int] = defaultdict(int)
        for turn_idx, turn_text in enumerate(query.turns):
            tp_t, fp_t, fn_t = _score_query_turn(
                turn_text,
                gt_by_turn.get(turn_idx, []),
                second_pass_detector,
            )
            for ph, n in tp_t.items():
                tp[ph] += n
                q_tp[ph] += n
            for ph, n in fp_t.items():
                fp[ph] += n
                q_fp[ph] += n
            for ph, n in fn_t.items():
                fn[ph] += n
                q_fn[ph] += n
        record: Dict[str, Tuple[int, int, int]] = {}
        for ph in set(q_tp) | set(q_fp) | set(q_fn):
            record[ph] = (q_tp.get(ph, 0), q_fp.get(ph, 0), q_fn.get(ph, 0))
        per_query_records.append(record)

    return _summarize_pr(tp, fp, fn), per_query_records


def detection_precision_recall(
    dataset: Sequence[EvalQuery],
    *,
    second_pass_detector: Optional[Any] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-placeholder precision and recall against ground-truth spans.

    Thin wrapper around :func:`detection_precision_recall_with_records`
    that drops the per-query records. Use the records variant when you
    want bootstrap CIs.
    """
    summary, _ = detection_precision_recall_with_records(
        dataset, second_pass_detector=second_pass_detector,
    )
    return summary


def detection_precision_recall_per_language(
    dataset: Sequence[EvalQuery],
    *,
    second_pass_detector: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per-(language × placeholder) precision and recall.

    Returns ``{language: {placeholder: {...}, "__macro__": {...},
    "__micro__": {...}}}``. Uses the same overlap-aware scorer as
    ``detection_precision_recall`` so per-language sums fold into the
    global counts: ``sum_lang(tp[ph]) == global_tp[ph]``.
    """
    by_lang_tp: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_lang_fp: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_lang_fn: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    languages: set = set()

    for query in dataset:
        lang = query.language or "unknown"
        languages.add(lang)
        gt_by_turn = _gt_spans_for_query(query)
        for turn_idx, turn_text in enumerate(query.turns):
            tp_t, fp_t, fn_t = _score_query_turn(
                turn_text,
                gt_by_turn.get(turn_idx, []),
                second_pass_detector,
            )
            for ph, n in tp_t.items():
                by_lang_tp[lang][ph] += n
            for ph, n in fp_t.items():
                by_lang_fp[lang][ph] += n
            for ph, n in fn_t.items():
                by_lang_fn[lang][ph] += n

    return {
        lang: _summarize_pr(by_lang_tp[lang], by_lang_fp[lang], by_lang_fn[lang])
        for lang in sorted(languages)
    }


# ----------------------------------------------------------------------
# M3b: detection P/R by length bucket × language × placeholder
# ----------------------------------------------------------------------

LENGTH_BUCKETS = ("sentence", "paragraph", "multi_paragraph", "structured")


def length_bucket_for_text(text: str) -> str:
    """Classify a raw text string into one of ``LENGTH_BUCKETS``.

    Priority order (first match wins):
      1. ``structured`` — markdown table pipes, ``≥3`` colon-and-space
         lines, or repeated tabs (key/value or log-like text).
      2. ``multi_paragraph`` — contains a blank-line separator.
      3. ``paragraph`` — multi-line OR longer than 1000 chars.
      4. ``sentence`` — short single line.
    """
    if not text:
        return "sentence"

    pipe_count = text.count("|")
    if pipe_count >= 6 and pipe_count >= text.count("\n"):
        return "structured"
    colon_lines = sum(1 for line in text.splitlines() if re.search(r":\s", line))
    if colon_lines >= 3:
        return "structured"
    if text.count("\t") >= 3:
        return "structured"

    if "\n\n" in text:
        return "multi_paragraph"
    if "\n" in text or len(text) > 1000:
        return "paragraph"
    return "sentence"


def length_bucket_for_query(query: EvalQuery) -> str:
    """Classify a query into one of ``LENGTH_BUCKETS``.

    Thin wrapper around :func:`length_bucket_for_text` that joins the
    query's turns with newlines before classifying. Imported-dataset
    evals call :func:`length_bucket_for_text` directly against the row
    text.
    """
    return length_bucket_for_text("\n".join(query.turns))


def detection_precision_recall_by_length_bucket(
    dataset: Sequence[EvalQuery],
    *,
    second_pass_detector: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """{length_bucket: {language: {placeholder: {...}, '__macro__', '__micro__'}}}.

    Sums across length buckets and languages reproduce the global
    detection P/R counts — useful as a self-check.
    """
    counts: Dict[Tuple[str, str], Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]] = {}
    keys: set = set()
    for query in dataset:
        bucket = length_bucket_for_query(query)
        lang = query.language or "unknown"
        key = (bucket, lang)
        keys.add(key)
        slot = counts.setdefault(
            key, (defaultdict(int), defaultdict(int), defaultdict(int)),
        )
        tp, fp, fn = slot
        gt_by_turn = _gt_spans_for_query(query)
        for turn_idx, turn_text in enumerate(query.turns):
            tp_t, fp_t, fn_t = _score_query_turn(
                turn_text,
                gt_by_turn.get(turn_idx, []),
                second_pass_detector,
            )
            for ph, n in tp_t.items():
                tp[ph] += n
            for ph, n in fp_t.items():
                fp[ph] += n
            for ph, n in fn_t.items():
                fn[ph] += n

    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for bucket, lang in sorted(keys):
        tp, fp, fn = counts[(bucket, lang)]
        out.setdefault(bucket, {})[lang] = _summarize_pr(tp, fp, fn)
    return out


# ----------------------------------------------------------------------
# M4: latency
# ----------------------------------------------------------------------

def latency_stats(results: Sequence[RunResult]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        grouped[r.config].append(r.elapsed_seconds)
    out: Dict[str, Dict[str, float]] = {}
    for config, samples in grouped.items():
        if not samples:
            continue
        out[config] = {
            "median": statistics.median(samples),
            "mean": statistics.fmean(samples),
            "p95": _p95(samples),
            "n": len(samples),
        }
    return out


def _p95(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[idx]


# ----------------------------------------------------------------------
# M5: leak guard trigger rate
# ----------------------------------------------------------------------

def leak_guard_trigger_rate(results: Sequence[RunResult]) -> Dict[str, float]:
    """Fraction of virtualization-config queries that triggered the leak guard at least once."""
    grouped: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        grouped[r.config].append(r)
    out: Dict[str, float] = {}
    for config, runs in grouped.items():
        triggered = sum(1 for r in runs if r.leak_guard_triggers > 0)
        out[config] = triggered / len(runs) if runs else 0.0
    return out


# ----------------------------------------------------------------------
# Shared CI'd P/R table renderer
# ----------------------------------------------------------------------

def render_detection_pr_table(
    summary: Dict[str, Dict[str, Any]],
    *,
    per_query_records: Optional[Sequence[Dict[str, Tuple[int, int, int]]]] = None,
    n_resamples: int = 1000,
) -> List[str]:
    """Render a per-placeholder P/R/F1 markdown table from a ``_summarize_pr``
    output. When ``per_query_records`` is supplied, P/R/F1 cells carry 95%
    bootstrap CIs (deterministic at seed=42). Without it the cells fall
    back to point estimates.

    ``summary`` is the canonical ``_summarize_pr`` shape::

        {detector: {"precision", "recall", "tp", "fp", "fn", "support"},
         "__macro__": …, "__micro__": …}

    Used by ``run_imported_dataset_eval`` and ``run_experiment._write_markdown``
    Table 2 — both end up rendering the same surface, so the renderer
    lives here.
    """
    # Local import to avoid a benchmarks→benchmarks import cycle at
    # module load (bootstrap imports nothing from metrics).
    from benchmarks.pii_evaluation.bootstrap import (
        bootstrap_ci,
        bootstrap_pr_f1,
    )

    placeholders = sorted(k for k in summary if not k.startswith("__"))
    with_ci = per_query_records is not None and len(per_query_records) > 0

    if with_ci:
        lines = [
            "| Placeholder | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | TP | FP | FN | Support |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| Placeholder | Precision | Recall | F1 | TP | FP | FN | Support |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

    for ph in placeholders:
        row = summary[ph]
        tp = int(row.get("tp", 0) or 0)
        fp = int(row.get("fp", 0) or 0)
        fn = int(row.get("fn", 0) or 0)
        support = tp + fn
        if with_ci:
            ph_records = [r.get(ph, (0, 0, 0)) for r in per_query_records]
            ci = bootstrap_pr_f1(ph_records, n_resamples=n_resamples)
            lines.append(
                f"| `{ph}` | {ci.precision.format_md()} | {ci.recall.format_md()} | "
                f"{ci.f1.format_md()} | {tp} | {fp} | {fn} | {support} |"
            )
        else:
            prec = float(row.get("precision", 0.0) or 0.0)
            rec = float(row.get("recall", 0.0) or 0.0)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            lines.append(
                f"| `{ph}` | {prec:.3f} | {rec:.3f} | {f1:.3f} | {tp} | {fp} | {fn} | {support} |"
            )

    macro = summary.get("__macro__")
    micro = summary.get("__micro__")
    if macro is not None or micro is not None:
        total_tp = int((micro or {}).get("tp") or sum(int(summary[ph].get("tp", 0) or 0) for ph in placeholders))
        total_fp = int((micro or {}).get("fp") or sum(int(summary[ph].get("fp", 0) or 0) for ph in placeholders))
        total_fn = int((micro or {}).get("fn") or sum(int(summary[ph].get("fn", 0) or 0) for ph in placeholders))
        total_support = total_tp + total_fn

        if with_ci:
            # Macro: bootstrap row resampling, recompute mean of per-placeholder rates.
            def _macro_p(rows: Sequence[Dict[str, Tuple[int, int, int]]]) -> float:
                if not placeholders:
                    return 0.0
                acc = 0.0
                for ph in placeholders:
                    t = sum(r.get(ph, (0, 0, 0))[0] for r in rows)
                    f = sum(r.get(ph, (0, 0, 0))[1] for r in rows)
                    acc += t / (t + f) if (t + f) else 0.0
                return acc / len(placeholders)

            def _macro_r(rows: Sequence[Dict[str, Tuple[int, int, int]]]) -> float:
                if not placeholders:
                    return 0.0
                acc = 0.0
                for ph in placeholders:
                    t = sum(r.get(ph, (0, 0, 0))[0] for r in rows)
                    fn_ = sum(r.get(ph, (0, 0, 0))[2] for r in rows)
                    acc += t / (t + fn_) if (t + fn_) else 0.0
                return acc / len(placeholders)

            def _macro_f1(rows: Sequence[Dict[str, Tuple[int, int, int]]]) -> float:
                p = _macro_p(rows)
                r = _macro_r(rows)
                return (2 * p * r / (p + r)) if (p + r) else 0.0

            macro_p_ci = bootstrap_ci(per_query_records, _macro_p, n_resamples=n_resamples)
            macro_r_ci = bootstrap_ci(per_query_records, _macro_r, n_resamples=n_resamples)
            macro_f1_ci = bootstrap_ci(per_query_records, _macro_f1, n_resamples=n_resamples)

            micro_records = [
                (
                    sum(v[0] for v in r.values()),
                    sum(v[1] for v in r.values()),
                    sum(v[2] for v in r.values()),
                )
                for r in per_query_records
            ]
            micro_ci = bootstrap_pr_f1(micro_records, n_resamples=n_resamples)

            lines.append(
                f"| **macro** | {macro_p_ci.format_md()} | {macro_r_ci.format_md()} | "
                f"{macro_f1_ci.format_md()} | {total_tp} | {total_fp} | {total_fn} | "
                f"{total_support} |"
            )
            lines.append(
                f"| **micro** | {micro_ci.precision.format_md()} | "
                f"{micro_ci.recall.format_md()} | {micro_ci.f1.format_md()} | "
                f"{total_tp} | {total_fp} | {total_fn} | {total_support} |"
            )
        else:
            if macro is not None:
                m_p = float(macro.get("precision", 0.0) or 0.0)
                m_r = float(macro.get("recall", 0.0) or 0.0)
                m_f1 = (2 * m_p * m_r / (m_p + m_r)) if (m_p + m_r) else 0.0
                lines.append(
                    f"| **macro** | {m_p:.3f} | {m_r:.3f} | {m_f1:.3f} | "
                    f"{total_tp} | {total_fp} | {total_fn} | {total_support} |"
                )
            if micro is not None:
                mi_p = float(micro.get("precision", 0.0) or 0.0)
                mi_r = float(micro.get("recall", 0.0) or 0.0)
                mi_f1 = (2 * mi_p * mi_r / (mi_p + mi_r)) if (mi_p + mi_r) else 0.0
                lines.append(
                    f"| **micro** | {mi_p:.3f} | {mi_r:.3f} | {mi_f1:.3f} | "
                    f"{total_tp} | {total_fp} | {total_fn} | {total_support} |"
                )
    return lines


# ----------------------------------------------------------------------
# Bootstrap CI bundle for headline M1/M2/M4/M5 metrics
# ----------------------------------------------------------------------

@dataclass
class PerConfigAtoms:
    """Per-query atoms for the four headline §3 metrics, per config.

    Each list has one entry per result that was actually scored (e.g. M1
    excludes hard negatives with no expected tool calls). Lists are
    independent — bootstrap each separately.
    """
    tool_success: List[bool]       # M1
    exposure_count: List[int]      # M2
    elapsed_seconds: List[float]   # M4
    leak_guard_triggered: List[bool]  # M5


def per_config_atoms(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[str, PerConfigAtoms]:
    """Extract per-query atoms for every config. Mirrors the inclusion
    rules of the aggregate metric functions exactly so bootstrap means
    match the point estimates."""
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        grouped[r.config].append(r)

    out: Dict[str, PerConfigAtoms] = {}
    for config, runs in grouped.items():
        successes: List[bool] = []
        exposures: List[int] = []
        elapseds: List[float] = []
        triggers: List[bool] = []
        for r in runs:
            query = by_id.get(r.query_id)
            if query is None:
                continue
            # M1 — inclusion rule: skip queries with no expected tool calls.
            if query.effective_tool_calls:
                all_matched = True
                for expected in query.effective_tool_calls:
                    hit = False
                    for inv in r.tool_invocations:
                        if inv.tool_name != expected.tool_name:
                            continue
                        if expected.expected_record_id and inv.record_id != expected.expected_record_id:
                            continue
                        hit = True
                        break
                    if not hit:
                        all_matched = False
                        break
                successes.append(all_matched)
            # M2, M4, M5 — every run counts.
            exposures.append(_literal_exposure_count(r, query))
            elapseds.append(r.elapsed_seconds)
            triggers.append(r.leak_guard_triggers > 0)
        out[config] = PerConfigAtoms(
            tool_success=successes,
            exposure_count=exposures,
            elapsed_seconds=elapseds,
            leak_guard_triggered=triggers,
        )
    return out


# ----------------------------------------------------------------------
# Per-detector latency contribution
# ----------------------------------------------------------------------

def latency_contribution_report(recorder: LatencyRecorder) -> Dict[str, Dict[str, float]]:
    """Pass-through to ``recorder.snapshot()`` with a stable shape."""
    return recorder.snapshot()
