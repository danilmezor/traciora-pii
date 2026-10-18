"""Argument-fidelity and cross-turn-stability metrics.

These supplement the five existing metrics in ``metrics.py`` by scoring:

  - whether each expected tool-call's *arguments* matched what the model
    actually sent (post-normalization), beyond just record-id match;
  - whether vault reference tokens for the same PII value remained
    stable across turns of a multi-turn query (so the framework
    rehydrates consistently and the leak guard sees the same token in
    follow-up turns);
  - per-(bucket × language) aggregated reporting of all metrics, which
    is the slice the results table is keyed on.

The metrics here only consume ``EvalQuery`` and ``RunResult`` objects;
no LLM is required.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from benchmarks.pii_evaluation.dataset import EvalQuery, ExpectedToolCall
from benchmarks.pii_evaluation.pipelines import RunResult
from benchmarks.pii_evaluation.tools import (
    _normalize_digits,
    _normalize_email,
    _normalize_order_id,
    _normalize_phone,
)


# ----------------------------------------------------------------------
# Argument exact match
# ----------------------------------------------------------------------

# Per-argument-name normalizer. Conservative: any argument we don't know
# about is compared as-is after string coercion.
_ARG_NORMALIZERS: Mapping[str, Callable[[str], str]] = {
    "phone": _normalize_phone,
    "email": _normalize_email,
    "name": lambda v: (v or "").strip().lower(),
    "ssn": _normalize_digits,
    "snils": _normalize_digits,
    "tax_id": _normalize_digits,
    "card_number": _normalize_digits,
    "value": _normalize_digits,
    "series_number": _normalize_digits,
    "order_id": _normalize_order_id,
    "ticket_id": _normalize_order_id,
    "id_type": lambda v: (v or "").strip().lower(),
    "new_slot": lambda v: (v or "").strip(),
    "record_id": lambda v: (v or "").strip(),
}


def _normalize_arg(arg_name: str, value: Any) -> str:
    s = "" if value is None else str(value)
    norm = _ARG_NORMALIZERS.get(arg_name)
    return norm(s) if norm else s.strip()


def argument_exact_match(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[str, Dict[str, float]]:
    """Per-config argument-fidelity score.

    For each query, walk every ``ExpectedToolCall`` and search the
    captured invocations for the first matching ``tool_name``. Compare
    each expected argument to the corresponding actual argument after
    passing both through the per-name normalizer
    (``_normalize_phone``, ``_normalize_email``, etc.). Two scores are
    reported per config:

      ``exact``   : fraction of queries where every argument of every
                    expected call matched.
      ``partial`` : fraction of queries where at least one argument of
                    each expected call matched.

    Queries with empty ``effective_tool_calls`` (hard negatives,
    security/stress) are excluded from the denominator.
    """
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        grouped[r.config].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for config, runs in grouped.items():
        considered = 0
        exact_hits = 0
        partial_hits = 0
        for r in runs:
            query = by_id.get(r.query_id)
            if query is None:
                continue
            expected_calls = query.effective_tool_calls
            if not expected_calls:
                continue
            considered += 1
            all_exact = True
            any_partial = True
            for expected in expected_calls:
                inv = next(
                    (i for i in r.tool_invocations if i.tool_name == expected.tool_name),
                    None,
                )
                if inv is None:
                    all_exact = False
                    any_partial = False
                    continue
                exact_for_call, partial_for_call = _compare_args(expected, inv.raw_args)
                all_exact = all_exact and exact_for_call
                any_partial = any_partial and partial_for_call
            if all_exact:
                exact_hits += 1
            if any_partial:
                partial_hits += 1
        out[config] = {
            "exact": exact_hits / considered if considered else 0.0,
            "partial": partial_hits / considered if considered else 0.0,
            "considered": considered,
        }
    return out


def _compare_args(
    expected: ExpectedToolCall, actual: Mapping[str, Any],
) -> Tuple[bool, bool]:
    """Return (all_match, any_match) for one expected call.

    Empty expected.arguments → both flags True (legacy queries that
    only tracked tool-name/record-id pass arg fidelity vacuously).
    """
    if not expected.arguments:
        return True, True
    matches = []
    for arg_name, expected_value in expected.arguments.items():
        actual_value = actual.get(arg_name, "")
        ok = _normalize_arg(arg_name, expected_value) == _normalize_arg(arg_name, actual_value)
        matches.append(ok)
    return all(matches), any(matches)


# ----------------------------------------------------------------------
# Cross-turn token stability
# ----------------------------------------------------------------------

_REF_TOKEN_RE = re.compile(r"([a-z_]+_ref:[a-z]{2}_[0-9a-f]{8})", re.IGNORECASE)


def cross_turn_token_stability(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[str, float]:
    """Fraction of multi-turn virtualization runs with stable ref tokens.

    For each multi-turn query under the virtualization config:
      1. Walk every ``sent_to_llm`` message in order.
      2. Collect all reference tokens of the form ``<type>_ref:<...>``.
      3. A query is "stable" iff every PII value that appears in a
         later turn is paired with the same ref token as the first
         turn that anonymized it.

    The comparison is done by counting *distinct ref tokens per
    message-content position*: if the same anonymized message is sent
    on two consecutive iterations and the ref tokens differ, that's a
    stability failure.

    Returns one score per config; non-virtualization configs are still
    reported (always 1.0 since they don't use ref tokens) so the table
    is uniform.
    """
    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        grouped[r.config].append(r)

    out: Dict[str, float] = {}
    for config, runs in grouped.items():
        eligible = 0
        stable = 0
        for r in runs:
            query = by_id.get(r.query_id)
            if query is None or query.bucket != "multi_turn":
                continue
            eligible += 1
            tokens_per_turn: List[List[str]] = [
                _REF_TOKEN_RE.findall(str(entry.get("content") or ""))
                for entry in r.sent_to_llm
                if str(entry.get("role") or "").lower() == "human"
            ]
            # Stability: a token introduced in turn k must reappear
            # identically in any later turn that includes the same
            # underlying anonymized message. Approximate by checking
            # that the multiset of tokens in turn 0 is a subset of every
            # subsequent turn that mentions any token at all.
            if not tokens_per_turn or len(tokens_per_turn) < 2:
                stable += 1
                continue
            t0 = set(tokens_per_turn[0])
            ok = True
            for later in tokens_per_turn[1:]:
                later_set = set(later)
                if later_set and not t0.issubset(later_set):
                    ok = False
                    break
            if ok:
                stable += 1
        out[config] = stable / eligible if eligible else 1.0
    return out


# ----------------------------------------------------------------------
# Per-(bucket × language) aggregated report
# ----------------------------------------------------------------------

def per_bucket_per_language_report(
    results: Sequence[RunResult],
    dataset: Sequence[EvalQuery],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Bucket × language × config aggregated report.

    Returns a dict keyed on ``(config, bucket, language)`` whose value
    holds the per-slice tool-call success rate, raw PII transmission
    count, leak-guard trigger rate, and median latency. The detection
    P/R figures stay global because they're computed on the dataset
    alone and slicing them adds noise without information.
    """
    from benchmarks.pii_evaluation import metrics as global_metrics

    by_id: Dict[str, EvalQuery] = {q.id: q for q in dataset}
    grouped: Dict[Tuple[str, str, str], List[RunResult]] = defaultdict(list)
    for r in results:
        q = by_id.get(r.query_id)
        if q is None:
            continue
        grouped[(r.config, q.bucket, q.language)].append(r)

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, runs in grouped.items():
        config, bucket, language = key
        # Subset the dataset to the queries this slice covers.
        slice_query_ids = {r.query_id for r in runs}
        slice_dataset = [q for q in dataset if q.id in slice_query_ids]
        success = global_metrics.tool_call_success_rate(runs, slice_dataset)
        exposure = global_metrics.model_pii_exposure_count(runs, slice_dataset)
        latency = global_metrics.latency_stats(runs)
        leak = global_metrics.leak_guard_trigger_rate(runs)
        out[key] = {
            "n": len(runs),
            "tool_call_success_rate": success.get(config, 0.0),
            "raw_pii_transmissions": exposure.get(config, 0),
            "median_latency_s": latency.get(config, {}).get("median", 0.0),
            "p95_latency_s": latency.get(config, {}).get("p95", 0.0),
            "leak_guard_trigger_rate": leak.get(config, 0.0),
        }
    return out
