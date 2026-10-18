"""Bootstrap confidence intervals for evaluation metrics.

The same recipe applies across §1 (cross-source eval), §2 (detector
ablation), and §3 (full framework eval): resample the per-query records
with replacement N times, recompute the metric on each resample, and
return the 2.5 / 97.5 percentiles alongside the point estimate.

We use a fixed seed (``DEFAULT_SEED = 42``) so the same eval inputs
always produce the same CI bounds. This is required for paper
reproducibility — a reviewer must be able to rerun and get the same
numbers byte-for-byte.

Two helpers:

* :func:`bootstrap_ci` — generic. Pass any aggregator function and a
  sequence of per-query records. The aggregator runs once per resample.

* :func:`bootstrap_pr_f1` — convenience wrapper for the very common
  case of TP/FP/FN counts per query. Returns CI for precision, recall,
  and F1 in one call.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, TypeVar

T = TypeVar("T")

DEFAULT_SEED = 42  # fixed for paper reproducibility
DEFAULT_N_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class CIResult:
    """Point estimate with bootstrap CI bounds."""
    point: float
    lower: float
    upper: float

    def format_md(self, *, decimals: int = 3) -> str:
        """Markdown rendering: ``0.913 [0.892, 0.931]``."""
        if self.point != self.point:  # NaN guard
            return "—"
        return f"{self.point:.{decimals}f} [{self.lower:.{decimals}f}, {self.upper:.{decimals}f}]"


def _percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interp percentile, mirrors numpy.percentile(interpolation='linear').

    No numpy dependency — pure-Python so the metrics module stays light.
    """
    if not samples:
        return float("nan")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def bootstrap_ci(
    per_query_records: Sequence[T],
    metric_fn: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> CIResult:
    """Bootstrap a scalar metric over per-query records.

    ``metric_fn(resampled_records) -> float`` is called ``n_resamples``
    times. Returns the point estimate (over the full input) and the
    two-sided ``confidence`` CI bounds (default 95%).

    Bootstrap math degenerates on empty inputs; we return NaN bounds in
    that case rather than raising, so callers can render "—" cleanly.
    """
    if not per_query_records:
        return CIResult(point=float("nan"), lower=float("nan"), upper=float("nan"))

    rng = random.Random(seed)
    records = list(per_query_records)
    n = len(records)

    point = metric_fn(records)

    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [records[rng.randrange(n)] for _ in range(n)]
        samples.append(metric_fn(resample))

    alpha = (1.0 - confidence) / 2.0
    lower = _percentile(samples, alpha)
    upper = _percentile(samples, 1.0 - alpha)
    return CIResult(point=point, lower=lower, upper=upper)


# ----------------------------------------------------------------------
# Convenience: P/R/F1 from per-query (tp, fp, fn) tuples
# ----------------------------------------------------------------------

PRRecord = Tuple[int, int, int]  # (tp, fp, fn) for one query


def _precision_from_counts(records: Sequence[PRRecord]) -> float:
    tp = sum(r[0] for r in records)
    fp = sum(r[1] for r in records)
    return tp / (tp + fp) if (tp + fp) else 0.0


def _recall_from_counts(records: Sequence[PRRecord]) -> float:
    tp = sum(r[0] for r in records)
    fn = sum(r[2] for r in records)
    return tp / (tp + fn) if (tp + fn) else 0.0


def _f1_from_counts(records: Sequence[PRRecord]) -> float:
    p = _precision_from_counts(records)
    r = _recall_from_counts(records)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


@dataclass(frozen=True)
class PRF1CI:
    precision: CIResult
    recall: CIResult
    f1: CIResult


def bootstrap_pr_f1(
    per_query_pr_records: Sequence[PRRecord],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> PRF1CI:
    """Bootstrap precision, recall, and F1 jointly over ``(tp, fp, fn)`` records.

    Uses the same resample for all three so the CIs are coherent (a
    given bootstrap iteration produces one (P, R, F1) triple).
    """
    if not per_query_pr_records:
        nan = CIResult(point=float("nan"), lower=float("nan"), upper=float("nan"))
        return PRF1CI(precision=nan, recall=nan, f1=nan)

    rng = random.Random(seed)
    records = list(per_query_pr_records)
    n = len(records)

    point_p = _precision_from_counts(records)
    point_r = _recall_from_counts(records)
    point_f1 = _f1_from_counts(records)

    samples_p: list[float] = []
    samples_r: list[float] = []
    samples_f1: list[float] = []
    for _ in range(n_resamples):
        resample = [records[rng.randrange(n)] for _ in range(n)]
        p = _precision_from_counts(resample)
        r = _recall_from_counts(resample)
        samples_p.append(p)
        samples_r.append(r)
        samples_f1.append((2 * p * r / (p + r)) if (p + r) else 0.0)

    alpha = (1.0 - confidence) / 2.0
    return PRF1CI(
        precision=CIResult(point=point_p, lower=_percentile(samples_p, alpha), upper=_percentile(samples_p, 1.0 - alpha)),
        recall=CIResult(point=point_r, lower=_percentile(samples_r, alpha), upper=_percentile(samples_r, 1.0 - alpha)),
        f1=CIResult(point=point_f1, lower=_percentile(samples_f1, alpha), upper=_percentile(samples_f1, 1.0 - alpha)),
    )
