"""Unit tests for the bootstrap CI helper.

We rely on this being deterministic (seed-controlled) and statistically
sound; failures here mean every reported CI in the paper is wrong.
"""
from __future__ import annotations

import math

from benchmarks.pii_evaluation.bootstrap import (
    CIResult,
    PRF1CI,
    _percentile,
    bootstrap_ci,
    bootstrap_pr_f1,
)


def test_percentile_basic_cases():
    assert _percentile([1.0], 0.5) == 1.0
    # Median of [1, 2, 3]
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    # 25th percentile of [1..5]
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25) == 2.0
    # Min / max
    assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0
    # Empty → NaN
    assert math.isnan(_percentile([], 0.5))


def test_bootstrap_ci_is_deterministic_under_fixed_seed():
    records = [1.0, 2.0, 3.0, 4.0, 5.0] * 20  # 100 records
    metric = lambda rs: sum(rs) / len(rs)  # mean

    r1 = bootstrap_ci(records, metric, n_resamples=200, seed=42)
    r2 = bootstrap_ci(records, metric, n_resamples=200, seed=42)
    assert r1 == r2


def test_bootstrap_ci_different_seeds_yield_different_intervals():
    records = list(range(50))
    metric = lambda rs: sum(rs) / len(rs)
    r1 = bootstrap_ci(records, metric, n_resamples=200, seed=1)
    r2 = bootstrap_ci(records, metric, n_resamples=200, seed=2)
    # Point is identical (full input); bounds differ.
    assert r1.point == r2.point
    assert (r1.lower, r1.upper) != (r2.lower, r2.upper)


def test_bootstrap_ci_point_matches_full_input_metric():
    records = [10.0, 20.0, 30.0, 40.0]
    metric = lambda rs: max(rs)
    r = bootstrap_ci(records, metric, n_resamples=100, seed=7)
    assert r.point == 40.0


def test_bootstrap_ci_empty_input_returns_nan_bounds():
    r = bootstrap_ci([], lambda rs: 1.0, n_resamples=10, seed=0)
    assert math.isnan(r.point)
    assert math.isnan(r.lower)
    assert math.isnan(r.upper)


def test_bootstrap_ci_bounds_contain_point_for_iid_data():
    """A 95% CI should usually contain the point estimate. For iid data
    where the bootstrap distribution is well-behaved, this holds for the
    full input's point estimate as well."""
    rng_records = [(i % 10) * 0.1 for i in range(200)]
    metric = lambda rs: sum(rs) / len(rs)
    r = bootstrap_ci(rng_records, metric, n_resamples=500, seed=42)
    assert r.lower <= r.point <= r.upper


def test_bootstrap_pr_f1_zero_support_returns_zero_point():
    """All-zero records (e.g. a placeholder with no support and no FPs)
    should return zeros, not NaN, to mirror legacy scoring."""
    records = [(0, 0, 0)] * 10
    ci = bootstrap_pr_f1(records, n_resamples=50, seed=0)
    assert ci.precision.point == 0.0
    assert ci.recall.point == 0.0
    assert ci.f1.point == 0.0


def test_bootstrap_pr_f1_perfect_classifier():
    """One TP per query, no FPs or FNs → P/R/F1 all 1.0; CIs degenerate."""
    records = [(1, 0, 0)] * 100
    ci = bootstrap_pr_f1(records, n_resamples=200, seed=42)
    assert ci.precision.point == 1.0
    assert ci.recall.point == 1.0
    assert ci.f1.point == 1.0
    # Tight CI — every resample is still perfect
    assert ci.precision.lower == 1.0
    assert ci.precision.upper == 1.0


def test_bootstrap_pr_f1_mixed_classifier():
    """50% TP / 25% FP / 25% FN over 100 records. Point estimates are
    deterministic; CI bounds should bracket them."""
    records = [(1, 0, 0)] * 50 + [(0, 1, 0)] * 25 + [(0, 0, 1)] * 25
    ci = bootstrap_pr_f1(records, n_resamples=500, seed=42)
    # tp=50, fp=25, fn=25 → P=R=F1=0.667
    assert abs(ci.precision.point - 50 / 75) < 1e-9
    assert abs(ci.recall.point - 50 / 75) < 1e-9
    assert abs(ci.f1.point - 50 / 75) < 1e-9
    # CIs span the point and are non-degenerate
    assert ci.precision.lower < ci.precision.point < ci.precision.upper
    assert ci.recall.lower < ci.recall.point < ci.recall.upper


def test_bootstrap_pr_f1_uses_same_resample_for_p_r_f1():
    """A given bootstrap iteration must produce one (P, R, F1) triple;
    F1 must equal 2PR/(P+R) for the same iteration. We can't observe a
    single iteration's tuple from outside, but we can confirm the
    point-estimate F1 matches 2PR/(P+R) from precision/recall points."""
    records = [(1, 0, 0)] * 40 + [(0, 1, 0)] * 10 + [(0, 0, 1)] * 10
    ci = bootstrap_pr_f1(records, n_resamples=200, seed=42)
    expected_f1 = 2 * ci.precision.point * ci.recall.point / (ci.precision.point + ci.recall.point)
    assert abs(ci.f1.point - expected_f1) < 1e-9


def test_ci_result_format_md():
    ci = CIResult(point=0.913, lower=0.892, upper=0.931)
    assert ci.format_md() == "0.913 [0.892, 0.931]"
    nan_ci = CIResult(point=float("nan"), lower=float("nan"), upper=float("nan"))
    assert nan_ci.format_md() == "—"
