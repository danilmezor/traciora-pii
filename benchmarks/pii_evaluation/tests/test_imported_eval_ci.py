"""Smoke test for bootstrap-CI rendering inside the imported-dataset harness.

We don't load OPF or hit the network here — we construct an EvalRun
directly with fixture per-row records and assert that the rendered
markdown contains the CI brackets.
"""
from __future__ import annotations

from benchmarks.pii_evaluation.run_imported_dataset_eval import (
    EvalRun,
    PerPlaceholder,
    _render_pr_block,
)


def _ev(tp: int = 0, fp: int = 0, fn: int = 0) -> PerPlaceholder:
    return PerPlaceholder(tp=tp, fp=fp, fn=fn)


def test_render_pr_block_includes_ci_brackets_when_per_row_records_provided():
    counts = {"[PERSON_NAME]": _ev(tp=80, fp=6, fn=2)}
    per_row = []
    for _ in range(80):
        per_row.append({"[PERSON_NAME]": (1, 0, 0)})
    for _ in range(6):
        per_row.append({"[PERSON_NAME]": (0, 1, 0)})
    for _ in range(2):
        per_row.append({"[PERSON_NAME]": (0, 0, 1)})

    lines = _render_pr_block(counts, per_row_records=per_row, n_resamples=50)

    table = "\n".join(lines)
    # Header is the CI variant.
    assert "Precision (95% CI)" in table
    # CI brackets appear in the data row.
    assert " [" in table and "]" in table
    # Macro / micro rows are also CI-formatted.
    assert "**macro**" in table
    assert "**micro**" in table


def test_render_pr_block_falls_back_to_point_estimates_without_records():
    counts = {"[X]": _ev(tp=5, fp=1, fn=2)}
    lines = _render_pr_block(counts)
    table = "\n".join(lines)
    # Plain header — no "(95% CI)".
    assert "Precision (95% CI)" not in table
    assert "| Precision |" in table


def test_render_pr_block_handles_zero_support_placeholder():
    """A placeholder with FPs but zero support shouldn't crash bootstrap."""
    counts = {"[LICENSE_PLATE]": _ev(tp=0, fp=3, fn=0)}
    per_row = [
        {"[LICENSE_PLATE]": (0, 1, 0)},
        {"[LICENSE_PLATE]": (0, 1, 0)},
        {"[LICENSE_PLATE]": (0, 1, 0)},
    ]
    lines = _render_pr_block(counts, per_row_records=per_row, n_resamples=20)
    table = "\n".join(lines)
    assert "0.000" in table  # recall is 0
