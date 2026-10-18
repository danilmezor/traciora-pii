"""Smoke test for Table 2 bootstrap CIs in run_experiment._write_markdown.

We don't run the full experiment harness — we construct the aggregated
dict directly and call _write_markdown to verify the CI surface lands
in the rendered output.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.pii_evaluation.dataset import EvalQuery
from benchmarks.pii_evaluation.run_experiment import _write_markdown


def test_table2_shows_ci_brackets_when_records_present(tmp_path: Path):
    aggregated = {
        "detector_configuration": {},
        "tool_call_success_rate": {},
        "model_pii_exposure_count": {},
        "latency_stats": {},
        "leak_guard_trigger_rate": {},
        "argument_exact_match": {},
        "cross_turn_token_stability": {},
        "per_config_atoms": {},
        # The Table 2 inputs:
        "detection_precision_recall": {
            "[PERSON_NAME]": {"precision": 0.95, "recall": 0.93, "tp": 80, "fp": 4, "fn": 6, "support": 86},
            "__macro__": {"precision": 0.95, "recall": 0.93, "tp": None, "fp": None, "fn": None, "support": None},
            "__micro__": {"precision": 0.95, "recall": 0.93, "tp": 80, "fp": 4, "fn": 6, "support": 86},
        },
        "detection_precision_recall_records": (
            [{"[PERSON_NAME]": (1, 0, 0)}] * 80
            + [{"[PERSON_NAME]": (0, 1, 0)}] * 4
            + [{"[PERSON_NAME]": (0, 0, 1)}] * 6
        ),
        "detection_precision_recall_per_language": {},
        "detection_precision_recall_by_length_bucket": {},
        "per_bucket_per_language": {},
        "security_stress_report": {},
        "model_pii_exposure_examples": {},
        "latency_contribution": {},
    }

    out = tmp_path / "results.md"
    _write_markdown(out, [], aggregated)
    text = out.read_text(encoding="utf-8")

    # Headline Table 2 section is present.
    assert "Table 2 — Detection sanity check" in text
    # CI'd header.
    assert "Precision (95% CI)" in text
    # CI brackets appear in a data row.
    person_row = next(
        line for line in text.splitlines() if "`[PERSON_NAME]`" in line
    )
    assert " [" in person_row and "]" in person_row


def test_table2_falls_back_to_point_estimates_without_records(tmp_path: Path):
    aggregated = {
        "detector_configuration": {},
        "tool_call_success_rate": {},
        "model_pii_exposure_count": {},
        "latency_stats": {},
        "leak_guard_trigger_rate": {},
        "argument_exact_match": {},
        "cross_turn_token_stability": {},
        "per_config_atoms": {},
        "detection_precision_recall": {
            "[X]": {"precision": 0.5, "recall": 0.7, "tp": 5, "fp": 5, "fn": 2, "support": 7},
        },
        # No detection_precision_recall_records key — Table 2 must
        # gracefully fall back to point estimates.
        "detection_precision_recall_per_language": {},
        "detection_precision_recall_by_length_bucket": {},
        "per_bucket_per_language": {},
        "security_stress_report": {},
        "model_pii_exposure_examples": {},
        "latency_contribution": {},
    }
    out = tmp_path / "results_nobootstrap.md"
    _write_markdown(out, [], aggregated)
    text = out.read_text(encoding="utf-8")

    assert "Table 2 — Detection sanity check" in text
    # No CI header when records are absent.
    assert "Precision (95% CI)" not in text
    assert "| Precision |" in text
