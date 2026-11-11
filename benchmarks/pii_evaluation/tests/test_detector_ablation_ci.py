"""Smoke test for the detector-ablation CI rendering path."""
from __future__ import annotations

from pathlib import Path

from benchmarks.pii_evaluation.run_detector_ablation import (
    AblationRow,
    _write_markdown,
)


def test_write_markdown_includes_ci_brackets_when_records_present(tmp_path: Path):
    row = AblationRow(
        name="regex_opf_routed",
        requested_mode="opf",
        opf_model="ROUTED",
        pr={
            "[PERSON_NAME]": {"precision": 0.95, "recall": 0.93, "tp": 80, "fp": 4, "fn": 6, "support": 86},
            "__macro__": {"precision": 0.95, "recall": 0.93, "tp": None, "fp": None, "fn": None, "support": None},
            "__micro__": {"precision": 0.95, "recall": 0.93, "tp": 80, "fp": 4, "fn": 6, "support": 86},
        },
        n_queries=100,
        total_s=12.34,
    )
    # Per-query records: 80 TP rows, 4 FP rows, 6 FN rows, plus 10 empty rows
    row.per_query_records = (
        [{"[PERSON_NAME]": (1, 0, 0)}] * 80
        + [{"[PERSON_NAME]": (0, 1, 0)}] * 4
        + [{"[PERSON_NAME]": (0, 0, 1)}] * 6
        + [{}] * 10
    )

    out = tmp_path / "ablation.md"
    _write_markdown(out, [row], dataset_size=100, n_resamples=50)
    text = out.read_text(encoding="utf-8")

    assert "macro P (95% CI)" in text
    assert " [" in text and "]" in text
    # The point estimate should still be present inside the CI cell.
    assert "0.95" in text or "0.94" in text  # accounts for bootstrap drift


def test_write_markdown_falls_back_to_points_when_row_skipped(tmp_path: Path):
    row = AblationRow(
        name="regex_llm",
        requested_mode="llm",
        opf_model="",
        pr={},
        n_queries=0,
        total_s=0.0,
        status="skipped: backend unavailable",
    )
    # per_query_records is empty — should not bootstrap, should show "—"
    out = tmp_path / "ablation_skipped.md"
    _write_markdown(out, [row], dataset_size=100, n_resamples=10)
    text = out.read_text(encoding="utf-8")
    assert "skipped" in text
    assert "—" in text
