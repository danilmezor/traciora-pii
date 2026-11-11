"""Smoke test for the stress-report archetype × config pivot."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.pii_evaluation.run_stress_report import (
    _archetype_for,
    pivot_security_stress_report,
    render_markdown,
    main,
)


def test_archetype_extraction_covers_known_workflows():
    assert _archetype_for("prompt_injection_for_raw_pii") == "prompt_injection"
    assert _archetype_for("prompt_injection_for_raw_pii_v2") == "prompt_injection"
    assert _archetype_for("forged_ref_token") == "forged_ref_token"
    assert _archetype_for("forged_ref_token_email") == "forged_ref_token"
    assert _archetype_for("zero_width_split_phone") == "zero_width_split"
    assert _archetype_for("code_switched_pii") == "code_switched"
    assert _archetype_for("hard_non_pii_mimic") == "hard_non_pii_mimic"
    assert _archetype_for("tool_exception_leakage") == "tool_exception_leakage"


def test_archetype_extraction_falls_back_to_raw_workflow():
    # Anything we haven't classified keeps its raw workflow name.
    assert _archetype_for("future_workflow_unmodeled") == "future_workflow_unmodeled"


def test_pivot_collapses_workflow_variants_into_archetype_totals():
    """Two prompt_injection workflow variants must roll up into one row
    per config with summed totals."""
    report = {
        "baseline|prompt_injection_for_raw_pii": {
            "total": 3, "passed": 0, "failed": 3, "raw_pii_transmissions": 5,
        },
        "baseline|prompt_injection_for_raw_pii_v2": {
            "total": 1, "passed": 0, "failed": 1, "raw_pii_transmissions": 2,
        },
        "virtualization|prompt_injection_for_raw_pii": {
            "total": 3, "passed": 3, "failed": 0, "raw_pii_transmissions": 0,
        },
        "virtualization|prompt_injection_for_raw_pii_v2": {
            "total": 1, "passed": 1, "failed": 0, "raw_pii_transmissions": 0,
        },
    }
    pivot = pivot_security_stress_report(report)
    assert pivot[("prompt_injection", "baseline")] == {
        "total": 4, "passed": 0, "failed": 4, "raw_pii_transmissions": 7,
    }
    assert pivot[("prompt_injection", "virtualization")] == {
        "total": 4, "passed": 4, "failed": 0, "raw_pii_transmissions": 0,
    }


def test_render_markdown_produces_archetype_rows_and_config_columns():
    pivot = {
        ("prompt_injection", "baseline"): {"total": 4, "passed": 0, "failed": 4, "raw_pii_transmissions": 7},
        ("prompt_injection", "virtualization"): {"total": 4, "passed": 4, "failed": 0, "raw_pii_transmissions": 0},
        ("tool_exception_leakage", "baseline"): {"total": 1, "passed": 1, "failed": 0, "raw_pii_transmissions": 1},
        ("tool_exception_leakage", "virtualization"): {"total": 1, "passed": 1, "failed": 0, "raw_pii_transmissions": 0},
    }
    md = render_markdown(pivot, slug="test")
    assert "Stress report — `test`" in md
    assert "prompt_injection" in md
    assert "tool_exception_leakage" in md
    assert "0% (0/4)" in md  # baseline prompt_injection pass rate
    assert "100% (4/4)" in md  # virtualization prompt_injection pass rate


def test_main_end_to_end_writes_markdown(tmp_path: Path):
    results_payload = {
        "aggregated": {
            "security_stress_report": {
                "baseline|prompt_injection_for_raw_pii": {
                    "total": 2, "passed": 0, "failed": 2, "raw_pii_transmissions": 4,
                },
                "virtualization|prompt_injection_for_raw_pii": {
                    "total": 2, "passed": 2, "failed": 0, "raw_pii_transmissions": 0,
                },
            },
        },
    }
    results_path = tmp_path / "results_demo.json"
    results_path.write_text(json.dumps(results_payload), encoding="utf-8")

    out_path = tmp_path / "stress_demo.md"
    rc = main(["--results", str(results_path), "--out", str(out_path)])
    assert rc == 0
    text = out_path.read_text(encoding="utf-8")
    assert "Stress report — `demo`" in text
    assert "prompt_injection" in text
    assert "0% (0/2)" in text
    assert "100% (2/2)" in text


def test_main_raises_when_report_missing(tmp_path: Path):
    results_path = tmp_path / "empty.json"
    results_path.write_text(json.dumps({"aggregated": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="security_stress_report"):
        main(["--results", str(results_path)])
