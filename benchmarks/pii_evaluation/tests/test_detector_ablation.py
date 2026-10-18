"""Tests for the detector-ablation driver."""
from __future__ import annotations

from typing import Any

import pytest

from benchmarks.pii_evaluation import metrics, run_detector_ablation
from benchmarks.pii_evaluation.dataset import EvalQuery, PIIGroundTruth
from traciora_pii.config import load_config
from traciora_pii._latency import LatencyRecorder


@pytest.fixture
def synthetic_dataset() -> list[EvalQuery]:
    return [
        EvalQuery(
            id="q-en",
            language="en",
            bucket="single_turn",
            turns=("Reach me at alice@example.com.",),
            pii_spans=(PIIGroundTruth("[EMAIL]", "alice@example.com"),),
        ),
        EvalQuery(
            id="q-de",
            language="de",
            bucket="single_turn",
            turns=("Bitte rufen Sie unter +49 30 123 4567 zurück.",),
            pii_spans=(PIIGroundTruth("[PHONE]", "+49 30 123 4567"),),
        ),
    ]


def test_regex_only_row_matches_metrics_detection_pr(synthetic_dataset):
    base_config = load_config()
    spec = run_detector_ablation.AblationSpec(
        name="regex_only", second_pass="none",
    )
    recorder = LatencyRecorder()
    row = run_detector_ablation._run_one(
        spec,
        base_config=base_config,
        dataset=synthetic_dataset,
        language_filter="all",
        recorder=recorder,
    )

    direct = metrics.detection_precision_recall(synthetic_dataset)
    assert row.status == "ok"
    assert row.pr["__macro__"]["precision"] == direct["__macro__"]["precision"]
    assert row.pr["__micro__"]["precision"] == direct["__micro__"]["precision"]
    assert row.pr["__micro__"]["recall"] == direct["__micro__"]["recall"]
    assert row.n_queries == len(synthetic_dataset)


def test_skipped_when_dependency_unavailable_returns_status_row(
    monkeypatch, synthetic_dataset,
):
    monkeypatch.setattr(
        run_detector_ablation,
        "build_second_pass_detector",
        lambda *a, **kw: None,
    )
    base_config = load_config()
    spec = run_detector_ablation.AblationSpec(
        name="regex_presidio", second_pass="presidio",
    )
    row = run_detector_ablation._run_one(
        spec,
        base_config=base_config,
        dataset=synthetic_dataset,
        language_filter="en",
        recorder=LatencyRecorder(),
    )
    assert row.status.startswith("skipped:")
    assert row.pr == {}


def test_routed_opf_falls_back_when_dependency_unavailable(
    monkeypatch, synthetic_dataset,
):
    monkeypatch.setattr(
        run_detector_ablation,
        "build_second_pass_detector",
        lambda *a, **kw: None,
    )
    base_config = load_config()
    spec = run_detector_ablation.AblationSpec(
        name="regex_opf_routed",
        second_pass="opf",
        opf_model=run_detector_ablation.ROUTED_OPF,
    )
    row = run_detector_ablation._run_one(
        spec,
        base_config=base_config,
        dataset=synthetic_dataset,
        language_filter="all",
        recorder=LatencyRecorder(),
    )
    assert row.status.startswith("skipped:")


def test_resolve_routed_opf_prefers_language_model_then_default():
    base_config = load_config()
    # ru entry exists in traciora.yaml; en typically does not until the fine-tune ships.
    assert run_detector_ablation._resolve_routed_opf(base_config, "ru") == "ru"
    assert run_detector_ablation._resolve_routed_opf(base_config, "en") in {
        "en", base_config.detector.opf.default_model,
    }
