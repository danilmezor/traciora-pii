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


def _broken_build(*args, **kwargs):
    raise RuntimeError("opf checkpoint not found")


def test_failed_construction_returns_error_row(monkeypatch, synthetic_dataset):
    monkeypatch.setattr(run_detector_ablation, "build_second_pass_detector", _broken_build)
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
    assert row.status == "error: RuntimeError: opf checkpoint not found"
    assert row.pr == {}


def test_routed_opf_failed_construction_returns_error_row(monkeypatch, synthetic_dataset):
    monkeypatch.setattr(run_detector_ablation, "build_second_pass_detector", _broken_build)
    base_config = load_config()
    spec = run_detector_ablation.AblationSpec(
        name="regex_opf_routed",
        second_pass="opf",
        opf_model=run_detector_ablation.ROUTED,
    )
    row = run_detector_ablation._run_one(
        spec,
        base_config=base_config,
        dataset=synthetic_dataset,
        language_filter="all",
        recorder=LatencyRecorder(),
    )
    assert row.status.startswith("error: RuntimeError: opf checkpoint not found")
    assert row.pr == {}


def test_resolve_routed_prefers_language_model_then_default():
    base_config = load_config()
    for registry in ("opf", "encoder"):
        assert run_detector_ablation._resolve_routed(base_config, "ru", registry) == "ru"
        default = getattr(base_config.detector, registry).default_model
        assert run_detector_ablation._resolve_routed(base_config, "xx", registry) == default


class _RegexOnlyDetector:
    """A second pass that finds nothing, so a routed row scores like regex-only."""

    def __init__(self) -> None:
        self.closed = False

    def detect(self, text: str) -> list:
        return []

    def close(self) -> None:
        self.closed = True


def _routed_row(monkeypatch, spec, dataset):
    calls: list[dict[str, Any]] = []
    built: list[_RegexOnlyDetector] = []

    def _recording_build(config, **kwargs):
        calls.append({"second_pass": config.detector.second_pass, **kwargs})
        built.append(_RegexOnlyDetector())
        return built[-1]

    monkeypatch.setattr(run_detector_ablation, "build_second_pass_detector", _recording_build)
    row = run_detector_ablation._run_one(
        spec,
        base_config=load_config(),
        dataset=dataset,
        language_filter="all",
        recorder=LatencyRecorder(),
    )
    assert all(d.closed for d in built)
    return row, calls


def test_routed_encoder_row_builds_per_language_and_aggregates_like_opf(
    monkeypatch, synthetic_dataset,
):
    specs = {s.name: s for s in run_detector_ablation.ABLATION_CONFIGS}
    encoder_spec = specs["regex_encoder_routed"]
    assert encoder_spec.second_pass == "encoder"
    assert encoder_spec.encoder_model == run_detector_ablation.ROUTED

    encoder_row, encoder_calls = _routed_row(monkeypatch, encoder_spec, synthetic_dataset)
    opf_row, opf_calls = _routed_row(monkeypatch, specs["regex_opf_routed"], synthetic_dataset)

    assert encoder_calls == [
        {"second_pass": "encoder", "encoder_model_name": "de"},
        {"second_pass": "encoder", "encoder_model_name": "en"},
    ]
    assert opf_calls == [
        {"second_pass": "opf", "opf_model_name": "de"},
        {"second_pass": "opf", "opf_model_name": "en"},
    ]
    assert encoder_row.status == "ok"
    assert encoder_row.encoder_model == run_detector_ablation.ROUTED
    assert encoder_row.opf_model == ""
    assert encoder_row.routing_decisions == {"de": "de", "en": "en"}
    assert encoder_row.pr == opf_row.pr
    assert encoder_row.per_query_records == opf_row.per_query_records
    assert encoder_row.n_queries == opf_row.n_queries == len(synthetic_dataset)
    assert run_detector_ablation._row_to_dict(encoder_row)["encoder_model"] == "ROUTED"
