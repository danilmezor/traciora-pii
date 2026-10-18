"""Check evaluation wiring without loading models or starting a GPU job."""
import sys

import pytest
import yaml

from fine_tuning import evaluate


def test_checkpoint_registered_and_report_copied(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(evaluate, "HARNESS_RESULTS_DIR", results)

    def run(cmd, *, env, cwd, check):
        assert cmd[:3] == [sys.executable, "-m", "benchmarks.pii_evaluation.run_detector_ablation"]
        assert cmd[cmd.index("--language") + 1] == "de"
        assert cmd[cmd.index("--only") + 1] == "regex_only,regex_opf_base,regex_opf_routed"
        config_path = cmd[cmd.index("--config") + 1]
        config = yaml.safe_load(open(config_path))
        assert config["detector"]["opf"]["models"]["de"] == {
            "checkpoint": str(checkpoint.resolve()), "policy": "de_comprehensive",
        }
        assert "base" in config["detector"]["opf"]["models"]
        assert env["PII_EVAL_MODEL_NAME"] == "de-test-eval"
        assert check is True
        (results / "detector_ablation_de-test-eval.md").write_text("report")

    monkeypatch.setattr(evaluate.subprocess, "run", run)
    report = evaluate.run_evaluate(
        checkpoint_dir=checkpoint, locale="de", slug="de-test", runs_dir=tmp_path / "run",
    )
    assert report.read_text() == "report"


def test_missing_report_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "HARNESS_RESULTS_DIR", tmp_path / "missing")
    monkeypatch.setattr(evaluate.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="report missing"):
        evaluate.run_evaluate(
            checkpoint_dir=tmp_path, locale="en", slug="test", runs_dir=tmp_path / "run",
        )
