"""Tests for the global ``traciora_pii.config`` module.

Covers:
  - YAML round-trip (load returns a frozen PIIConfig matching the file)
  - Missing-file fallback to built-in defaults
  - Env-var override precedence: env > local YAML > YAML > built-in default
  - CLI override via ``with_overrides`` returns a new instance, original unchanged
  - Removed env vars surface a clear, actionable RuntimeError
  - OPF model registry: unknown model name raises with the available keys listed
  - Adding a synthetic model entry at runtime makes it pickable without
    touching detector code
  - Secret indirection: ``api_key_env`` resolves at call time via os.environ
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pydantic import ValidationError

from traciora_pii.config import (
    EncoderConfig,
    EncoderModelEntry,
    OPFConfig,
    OPFModelEntry,
    PIIConfig,
    load_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def _reset_config_cache():
    reset_config()
    yield
    reset_config()


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    path = tmp_path / "traciora.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            detector:
              second_pass: opf
              opf:
                default_model: ru
                models:
                  base:
                    checkpoint: openai/privacy-filter
                    policy: en_comprehensive
                  ru:
                    checkpoint: pii-anon/opf-ru-v2
                    policy: ru_comprehensive
                device: cpu
            eval:
              llm:
                enabled: true
                model: gpt-4
            """,
        ),
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def test_yaml_round_trip(yaml_path):
    cfg = load_config(yaml_path)
    assert cfg.detector.second_pass == "opf"
    assert cfg.detector.opf.default_model == "ru"
    assert cfg.detector.opf.models["ru"].checkpoint == "pii-anon/opf-ru-v2"
    assert cfg.detector.opf.models["ru"].policy == "ru_comprehensive"
    assert cfg.eval.llm.enabled is True
    assert cfg.eval.llm.model == "gpt-4"


def test_missing_file_uses_built_in_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACIORA_PII_CONFIG", raising=False)
    nonexistent = tmp_path / "absent.yaml"
    cfg = load_config(nonexistent)
    # Defaults from src/traciora_pii/config.py
    assert cfg.detector.second_pass == "none"
    assert cfg.detector.opf.default_model == "base"
    assert cfg.eval.llm.enabled is False


def test_load_config_layers_local_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("PII_SECOND_PASS", raising=False)
    (tmp_path / "traciora.yaml").write_text(
        "detector:\n  second_pass: none\n  opf:\n    device: cpu\n    default_model: base\n",
        encoding="utf-8",
    )
    (tmp_path / "traciora.local.yaml").write_text(
        "detector:\n  second_pass: opf\n  opf:\n    device: mps\n", encoding="utf-8",
    )
    cfg = load_config(tmp_path / "traciora.yaml")
    assert cfg.detector.second_pass == "opf"
    assert cfg.detector.opf.device == "mps"
    # Deep merge: keys the local file does not mention keep the base value.
    assert cfg.detector.opf.default_model == "base"
    # Env still beats the local file.
    monkeypatch.setenv("PII_SECOND_PASS", "none")
    assert load_config(tmp_path / "traciora.yaml").detector.second_pass == "none"


def test_returned_config_is_frozen(yaml_path):
    cfg = load_config(yaml_path)
    with pytest.raises(Exception):
        cfg.detector.second_pass = "llm"  # frozen


# ----------------------------------------------------------------------
# Env override precedence
# ----------------------------------------------------------------------

def test_env_override_beats_yaml(yaml_path, monkeypatch):
    monkeypatch.setenv("PII_SECOND_PASS", "llm")
    cfg = load_config(yaml_path)
    assert cfg.detector.second_pass == "llm"


def test_env_override_coerces_bool(yaml_path, monkeypatch):
    monkeypatch.setenv("PII_EVAL_LLM_ENABLED", "false")
    cfg = load_config(yaml_path)
    assert cfg.eval.llm.enabled is False


def test_env_override_coerces_float(yaml_path, monkeypatch):
    monkeypatch.setenv("PII_LLM_TIMEOUT", "42.5")
    cfg = load_config(yaml_path)
    assert cfg.detector.llm.timeout_seconds == 42.5


def test_empty_env_var_does_not_override(yaml_path, monkeypatch):
    monkeypatch.setenv("PII_SECOND_PASS", "")  # empty string == unset for our purposes
    cfg = load_config(yaml_path)
    assert cfg.detector.second_pass == "opf"


def test_with_overrides_returns_new_instance(yaml_path):
    cfg = load_config(yaml_path)
    cfg2 = cfg.with_overrides({"detector": {"second_pass": "none"}})
    assert cfg2.detector.second_pass == "none"
    # Original is untouched.
    assert cfg.detector.second_pass == "opf"
    # Untouched fields persist.
    assert cfg2.detector.opf.default_model == "ru"


# ----------------------------------------------------------------------
# Removed env vars surface loudly
# ----------------------------------------------------------------------

def test_legacy_pii_opf_enabled_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PII_OPF_ENABLED", "true")
    monkeypatch.delenv("TRACIORA_PII_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.yaml")
    assert "PII_OPF_ENABLED" in str(excinfo.value)
    assert "PII_SECOND_PASS=opf" in str(excinfo.value)


def test_legacy_finetuned_model_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PII_OPF_FINETUNED_MODEL", "/some/path")
    monkeypatch.delenv("TRACIORA_PII_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.yaml")
    assert "PII_OPF_FINETUNED_MODEL" in str(excinfo.value)


# ----------------------------------------------------------------------
# OPF model registry
# ----------------------------------------------------------------------

def test_opf_unknown_model_lists_available(yaml_path):
    from traciora_pii.opf_pii_detector import OPFPIIDetector
    cfg = load_config(yaml_path)
    with pytest.raises(KeyError) as excinfo:
        OPFPIIDetector.from_config(cfg.detector.opf, model_name="nonexistent")
    msg = str(excinfo.value)
    assert "nonexistent" in msg
    assert "base" in msg
    assert "ru" in msg


def test_opf_registry_extends_without_code_change():
    """Adding a synthetic 5th model only requires a new registry entry."""
    cfg = PIIConfig(
        detector={
            "second_pass": "opf",
            "opf": {
                "default_model": "synthetic",
                "models": {
                    "synthetic": {"checkpoint": "/tmp/fake", "policy": "ru_names_addresses"},
                },
            },
        },
    )
    assert cfg.detector.opf.default_model == "synthetic"
    assert cfg.detector.opf.models["synthetic"].checkpoint == "/tmp/fake"


# ----------------------------------------------------------------------
# Secret indirection
# ----------------------------------------------------------------------

def test_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("MY_FAKE_KEY", "sk-secret-value")
    cfg = PIIConfig()
    assert cfg.secret("MY_FAKE_KEY") == "sk-secret-value"


def test_secret_returns_none_when_unset_and_optional(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    cfg = PIIConfig()
    assert cfg.secret("DEFINITELY_NOT_SET") is None


def test_secret_required_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    cfg = PIIConfig()
    with pytest.raises(RuntimeError) as excinfo:
        cfg.secret("DEFINITELY_NOT_SET", required=True)
    assert "DEFINITELY_NOT_SET" in str(excinfo.value)


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------

def test_unknown_field_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "detector:\n  second_pass: opf\n  unknown_field: 42\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    assert "validation" in str(excinfo.value).lower() or "unknown_field" in str(excinfo.value)


def test_opf_entry_requires_policy(tmp_path):
    path = tmp_path / "missing-policy.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            detector:
              second_pass: opf
              opf:
                default_model: thing
                models:
                  thing:
                    checkpoint: foo/bar
            """,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(path)


def test_opf_config_accepts_mlx_device_and_bits():
    cfg = OPFConfig(device="mlx", mlx_bits=8)
    assert cfg.device == "mlx"
    assert cfg.mlx_bits == 8
    assert OPFConfig(device="mlx").mlx_bits is None


def test_opf_config_rejects_other_bits():
    with pytest.raises(ValidationError, match="8"):
        OPFConfig(device="mlx", mlx_bits=4)


def test_encoder_entry_requires_revision_for_repo_id():
    with pytest.raises(ValidationError, match="revision"):
        EncoderModelEntry(checkpoint="Horizon-Labs/pii-redactor-small", family="horizon")
    with pytest.raises(ValidationError, match="revision"):
        EncoderModelEntry(checkpoint="Horizon-Labs/pii-redactor-small", revision="main",
                          family="horizon")
    entry = EncoderModelEntry(checkpoint="Horizon-Labs/pii-redactor-small",
                              revision="b9d834a12fc946a53551be95ba20ed9c11e9a553",
                              family="horizon")
    assert entry.revision == "b9d834a12fc946a53551be95ba20ed9c11e9a553"


def test_encoder_entry_local_by_syntax_not_by_current_directory(tmp_path, monkeypatch):
    for checkpoint in ("/opt/models/rubert", "~/models/rubert", "./models/rubert"):
        entry = EncoderModelEntry(checkpoint=checkpoint, family="rubert")
        assert entry.is_local and entry.revision is None
    (tmp_path / "org" / "name").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError, match="repo id"):
        EncoderModelEntry(checkpoint="org/name", family="rubert")
    assert EncoderModelEntry(checkpoint="./org/name", family="rubert").is_local


def test_encoder_family_and_device_are_closed():
    with pytest.raises(ValidationError) as excinfo:
        EncoderModelEntry(checkpoint="/opt/models/bert", family="bert")
    assert "rubert" in str(excinfo.value) and "horizon" in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        EncoderConfig(device="mps")
    assert "cpu" in str(excinfo.value) and "cuda" in str(excinfo.value)


def test_encoder_env_overrides(yaml_path, monkeypatch, tmp_path):
    monkeypatch.setenv("PII_ENCODER_DEVICE", "cuda")
    monkeypatch.setenv("PII_ENCODER_MODEL", "en")
    monkeypatch.setenv("PII_ENCODER_CACHE_DIR", str(tmp_path))
    cfg = load_config(yaml_path)
    assert cfg.detector.encoder.device == "cuda"
    assert cfg.detector.encoder.default_model == "en"
    assert cfg.detector.encoder.cache_dir == str(tmp_path)
