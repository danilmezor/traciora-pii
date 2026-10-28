"""Single source of truth for traciora_pii runtime configuration.

The library and benchmarks read every flag, model name, endpoint, path,
and timeout through ``PIIConfig`` rather than scattered
``os.getenv`` calls. The committed default lives at ``traciora.yaml`` in
the repo root; users override on a per-invocation basis with env vars
(see :data:`_ENV_OVERRIDES`) or CLI flags.

Precedence
----------

For every leaf field::

    CLI flag  >  env var  >  YAML  >  built-in default

CLI flags are applied at the call site by the consuming script via
``config.with_overrides({...})``. The library never reads ``sys.argv``.

Secrets
-------

API keys, Fernet keys, and similar secrets never appear in the YAML.
Each consumer field that needs a secret carries an ``api_key_env`` (or
``encryption_key_env``) sibling that names the env var to read at the
point of use. The YAML is therefore safe to commit.

Example
-------

>>> cfg = load_config()
>>> cfg.detector.second_pass
'none'
>>> cfg.detector.opf.models["base"].checkpoint
'openai/privacy-filter'
>>> cfg2 = cfg.with_overrides({"detector": {"second_pass": "opf"}})
>>> cfg2.detector.second_pass
'opf'

Env override example::

    PII_SECOND_PASS=opf python -m benchmarks.pii_evaluation.run_experiment
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .policies.schemas import VaultCategory


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

class OPFModelEntry(BaseModel):
    """One OPF model registered for selection by name.

    ``checkpoint`` is either a HuggingFace repo id (``org/name``) or an
    absolute local path. ``policy`` names a YAML under
    ``src/traciora_pii/policies/opf/<policy>.yaml`` whose ``label_map``
    is consumed by the OPF detector. Adding a new model to the registry
    is a one-line YAML edit; no Python edit required.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    checkpoint: str
    policy: str                          # name of a policy under policies/opf/
    description: str = ""


class OPFConfig(BaseModel):
    """OPF (token-classification) second-pass detector wiring."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    default_model: str = "base"
    models: dict[str, OPFModelEntry] = Field(default_factory=dict)
    device: str = "cpu"                  # cpu | cuda | mps | mlx
    # mlx only: None runs the converted checkpoint in bf16, 8 in 8-bit affine (group 64).
    mlx_bits: int | None = None
    prefilter: bool = False
    cache_dir: str | None = None

    @field_validator("mlx_bits")
    @classmethod
    def _mlx_bits_known(cls, value: int | None) -> int | None:
        if value not in (None, 8):
            raise ValueError(f"mlx_bits must be null (bf16) or 8, got {value!r}")
        return value


class EncoderModelEntry(BaseModel):
    """One encoder token classifier registered for selection by name.

    ``checkpoint`` is a HuggingFace repo id (``org/name``) pinned by a
    40-hex ``revision``, or a local path that needs no revision, told by its
    leading ``/``, ``~`` or ``.`` (a relative directory is written ``./dir``). ``family`` names the
    label scheme the detector reads from the classifier head.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    checkpoint: str
    revision: str | None = None
    family: str
    description: str = ""

    @field_validator("family")
    @classmethod
    def _family_known(cls, value: str) -> str:
        if value not in ("rubert", "horizon"):
            raise ValueError(f"family must be rubert or horizon, got {value!r}")
        return value

    @property
    def is_local(self) -> bool:
        return self.checkpoint.startswith(("/", "~", "."))

    @model_validator(mode="after")
    def _repo_id_pinned(self) -> "EncoderModelEntry":
        if not self.is_local and not re.fullmatch(r"[0-9a-f]{40}", self.revision or ""):
            raise ValueError(
                f"checkpoint {self.checkpoint!r} is a repo id; revision must be a "
                f"40-hex commit, got {self.revision!r}"
            )
        return self


class EncoderConfig(BaseModel):
    """Encoder (BERT-family token-classification) second-pass detector wiring."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    default_model: str = "ru"
    models: dict[str, EncoderModelEntry] = Field(default_factory=dict)
    device: str = "cpu"
    threshold: float = 0.5
    threads: int | None = None
    cache_dir: str | None = None

    @field_validator("device")
    @classmethod
    def _device_known(cls, value: str) -> str:
        if value not in ("cpu", "cuda"):
            raise ValueError(f"device must be cpu or cuda, got {value!r}")
        return value


class LLMConfig(BaseModel):
    """OpenAI-compatible LLM second-pass detector wiring.

    Defaults to OpenRouter's OpenAI-compatible endpoint so any model
    (Nvidia Nemotron, Llama, Mistral, Gemma, …) can be selected by
    prefix-routed name at runtime. There is intentionally **no default
    model** — every run that enables ``detector.second_pass: llm`` must
    pass ``--detector-llm-model <name>`` on the CLI so reviewers can see
    exactly which model produced each detection result.

    Works equally well against a local OpenAI-compatible server
    (LM Studio, Ollama, vLLM): set ``base_url`` to that endpoint and the
    detector's HTTP protocol is unchanged.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    base_url: str = "https://openrouter.ai/api/v1"
    model: str | None = None
    # Reuses the eval-LLM key by default because both calls go to the
    # same OpenRouter account. Override if you need a separate key.
    api_key_env: str = "PII_EVAL_API_KEY"
    timeout_seconds: float = 30.0
    prefilter: bool = True
    # 1024 (not 256) so reasoning-mode models (qwen3.5, deepseek-r1) have
    # headroom to finish their hidden chain-of-thought before emitting
    # the JSON answer. Non-reasoning models cap out well below this and
    # pay no extra cost; the harness only bills emitted tokens.
    max_tokens: int = 1024


class PresidioConfig(BaseModel):
    """Microsoft Presidio second-pass detector wiring.

    ``language`` is the spaCy/Presidio language code (``en`` / ``de`` /
    ``ru`` etc.) — Presidio loads one NER backbone per language, so a
    detector instance is single-language. The benchmark harness builds
    one per query language when running trilingual ablations.

    ``nlp_model`` is the spaCy model name Presidio's NlpEngine loads
    for ``language``. When unset (``""``) the adapter picks a sensible
    default (``en_core_web_lg`` / ``de_core_news_lg`` / ``ru_core_news_lg``).
    Override per language to swap ``_lg`` for ``_md`` / ``_sm``.

    ``label_map`` overrides the package default (``PERSON`` only) when
    set; an empty dict keeps the package default. ``entities`` is the
    Presidio recognizer allowlist passed into ``analyze()``; an empty
    list keeps the labels derived from ``label_map``.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    language: str = "en"
    score_threshold: float = 0.5
    label_map: dict[str, str] = Field(default_factory=dict)
    entities: list = Field(default_factory=list)
    prefilter: bool = False
    nlp_model: str = ""


class DetectorConfig(BaseModel):
    """First-pass regex + optional second-pass detector configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    excluded_categories: list[VaultCategory] = Field(default_factory=list)
    second_pass: str = "none"
    opf: OPFConfig = Field(default_factory=OPFConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    presidio: PresidioConfig = Field(default_factory=PresidioConfig)
    # Exactly one of regex_policy / regex_policies should be set. Single-
    # policy keeps the historical behavior; the list form unions multiple
    # locales' patterns so the regex layer catches typed identifiers
    # regardless of the surrounding language. Default: single "ru" for
    # backward compatibility with existing configs.
    regex_policy: str | None = "ru"
    regex_policies: list[str] | None = None
    masking_enabled: bool = True
    leak_guard_enabled: bool = True

    @model_validator(mode="after")
    def _policy_xor(self) -> "DetectorConfig":
        if self.regex_policies is not None and self.regex_policy is not None and self.regex_policy != "ru":
            # User explicitly set both -> ambiguous.
            raise ValueError(
                "detector.regex_policy and detector.regex_policies are mutually exclusive; "
                "set exactly one."
            )
        return self


class EvalLLMConfig(BaseModel):
    """Agent-pipeline LLM client used by the benchmark harness.

    Defaults to OpenRouter's OpenAI-compatible endpoint so any model
    (Anthropic / OpenAI / Google / DeepSeek / etc.) can be selected by
    prefix-routed model name at runtime. There is intentionally **no
    default model** — every run must pass ``--llm-model <name>`` on the
    CLI so reviewers can see exactly which model produced each result
    instead of inheriting a hidden default.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    # Optional: enforce-at-build-time in _build_eval_llm rather than at
    # config-load time so dataset-only / regex-only runs don't trip on
    # the absence of a model.
    model: str | None = None
    api_key_env: str = "PII_EVAL_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int = 512


class EvalConfig(BaseModel):
    """Benchmark eval harness wiring (results dir, agent loop, LLM)."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    llm: EvalLLMConfig = Field(default_factory=EvalLLMConfig)
    results_dir: str = "benchmarks/pii_evaluation/results"
    max_agent_iterations: int = 4


class VaultConfig(BaseModel):
    """Vault encryption wiring; key is read from env at session open."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    encryption_key_env: str = "PII_VAULT_KEY"


class PIIConfig(BaseModel):
    """Top-level config object loaded from ``traciora.yaml``.

    Frozen so consumers cannot mutate it after load — pass a
    transformed copy via :meth:`with_overrides` when you need a
    derived view (e.g. CLI flag overrides at the call site).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)

    def with_overrides(self, updates: Mapping[str, Any]) -> "PIIConfig":
        """Return a copy with ``updates`` deep-merged into the config.

        ``updates`` is a (possibly nested) dict; missing keys are left
        unchanged. Used by CLI consumers to apply argparse-derived
        overrides without rebuilding the whole config from scratch.
        """
        merged = _deep_merge(self.model_dump(), dict(updates))
        return PIIConfig.model_validate(merged)

    def secret(self, env_var_name: str, *, required: bool = False) -> str | None:
        """Resolve a secret by env-var name at the point of use.

        ``required=True`` raises a clear ``RuntimeError`` if the env
        var is unset; the default (``False``) returns ``None`` so
        consumers that *might* need a key can branch.
        """
        value = os.getenv(env_var_name)
        if not value and required:
            raise RuntimeError(
                f"Required secret env var {env_var_name!r} is not set. "
                f"Add it to .env (which is gitignored) or export it in the shell.",
            )
        return value


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

# Each entry: (env-var-name, dotted-path-into-config, coercer)
# The dotted path is split on '.' and the value is set on a nested dict
# before model_validate runs, so the env value goes through Pydantic's
# normal type coercion + validation.
_ENV_OVERRIDES: tuple = (
    ("PII_SECOND_PASS", "detector.second_pass", str),
    ("PII_OPF_MODEL", "detector.opf.default_model", str),
    ("PII_OPF_DEVICE", "detector.opf.device", str),
    ("PII_OPF_PREFILTER_ENABLED", "detector.opf.prefilter", "bool"),
    ("PII_OPF_CACHE_DIR", "detector.opf.cache_dir", str),
    ("PII_ENCODER_MODEL", "detector.encoder.default_model", str),
    ("PII_ENCODER_DEVICE", "detector.encoder.device", str),
    ("PII_ENCODER_CACHE_DIR", "detector.encoder.cache_dir", str),
    ("PII_LLM_BASE_URL", "detector.llm.base_url", str),
    ("PII_LLM_MODEL", "detector.llm.model", str),
    ("PII_LLM_TIMEOUT", "detector.llm.timeout_seconds", float),
    ("PII_LLM_PREFILTER", "detector.llm.prefilter", "bool"),
    ("PII_PRESIDIO_LANGUAGE", "detector.presidio.language", str),
    ("PII_PRESIDIO_SCORE_THRESHOLD", "detector.presidio.score_threshold", float),
    ("PII_PRESIDIO_PREFILTER", "detector.presidio.prefilter", "bool"),
    ("PII_REGEX_POLICY", "detector.regex_policy", str),
    ("PII_MASKING_ENABLED", "detector.masking_enabled", "bool"),
    ("PII_LEAK_GUARD_ENABLED", "detector.leak_guard_enabled", "bool"),
    ("PII_EVAL_LLM_ENABLED", "eval.llm.enabled", "bool"),
    ("PII_EVAL_MODEL_NAME", "eval.llm.model", str),
    ("PII_EVAL_BASE_URL", "eval.llm.base_url", str),
)

# Env vars that were removed entirely in the routed-OPF cleanup. If a
# user still has them set (e.g. from CI configs), we surface a clear
# error rather than silently ignoring the value.
_REMOVED_ENV_VARS: Mapping[str, str] = {
    "PII_OPF_ENABLED": "use detector.second_pass=opf in traciora.yaml or PII_SECOND_PASS=opf",
    "PII_OPF_BASE_MODEL": "register an entry in detector.opf.models and select it via --opf-model",
    "PII_OPF_FINETUNED_MODEL": "register an entry in detector.opf.models and select it via --opf-model",
    "PII_OPF_FINETUNED_HF_REPO": "set the entry's `checkpoint:` field in detector.opf.models",
    "PII_OPF_FINETUNED_HF_AUTO_DOWNLOAD": "removed — HuggingFace download happens via huggingface-cli",
    "PII_OPF_POLICY": "set the entry's `policy:` field in detector.opf.models",
    "PII_OPF_POLICY_PATH": "set the entry's `policy:` field in detector.opf.models",
    "PII_OPF_LABEL_MODE": "set the entry's `policy:` field in detector.opf.models",
    "PII_REGEX_POLICY_PATH": "use detector.regex_policy with an absolute path",
}


def _coerce_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _coerce(raw: str, kind: Any) -> Any:
    if kind == "bool":
        return _coerce_bool(raw)
    return kind(raw)


def _set_nested(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cursor = target
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — overrides win at every leaf."""
    out = dict(base)
    for key, value in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_yaml_path(explicit: Path | None) -> Path | None:
    """Discover the YAML file location.

    Order:
      1. ``explicit`` argument
      2. ``$TRACIORA_PII_CONFIG`` env var
      3. ``./traciora.yaml`` next to CWD
      4. ``traciora.yaml`` at the repo root (parent of the package's
         ``src/`` directory)
    Returns ``None`` if no file is found — the loader then returns the
    built-in defaults.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env_path = os.getenv("TRACIORA_PII_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    cwd_path = Path.cwd() / "traciora.yaml"
    if cwd_path.is_file():
        return cwd_path
    # Walk up from this module to find the repo-root traciora.yaml.
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "traciora.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(yaml_path: Path | None = None) -> PIIConfig:
    """Load and validate the global config.

    Precedence: env vars > ``<stem>.local.yaml`` > the selected YAML > defaults.
    Returns a frozen :class:`PIIConfig`. Use
    :meth:`PIIConfig.with_overrides` for per-call-site
    customization (CLI flags, programmatic profile switches).
    """
    _check_removed_env_vars()
    raw: dict[str, Any] = {}
    path = _resolve_yaml_path(yaml_path)
    if path is not None:
        # ``<stem>.local.yaml`` beside the selected file is merged over it.
        for layer in (path, path.with_name(f"{path.stem}.local.yaml")):
            if not layer.is_file():
                continue
            loaded = yaml.safe_load(layer.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{layer} top-level must be a mapping")
            raw = _deep_merge(raw, loaded)
    # Apply env overrides on top of the YAML.
    for env_var, dotted, kind in _ENV_OVERRIDES:
        raw_val = os.getenv(env_var)
        if raw_val is None or raw_val == "":
            continue
        try:
            _set_nested(raw, dotted, _coerce(raw_val, kind))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"failed to apply env override {env_var}={raw_val!r}: {exc}",
            )
    try:
        return PIIConfig.model_validate(raw)
    except ValidationError as exc:
        if path is not None:
            raise ValueError(
                f"config at {path} failed validation:\n{exc}",
            ) from exc
        raise


def _check_removed_env_vars() -> None:
    """Fail fast if a user has stale env vars from before the cleanup."""
    leftovers = []
    for name, message in _REMOVED_ENV_VARS.items():
        if os.getenv(name):
            leftovers.append(f"  {name}: {message}")
    if leftovers:
        raise RuntimeError(
            "These env vars were removed in the routed-OPF cleanup. "
            "Update your .env / shell to use the config-driven equivalents:\n"
            + "\n".join(leftovers),
        )


# Module-level cached config — most callers want the global default.
_CACHED: PIIConfig | None = None


def get_config() -> PIIConfig:
    """Return the process-global config, loading on first call.

    Tests that need a fresh load should call :func:`reset_config` first
    or call :func:`load_config` directly with an explicit path.
    """
    global _CACHED
    if _CACHED is None:
        _CACHED = load_config()
    return _CACHED


def reset_config() -> None:
    """Clear the cached config — used by tests."""
    global _CACHED
    _CACHED = None


__all__ = [
    "DetectorConfig",
    "EncoderConfig",
    "EncoderModelEntry",
    "EvalConfig",
    "EvalLLMConfig",
    "LLMConfig",
    "OPFConfig",
    "OPFModelEntry",
    "PIIConfig",
    "VaultConfig",
    "get_config",
    "load_config",
    "reset_config",
]
