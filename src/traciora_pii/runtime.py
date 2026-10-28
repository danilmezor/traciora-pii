"""Explicit Traciora configuration and one process-owned second-pass detector."""
from __future__ import annotations

import atexit
import json
import threading
from pathlib import Path

from .config import PIIConfig
from .encoder_checkpoint import ensure_encoder_checkpoint, validate_encoder_checkpoint
from .policies.loader import load_opf_policy

LANGUAGES = ("en", "de", "ru")
_HORIZON = {"checkpoint": "Horizon-Labs/pii-redactor-small",
            "revision": "b9d834a12fc946a53551be95ba20ed9c11e9a553", "family": "horizon"}
ENCODER_MODELS = {
    "ru": {"checkpoint": "redmadrobot-rnd/rubert-base-pii-ner",
           "revision": "c802e8cd26f85d1cf920973ea6f83965a0618d63", "family": "rubert",
           "description": "Russian BERT token classifier, 21 PII types (Apache-2.0)."},
    "en": {**_HORIZON,
           "description": "Multilingual ModernBERT PII redactor, 29 types (Apache-2.0)."},
    "de": _HORIZON,
}
# One sentence per language; a working second pass finds the name in it.
SELF_CHECK_SENTENCES = {
    "en": "Please call Maria Schneider tomorrow.",
    "de": "Bitte rufen Sie morgen Maria Schneider an.",
    "ru": "Пожалуйста, позвоните завтра Марии Шнайдер.",
}


class SecondPassUnavailable(RuntimeError):
    """The configured second pass cannot start or finds no name on the self-check sentence."""


def traciora_config(language: str = "en") -> PIIConfig:
    """Build an installed-package profile without reading YAML or environment."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported PII language {language!r}; choose {LANGUAGES}")
    return PIIConfig.model_validate({"detector": {
        "regex_policy": language,
        "excluded_categories": ["url"],
        "opf": {"default_model": language, "models": {
            language: {"checkpoint": f"pii-anon/opf-{language}-v2",
                       "policy": f"{language}_comprehensive"},
        }},
        "encoder": {"default_model": language, "models": {language: ENCODER_MODELS[language]}},
    }})

SECOND_PASS_MODES = ("none", "opf", "encoder")


def _cached_checkpoint(config: PIIConfig) -> Path:
    cfg = config.detector.opf
    checkpoint = cfg.models[cfg.default_model].checkpoint
    path = Path(checkpoint).expanduser()
    if path.is_dir() or path.is_absolute() or checkpoint.startswith((".", "~")):
        return path
    root = Path(cfg.cache_dir or "~/.cache/traciora_pii/models/opf").expanduser()
    return root / checkpoint.replace("/", "--")


def validate_checkpoint(path: Path, policy: str) -> None:
    """Reject incomplete checkpoints or a classifier head with the wrong labels."""
    if not path.is_dir() or not any(path.glob("*.safetensors")):
        raise ValueError(f"Incomplete OPF checkpoint at {path}")
    metadata = json.loads((path / "config.json").read_text())
    actual = set(metadata.get("span_class_names", ())) - {"O"}
    expected = set(load_opf_policy(policy).label_map)
    if actual != expected:
        raise ValueError(f"OPF labels do not match {policy}: missing={sorted(expected-actual)}, "
                         f"unexpected={sorted(actual-expected)}")


def _check_mode(config: PIIConfig) -> str:
    mode = config.detector.second_pass
    if mode not in SECOND_PASS_MODES:
        raise ValueError(f"Shared Traciora runtime supports second_pass="
                         f"{', '.join(SECOND_PASS_MODES)}; got {mode!r}")
    return mode


def prepare_second_pass(config: PIIConfig) -> Path | None:
    """Download only the selected model; call at deployment/startup, never in a request.

    ``opf`` with ``device: mlx`` also converts the checkpoint to the MLX layout beside
    itself when that copy is absent or was made from other weights; the converted
    path is returned. ``encoder`` downloads the pinned revision and validates it.
    """
    mode = _check_mode(config)
    if mode == "none":
        return None
    if mode == "encoder":
        cfg = config.detector.encoder
        entry = cfg.models[cfg.default_model]
        path = ensure_encoder_checkpoint(entry, cfg.cache_dir)
        validate_encoder_checkpoint(path, entry)
        return path
    from .opf_pii_detector import _resolve_opf_checkpoint
    cfg = config.detector.opf
    entry = cfg.models[cfg.default_model]
    path = Path(_resolve_opf_checkpoint(entry.checkpoint, cache_dir=cfg.cache_dir))
    validate_checkpoint(path, entry.policy)
    if cfg.device != "mlx":
        return path
    from . import opf_mlx_convert
    converted = opf_mlx_convert.converted_path(path, cfg.mlx_bits)
    if not opf_mlx_convert.is_current(path, converted, cfg.mlx_bits):
        opf_mlx_convert.convert_checkpoint(path, converted, bits=cfg.mlx_bits)
    return converted


class SharedSecondPass:
    """Serialize inference on a detector borrowed by multiple session virtualizers."""

    def __init__(self, detector):
        self._detector = detector
        self._lock = threading.Lock()

    def detect(self, text: str) -> list:
        with self._lock:
            return self._detector.detect(text)

    def close(self) -> None:
        with self._lock:
            self._detector.close()


_lock = threading.Lock()
_identity: str | None = None
_shared: SharedSecondPass | None = None
_failure: Exception | None = None


def get_shared_second_pass(config: PIIConfig) -> SharedSecondPass | None:
    """Load one selected, already-prefetched model. Never perform network I/O.

    A process cannot switch configuration after first use: restart it to change
    backend/language/model. A second pass that cannot be opened, finds no name on the
    self-check sentence, or cannot run it raises, and keeps raising until restart
    without retrying the load. There is no regex-only fallback.
    """
    global _identity, _shared, _failure
    mode = _check_mode(config)
    if mode == "none":
        return None
    section = config.detector.opf if mode == "opf" else config.detector.encoder
    identity = f"{mode}:{section.model_dump_json()}"
    with _lock:
        if _identity is not None:
            if identity != _identity:
                raise RuntimeError("PII model configuration changed; restart the process")
            if _failure is not None:
                raise _failure
            return _shared
        _identity = identity
        try:
            if mode == "opf":
                from .opf_pii_detector import OPFPIIDetector
                entry = section.models[section.default_model]
                policy = load_opf_policy(entry.policy)
                path = _cached_checkpoint(config)
                validate_checkpoint(path, entry.policy)
                detector = OPFPIIDetector(checkpoint=str(path), device=section.device,
                                          mlx_bits=section.mlx_bits,
                                          prefilter_enabled=section.prefilter,
                                          label_map=policy.label_map)
                language, name_placeholder = policy.locale, policy.label_map["private_person"]
                hint = " On mps set OPF_MOE_TRITON=0; on cuda check the Triton build."
            else:
                from .encoder_pii_detector import EncoderPIIDetector
                language, name_placeholder, hint = section.default_model, "[PERSON_NAME]", ""
                if language not in LANGUAGES:
                    raise ValueError(f"encoder model {language!r} has no self-check sentence; "
                                     f"register it under one of {LANGUAGES}")
                detector = EncoderPIIDetector.from_config(section)
        except Exception as exc:
            _failure = SecondPassUnavailable(
                f"PII second pass is configured but cannot start (model={section.default_model}, "
                f"device={section.device}): {type(exc).__name__}: {exc}. "
                f"Prepare the model before startup.")
            raise _failure from exc
        try:
            findings = detector.detect(SELF_CHECK_SENTENCES[language])
            if not any(f.placeholder == name_placeholder for f in findings):
                raise SecondPassUnavailable(
                    f"PII second pass found nothing on the self-check sentence "
                    f"(device={section.device}, model={section.default_model}); the detector "
                    f"cannot be trusted.{hint}")
        except Exception as exc:
            detector.close()
            _failure = exc
            raise
        _shared = SharedSecondPass(detector)
        return _shared


def close_shared_second_pass() -> None:
    """Release the process-owned detector at shutdown (also useful for isolated tests)."""
    global _identity, _shared, _failure
    with _lock:
        if _shared is not None:
            _shared.close()
        _shared = None
        _identity = None
        _failure = None


atexit.register(close_shared_second_pass)
