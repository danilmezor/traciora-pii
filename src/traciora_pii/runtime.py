"""Explicit Traciora configuration and one process-owned OPF detector."""
from __future__ import annotations

import atexit
import json
import threading
from pathlib import Path

from .config import PIIConfig
from .policies.loader import load_opf_policy

LANGUAGES = ("en", "de", "ru")
# One sentence per OPF policy locale; a working second pass finds the name in it.
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
    }})


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


def prepare_opf(config: PIIConfig) -> Path | None:
    """Download only the selected model; call at deployment/startup, never in a request."""
    if config.detector.second_pass != "opf":
        return None
    from .opf_pii_detector import _resolve_opf_checkpoint
    cfg = config.detector.opf
    entry = cfg.models[cfg.default_model]
    path = Path(_resolve_opf_checkpoint(entry.checkpoint, cache_dir=cfg.cache_dir))
    validate_checkpoint(path, entry.policy)
    return path


class SharedOPFDetector:
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
_shared: SharedOPFDetector | None = None
_failure: Exception | None = None


def get_shared_second_pass(config: PIIConfig) -> SharedOPFDetector | None:
    """Load one selected, already-prefetched model. Never perform network I/O.

    A process cannot switch configuration after first use: restart it to change
    language/model. A second pass that cannot be opened, finds no name on the
    self-check sentence, or cannot run it raises, and keeps raising until restart
    without retrying the load. There is no regex-only fallback.
    """
    global _identity, _shared, _failure
    if config.detector.second_pass == "none":
        return None
    if config.detector.second_pass != "opf":
        raise ValueError("Shared Traciora runtime supports second_pass=none or opf")
    identity = config.detector.opf.model_dump_json()
    with _lock:
        if _identity is not None:
            if identity != _identity:
                raise RuntimeError("PII model configuration changed; restart the process")
            if _failure is not None:
                raise _failure
            return _shared
        _identity = identity
        cfg = config.detector.opf
        entry = cfg.models[cfg.default_model]
        policy = load_opf_policy(entry.policy)
        try:
            from .opf_pii_detector import OPFPIIDetector
            path = _cached_checkpoint(config)
            validate_checkpoint(path, entry.policy)
            detector = OPFPIIDetector(checkpoint=str(path), device=cfg.device,
                                      prefilter_enabled=cfg.prefilter,
                                      label_map=policy.label_map)
        except Exception as exc:
            _failure = SecondPassUnavailable(
                f"PII second pass is configured but cannot start (model={cfg.default_model}, "
                f"device={cfg.device}): {type(exc).__name__}: {exc}. "
                f"Prepare the model before startup.")
            raise _failure from exc
        try:
            name_placeholder = policy.label_map["private_person"]
            findings = detector.detect(SELF_CHECK_SENTENCES[policy.locale])
            if not any(f.placeholder == name_placeholder for f in findings):
                raise SecondPassUnavailable(
                    f"PII second pass found nothing on the self-check sentence "
                    f"(device={cfg.device}, model={cfg.default_model}); the detector cannot "
                    f"be trusted. On mps set OPF_MOE_TRITON=0; on cuda check the Triton build.")
        except Exception as exc:
            detector.close()
            _failure = exc
            raise
        _shared = SharedOPFDetector(detector)
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
