"""Pre-model PII leak detection guard.

Scans payloads destined for an external LLM for raw PII values that should
have been replaced by ref tokens.  Raises ``PIILeakError`` on detection,
preventing the call from proceeding.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import PIIConfig

import logging

from .pii import detect_pii
from .pii_vault import PIIVaultStore

logger = logging.getLogger(__name__)


class PIILeakError(Exception):
    """Raised when raw PII is found in a model-bound payload."""

    def __init__(self, pii_type: str, preview: str):
        self.pii_type = pii_type
        self.preview = preview
        super().__init__(f"PII leak detected: type={pii_type} preview={preview}")


def _is_enabled() -> bool:
    """Whether the leak guard is active for this process.

    Direct env read first (so ``monkeypatch.setenv`` and similar take
    effect immediately without resetting the config cache), then falls
    back to ``PIIConfig.detector.leak_guard_enabled`` when the env
    var is unset.
    """
    raw = os.getenv("PII_LEAK_GUARD_ENABLED")
    if raw is not None and raw != "":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    from .config import get_config
    return get_config().detector.leak_guard_enabled


def extract_strings(payload: Any, skip_block: Callable[[Any], bool] | None = None) -> list[str]:
    """Recursively collect all string values from a nested payload."""
    strings: list[str] = []
    if skip_block is not None and skip_block(payload):
        return strings
    if isinstance(payload, str):
        strings.append(payload)
    elif isinstance(payload, dict):
        for v in payload.values():
            strings.extend(extract_strings(v, skip_block))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            strings.extend(extract_strings(item, skip_block))
    return strings


def assert_model_safe(
    payload: Any,
    session_scope_key: str,
    vault: PIIVaultStore,
    *,
    config: PIIConfig | None = None,
    skip_block: Callable[[Any], bool] | None = None,
) -> None:
    """Check that *payload* contains no raw PII that has a vault token.

    Only raises if the exact raw value is tracked by the vault for this
    session — random digit sequences that happen to match PII patterns
    but aren't in the vault are ignored.
    """
    if not (config.detector.leak_guard_enabled if config is not None else _is_enabled()):
        return

    all_tokens = vault.resolve_all_tokens(session_scope_key)
    if not all_tokens:
        return  # Nothing in vault to leak

    raw_values = set(all_tokens.values())

    for text in extract_strings(payload, skip_block):
        if not text:
            continue
        findings = detect_pii(text, config=config)
        for finding in findings:
            raw_span = text[finding.start:finding.end]
            # Only flag if the raw value is actually tracked in the vault
            if raw_span in raw_values:
                preview = raw_span[:3] + "***" if len(raw_span) > 3 else raw_span
                logger.warning(
                    "PII leak guard triggered: session=%s type=%s preview=%s",
                    session_scope_key, finding.placeholder, preview,
                )
                raise PIILeakError(pii_type=finding.placeholder, preview=preview)
