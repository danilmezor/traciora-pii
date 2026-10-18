from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pii import PIIFinding, redact_pii_text
import logging

if TYPE_CHECKING:
    from .config import PIIConfig

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    text: str
    findings: list[PIIFinding]


class PIIRedactor:
    """Destructive redactor: regex-only PII masking for the legacy
    benchmark baseline. The framework's preferred path is reversible
    virtualization (see :class:`PIIVirtualizer`); this redactor exists
    only so the eval harness can report a "destructive baseline" pipeline
    alongside virtualization.

    LLM-based redaction now lives at the architecturally correct layer —
    the regex-then-LLM second-pass detector (``detector.second_pass: llm``,
    see :class:`traciora_pii.llm_pii_detector.LLMPIIDetector`). Running an LLM
    *after* destructive redaction would only see already-masked text and
    cannot recover the semantic entities the regex missed.
    """

    def __init__(self, *, enabled: bool = True, excluded_categories: Iterable[str] | None = None):
        from ._masking_policy import resolve_exclusions
        self._excluded_categories = resolve_exclusions(excluded_categories)
        self.enabled = enabled

    @classmethod
    def from_env(cls) -> "PIIRedactor":
        """Legacy constructor — reads PII_MASKING_ENABLED directly."""
        enabled = os.getenv("PII_MASKING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        return cls(enabled=enabled)

    @classmethod
    def from_config(cls, config: "PIIConfig") -> "PIIRedactor":
        """Build from a :class:`PIIConfig`.

        ``config.detector.masking_enabled`` flips the redactor on/off.
        """
        return cls(
            enabled=config.detector.masking_enabled,
            excluded_categories=config.detector.excluded_categories,
        )

    def redact_text(self, text: str) -> RedactionResult:
        if not self.enabled or not text:
            return RedactionResult(text=text, findings=[])
        regex_redacted, findings = redact_pii_text(
            text, excluded_categories=self._excluded_categories,
        )
        return RedactionResult(text=regex_redacted, findings=findings)

    def redact_structured(self, payload: Any) -> Any:
        if not self.enabled:
            return payload
        if isinstance(payload, str):
            return self.redact_text(payload).text
        if isinstance(payload, list):
            return [self.redact_structured(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.redact_structured(item) for item in payload)
        if isinstance(payload, dict):
            return {str(key): self.redact_structured(value) for key, value in payload.items()}
        return payload

    def redact_message_for_audit(self, message: str) -> tuple[int, str]:
        safe = self.redact_text(message).text if self.enabled else message
        return len(message or ""), safe


_DEFAULT_REDACTOR: PIIRedactor | None = None


def get_pii_redactor() -> PIIRedactor:
    global _DEFAULT_REDACTOR
    if _DEFAULT_REDACTOR is None:
        _DEFAULT_REDACTOR = PIIRedactor.from_env()
        logger.info("PII redactor initialized enabled=%s", _DEFAULT_REDACTOR.enabled)
    return _DEFAULT_REDACTOR


def redact_text(text: str) -> str:
    return get_pii_redactor().redact_text(text).text


def redact_structured(payload: dict[str, Any]) -> dict[str, Any]:
    redactor = get_pii_redactor()
    masked = redactor.redact_structured(payload)
    return masked if isinstance(masked, dict) else {"payload": masked}
