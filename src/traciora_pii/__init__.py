"""Public API for traciora_pii."""

from .pii import PIIDetector, PIIFinding, detect_pii, redact_pii_text, scrub
from .pii_leak_guard import PIILeakError, assert_model_safe
from .pii_tool_broker import make_pii_brokered_tool
from .pii_vault import PIIVaultStore, REF_TOKEN_PATTERN
from .pii_virtualizer import PIIVirtualizer, StreamingRefRehydrator

__all__ = [
    "traciora_config",
    "get_shared_second_pass",
    "prepare_opf",
    "scrub",
    "PIIDetector",
    "PIIFinding",
    "detect_pii",
    "redact_pii_text",
    "PIILeakError",
    "assert_model_safe",
    "make_pii_brokered_tool",
    "PIIVaultStore",
    "REF_TOKEN_PATTERN",
    "PIIVirtualizer",
    "StreamingRefRehydrator",
]

from .runtime import traciora_config, get_shared_second_pass, prepare_opf
