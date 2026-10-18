"""PII virtualization layer: anonymize text with reversible ref tokens
and rehydrate them back to real values.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
import unicodedata
from typing import TYPE_CHECKING, Any, Protocol

from ._latency import time_detector
from .pii import PIIFinding, detect_pii
from .pii_vault import PARTIAL_REF_TOKEN_PATTERN, PIIVaultStore, REF_TOKEN_PATTERN
import logging

if TYPE_CHECKING:
    from .config import PIIConfig

logger = logging.getLogger(__name__)


# Zero-width and invisible-formatting characters that attackers use to split
# PII across digit boundaries (e.g. `+49 30 1<U+200B>23 4567`) so the regex
# layer fails to match. Stripped before any other processing.
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")


def _normalize_input(text: str) -> str:
    """Strip zero-width evasion markers and NFKC-normalize."""
    if not text:
        return text
    return unicodedata.normalize("NFKC", _ZERO_WIDTH_RE.sub("", text))


class _SecondPassDetector(Protocol):
    """Duck-typed protocol for a second-pass PII detector.

    ``OPFPIIDetector`` and ``LLMPIIDetector`` both match this shape.
    """

    def detect(self, text: str) -> list:  # returns List[LLMFinding]
        ...


class PIIVirtualizer:
    """Anonymize PII in text/structured data with vault-backed ref tokens,
    and rehydrate refs back to raw values.

    The first pass is always the regex layer (``detect_pii``). An optional
    second pass runs a bidirectional token classifier (a fine-tuned OPF
    model) over the regex-tokenized output to catch names, addresses, and
    other entities regex structurally can't handle. The fine-tune training
    includes regex placeholder tokens as hard negatives, so the second
    pass does not re-tag content the regex already tokenized.
    """

    def __init__(
        self,
        vault: PIIVaultStore,
        *,
        second_pass_detector: _SecondPassDetector | None = None,
        excluded_categories: Iterable[str] | None = None,
    ):
        from ._masking_policy import resolve_exclusions
        self._excluded_categories = resolve_exclusions(excluded_categories)
        self._vault = vault
        self._second_pass = second_pass_detector

    # ------------------------------------------------------------------
    # Anonymize (raw PII → ref tokens)
    # ------------------------------------------------------------------

    def anonymize_text(self, session_scope_key: str, text: str) -> str:
        """Replace raw PII spans in *text* with deterministic ref tokens.

        Two passes compose: regex first, then optional OPF second-pass
        over the regex-tokenized output. Both passes write through the
        same vault, so downstream rehydration is identical.
        """
        if not text:
            return text

        text = _normalize_input(text)
        findings = detect_pii(text, excluded_categories=self._excluded_categories)
        merged = _merge_findings(findings) if findings else []

        # Build result from right to left to preserve offsets
        result = text
        for finding in reversed(merged):
            raw = text[finding.start:finding.end]
            ref_token = self._vault.get_or_create_token(
                session_scope_key,
                finding.placeholder,
                raw,
                source="anonymize",
            )
            result = result[:finding.start] + ref_token + result[finding.end:]

        # Second pass — OPF detector on the regex-tokenized text.
        # Fail-safe: any error falls back to the regex-only output.
        if self._second_pass is not None:
            try:
                result = self._apply_second_pass(session_scope_key, result)
            except Exception as exc:  # noqa: BLE001 — second-pass fail-safe; regex output is the fallback
                logger.warning(
                    "PIIVirtualizer second-pass failed; falling back to regex-only output. "
                    "error_type=%s",
                    type(exc).__name__,
                )
        return result

    def _apply_second_pass(self, session_scope_key: str, text: str) -> str:
        """Run the second-pass detector over *text* and substitute findings."""
        with time_detector(f"second_pass_{type(self._second_pass).__name__}"):
            opf_findings = self._second_pass.detect(text) or []
        if not opf_findings:
            return text

        # Pre-compute the set of existing ref-token spans in the text.
        # The second-pass detector's training includes these as hard
        # negatives, but empirically the model can still emit spans that
        # overlap a ref token (e.g. tagging the hex suffix of
        # ``phone_ref:ph_ad3a2c1a`` as a different class). Any such
        # overlap would corrupt the vault — skip defensively.
        ref_spans = [(m.start(), m.end()) for m in REF_TOKEN_PATTERN.finditer(text)]

        def _overlaps_ref(s: int, e: int) -> bool:
            return any(s < re and rs < e for rs, re in ref_spans)

        # Sort ascending then walk right-to-left for offset stability.
        from ._masking_policy import category_is_excluded
        ordered = sorted(
            (f for f in opf_findings
             if not category_is_excluded(f.placeholder, self._excluded_categories)),
            key=lambda f: (f.start, f.end),
        )
        result = text
        for finding in reversed(ordered):
            raw = result[finding.start:finding.end]
            if not raw:
                continue
            if _overlaps_ref(finding.start, finding.end):
                logger.debug(
                    "PIIVirtualizer dropping OPF span that overlaps existing ref token: "
                    "placeholder=%s span=%r",
                    finding.placeholder,
                    raw[:60],
                )
                continue
            ref_token = self._vault.get_or_create_token(
                session_scope_key,
                finding.placeholder,
                raw,
                source="anonymize-opf",
                normalization_key=finding.lemma,
            )
            result = result[:finding.start] + ref_token + result[finding.end:]
        return result

    def anonymize_structured(self, session_scope_key: str, payload: Any) -> Any:
        """Recursively anonymize PII in dicts, lists, and strings."""
        return self._walk(payload, lambda s: self.anonymize_text(session_scope_key, s))

    # ------------------------------------------------------------------
    # Rehydrate (ref tokens → raw PII)
    # ------------------------------------------------------------------

    def rehydrate_text(self, session_scope_key: str, text: str) -> str:
        """Replace ref tokens in *text* with their raw PII values."""
        if not text:
            return text

        def _replace(match: re.Match) -> str:
            ref = match.group(0)
            raw = self._vault.resolve_token(session_scope_key, ref)
            if raw is None:
                logger.warning("PII vault: unresolved ref token %s", ref)
                return ref  # Leave unresolved refs as-is
            return raw

        return REF_TOKEN_PATTERN.sub(_replace, text)

    def rehydrate_structured(self, session_scope_key: str, payload: Any) -> Any:
        """Recursively rehydrate ref tokens in dicts, lists, and strings."""
        return self._walk(payload, lambda s: self.rehydrate_text(session_scope_key, s))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(obj: Any, transform: callable) -> Any:
        if isinstance(obj, str):
            return transform(obj)
        if isinstance(obj, dict):
            return {k: PIIVirtualizer._walk(v, transform) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            result = [PIIVirtualizer._walk(item, transform) for item in obj]
            return type(obj)(result) if isinstance(obj, tuple) else result
        return obj


def build_second_pass_detector(
    config: "PIIConfig",
    *,
    opf_model_name: str | None = None,
) -> _SecondPassDetector | None:
    """Build the second-pass detector configured under ``detector.second_pass``.

    Supported modes: ``"none"`` (returns None), ``"opf"`` (OPFPIIDetector;
    ``opf_model_name`` or ``detector.opf.default_model`` picks the entry
    from ``detector.opf.models``), ``"llm"`` (LLMPIIDetector), and
    ``"presidio"`` (PresidioPIIDetector). Missing optional extras or
    construction failures log a warning and return None so the caller
    falls back to regex-only detection.
    """
    mode = config.detector.second_pass.strip().lower()
    if mode == "none":
        return None
    if mode == "opf":
        try:
            from .opf_pii_detector import OPFPIIDetector
        except ImportError as exc:
            logger.warning(
                "second_pass=opf requested but the `opf` extra is not installed; "
                "install with `pip install -e '.[opf]'`. Falling back to regex-only. "
                "error=%s",
                exc,
            )
            return None
        try:
            detector = OPFPIIDetector.from_config(
                config.detector.opf, model_name=opf_model_name,
            )
        except Exception as exc:  # noqa: BLE001 — construction failure fallback; see warning log
            logger.warning(
                "Failed to construct OPFPIIDetector; falling back to regex-only. "
                "error_type=%s error=%s",
                type(exc).__name__, exc,
            )
            return None
        logger.info(
            "PIIVirtualizer second-pass: opf model=%s device=%s",
            opf_model_name or config.detector.opf.default_model,
            config.detector.opf.device,
        )
        return detector
    if mode == "llm":
        try:
            from .llm_pii_detector import LLMPIIDetector
        except ImportError as exc:
            logger.warning(
                "second_pass=llm requested but module unavailable; "
                "falling back to regex-only. error=%s",
                exc,
            )
            return None
        try:
            detector = LLMPIIDetector.from_config(config.detector.llm)
        except Exception as exc:  # noqa: BLE001 — construction failure fallback; see warning log
            logger.warning(
                "Failed to construct LLMPIIDetector; falling back to regex-only. "
                "error_type=%s error=%s",
                type(exc).__name__, exc,
            )
            return None
        logger.info(
            "PIIVirtualizer second-pass: llm base_url=%s model=%s",
            config.detector.llm.base_url,
            config.detector.llm.model,
        )
        return detector
    if mode == "presidio":
        try:
            from .presidio_pii_detector import PresidioPIIDetector
        except ImportError as exc:
            logger.warning(
                "second_pass=presidio requested but the `presidio` extra is not installed; "
                "install with `pip install -e '.[presidio]'` and download the spaCy model "
                "for your language (e.g. `python -m spacy download en_core_web_sm`). "
                "Falling back to regex-only. error=%s",
                exc,
            )
            return None
        try:
            detector = PresidioPIIDetector.from_config(config.detector.presidio)
        except Exception as exc:  # noqa: BLE001 — construction failure fallback; see warning log
            logger.warning(
                "Failed to construct PresidioPIIDetector; falling back to regex-only. "
                "error_type=%s error=%s",
                type(exc).__name__, exc,
            )
            return None
        logger.info(
            "PIIVirtualizer second-pass: presidio language=%s score_threshold=%s",
            config.detector.presidio.language,
            config.detector.presidio.score_threshold,
        )
        return detector
    raise ValueError(
        f"detector.second_pass={mode!r} is not a valid mode. "
        "Expected one of: none, opf, llm, presidio.",
    )


def _merge_findings(findings):
    """Merge overlapping PII findings, keeping the first detector's placeholder."""
    if not findings:
        return findings
    merged = []
    for f in findings:
        if not merged:
            merged.append(f)
            continue
        last = merged[-1]
        if f.start <= last.end:
            merged[-1] = PIIFinding(
                detector=last.detector,
                start=last.start,
                end=max(last.end, f.end),
                placeholder=last.placeholder,
            )
        else:
            merged.append(f)
    return merged


class StreamingRefRehydrator:
    """Buffered streaming rehydrator for SSE/WebSocket token streams.

    Buffers chunks to handle ref tokens that arrive split across chunks.
    """

    _MAX_BUFFER = 64  # Longer than any possible ref token

    def __init__(self, virtualizer: PIIVirtualizer, session_scope_key: str):
        self._virtualizer = virtualizer
        self._session_scope_key = session_scope_key
        self._buffer = ""

    # Partial ref detection: matches incomplete ref token at end of buffer.
    # Derived from the same policy that builds REF_TOKEN_PATTERN so the two
    # matchers can't drift as the taxonomy evolves (see
    # policies.loader.build_partial_ref_token_pattern).
    _PARTIAL_RE = PARTIAL_REF_TOKEN_PATTERN

    def feed(self, chunk: str) -> str:
        """Feed a chunk from the stream. Returns safe-to-emit rehydrated text."""
        if not chunk:
            return ""
        self._buffer += chunk

        # Find all complete ref tokens in buffer
        safe_end = len(self._buffer)
        partial_match = self._PARTIAL_RE.search(self._buffer)
        if partial_match:
            candidate = partial_match.group(0)
            # Only hold back if the candidate looks like it could become a ref token
            if not REF_TOKEN_PATTERN.fullmatch(candidate):
                safe_end = partial_match.start()

        safe_text = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]

        # Truncate buffer if it grows too large (not a real ref token)
        if len(self._buffer) > self._MAX_BUFFER:
            safe_text += self._buffer
            self._buffer = ""

        if not safe_text:
            return ""

        return self._virtualizer.rehydrate_text(self._session_scope_key, safe_text)

    def flush(self) -> str:
        """Flush remaining buffer at stream end."""
        if not self._buffer:
            return ""
        remaining = self._buffer
        self._buffer = ""
        return self._virtualizer.rehydrate_text(self._session_scope_key, remaining)
