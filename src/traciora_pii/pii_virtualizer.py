"""PII virtualization layer: anonymize text with reversible ref tokens
and rehydrate them back to real values.
"""
from __future__ import annotations

import bisect
import logging
import re
import unicodedata
import weakref
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

from ._latency import time_detector
from .pii import PIIFinding, detect_pii
from .pii_vault import (
    _SEPARATORS_RE,
    NORMALIZE_SEPARATORS,
    PARTIAL_REF_TOKEN_PATTERN,
    REF_TOKEN_PATTERN,
    KnownValue,
    KnownValues,
    PIIVaultStore,
    _normalize_value,
)

if TYPE_CHECKING:
    from .config import PIIConfig

logger = logging.getLogger(__name__)


# Zero-width and invisible-formatting characters that attackers use to split
# PII across digit boundaries (e.g. `+49 30 1<U+200B>23 4567`) so the regex
# layer fails to match. Stripped before any other processing.
_ZERO_WIDTH_RE = re.compile("[\u200b‌‍⁠﻿]")


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

    Before any detector runs, every value the session already tokenized is
    replaced by its token wherever it appears in the text, so a value never
    goes back to the model because a detector missed it this time.

    The first detector pass is always the regex layer (``detect_pii``). An optional
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
        config: PIIConfig | None = None,
        map_text: Callable[[Any, Callable[[str], str]], Any] | None = None,
    ):
        from ._masking_policy import resolve_exclusions
        self._config = config
        self._map_text = map_text
        self._excluded_categories = resolve_exclusions(
            config.detector.excluded_categories
            if config is not None and excluded_categories is None else excluded_categories
        )
        self._vault = vault
        self._second_pass = second_pass_detector
        # Matcher per cached vault entry: it lives exactly as long as the
        # vault keeps that session's decrypted values, and a session that
        # gains a token gets a new entry and so a new matcher.
        self._known_matchers: weakref.WeakKeyDictionary[KnownValues, _KnownValueMatcher | None] = (
            weakref.WeakKeyDictionary()
        )

    # ------------------------------------------------------------------
    # Anonymize (raw PII → ref tokens)
    # ------------------------------------------------------------------

    def _replace_known_values(self, session_scope_key: str, text: str) -> str:
        known = self._vault.known_values(session_scope_key)
        if not known.values:
            return text
        try:
            matcher = self._known_matchers[known]
        except KeyError:
            matcher = _KnownValueMatcher.build(known.values, self._excluded_categories)
            self._known_matchers[known] = matcher
        return matcher.replace(text) if matcher is not None else text

    def anonymize_text(self, session_scope_key: str, text: str) -> str:
        """Replace raw PII spans in *text* with deterministic ref tokens.

        Three passes compose. Every value the session's vault already holds
        is replaced by its token wherever it appears; then the regex layer;
        then the optional second pass over the tokenized output. The
        detector passes write through the same vault, so downstream
        rehydration is identical.
        """
        if not text:
            return text

        text = _normalize_input(text)
        # Values this session already holds become their tokens first, so a
        # value the detectors would miss in this text (a plate without its
        # keyword, a name the model does not catch twice) never goes back
        # out as text.
        text = self._replace_known_values(session_scope_key, text)
        findings = detect_pii(text, excluded_categories=self._excluded_categories, config=self._config)
        merged = _merge_findings(findings) if findings else []

        # Build result from right to left to preserve offsets. One vault
        # transaction for the whole loop: one commit, nothing left on failure.
        result = text
        with self._vault.batch():
            for finding in reversed(merged):
                raw = text[finding.start:finding.end]
                ref_token = self._vault.get_or_create_token(
                    session_scope_key,
                    finding.placeholder,
                    raw,
                    source="anonymize",
                )
                result = result[:finding.start] + ref_token + result[finding.end:]

        # Second pass on the regex-tokenized text. A failure propagates: a
        # turn the configured detector did not cover is never returned.
        if self._second_pass is not None:
            result = self._apply_second_pass(session_scope_key, result)
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
        # Inference above runs outside the vault lock; the token writes share
        # one transaction of their own.
        result = text
        with self._vault.batch():
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

    def anonymize_content(self, session_scope_key: str, content: Any) -> Any:
        mapper = self._map_text or self._walk
        return mapper(content, lambda text: self.anonymize_text(session_scope_key, text))

    def rehydrate_content(self, session_scope_key: str, content: Any) -> Any:
        mapper = self._map_text or self._walk
        return mapper(content, lambda text: self.rehydrate_text(session_scope_key, text))

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
    config: PIIConfig,
    *,
    opf_model_name: str | None = None,
    encoder_model_name: str | None = None,
) -> _SecondPassDetector | None:
    """Build the second-pass detector configured under ``detector.second_pass``.

    Supported modes: ``"none"`` (returns None), ``"opf"`` (OPFPIIDetector;
    ``opf_model_name`` or ``detector.opf.default_model`` picks the entry
    from ``detector.opf.models``), ``"encoder"`` (EncoderPIIDetector;
    ``encoder_model_name`` or ``detector.encoder.default_model`` picks the
    entry from ``detector.encoder.models``), ``"llm"`` (LLMPIIDetector), and
    ``"presidio"`` (PresidioPIIDetector). The detector is a compliance
    control: a missing extra or a construction failure raises, never
    degrades to regex-only.
    """
    mode = config.detector.second_pass.strip().lower()
    if mode == "none":
        return None
    if mode == "opf":
        from .opf_pii_detector import OPFPIIDetector

        detector = OPFPIIDetector.from_config(config.detector.opf, model_name=opf_model_name)
        logger.info(
            "PIIVirtualizer second-pass: opf model=%s device=%s",
            opf_model_name or config.detector.opf.default_model,
            config.detector.opf.device,
        )
        return detector
    if mode == "encoder":
        from .encoder_pii_detector import EncoderPIIDetector

        detector = EncoderPIIDetector.from_config(config.detector.encoder,
                                                  model_name=encoder_model_name)
        logger.info(
            "PIIVirtualizer second-pass: encoder model=%s device=%s",
            encoder_model_name or config.detector.encoder.default_model,
            config.detector.encoder.device,
        )
        return detector
    if mode == "llm":
        from .llm_pii_detector import LLMPIIDetector

        detector = LLMPIIDetector.from_config(config.detector.llm)
        logger.info(
            "PIIVirtualizer second-pass: llm base_url=%s model=%s",
            config.detector.llm.base_url,
            config.detector.llm.model,
        )
        return detector
    if mode == "presidio":
        from .presidio_pii_detector import PresidioPIIDetector

        detector = PresidioPIIDetector.from_config(config.detector.presidio)
        logger.info(
            "PIIVirtualizer second-pass: presidio language=%s score_threshold=%s",
            config.detector.presidio.language,
            config.detector.presidio.score_threshold,
        )
        return detector
    raise ValueError(
        f"detector.second_pass={mode!r} is not a valid mode. "
        "Expected one of: none, opf, encoder, llm, presidio.",
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


# Known-value pass. A value matches where its significant characters (the
# value without the separators the vault's normaliser drops) appear in order,
# with any run of those separators between them: the same rule that gives two
# spellings one token. Matching is case-insensitive for values with a digit or
# an "@" (phones, emails, plates, IDs), where case means nothing, and exact for
# letter-only values (names), where "Mark" and "mark" differ.
_KNOWN_SEPARATOR = NORMALIZE_SEPARATORS + "*"
# A value glued to a word character by one of these is part of a longer token
# ("anna.lee@…", "+1-555-…", "12.03.2026", "Anna-Maria"), not the value.
_KNOWN_JOINER = r"[.\-+@/]"
_KNOWN_LEFT = rf"(?<!\w)(?<!\w{_KNOWN_JOINER})"
_KNOWN_RIGHT = rf"(?!\w)(?!{_KNOWN_JOINER}\w)"
# Shorter values are too likely to be a different thing that looks the same:
# a count, a year, an amount, a common word.
_KNOWN_MIN_CHARS = 4
_KNOWN_MIN_DIGITS = 6

try:  # Python 3.11+: the compiler behind re.compile, without its cache.
    from re._compiler import compile as _compile_uncached
except ImportError:  # pragma: no cover
    _compile_uncached = re.compile


def _significant(value: str) -> str:
    return _SEPARATORS_RE.sub("", value.strip())


def _long_enough(significant: str) -> bool:
    if significant.lstrip("+").isdigit():
        return len(significant.lstrip("+")) >= _KNOWN_MIN_DIGITS
    return len(significant) >= _KNOWN_MIN_CHARS


def _trie_pattern(values: Iterable[str]) -> str:
    """A regex matching any of *values*, separators allowed between characters.

    Built as a trie so a position fails after one character test instead of
    one per value; a value that continues into a longer one is tried longest
    first and falls back to the shorter one.
    """
    root: dict = {}
    for value in values:
        node = root
        for char in value:
            node = node.setdefault(char, {})
        node[""] = {}

    def emit(node: dict) -> str:
        branches = []
        for char in sorted(k for k in node if k):
            child = node[char]
            branch = re.escape(char)
            # A run of single-child nodes is one flat sequence, so nesting
            # grows only where values diverge.
            while len(child) == 1 and "" not in child:
                (char,) = child
                branch += _KNOWN_SEPARATOR + re.escape(char)
                child = child[char]
            if len(child) > ("" in child):
                rest = f"{_KNOWN_SEPARATOR}(?:{emit(child)})"
                branch += f"(?:{rest})?" if "" in child else rest
            branches.append(branch)
        return "|".join(branches)

    return emit(root)


class _KnownValueMatcher:
    """One compiled pattern over a session's known values."""

    def __init__(self, pattern: re.Pattern, folded: dict[str, str], exact: dict[str, str]):
        self._pattern = pattern
        self._folded = folded  # vault-normalized value -> token
        self._exact = exact    # significant characters -> token

    @classmethod
    def build(cls, values: Iterable[KnownValue], excluded_categories) -> _KnownValueMatcher | None:
        from ._masking_policy import category_is_excluded

        folded: dict[str, str] = {}
        exact: dict[str, str] = {}
        # Oldest first: when two tokens share a value, the first one wins.
        for value in values:
            if category_is_excluded(value.pii_type, excluded_categories):
                continue
            raw = _normalize_input(value.raw_value)
            significant = _significant(raw)
            if not _long_enough(significant):
                continue
            if "@" in significant or any(c.isdigit() for c in significant):
                folded.setdefault(_normalize_value(raw), value.ref_token)
            else:
                exact.setdefault(significant, value.ref_token)
        if not folded and not exact:
            return None
        union = "|".join(
            ([f"(?i:{_trie_pattern(folded)})"] if folded else [])
            + ([_trie_pattern(exact)] if exact else [])
        )
        # Compiled outside re's module cache: that cache is process-wide, would
        # keep up to 512 of these large patterns, and with them the session's
        # raw values after the session is closed.
        pattern = _compile_uncached(f"{_KNOWN_LEFT}(?:{union}){_KNOWN_RIGHT}", 0)
        return cls(pattern, folded, exact)

    def replace(self, text: str) -> str:
        token_spans = [m.span() for m in REF_TOKEN_PATTERN.finditer(text)]
        token_starts = [start for start, _ in token_spans]
        parts: list[str] = []
        pos = 0
        for match in self._pattern.finditer(text):
            start, end = match.span()
            i = bisect.bisect_right(token_starts, start) - 1
            if (i >= 0 and start < token_spans[i][1]) or (
                i + 1 < len(token_spans) and token_spans[i + 1][0] < end
            ):
                continue  # never inside an existing token
            span = match.group()
            token = self._folded.get(_normalize_value(span)) or self._exact.get(_significant(span))
            if token is None:
                continue
            parts.append(text[pos:start])
            parts.append(token)
            pos = end
        if not parts:
            return text
        parts.append(text[pos:])
        return "".join(parts)


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
