"""
First-pass regex PII detection.

The default policy is loaded lazily on first use and cached for the process.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PIIConfig
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ._latency import time_detector


@dataclass(frozen=True)
class PIIDetector:
    name: str
    pattern: re.Pattern
    placeholder: str
    keywords: tuple[str, ...] = ()
    lookaround_chars: int = 40
    # Optional post-match validator. When set, a regex hit is only kept
    # if `validator(matched_text)` returns True. Used to apply structural
    # checks (e.g. Luhn for credit cards) that cannot be expressed in regex.
    validator: Callable[[str], bool] | None = None

    def is_keyword_anchored(self) -> bool:
        return bool(self.keywords)


@dataclass(frozen=True)
class PIIFinding:
    detector: str
    start: int
    end: int
    placeholder: str


# The fixed groups identify the UUID even when a neighboring phone starts or ends in a digit.
_UUID_BODY = r"(?<![a-f])[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
_UUID = re.compile(rf"{_UUID_BODY}(?![a-f])", re.IGNORECASE | re.ASCII)
_UUID_BEFORE_IBAN = re.compile(
    rf"(?P<uuid>{_UUID_BODY})(?P<iban>[a-f][a-z]\d{{2}}[a-z0-9]{{10,30}})(?!\w)",
    re.IGNORECASE | re.ASCII,
)
_GROUPED_CARD_START = re.compile(r"[^\W\d]\d{4}[ -]")
_GROUPED_CARD_END = re.compile(r"\d{4}[ -]\d{4}[^\W\d]")


def _blank_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    for start, end in reversed(list(spans)):
        text = text[:start] + "#" * (end - start) + text[end:]
    return text


def _uuid_spans(text: str) -> list[tuple[int, int]]:
    if "-" not in text:
        return []
    spans = [match.span() for match in _UUID.finditer(text)]
    # A neighboring valid IBAN can start with a hex letter. Keep that boundary
    # without treating a longer hex identifier as a UUID.
    for match in _UUID_BEFORE_IBAN.finditer(_blank_spans(text, spans)):
        iban = match["iban"].upper()
        digits = "".join(ch if ch.isdigit() else str(ord(ch) - ord("A") + 10) for ch in iban[4:] + iban[:4])
        if int(digits) % 97 == 1:
            spans.append(match.span("uuid"))
    return sorted(spans)

def _without_refs(text: str) -> str:
    """Keep offsets and boundaries when a neighboring value has already become a vault token."""
    if "_ref:" not in text:
        return text
    from .pii_vault import REF_TOKEN_PATTERN
    return REF_TOKEN_PATTERN.sub(lambda match: "x" * len(match.group()), text)


_DEFAULT_DETECTORS_CACHE: list[PIIDetector] | None = None

# Zero-width characters split an identifier for the patterns
# (`+7 495 0<U+200B>14 2233`) without changing what a reader sees.
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\ufeff")


def _get_default_detectors() -> list[PIIDetector]:
    """Lazy-load and cache the default regex policy's compiled detectors.

    Reads ``detector.regex_policies`` (list, preferred for multi-locale
    coverage) or ``detector.regex_policy`` (single-policy form) from the
    runtime config. Cached for the process; call
    :func:`reset_default_detectors_cache` after a config switch in tests.
    """
    global _DEFAULT_DETECTORS_CACHE
    if _DEFAULT_DETECTORS_CACHE is None:
        from .policies.loader import (
            compile_multi_regex_policy,
            compile_regex_policy,
            load_regex_policy,
        )
        try:
            from .config import load_config
            cfg = load_config()
            policies = cfg.detector.regex_policies
            policy = cfg.detector.regex_policy
        except Exception:  # noqa: BLE001 — config layer may not be importable
            policies = None
            policy = "ru"
        if policies:
            _DEFAULT_DETECTORS_CACHE = compile_multi_regex_policy(policies)
        else:
            _DEFAULT_DETECTORS_CACHE = compile_regex_policy(
                load_regex_policy(policy or "ru")
            )
    return _DEFAULT_DETECTORS_CACHE


def reset_default_detectors_cache() -> None:
    """Drop the cached default detectors. Next call reloads from policy.

    Useful in tests that mutate ``PII_REGEX_POLICY*`` env vars between
    cases. Not part of the runtime hot path.
    """
    global _DEFAULT_DETECTORS_CACHE
    _DEFAULT_DETECTORS_CACHE = None


def _keyword_ok(text: str, finding_start: int, finding_end: int, detector: PIIDetector) -> bool:
    if not detector.is_keyword_anchored():
        return True
    start = max(0, finding_start - detector.lookaround_chars)
    end = min(len(text), finding_end + detector.lookaround_chars)
    snippet = text[start:end].lower()
    return any(keyword in snippet for keyword in detector.keywords)


@lru_cache(maxsize=16)
def _compile_policies(specs: tuple[str, ...]) -> tuple[PIIDetector, ...]:
    from .policies.loader import compile_multi_regex_policy
    return tuple(compile_multi_regex_policy(specs))


def configured_detectors(config: PIIConfig) -> tuple[PIIDetector, ...]:
    return _compile_policies(tuple(config.detector.regex_policies or
                                   [config.detector.regex_policy or "ru"]))


def detect_pii(
    text: str, detectors: Iterable[PIIDetector] | None = None, *,
    excluded_categories: Iterable[str] | None = None,
    config: PIIConfig | None = None,
) -> list[PIIFinding]:
    if not text:
        return []

    from ._masking_policy import category_is_excluded, resolve_exclusions
    excluded = resolve_exclusions(
        config.detector.excluded_categories
        if config is not None and excluded_categories is None else excluded_categories
    )

    with time_detector("regex"):
        # Match on a copy without zero-width characters; `kept[i]` is the
        # original index of the copy's i-th character, so spans point into
        # *text* and cover any zero-width characters inside the value.
        if _ZERO_WIDTH.isdisjoint(text):
            scan, kept = text, None
        else:
            kept = [i for i, ch in enumerate(text) if ch not in _ZERO_WIDTH]
            scan = "".join(text[i] for i in kept)
        # A numeric matcher can consume the end of an ID and a real neighboring
        # value in one match. Whole emails and URLs may legitimately contain IDs.
        refs_blanked = _without_refs(scan)
        identifiers = _uuid_spans(refs_blanked)
        # Keep word boundaries on both sides without supplying a phone or card separator.
        uuid_scan = _blank_spans(scan, identifiers)
        phone_scan = _blank_spans(refs_blanked, identifiers)
        # A grouped card can start after a one-character join between it and
        # a UUID. Keep that character literal and validate the card as usual.
        card_joins = [(end, end + 1) for _, end in identifiers if _GROUPED_CARD_START.match(scan, end)]
        card_joins.extend((start - 1, start) for start, _ in identifiers
                          if _GROUPED_CARD_END.fullmatch(scan, max(0, start - 10), start))
        card_scan = _blank_spans(uuid_scan, card_joins)
        findings: list[PIIFinding] = []
        active = (list(detectors) if detectors is not None else
                  configured_detectors(config) if config is not None else _get_default_detectors())
        for detector in active:
            if category_is_excluded(detector.placeholder, excluded):
                continue
            embeds_ids = detector.placeholder in ("[EMAIL]", "[URL]")
            if embeds_ids:
                source = scan
            elif detector.placeholder == "[PHONE]":
                source = phone_scan
            elif detector.placeholder == "[CARD]":
                source = card_scan
            else:
                source = uuid_scan
            for match in detector.pattern.finditer(source):
                start, end = match.span()
                if not embeds_ids and any(start < right and left < end for left, right in identifiers):
                    continue
                if not _keyword_ok(scan, start, end, detector):
                    continue
                if detector.validator is not None and not detector.validator(scan[start:end]):
                    continue
                if kept is not None:
                    start, end = kept[start], kept[end - 1] + 1
                findings.append(
                    PIIFinding(
                        detector=detector.name,
                        start=start,
                        end=end,
                        placeholder=detector.placeholder,
                    )
                )

        findings.sort(key=lambda item: (item.start, item.end))
        return findings


def redact_pii_text(
    text: str, detectors: Iterable[PIIDetector] | None = None, *,
    excluded_categories: Iterable[str] | None = None,
    config: PIIConfig | None = None,
) -> tuple[str, list[PIIFinding]]:
    if not text:
        return text, []

    findings = detect_pii(text, detectors=detectors, excluded_categories=excluded_categories, config=config)
    if not findings:
        return text, []

    # Merge overlaps and keep placeholder from the first detector in the merged span.
    merged: list[PIIFinding] = []
    for finding in findings:
        if not merged:
            merged.append(finding)
            continue
        last = merged[-1]
        if finding.start <= last.end:
            merged[-1] = PIIFinding(
                detector=last.detector,
                start=last.start,
                end=max(last.end, finding.end),
                placeholder=last.placeholder,
            )
        else:
            merged.append(finding)

    chunks: list[str] = []
    cursor = 0
    for finding in merged:
        chunks.append(text[cursor:finding.start])
        chunks.append(finding.placeholder)
        cursor = finding.end
    chunks.append(text[cursor:])
    return "".join(chunks), merged


_LONG_NUMBER = PIIDetector("long_number", re.compile(r"(?<![\w-])\d{10,}(?!\w)"), "[NUMBER]")


def scrub(text: str, *, config: PIIConfig | None = None) -> str:
    """Regex-only logging policy, independent of conversational exclusions."""
    if not text:
        return text
    detectors = configured_detectors(config) if config is not None else _get_default_detectors()
    return redact_pii_text(text, detectors=(*detectors, _LONG_NUMBER), excluded_categories=[])[0]
