"""
First-pass regex PII detection.

The default policy is loaded lazily on first use and cached for the process.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable, Iterable

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


_DEFAULT_DETECTORS_CACHE: list[PIIDetector] | None = None


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


def detect_pii(
    text: str, detectors: Iterable[PIIDetector] | None = None, *,
    excluded_categories: Iterable[str] | None = None,
) -> list[PIIFinding]:
    if not text:
        return []

    from ._masking_policy import category_is_excluded, resolve_exclusions
    excluded = resolve_exclusions(excluded_categories)

    with time_detector("regex"):
        findings: list[PIIFinding] = []
        active = list(detectors) if detectors is not None else _get_default_detectors()
        for detector in active:
            if category_is_excluded(detector.placeholder, excluded):
                continue
            for match in detector.pattern.finditer(text):
                start, end = match.span()
                if not _keyword_ok(text, start, end, detector):
                    continue
                if detector.validator is not None and not detector.validator(text[start:end]):
                    continue
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
) -> tuple[str, list[PIIFinding]]:
    if not text:
        return text, []

    findings = detect_pii(text, detectors=detectors, excluded_categories=excluded_categories)
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
