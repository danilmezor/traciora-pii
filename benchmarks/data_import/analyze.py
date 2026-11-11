"""Compute audit statistics over a stream of NormalizedExample rows.

The output is a single :class:`Analysis` dataclass that ``report.py`` turns
into Markdown. Two design rules:

1. Streaming-friendly: we accumulate counters in one pass; we never call
   ``list(stream)`` if it can be avoided. This matters for large HF splits
   such as Nemotron-PII.
2. Deterministic sample selection: the report shows representative and
   "interesting" (max-span-count, malformed) examples. We pick those by
   running statistic, not by RNG, so two runs on the same data produce
   byte-identical reports.
"""
from __future__ import annotations

import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .loaders import NormalizedExample, Span


# A span is "value-mismatched" if text[start:end] != value. The dataset
# claims a span at (start, end) with a given value; if slicing the text
# disagrees, either the offsets or the value is wrong.
@dataclass
class QualityCounts:
    rows: int = 0
    rows_with_no_spans: int = 0
    rows_with_overlapping_spans: int = 0
    rows_with_value_mismatch: int = 0
    rows_with_out_of_bounds_span: int = 0
    rows_with_dropped_spans: int = 0  # set by loader if span dict was malformed
    rows_with_non_nfc_text: int = 0
    rows_with_control_chars: int = 0
    total_spans: int = 0
    total_value_mismatches: int = 0
    total_out_of_bounds: int = 0
    total_overlap_pairs: int = 0


@dataclass
class Analysis:
    source: str
    locale: str
    n_rows: int
    label_counts: Counter
    label_examples: dict[str, str]  # label -> one example value
    text_len_chars: dict[str, float]  # min/p25/p50/p75/p95/max/mean
    text_len_words: dict[str, float]
    spans_per_row: dict[str, float]
    span_len_chars: dict[str, float]
    quality: QualityCounts
    samples: list[NormalizedExample] = field(default_factory=list)
    # Reservoir of "weird" examples: max-span-count, smallest, with-mismatch.
    flagged: dict[str, NormalizedExample] = field(default_factory=dict)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: 0.0 for k in ("min", "p25", "p50", "p75", "p95", "max", "mean")}
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {
        "min": s[0],
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def _spans_overlap(a: Span, b: Span) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def _row_overlap_count(spans: list[Span]) -> int:
    """Count overlapping pairs (O(n^2) is fine: PII rows have <50 spans)."""
    pairs = 0
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            if _spans_overlap(a, b):
                pairs += 1
    return pairs


def _has_control_chars(text: str) -> bool:
    """Detect non-whitespace control characters (Cc category, excluding \\t\\n\\r)."""
    return any(unicodedata.category(c) == "Cc" and c not in "\t\n\r" for c in text)


def analyze(
    stream: Iterable[NormalizedExample],
    *,
    source: str,
    locale: str,
    n_random_samples: int = 5,
) -> Analysis:
    label_counts: Counter[str] = Counter()
    label_examples: dict[str, str] = {}
    text_chars: list[float] = []
    text_words: list[float] = []
    spans_per: list[float] = []
    span_lens: list[float] = []
    q = QualityCounts()

    samples: list[NormalizedExample] = []
    flagged: dict[str, NormalizedExample] = {}
    max_spans_seen = -1

    for i, ex in enumerate(stream):
        q.rows += 1
        text = ex.text
        text_chars.append(len(text))
        text_words.append(len(text.split()))
        spans_per.append(len(ex.spans))

        if not ex.spans:
            q.rows_with_no_spans += 1
            flagged.setdefault("no_spans", ex)

        if ex.meta.get("n_dropped_spans", 0):
            q.rows_with_dropped_spans += 1
            flagged.setdefault("dropped_spans", ex)

        # NFC normalization check: if NFC(text) != text the corpus is not
        # consistently normalized, which trips downstream tokenizer-based
        # NER detectors.
        if unicodedata.normalize("NFC", text) != text:
            q.rows_with_non_nfc_text += 1
            flagged.setdefault("non_nfc", ex)

        if _has_control_chars(text):
            q.rows_with_control_chars += 1
            flagged.setdefault("control_chars", ex)

        # Per-row span analysis.
        row_mismatch = 0
        row_oob = 0
        for sp in ex.spans:
            label_counts[sp.label] += 1
            label_examples.setdefault(sp.label, sp.value)
            span_lens.append(max(0, sp.end - sp.start))

            if sp.start < 0 or sp.end > len(text) or sp.start >= sp.end:
                row_oob += 1
                continue

            sliced = text[sp.start : sp.end]
            if sliced != sp.value:
                row_mismatch += 1

        if row_mismatch:
            q.rows_with_value_mismatch += 1
            q.total_value_mismatches += row_mismatch
            flagged.setdefault("value_mismatch", ex)
        if row_oob:
            q.rows_with_out_of_bounds_span += 1
            q.total_out_of_bounds += row_oob
            flagged.setdefault("out_of_bounds", ex)

        overlap_pairs = _row_overlap_count(ex.spans)
        if overlap_pairs:
            q.rows_with_overlapping_spans += 1
            q.total_overlap_pairs += overlap_pairs
            flagged.setdefault("overlap", ex)

        if len(ex.spans) > max_spans_seen:
            max_spans_seen = len(ex.spans)
            flagged["max_spans"] = ex

        # Deterministic spread sample: take every 199th row, capped at
        # n_random_samples. Stable across reruns; spreads samples through
        # the stream rather than clustering at the head.
        if i % 199 == 0 and len(samples) < n_random_samples:
            samples.append(ex)

    q.total_spans = sum(label_counts.values())

    return Analysis(
        source=source,
        locale=locale,
        n_rows=q.rows,
        label_counts=label_counts,
        label_examples=label_examples,
        text_len_chars=_summary(text_chars),
        text_len_words=_summary(text_words),
        spans_per_row=_summary(spans_per),
        span_len_chars=_summary(span_lens),
        quality=q,
        samples=samples,
        flagged=flagged,
    )
