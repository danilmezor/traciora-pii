"""Loaders that pull HF PII datasets and normalize them to a common schema.

The source datasets use slightly different span schemas:

* ``nvidia/Nemotron-PII`` stores stringified span dicts with
  ``{label, start, end, text}``.
* ``wolframko/russian-pii-66k`` stores ``{label, start, end, value}``
  inside a ``privacy_mask`` list.

Both are normalized into ``NormalizedExample`` / ``Span`` so downstream
audits and eval harnesses can share transform code.

Loaders return an iterator of :class:`NormalizedExample` so we can stream
through arbitrarily large splits without holding the entire dataset in
memory. Callers that need a list materialize via ``list(...)``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterator

try:  # pragma: no cover - import guard
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "benchmarks.data_import requires the 'datasets' library. "
        "Install with: pip install -e '.[data-import]'"
    ) from exc


AI4PRIVACY_REPO = "ai4privacy/pii-masking-openpii-1m"
NEMOTRON_REPO = "nvidia/Nemotron-PII"
WOLFRAMKO_REPO = "wolframko/russian-pii-66k"


@dataclass(frozen=True)
class Span:
    label: str
    value: str
    start: int
    end: int


@dataclass
class NormalizedExample:
    id: str
    locale: str
    text: str
    spans: list[Span]
    source: str
    meta: dict = field(default_factory=dict)


def _coerce_span(raw: dict) -> Span | None:
    """Best-effort coercion of an ai4privacy / wolframko span dict.

    Returns ``None`` if the row is missing required fields rather than
    raising - we want loader noise surfaced by ``analyze.py`` as a quality
    metric, not a hard crash.
    """
    try:
        return Span(
            label=str(raw["label"]),
            value=str(raw.get("value", "")),
            start=int(raw["start"]),
            end=int(raw["end"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _coerce_nemotron_span(raw: dict) -> Span | None:
    """Coerce a Nemotron span dict.

    Nemotron uses ``text`` for the span value, and some values are numeric
    scalars rather than strings. We stringify values so the downstream span
    invariant remains ``text[start:end] == span.value`` when the dataset
    value is exact.
    """
    try:
        return Span(
            label=str(raw["label"]),
            value=str(raw.get("text", "")),
            start=int(raw["start"]),
            end=int(raw["end"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_ai4privacy(
    locale: str,
    *,
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
) -> Iterator[NormalizedExample]:
    """Stream rows from ai4privacy/pii-masking-openpii-1m for a given locale.

    The HF dataset stores its language code in a ``language`` column. We
    filter inside the iterator (HF's filter() works on streaming datasets
    but materializes a generator anyway, so we just inline it).
    """
    if locale not in {"en", "de"}:
        raise ValueError(
            f"ai4privacy locale must be one of {{'en','de'}} for this project, got {locale!r}"
        )

    ds = load_dataset(AI4PRIVACY_REPO, split=split, streaming=streaming)
    yielded = 0
    for i, row in enumerate(ds):
        if row.get("language") != locale:
            continue
        if limit is not None and yielded >= limit:
            break

        spans_raw = row.get("privacy_mask") or []
        spans = [s for s in (_coerce_span(r) for r in spans_raw) if s is not None]
        yield NormalizedExample(
            id=f"ai4p-{locale}-{row.get('uid', i)}",
            locale=locale,
            text=row.get("source_text", ""),
            spans=spans,
            source="ai4privacy",
            meta={
                "region": row.get("region"),
                "script": row.get("script"),
                "split": row.get("split"),
                "n_dropped_spans": len(spans_raw) - len(spans),
            },
        )
        yielded += 1


def load_nemotron_pii(
    *,
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
) -> Iterator[NormalizedExample]:
    """Stream rows from nvidia/Nemotron-PII.

    The dataset is English / US-locale in practice (`locale == "us"` on
    sampled rows). Its ``spans`` column is stored as a stringified Python
    list, not a list column, so the loader parses it with ``ast.literal_eval``.
    """
    ds = load_dataset(NEMOTRON_REPO, split=split, streaming=streaming)
    yielded = 0
    for i, row in enumerate(ds):
        if limit is not None and yielded >= limit:
            break

        spans_raw_value = row.get("spans") or "[]"
        try:
            spans_raw = ast.literal_eval(spans_raw_value)
        except (SyntaxError, ValueError):
            spans_raw = []
            parse_failed = True
        else:
            parse_failed = False

        spans = [s for s in (_coerce_nemotron_span(r) for r in spans_raw) if s is not None]
        yield NormalizedExample(
            id=f"nemotron-en-{row.get('uid', i)}",
            locale="en",
            text=row.get("text", ""),
            spans=spans,
            source="nemotron",
            meta={
                "hf_locale": row.get("locale"),
                "domain": row.get("domain"),
                "document_type": row.get("document_type"),
                "document_format": row.get("document_format"),
                "split": split,
                "span_parse_failed": parse_failed,
                "n_dropped_spans": len(spans_raw) - len(spans) if not parse_failed else 1,
            },
        )
        yielded += 1


def load_wolframko_ru(
    *,
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
) -> Iterator[NormalizedExample]:
    """Stream rows from wolframko/russian-pii-66k.

    Schema mirrors ai4privacy's ``privacy_mask`` shape; locale is hardcoded
    'ru' on every row.
    """
    ds = load_dataset(WOLFRAMKO_REPO, split=split, streaming=streaming)
    yielded = 0
    for i, row in enumerate(ds):
        if limit is not None and yielded >= limit:
            break

        spans_raw = row.get("privacy_mask") or []
        spans = [s for s in (_coerce_span(r) for r in spans_raw) if s is not None]
        yield NormalizedExample(
            id=f"wolframko-ru-{i}",
            locale="ru",
            text=row.get("source_text", ""),
            spans=spans,
            source="wolframko",
            meta={
                "language": row.get("language"),
                "locale_field": row.get("locale"),
                "n_dropped_spans": len(spans_raw) - len(spans),
            },
        )
        yielded += 1
