"""Third-party PII dataset import + analysis.

Pulls public PII benchmarks from Hugging Face, normalizes them to a common
in-memory schema, and emits per-dataset audit reports (label distribution,
text/span length stats, span-quality checks).

Used to evaluate ``traciora_pii`` on data not produced by ``data_generator``,
addressing the "trained on the test set" reviewer concern. The legality
constraint on real-world PII corpora is acknowledged: every dataset here is
itself synthetic, but produced independently of this project.

Sources:
* nvidia/Nemotron-PII (en/us) - 200k synthetic English rows, stringified spans.
* wolframko/russian-pii-66k (ru) - 65k synthetic Russian rows, span-only.

Schema after normalization::

    NormalizedExample(
        id: str,
        locale: str,
        text: str,
        spans: list[Span(label, value, start, end)],
        source: str,            # "nemotron" | "wolframko"
        meta: dict,             # arbitrary source-specific fields
    )
"""
from __future__ import annotations

from .loaders import (
    NormalizedExample,
    Span,
    load_nemotron_pii,
    load_wolframko_ru,
)

__all__ = [
    "NormalizedExample",
    "Span",
    "load_nemotron_pii",
    "load_wolframko_ru",
]
