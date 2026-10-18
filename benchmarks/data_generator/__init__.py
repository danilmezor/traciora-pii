"""Slot-template PII benchmark generator.

Reads per-locale seed YAML files (``seeds/<locale>/*.yaml``), renders
slot-templates to produce labeled examples, optionally applies adversarial
noise, and emits JSONL or ``EvalQuery`` records.

Public API:

    from benchmarks.data_generator import Pipeline, load_seed_bundle

    pipeline = Pipeline(seeds=load_seed_bundle("ru"), seed=42)
    for example in pipeline.run(100):
        ...

Adding a locale: drop a new ``seeds/<locale>/`` directory; no Python edit
required as long as the slots used appear in ``_SLOT_METHODS`` in
``injectors.py``.
"""
from __future__ import annotations

from .injectors import SlotFillEngine, Span
from .noise import NoiseApplier, NoiseConfig
from .pipeline import (
    GeneratedExample,
    Pipeline,
    split_examples,
    to_eval_query,
    write_jsonl,
)
from .seeds import SeedBundle, load_all_locales, load_seed_bundle, load_shared_seeds

__all__ = [
    "GeneratedExample",
    "NoiseApplier",
    "NoiseConfig",
    "Pipeline",
    "SeedBundle",
    "SlotFillEngine",
    "Span",
    "load_all_locales",
    "load_seed_bundle",
    "load_shared_seeds",
    "split_examples",
    "to_eval_query",
    "write_jsonl",
]
