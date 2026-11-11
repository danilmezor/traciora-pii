"""Datasheet emitter for generated benchmark artifacts.

Produces a per-locale Markdown datasheet that documents how a benchmark
was generated, what it contains, and how it should (or should not) be
used. Follows the structure from Gebru et al., "Datasheets for Datasets"
(2018) — Motivation, Composition, Collection, Preprocessing, Uses,
Distribution, Maintenance.

Dataset users consume this directly. Keep the rendered output stable so two runs against the
same metadata diff cleanly.
"""
from __future__ import annotations

import collections
import datetime
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .pipeline import GeneratedExample


# ======================================================================
# Metadata record
# ======================================================================


@dataclass
class DatasheetMetadata:
    """Frozen metadata snapshot of one benchmark build.

    Populated by ``build_release`` after a Pipeline run completes; passed
    to ``render_datasheet`` to produce the user-facing Markdown.
    """

    name: str                                # "traciora_pii-bench-ru"
    version: str                             # "v1"
    license: str                             # "CC-BY-4.0"
    git_sha: str                             # commit-pinned generator version
    seed: int
    locale: str
    scaffold_sources: List[Dict[str, str]] = field(default_factory=list)
    placeholder_coverage: Dict[str, int] = field(default_factory=dict)
    profile_distribution: Dict[str, int] = field(default_factory=dict)
    n_train: int = 0
    n_dev: int = 0
    n_test: int = 0
    spacy_scrubbing_enabled: bool = False
    noise_config: Optional[Dict[str, float]] = None
    generation_timestamp: str = ""


# ======================================================================
# Public API
# ======================================================================


def compute_coverage(examples: Iterable[GeneratedExample]) -> Dict[str, int]:
    """Count how many gold spans were emitted per placeholder."""
    counts: collections.Counter = collections.Counter()
    for ex in examples:
        for span in ex.spans:
            counts[span.placeholder] += 1
    return dict(counts)


def compute_profile_distribution(examples: Iterable[GeneratedExample]) -> Dict[str, int]:
    """Count how often each profile was sampled."""
    counts: collections.Counter = collections.Counter(ex.profile for ex in examples)
    return dict(counts)


def detect_git_sha() -> str:
    """Best-effort ``git rev-parse HEAD``; empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return ""


def render_datasheet(meta: DatasheetMetadata) -> str:
    """Return a Markdown datasheet for ``meta``.

    Sections are ordered to match the Gebru et al. (2018) outline. Output
    is deterministic given identical metadata so two runs produce the same
    bytes.
    """
    sections = [
        _header(meta),
        _motivation(meta),
        _composition(meta),
        _collection(meta),
        _preprocessing(meta),
        _uses(meta),
        _distribution(meta),
        _maintenance(meta),
    ]
    return "\n\n".join(s.rstrip() for s in sections) + "\n"


# ======================================================================
# Section renderers
# ======================================================================


def _header(meta: DatasheetMetadata) -> str:
    ts = meta.generation_timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return (
        f"# {meta.name} {meta.version}\n"
        f"\n"
        f"Datasheet for the {meta.locale.upper()} synthetic-PII benchmark "
        f"shipped alongside the `traciora_pii` framework. Format follows "
        f"Gebru et al. (2018), \"Datasheets for Datasets\".\n"
        f"\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Name | `{meta.name}` |\n"
        f"| Version | `{meta.version}` |\n"
        f"| Locale | `{meta.locale}` |\n"
        f"| License | `{meta.license}` |\n"
        f"| Generation seed | `{meta.seed}` |\n"
        f"| Generator commit | `{meta.git_sha or 'unknown'}` |\n"
        f"| Generated on | {ts} |\n"
        f"| Train / dev / test | {meta.n_train} / {meta.n_dev} / {meta.n_test} |\n"
        f"| spaCy scrubbing | `{meta.spacy_scrubbing_enabled}` |\n"
    )


def _motivation(meta: DatasheetMetadata) -> str:
    return (
        f"## Motivation\n"
        f"\n"
        f"Real personally identifiable information cannot be released by "
        f"definition. This benchmark addresses the resulting evaluation "
        f"gap with a hybrid approach: real linguistic context is drawn "
        f"from independent published corpora (see *Collection process* "
        f"below), while every PII identifier is synthetic but format- "
        f"and checksum-correct. The result is naturalistic prose that "
        f"never carries a real subject's data.\n"
        f"\n"
        f"This artifact is intended to support: (a) evaluation of "
        f"locale-aware PII detectors, (b) cross-source generalization "
        f"studies (independent of the generator that produced "
        f"`ai4privacy/pii-masking-openpii-1m`, "
        f"`nvidia/Nemotron-PII`, or other published synthetic sets), "
        f"and (c) reproducibility of the numbers in the accompanying paper."
    )


def _composition(meta: DatasheetMetadata) -> str:
    cov_lines = "\n".join(
        f"| `{ph}` | {n} |"
        for ph, n in sorted(meta.placeholder_coverage.items(), key=lambda kv: -kv[1])
    ) or "| (no spans) | 0 |"
    profile_lines = "\n".join(
        f"| `{p}` | {n} |"
        for p, n in sorted(meta.profile_distribution.items(), key=lambda kv: -kv[1])
    ) or "| (no profiles) | 0 |"
    return (
        f"## Composition\n"
        f"\n"
        f"Each instance is one labeled sentence with character-offset "
        f"gold spans. Format: JSONL with the schema declared in "
        f"`benchmarks.data_generator.pipeline.GeneratedExample`. The "
        f"invariant `text[span.start:span.end] == span.value` holds for "
        f"every emitted span and is enforced by the generator's smoke "
        f"tests.\n"
        f"\n"
        f"### Placeholder coverage\n"
        f"\n"
        f"| Placeholder | Span count |\n"
        f"|---|---|\n"
        f"{cov_lines}\n"
        f"\n"
        f"### Profile distribution\n"
        f"\n"
        f"| Profile | Examples |\n"
        f"|---|---|\n"
        f"{profile_lines}"
    )


def _collection(meta: DatasheetMetadata) -> str:
    if meta.scaffold_sources:
        rows = "\n".join(
            f"| `{s.get('id', '')}` | `{s.get('config', '')}` | `{s.get('license', '')}` |"
            for s in meta.scaffold_sources
        )
        sources_block = (
            f"| Source | Config | License |\n"
            f"|---|---|---|\n"
            f"{rows}\n"
        )
    else:
        sources_block = (
            "Scaffold source: hand-curated stub sentences shipped with the "
            "generator (no external dataset was streamed).\n"
        )
    return (
        f"## Collection process\n"
        f"\n"
        f"Scaffold sentences were drawn from the sources below. The "
        f"generator filters them with `filters.is_clean_prose` to drop "
        f"markup, list items, and short fragments, then injects synthetic "
        f"PII via locale-specific templates declared in "
        f"`injectors.py`. Identifiers are produced by deterministic "
        f"checksum-correct algorithms in `faker_providers.py` (see "
        f"`README.md` for the standards cited per identifier type).\n"
        f"\n"
        f"{sources_block}"
    )


def _preprocessing(meta: DatasheetMetadata) -> str:
    noise_block = "Adversarial noise: **disabled**."
    if meta.noise_config:
        rates = ", ".join(f"`{k}`={v}" for k, v in meta.noise_config.items())
        noise_block = f"Adversarial noise applied: {rates}."
    return (
        f"## Preprocessing / cleaning / labeling\n"
        f"\n"
        f"* Sentence segmentation: regex-based; replaceable with razdel/"
        f"spaCy in a future revision.\n"
        f"* Quality filter: length 30–280 chars, ≥1 finite verb (locale-"
        f"aware heuristic), no markup or URL leakage.\n"
        f"* Pre-injection scrubbing: "
        f"{'enabled' if meta.spacy_scrubbing_enabled else 'disabled'}. "
        f"When enabled, spaCy NER is run over the scaffold and PERSON / "
        f"address-shaped LOC entities are replaced with locale-appropriate "
        f"synthetic equivalents; gold spans for those replacements are "
        f"emitted alongside injected spans.\n"
        f"* {noise_block}\n"
        f"* Span invariant: enforced by the smoke test "
        f"`test_span_invariant`."
    )


def _uses(meta: DatasheetMetadata) -> str:
    return (
        f"## Uses\n"
        f"\n"
        f"### Intended\n"
        f"\n"
        f"* Evaluation of regex- and ML-based PII detectors across the "
        f"declared placeholder taxonomy.\n"
        f"* Cross-generator generalization studies. Train on "
        f"`ai4privacy/pii-masking-openpii-1m` or similar, evaluate here.\n"
        f"* Stress evaluation under adversarial perturbation when paired "
        f"with `NoiseConfig`-enabled runs.\n"
        f"\n"
        f"### Not intended\n"
        f"\n"
        f"* Training a detector that will then be evaluated on this same "
        f"benchmark. The synthetic-ID distributions are shaped by the "
        f"generator's own checksums; co-evaluating the trainer on the "
        f"trainer's distribution overstates real-world performance.\n"
        f"* Any inference about real individuals. All identifiers are "
        f"random; collisions with a real person are accidental.\n"
        f"* Pseudonymization of real PII. Use the upstream framework's "
        f"vault for that — this artifact is detection-only ground truth."
    )


def _distribution(meta: DatasheetMetadata) -> str:
    return (
        f"## Distribution\n"
        f"\n"
        f"Released under `{meta.license}`. Reproducible from seed "
        f"`{meta.seed}` and generator commit `{meta.git_sha or 'unknown'}` "
        f"via `python -m benchmarks.data_generator.build_release`. The "
        f"accompanying `MANIFEST.json` lists sha256 of every file in the "
        f"release; verify before consuming."
    )


def _maintenance(meta: DatasheetMetadata) -> str:
    return (
        f"## Maintenance\n"
        f"\n"
        f"Versioning: incrementing `version` produces a new artifact set; "
        f"older versions are not retroactively patched. Generator-policy "
        f"contract drift is detected by `test_policy_catches_generator_output` "
        f"in CI, which gates merges that would silently invalidate "
        f"existing benchmark numbers."
    )
