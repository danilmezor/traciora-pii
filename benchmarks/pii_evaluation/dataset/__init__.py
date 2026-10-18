"""Trilingual evaluation dataset for the PII virtualization framework.

The dataset is a *parallel trilingual* (EN / DE / RU) suite for
PII evaluation. Each scenario is authored once as a language-agnostic
skeleton in ``scenarios/<bucket>.yaml`` and rendered into three
near-translation ``EvalQuery`` instances sharing a ``scenario_id`` —
only the language and locale-specific PII type vary across the triplet.

Public surface
--------------

  - ``PIIGroundTruth``    — one span of ground-truth PII inside a user turn
  - ``ExpectedToolCall``  — one expected tool invocation
  - ``InjectedPII``       — PII that a tool *result* introduces and that a
                            later turn must reference (cross-turn-taint)
  - ``BoundaryPIIValue``  — raw PII value that could cross the model
                            boundary and must be counted independently of
                            detector success
  - ``EvalQuery``         — one evaluation query, possibly multi-turn
  - ``load_dataset()``    — returns the rendered EvalQuery tuple
  - ``load_fixtures()``   — returns the per-scenario synthetic CRM records
                            that the tools should be primed with before
                            running the experiment

Determinism & reproducibility
-----------------------------

By default ``load_dataset()`` reads the frozen JSONL at
``data/dataset_v1.jsonl``. To re-render from YAML use
``python -m benchmarks.pii_evaluation.dataset.render --freeze``; to
verify reproducibility use ``--check`` to compare against the committed
``data/dataset_v1.sha256``.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PIIGroundTruth:
    """A single PII span the detector is expected to find.

    ``placeholder`` is the canonical bracketed type (e.g. ``"[PHONE]"``)
    matching what the production detector library emits.
    ``value`` is the literal substring as it appears in the user turn.
    ``turn`` is the zero-based index of the user turn it appears in.

    Scoring uses ``placeholder`` + span overlap rather than detector-name
    + exact-value equality, because the production detectors run on
    merged findings (see ``redact_pii_text`` in ``pii.py``) and because
    some patterns (notably ``us_phone``) capture a strict substring of
    the user-typed value.
    """
    placeholder: str
    value: str
    turn: int = 0


# An immutable empty mapping used as the default for ExpectedToolCall.arguments
# so that frozen dataclasses can share a hashable, type-correct singleton.
_EMPTY_ARGS: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class ExpectedToolCall:
    """One expected tool invocation in a scenario.

    Multiple ``ExpectedToolCall`` objects in a query express a sequence
    of tool calls the model is expected to make across turns (e.g.
    lookup on turn 0, update on turn 1).

    ``arguments`` carries the *post-normalization-equivalent* literal
    values the tool should receive, used by the
    ``argument_exact_match`` metric to score full argument fidelity
    beyond a single record id. Phone / email / digit values are
    compared after passing both sides through
    ``tools._normalize_phone`` / ``_normalize_email`` /
    ``_normalize_digits``, so the score is robust to formatting
    variation.
    """
    turn_index: int
    tool_name: str
    arguments: Mapping[str, str] = field(default_factory=lambda: _EMPTY_ARGS)
    expected_record_id: str = ""


@dataclass(frozen=True)
class InjectedPII:
    """PII that a tool *result* introduces, for cross-turn-taint cases.

    The tool returns a synthetic record whose
    ``from_record_field`` (e.g. ``email``, ``phone``, ``address``)
    carries fresh PII the model is expected to reference in
    ``must_appear_in_turn`` (a later user turn). This stresses three
    things at once: vault-token rehydration on inbound tool results,
    leak-guard fidelity on PII the model reads from a tool, and
    cross-turn vault-token stability when the same PII is re-tokenized
    on the follow-up turn.

    ``turn_index`` is the turn at which the producing tool runs.
    ``value`` is the literal that will appear both in the canned record
    returned by the tool *and* in the user text of
    ``must_appear_in_turn``. The renderer asserts both invariants when
    materializing the dataset.
    """
    turn_index: int
    placeholder: str
    value: str
    must_appear_in_turn: int


@dataclass(frozen=True)
class BoundaryPIIValue:
    """A raw PII value that must not cross the external-model boundary.

    This includes explicit user-turn spans, PII introduced by tool
    results, and PII-bearing fields in the fixture records a scenario's
    expected tool calls return. Exposure scoring uses these literals
    directly instead of re-running a detector, so detector misses cannot
    turn into false-zero privacy results.
    """
    placeholder: str
    value: str
    source: str                                  # "user_turn" | "injected_pii" | "fixture_result"
    record_id: Optional[str] = None


@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query, possibly multi-turn.

    Trilingual scenarios are joined across languages by
    ``scenario_id``; each scenario_id materializes to exactly three
    ``EvalQuery`` instances (``language ∈ {en, de, ru}``) sharing the
    same ``bucket``, ``workflow``, turn count, expected_tool_calls
    signature, and placeholder multiset.
    """
    id: str
    language: str                                       # "en" | "de" | "ru"
    bucket: str                                         # for grouping in the report
    turns: Tuple[str, ...]                              # user turns; assistant turns are produced by the loop
    pii_spans: Tuple[PIIGroundTruth, ...]               # ground truth across all turns
    scenario_id: str = ""                               # joins EN/DE/RU triplets
    workflow: str = ""                                  # workflow archetype
    expected_tool_calls: Tuple[ExpectedToolCall, ...] = ()
    injected_pii: Tuple[InjectedPII, ...] = ()
    boundary_pii_values: Tuple[BoundaryPIIValue, ...] = ()
    # Optional subset of tool names available to the agent for this query.
    # If empty, the agent sees the full toolset. Used by security-stress
    # scenarios (e.g. scn-sec-750) where the test requires routing to a
    # specific tool that would otherwise lose against a "normal" alternative.
    available_tools: Tuple[str, ...] = ()

    @property
    def effective_tool_calls(self) -> Tuple[ExpectedToolCall, ...]:
        """Tool-call sequence to score against."""
        return self.expected_tool_calls

    @property
    def effective_scenario_id(self) -> str:
        """Stable scenario id; falls back to ``id`` for legacy entries."""
        return self.scenario_id or self.id


# ----------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------

_FROZEN_JSONL = Path(__file__).resolve().parent / "data" / "dataset_v1.jsonl"


def load_dataset(*, regenerate: bool = False) -> Tuple[EvalQuery, ...]:
    """Public accessor — returns the trilingual evaluation suite.

    By default reads the frozen JSONL at ``data/dataset_v1.jsonl`` to
    guarantee paper reproducibility. Pass ``regenerate=True`` to
    re-render from the source YAMLs (used by tests and by ``--freeze``).

    The dataset is registered with the synthetic CRM tool store as a
    side-effect: ``benchmarks.pii_evaluation.tools.register_fixtures``
    is called so that ``customer_lookup_by_phone`` etc. find the
    per-scenario records the renderer materialized.
    """
    queries, fixtures = _load_internal(regenerate=regenerate)
    if fixtures:
        # Importing tools here avoids a circular import at module load.
        from benchmarks.pii_evaluation.tools import register_fixtures
        register_fixtures(fixtures)
    return tuple(queries)


def load_fixtures(*, regenerate: bool = False) -> Tuple:
    """Return the per-scenario synthetic records (without registering)."""
    _, fixtures = _load_internal(regenerate=regenerate)
    return tuple(fixtures)


def _load_internal(*, regenerate: bool) -> Tuple[List[EvalQuery], List]:
    if regenerate or not _FROZEN_JSONL.exists():
        # Re-render from YAML. Importing here keeps top-level import light.
        from benchmarks.pii_evaluation.dataset.render import render_all
        return render_all()
    from benchmarks.pii_evaluation.dataset.render import load_frozen
    return load_frozen(_FROZEN_JSONL)


__all__ = [
    "EvalQuery",
    "ExpectedToolCall",
    "InjectedPII",
    "BoundaryPIIValue",
    "PIIGroundTruth",
    "load_dataset",
    "load_fixtures",
]
