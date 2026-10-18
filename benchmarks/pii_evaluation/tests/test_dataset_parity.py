"""Parity, span-invariant, and determinism tests for the trilingual dataset.

Run with::

    pytest benchmarks/pii_evaluation/tests/

The tests exercise the renderer (not the frozen JSONL) so that adding a
new scenario to a YAML cannot silently break parity.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Mapping

import pytest

from benchmarks.pii_evaluation.dataset import EvalQuery
from benchmarks.pii_evaluation.dataset.render import (
    LOCALES,
    render_all,
    serialize,
)


@pytest.fixture(scope="module")
def rendered():
    queries, fixtures = render_all()
    return queries, fixtures


# ----------------------------------------------------------------------
# Parity: every scenario_id renders to all 3 locales
# ----------------------------------------------------------------------

def test_every_scenario_has_all_three_locales(rendered):
    queries, _fixtures = rendered
    by_scenario: defaultdict[str, set] = defaultdict(set)
    for q in queries:
        by_scenario[q.scenario_id].add(q.language)
    expected = set(LOCALES)
    missing = {sid: expected - langs for sid, langs in by_scenario.items() if langs != expected}
    assert not missing, f"scenarios missing locales: {missing}"


def test_triplets_share_bucket_and_workflow(rendered):
    queries, _fixtures = rendered
    by_scenario: defaultdict[str, list] = defaultdict(list)
    for q in queries:
        by_scenario[q.scenario_id].append(q)
    for sid, triplet in by_scenario.items():
        buckets = {q.bucket for q in triplet}
        workflows = {q.workflow for q in triplet}
        turn_counts = {len(q.turns) for q in triplet}
        call_count = {len(q.expected_tool_calls) for q in triplet}
        call_names = {tuple(c.tool_name for c in q.expected_tool_calls) for q in triplet}
        assert len(buckets) == 1, f"{sid}: mixed buckets {buckets}"
        assert len(workflows) == 1, f"{sid}: mixed workflows {workflows}"
        assert len(turn_counts) == 1, f"{sid}: mixed turn counts {turn_counts}"
        assert len(call_count) == 1, f"{sid}: mixed call counts {call_count}"
        assert len(call_names) == 1, f"{sid}: mixed tool names per call {call_names}"


_NATIONAL_ID_PLACEHOLDERS = {
    "[PERSONAL_ID]", "[DE_STEUER_ID]", "[RU_SNILS]", "[RU_INN]",
}


def _canonicalize_placeholder(p: str) -> str:
    """Normalize per-locale national-id placeholders to a single tag.

    Verify_national_id workflows produce a different concrete tag per
    locale (US_SSN / DE_STEUER_ID / RU_SNILS), but the workflow shape is
    identical across the triplet. Map all national-id tags to the
    sentinel ``[NATIONAL_ID]`` for the multiset parity check; concrete
    detection P/R is still scored against the locale-specific tag.
    """
    if p in _NATIONAL_ID_PLACEHOLDERS:
        return "[NATIONAL_ID]"
    return p


def test_triplets_share_placeholder_multiset(rendered):
    """Triplet placeholder multisets match after national-id canonicalization.

    Verify_national_id scenarios use locale-specific tags ([PERSONAL_ID]
    in EN/RU passport-or-SSN cases, [DE_STEUER_ID] in DE, [RU_SNILS] in
    RU SNILS cases) — the workflow shape is identical, only the concrete
    national-id type differs. We canonicalize all national-id tags to
    ``[NATIONAL_ID]`` before comparing.
    """
    queries, _fixtures = rendered
    by_scenario: defaultdict[str, list] = defaultdict(list)
    for q in queries:
        by_scenario[q.scenario_id].append(q)
    for sid, triplet in by_scenario.items():
        multisets = {
            frozenset(
                Counter(_canonicalize_placeholder(s.placeholder) for s in q.pii_spans).items()
            )
            for q in triplet
        }
        assert len(multisets) == 1, (
            f"{sid}: placeholder multisets differ across locales — "
            f"{[Counter(_canonicalize_placeholder(s.placeholder) for s in q.pii_spans) for q in triplet]}"
        )


# ----------------------------------------------------------------------
# Span invariant: every PIIGroundTruth.value appears in its turn text
# ----------------------------------------------------------------------

def test_span_invariant(rendered):
    queries, _fixtures = rendered
    for q in queries:
        for span in q.pii_spans:
            turn_text = q.turns[span.turn]
            assert span.value in turn_text, (
                f"{q.id}: span value {span.value!r} not present in turn "
                f"{span.turn} text {turn_text!r}"
            )


# ----------------------------------------------------------------------
# Cross-turn-taint: injected_pii.value appears in must_appear_in_turn text
# ----------------------------------------------------------------------

def test_injected_pii_appears_in_target_turn(rendered):
    queries, _fixtures = rendered
    for q in queries:
        for inj in q.injected_pii:
            target_turn = q.turns[inj.must_appear_in_turn]
            assert inj.value in target_turn, (
                f"{q.id}: injected value {inj.value!r} expected in turn "
                f"{inj.must_appear_in_turn} but absent in {target_turn!r}"
            )


# ----------------------------------------------------------------------
# Deterministic re-render
# ----------------------------------------------------------------------

def test_render_is_byte_stable():
    """Re-rendering with the same source produces the same JSONL bytes."""
    queries_a, fixtures_a = render_all()
    queries_b, fixtures_b = render_all()
    payload_a = serialize(queries_a, fixtures_a)
    payload_b = serialize(queries_b, fixtures_b)
    assert payload_a == payload_b
    digest_a = hashlib.sha256(payload_a.encode("utf-8")).hexdigest()
    digest_b = hashlib.sha256(payload_b.encode("utf-8")).hexdigest()
    assert digest_a == digest_b


# ----------------------------------------------------------------------
# Frozen artifact matches a fresh re-render
# ----------------------------------------------------------------------

def test_frozen_artifact_matches_render():
    """The committed dataset_v1.sha256 must match what render_all produces.

    If this test fails after editing scenarios, run::

        python -m benchmarks.pii_evaluation.dataset.render --freeze

    and commit the updated JSONL + sha256.
    """
    from pathlib import Path
    sha_path = (
        Path(__file__).resolve().parents[1]
        / "dataset" / "data" / "dataset_v1.sha256"
    )
    if not sha_path.exists():
        pytest.skip("dataset_v1.sha256 not committed yet")
    expected = sha_path.read_text(encoding="utf-8").strip()
    queries, fixtures = render_all()
    payload = serialize(queries, fixtures)
    actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert actual == expected, (
        f"frozen dataset is stale. Expected sha={expected}, got sha={actual}. "
        "Re-run python -m benchmarks.pii_evaluation.dataset.render --freeze."
    )


# ----------------------------------------------------------------------
# expected_tool_calls type / shape sanity
# ----------------------------------------------------------------------

def test_expected_tool_calls_arguments_are_strings(rendered):
    queries, _fixtures = rendered
    for q in queries:
        for call in q.expected_tool_calls:
            assert isinstance(call.arguments, Mapping), q.id
            for k, v in call.arguments.items():
                assert isinstance(k, str), f"{q.id}: arg key not str: {k!r}"
                assert isinstance(v, str), f"{q.id}: arg value not str: {v!r}"
            assert call.turn_index < len(q.turns), q.id


def test_legacy_id_tools_shim_matches_unified(rendered):
    """background_check_by_ssn / snils_record_lookup / passport_validity_check
    must produce identical results to verify_national_id with matching id_type."""
    from benchmarks.pii_evaluation.tools import InvocationLog, make_tools
    log = InvocationLog()
    tools = {t.name: t for t in make_tools(log)}
    # SSN
    legacy = tools["background_check_by_ssn"].invoke({"ssn": "123-45-6789"})
    unified = tools["verify_national_id"].invoke({"id_type": "us_ssn", "value": "123-45-6789"})
    assert legacy == unified, "SSN shim diverged from verify_national_id"
    # SNILS
    legacy = tools["snils_record_lookup"].invoke({"snils": "112-233-445 95"})
    unified = tools["verify_national_id"].invoke({"id_type": "ru_snils", "value": "112-233-445 95"})
    assert legacy == unified, "SNILS shim diverged from verify_national_id"
    # Passport
    legacy = tools["passport_validity_check"].invoke({"series_number": "4509 123456"})
    unified = tools["verify_national_id"].invoke({"id_type": "ru_passport", "value": "4509 123456"})
    assert legacy == unified, "Passport shim diverged from verify_national_id"


def test_load_dataset_registers_fixtures():
    """load_dataset() should register fixtures so customer_lookup_by_phone
    finds the rendered records."""
    from benchmarks.pii_evaluation.dataset import load_dataset
    from benchmarks.pii_evaluation.tools import _PHONE_INDEX  # noqa: WPS437
    queries = load_dataset()
    # At least the rendered customers should now be in _PHONE_INDEX.
    rendered_phone_count = sum(
        1 for q in queries
        for c in q.expected_tool_calls
        if c.tool_name == "customer_lookup_by_phone"
    )
    assert rendered_phone_count > 0, "no phone-lookup queries rendered"
    assert len(_PHONE_INDEX) >= 5, (
        f"expected at least 5 phone-indexed customers post-load, "
        f"found {len(_PHONE_INDEX)}"
    )


def test_expected_tool_calls_resolve_expected_records(rendered):
    """Every gold tool call must execute against the rendered fixture store."""
    import json

    from benchmarks.pii_evaluation.tools import InvocationLog, make_tools, register_fixtures

    queries, fixtures = rendered
    register_fixtures(fixtures)
    log = InvocationLog()
    tools = {t.name: t for t in make_tools(log)}
    failures = []

    for q in queries:
        for call in q.expected_tool_calls:
            if not call.expected_record_id:
                continue
            tool_obj = tools.get(call.tool_name)
            if tool_obj is None:
                failures.append(f"{q.id}: missing tool {call.tool_name}")
                continue
            result = tool_obj.invoke(dict(call.arguments))
            try:
                parsed = json.loads(result)
            except (TypeError, json.JSONDecodeError):
                parsed = {}
            record_id = parsed.get("record_id") if isinstance(parsed, dict) else None
            if record_id != call.expected_record_id:
                failures.append(
                    f"{q.id}: {call.tool_name} expected {call.expected_record_id}, "
                    f"got {record_id!r} from args={dict(call.arguments)!r} result={result!r}"
                )

    assert not failures, "\n".join(failures[:20])
