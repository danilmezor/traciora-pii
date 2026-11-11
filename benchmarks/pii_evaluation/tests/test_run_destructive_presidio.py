"""Smoke test for the Presidio-as-framework pipeline.

We don't load real Presidio (heavyweight install + spaCy model). Instead
we hand the pipeline fake analyzer/anonymizer stand-ins that exercise the
same code path. Real Presidio is exercised end-to-end in the integration
tests when the ``[presidio]`` extra is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pytest

from benchmarks.pii_evaluation.dataset import EvalQuery
from benchmarks.pii_evaluation.pipelines import RunResult, run_destructive_presidio


@dataclass
class _FakeAnalyzerResult:
    """Matches presidio_analyzer.RecognizerResult's read interface."""
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class _FakeAnonymizedText:
    text: str


class _FakeAnalyzer:
    def __init__(self, return_results: List[_FakeAnalyzerResult]) -> None:
        self.return_results = return_results
        self.calls: List[tuple] = []

    def analyze(self, text: str, language: str) -> List[_FakeAnalyzerResult]:
        self.calls.append((text, language))
        return self.return_results


class _FakeAnonymizer:
    def __init__(self, replacement_text: str) -> None:
        self.replacement_text = replacement_text

    def anonymize(self, text: str, analyzer_results: List[_FakeAnalyzerResult]) -> _FakeAnonymizedText:
        return _FakeAnonymizedText(text=self.replacement_text)


class _NoOpLLM:
    """LLM stand-in that returns a single final response immediately."""

    def invoke(self, messages: List[Any]) -> Any:
        from langchain_core.messages import AIMessage
        return AIMessage(content="ok")

    def bind_tools(self, tools: List[Any]) -> "_NoOpLLM":
        return self


def _query(text: str) -> EvalQuery:
    return EvalQuery(
        id="presidio-test-1",
        bucket="single_turn",
        language="en",
        turns=(text,),
        pii_spans=(),
        expected_tool_calls=(),
        boundary_pii_values=(),
    )


def test_run_destructive_presidio_anonymizes_user_input():
    analyzer = _FakeAnalyzer(return_results=[
        _FakeAnalyzerResult(entity_type="PERSON", start=0, end=5, score=0.99),
    ])
    anonymizer = _FakeAnonymizer(replacement_text="<PERSON> is a customer")
    llm = _NoOpLLM()

    result = run_destructive_presidio(
        query=_query("Alice is a customer"),
        llm=llm,
        analyzers={"en": analyzer},
        anonymizer=anonymizer,
        language="en",
    )

    assert result.config == "destructive_presidio"
    # The text that reached the LLM should be the anonymized version.
    assert len(result.sent_to_llm) > 0
    user_msg = next(
        (e for e in result.sent_to_llm if e.get("role") == "human"),
        None,
    )
    assert user_msg is not None
    assert "<PERSON>" in user_msg.get("content", "")
    assert "Alice" not in user_msg.get("content", "")
    assert analyzer.calls == [("Alice is a customer", "en")]


def test_run_destructive_presidio_passes_raw_when_analyzer_fails():
    """Realistic failure mode: analyzer crashes, we log and pass raw —
    the deployment leaks PII, which is the point of having this baseline."""
    class CrashingAnalyzer:
        def analyze(self, **kwargs):
            raise RuntimeError("simulated spaCy load failure")

    result = run_destructive_presidio(
        query=_query("Bob has SSN 123-45-6789"),
        llm=_NoOpLLM(),
        analyzers={"en": CrashingAnalyzer()},
        anonymizer=_FakeAnonymizer(replacement_text="should-not-be-used"),
    )
    user_msg = next(
        (e for e in result.sent_to_llm if e.get("role") == "human"),
        None,
    )
    assert user_msg is not None
    # Raw text passes through on failure.
    assert "Bob" in user_msg.get("content", "")
    assert "123-45-6789" in user_msg.get("content", "")


def test_run_destructive_presidio_skips_anonymize_when_no_results():
    """No PII detected → no anonymizer call → text passes through unchanged."""
    analyzer = _FakeAnalyzer(return_results=[])
    anonymizer = _FakeAnonymizer(replacement_text="<SHOULD-NOT-APPEAR>")

    result = run_destructive_presidio(
        query=_query("the weather is fine today"),
        llm=_NoOpLLM(),
        analyzers={"en": analyzer},
        anonymizer=anonymizer,
    )
    user_msg = next(
        (e for e in result.sent_to_llm if e.get("role") == "human"),
        None,
    )
    assert user_msg is not None
    assert "the weather is fine today" in user_msg.get("content", "")
    assert "<SHOULD-NOT-APPEAR>" not in user_msg.get("content", "")


def test_run_destructive_presidio_dispatches_by_language():
    """DE query must hit the DE analyzer, not EN. This is the regression
    the multi-language wiring exists to prevent."""
    en_analyzer = _FakeAnalyzer(return_results=[])
    de_analyzer = _FakeAnalyzer(return_results=[
        _FakeAnalyzerResult(entity_type="PERSON", start=0, end=5, score=0.99),
    ])
    de_query = EvalQuery(
        id="de-test-1",
        bucket="single_turn",
        language="de",
        turns=("Klaus ist Kunde",),
        pii_spans=(),
        expected_tool_calls=(),
        boundary_pii_values=(),
    )

    run_destructive_presidio(
        query=de_query,
        llm=_NoOpLLM(),
        analyzers={"en": en_analyzer, "de": de_analyzer},
        anonymizer=_FakeAnonymizer(replacement_text="<PERSON> ist Kunde"),
        language="de",
    )

    assert en_analyzer.calls == [], "EN analyzer should not be called for a DE query"
    assert de_analyzer.calls == [("Klaus ist Kunde", "de")]


def test_run_destructive_presidio_falls_back_to_en_with_warning(caplog):
    """When the query language isn't registered, fall back to EN with a
    warning rather than crashing."""
    import logging
    en_analyzer = _FakeAnalyzer(return_results=[])

    fr_query = EvalQuery(
        id="fr-test-1",
        bucket="single_turn",
        language="fr",
        turns=("Bonjour Pierre",),
        pii_spans=(),
        expected_tool_calls=(),
        boundary_pii_values=(),
    )

    with caplog.at_level(logging.WARNING):
        run_destructive_presidio(
            query=fr_query,
            llm=_NoOpLLM(),
            analyzers={"en": en_analyzer},
            anonymizer=_FakeAnonymizer(replacement_text="ignored"),
            language="fr",
        )

    assert en_analyzer.calls == [("Bonjour Pierre", "en")]
    assert any("not registered for language='fr'" in rec.message for rec in caplog.records)


def test_run_destructive_presidio_raises_on_empty_analyzers():
    with pytest.raises(ValueError, match="at least one analyzer"):
        run_destructive_presidio(
            query=_query("x"),
            llm=_NoOpLLM(),
            analyzers={},
            anonymizer=_FakeAnonymizer(replacement_text="ignored"),
        )
