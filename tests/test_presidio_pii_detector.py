"""Unit tests for PresidioPIIDetector with a fake analyzer engine.

Most assertions inject a stubbed engine so the test suite stays
hermetic — the real ``presidio_analyzer`` install pulls in spaCy and
hundreds of MB of language models we do not want to require for every
contributor. The optional ``TestPresidioRealEngine`` block exercises a
live AnalyzerEngine if the dependency is installed and a spaCy English
model is available; it is skipped otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pytest

from traciora_pii._detector_common import LLMFinding
from traciora_pii.config import PresidioConfig
from traciora_pii.presidio_pii_detector import (
    DEFAULT_LABEL_MAP,
    LOCATION_LABEL_MAP,
    WIDE_LABEL_MAP,
    PresidioPIIDetector,
    _shim_protobuf_message_factory,
)


# ----------------------------------------------------------------------
# Fake engine — duck-types presidio_analyzer.AnalyzerEngine.analyze
# ----------------------------------------------------------------------


@dataclass
class _FakeRecognizerResult:
    entity_type: str
    start: int
    end: int
    score: float = 0.85


class _FakeEngine:
    """Captures analyze() kwargs and returns a canned response list."""

    def __init__(self, responses: Iterable[_FakeRecognizerResult]):
        self._responses = list(responses)
        self.calls: List[dict] = []
        self.raise_next: Optional[Exception] = None

    def analyze(
        self,
        *,
        text: str,
        language: str,
        entities: Sequence[str],
        score_threshold: float,
    ) -> List[_FakeRecognizerResult]:
        self.calls.append(
            {
                "text": text,
                "language": language,
                "entities": list(entities),
                "score_threshold": score_threshold,
            }
        )
        if self.raise_next is not None:
            exc, self.raise_next = self.raise_next, None
            raise exc
        return list(self._responses)


def _make_detector(
    responses: Iterable[_FakeRecognizerResult] = (),
    *,
    language: str = "en",
    score_threshold: float = 0.5,
    label_map=None,
    entities=None,
    prefilter_enabled: bool = False,
) -> tuple[PresidioPIIDetector, _FakeEngine]:
    engine = _FakeEngine(responses)
    detector = PresidioPIIDetector(
        language=language,
        score_threshold=score_threshold,
        label_map=label_map,
        entities=entities,
        prefilter_enabled=prefilter_enabled,
        engine=engine,
    )
    return detector, engine


# ----------------------------------------------------------------------
# detect() behavior
# ----------------------------------------------------------------------


class TestDetect:
    def test_returns_empty_for_empty_input(self):
        detector, engine = _make_detector()
        assert detector.detect("") == []
        assert engine.calls == []

    def test_maps_person_to_person_name_with_lemma(self):
        text = "Find Marcus Holloway in the system."
        offset = text.index("Marcus Holloway")
        detector, _ = _make_detector(
            [_FakeRecognizerResult("PERSON", offset, offset + len("Marcus Holloway"))],
        )

        findings = detector.detect(text)

        assert len(findings) == 1
        f = findings[0]
        assert isinstance(f, LLMFinding)
        assert f.detector == "presidio"
        assert f.placeholder == "[PERSON_NAME]"
        assert text[f.start:f.end] == "Marcus Holloway"
        # English path strips title/possessive and reorders "lastname firstname".
        assert f.lemma == "holloway marcus"

    def test_default_map_drops_location_but_opt_in_keeps_it(self):
        text = "Ship the package to 1742 Magnolia Avenue."
        offset = text.index("1742 Magnolia Avenue")

        # Default map is PERSON-only — Presidio's noisy LOCATION recognizer
        # does not get mapped to [STREET_ADDRESS] under the package default.
        default_detector, _ = _make_detector(
            [_FakeRecognizerResult("LOCATION", offset, offset + len("1742 Magnolia Avenue"))],
        )
        assert default_detector.detect(text) == []

        # Opt-in via LOCATION_LABEL_MAP restores the previous behavior.
        opted_in_detector, _ = _make_detector(
            [_FakeRecognizerResult("LOCATION", offset, offset + len("1742 Magnolia Avenue"))],
            label_map=LOCATION_LABEL_MAP,
        )
        findings = opted_in_detector.detect(text)
        assert len(findings) == 1
        assert findings[0].placeholder == "[STREET_ADDRESS]"
        assert findings[0].lemma == "1742 magnolia avenue"

    def test_unmapped_entity_dropped(self):
        text = "Apple released a new product yesterday."
        offset = text.index("Apple")
        detector, _ = _make_detector(
            [_FakeRecognizerResult("ORGANIZATION", offset, offset + len("Apple"))],
        )

        assert detector.detect(text) == []

    def test_invalid_span_offsets_dropped_and_counted(self):
        text = "Some text"
        detector, _ = _make_detector(
            [
                _FakeRecognizerResult("PERSON", -1, 5),
                _FakeRecognizerResult("PERSON", 5, 5),
                _FakeRecognizerResult("PERSON", 0, 9999),
            ],
        )

        assert detector.detect(text) == []
        assert detector.hallucinated_drops == 3

    def test_analyze_exception_returns_empty_no_raise(self):
        detector, engine = _make_detector(
            [_FakeRecognizerResult("PERSON", 0, 5)],
        )
        engine.raise_next = RuntimeError("boom")

        # Must degrade gracefully; the virtualizer pipeline must not crash
        # because Presidio raised on a single input.
        assert detector.detect("Hello world") == []

    def test_analyze_called_with_configured_kwargs(self):
        text = "input text"
        detector, engine = _make_detector(
            [],
            language="de",
            score_threshold=0.7,
            entities=["PERSON", "LOCATION", "PHONE_NUMBER"],
            label_map={"PERSON": "[PERSON_NAME]", "LOCATION": "[STREET_ADDRESS]"},
        )

        detector.detect(text)

        assert engine.calls == [
            {
                "text": text,
                "language": "de",
                "entities": ["PERSON", "LOCATION", "PHONE_NUMBER"],
                "score_threshold": 0.7,
            }
        ]

    def test_lm_calls_counter_increments_only_when_engine_invoked(self):
        detector, _ = _make_detector([])
        for _ in range(3):
            detector.detect("Find Marcus Holloway today.")
        assert detector.lm_calls == 3
        assert detector.prefilter_skips == 0

    def test_prefilter_skips_obvious_negatives(self):
        # "v3.14.2" has no capitalized name nor address keyword → prefilter
        # returns False and the engine is never called.
        detector, engine = _make_detector([], prefilter_enabled=True)
        assert detector.detect("rolled back to v3.14.2") == []
        assert engine.calls == []
        assert detector.prefilter_skips == 1


# ----------------------------------------------------------------------
# from_config
# ----------------------------------------------------------------------


def _install_fake_presidio_modules(monkeypatch):
    """Install stub presidio_analyzer + .nlp_engine modules.

    The adapter imports both ``AnalyzerEngine`` and ``NlpEngineProvider``
    when no engine is injected. Tests that exercise the construction
    path need both stubs present.
    """
    import sys
    import types

    captured: dict = {}

    class _StubAnalyzerEngine:
        def __init__(self, **kwargs):
            captured["analyzer_kwargs"] = kwargs

        def analyze(self, **_kwargs):
            return []

    class _StubNlpEngine:
        pass

    class _StubProvider:
        def __init__(self, *, nlp_configuration):
            captured["nlp_configuration"] = nlp_configuration

        def create_engine(self):
            return _StubNlpEngine()

    parent = types.ModuleType("presidio_analyzer")
    parent.AnalyzerEngine = _StubAnalyzerEngine
    nlp_module = types.ModuleType("presidio_analyzer.nlp_engine")
    nlp_module.NlpEngineProvider = _StubProvider
    parent.nlp_engine = nlp_module

    monkeypatch.setitem(sys.modules, "presidio_analyzer", parent)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp_module)
    return captured


class TestFromConfig:
    def test_default_config_uses_default_label_map(self, monkeypatch):
        captured = _install_fake_presidio_modules(monkeypatch)

        cfg = PresidioConfig()
        detector = PresidioPIIDetector.from_config(cfg)

        assert captured["analyzer_kwargs"]["supported_languages"] == [cfg.language]
        assert detector._label_map == DEFAULT_LABEL_MAP
        assert sorted(detector._entities) == sorted(DEFAULT_LABEL_MAP)
        # Default EN model name resolved without explicit nlp_model.
        assert detector.nlp_model == "en_core_web_lg"
        assert captured["nlp_configuration"]["models"] == [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
        ]

    def test_ru_language_auto_picks_ru_spacy_model(self, monkeypatch):
        captured = _install_fake_presidio_modules(monkeypatch)

        cfg = PresidioConfig(language="ru")
        detector = PresidioPIIDetector.from_config(cfg)

        assert detector.nlp_model == "ru_core_news_lg"
        assert captured["nlp_configuration"]["models"] == [
            {"lang_code": "ru", "model_name": "ru_core_news_lg"},
        ]

    def test_custom_nlp_model_overrides_default(self, monkeypatch):
        captured = _install_fake_presidio_modules(monkeypatch)

        cfg = PresidioConfig(language="de", nlp_model="de_core_news_md")
        PresidioPIIDetector.from_config(cfg)

        assert captured["nlp_configuration"]["models"] == [
            {"lang_code": "de", "model_name": "de_core_news_md"},
        ]

    def test_unknown_language_without_nlp_model_raises(self, monkeypatch):
        _install_fake_presidio_modules(monkeypatch)

        cfg = PresidioConfig(language="zz")
        with pytest.raises(ValueError, match=r"No spaCy model registered"):
            PresidioPIIDetector.from_config(cfg)

    def test_custom_label_map_propagates(self, monkeypatch):
        _install_fake_presidio_modules(monkeypatch)

        cfg = PresidioConfig(
            language="de",
            score_threshold=0.6,
            label_map=dict(WIDE_LABEL_MAP),
        )
        detector = PresidioPIIDetector.from_config(cfg)

        assert detector.language == "de"
        assert detector.score_threshold == 0.6
        assert detector._label_map == WIDE_LABEL_MAP

    def test_missing_dependency_raises_helpful_import_error(self, monkeypatch):
        # Hide presidio_analyzer from the import system to verify we
        # produce an actionable error message.
        import sys

        monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
        with pytest.raises(ImportError, match=r"presidio.*extra"):
            PresidioPIIDetector(language="en")


# ----------------------------------------------------------------------
# build_second_pass_detector wiring
# ----------------------------------------------------------------------


class TestBuilderWiring:
    def test_build_second_pass_detector_with_presidio_mode(self, monkeypatch):
        _install_fake_presidio_modules(monkeypatch)

        from traciora_pii.config import PIIConfig
        from traciora_pii.pii_virtualizer import build_second_pass_detector

        cfg = PIIConfig().with_overrides({"detector": {"second_pass": "presidio"}})
        detector = build_second_pass_detector(cfg)
        assert isinstance(detector, PresidioPIIDetector)

    def test_build_second_pass_detector_falls_back_when_extra_missing(self, monkeypatch, caplog):
        import sys

        # Force the import to fail.
        monkeypatch.setitem(sys.modules, "presidio_analyzer", None)

        from traciora_pii.config import PIIConfig
        from traciora_pii.pii_virtualizer import build_second_pass_detector

        cfg = PIIConfig().with_overrides({"detector": {"second_pass": "presidio"}})
        with caplog.at_level("WARNING"):
            detector = build_second_pass_detector(cfg)

        assert detector is None
        assert any("presidio" in rec.message.lower() for rec in caplog.records)


# ----------------------------------------------------------------------
# Protobuf compatibility shim
# ----------------------------------------------------------------------


class TestProtobufShim:
    """The shim's job is to bridge the protobuf 5.x removal of
    ``MessageFactory.GetPrototype`` so Presidio's recognizer registry
    initialization does not flood stderr. We exercise the patch by
    constructing a fake module that mirrors protobuf 5.x's surface
    (``GetMessageClass`` present, ``GetPrototype`` removed)."""

    def test_patches_class_when_method_missing(self, monkeypatch):
        import sys
        import types

        class _FakeFactory:
            pass

        sentinel = object()

        def _fake_get_message_class(descriptor):
            return sentinel

        fake_pkg = types.ModuleType("google")
        fake_protobuf = types.ModuleType("google.protobuf")
        fake_mf = types.ModuleType("google.protobuf.message_factory")
        fake_mf.MessageFactory = _FakeFactory
        fake_mf.GetMessageClass = _fake_get_message_class
        fake_protobuf.message_factory = fake_mf
        fake_pkg.protobuf = fake_protobuf

        monkeypatch.setitem(sys.modules, "google", fake_pkg)
        monkeypatch.setitem(sys.modules, "google.protobuf", fake_protobuf)
        monkeypatch.setitem(sys.modules, "google.protobuf.message_factory", fake_mf)

        _shim_protobuf_message_factory()

        instance = _FakeFactory()
        assert hasattr(instance, "GetPrototype")
        assert instance.GetPrototype("any-descriptor") is sentinel

    def test_noop_when_method_already_present(self, monkeypatch):
        import sys
        import types

        original_marker = object()

        class _FakeFactory:
            def GetPrototype(self, descriptor):
                return original_marker

        fake_pkg = types.ModuleType("google")
        fake_protobuf = types.ModuleType("google.protobuf")
        fake_mf = types.ModuleType("google.protobuf.message_factory")
        fake_mf.MessageFactory = _FakeFactory
        fake_mf.GetMessageClass = lambda _d: object()
        fake_protobuf.message_factory = fake_mf
        fake_pkg.protobuf = fake_protobuf

        monkeypatch.setitem(sys.modules, "google", fake_pkg)
        monkeypatch.setitem(sys.modules, "google.protobuf", fake_protobuf)
        monkeypatch.setitem(sys.modules, "google.protobuf.message_factory", fake_mf)

        _shim_protobuf_message_factory()

        # The original implementation is preserved, not shadowed.
        assert _FakeFactory().GetPrototype(None) is original_marker

    def test_silent_when_protobuf_unavailable(self, monkeypatch):
        import sys

        # Force the import to fail; the shim must not raise.
        monkeypatch.setitem(sys.modules, "google.protobuf.message_factory", None)
        _shim_protobuf_message_factory()  # no exception


# ----------------------------------------------------------------------
# Optional integration test against a real Presidio installation
# ----------------------------------------------------------------------


@pytest.mark.integration
class TestPresidioRealEngine:
    """Live-engine assertions, gated by the optional dependency.

    Run these with:
        pip install -e '.[presidio]'
        python -m spacy download en_core_web_sm
        pytest tests/test_presidio_pii_detector.py::TestPresidioRealEngine
    """

    def test_real_engine_detects_obvious_person_name(self):
        pytest.importorskip("presidio_analyzer")
        spacy = pytest.importorskip("spacy")
        if not spacy.util.is_package("en_core_web_sm"):
            pytest.skip("spaCy en_core_web_sm not installed; "
                        "run `python -m spacy download en_core_web_sm`")

        detector = PresidioPIIDetector(
            language="en", score_threshold=0.4, nlp_model="en_core_web_sm",
        )
        text = "Please contact Marcus Holloway about the rebooking."
        findings = detector.detect(text)

        assert any(f.placeholder == "[PERSON_NAME]" for f in findings), (
            f"expected at least one PERSON_NAME finding, got {findings!r}"
        )
        detector.close()
