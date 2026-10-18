"""Microsoft Presidio second-pass PII detector.

Sibling of ``OPFPIIDetector`` and ``LLMPIIDetector`` that swaps the
backend for Microsoft's Presidio Analyzer. Presidio combines a curated
recognizer registry (regex + checksum + context-aware) with a spaCy NER
backbone for ``PERSON`` / ``LOCATION`` / ``ORGANIZATION`` spans.

Why it exists here
------------------

The detector ablation table needs a third-party baseline alongside our
own regex+OPF stack to avoid a strawman comparison. Presidio is the
de-facto industry baseline for "destructive" PII redaction in non-LLM
pipelines, and reviewers will reasonably ask how the framework's
detector layer compares to it on the same dataset.

Detector contract
-----------------

Implements the duck-typed ``_SecondPassDetector`` protocol consumed by
``PIIVirtualizer``:

  * ``detect(text) -> List[LLMFinding]`` — span offsets into ``text``
    plus a canonical ``lemma`` for vault keying.
  * ``close()`` — release the analyzer engine.

Design notes
------------

  * Lemma normalization for ``[PERSON_NAME]`` reuses the shared
    ``normalize_person_lemma`` helper so cross-turn vault keys collapse
    the same way they do for OPF / LLM detectors.
  * Entities outside the configured ``label_map`` are dropped silently —
    Presidio's native taxonomy is broader than our placeholder vocabulary
    and the regex first pass has already tokenized phones, emails, etc.
    By default we keep only ``PERSON`` and ``LOCATION`` so this detector
    targets the same gap the OPF/LLM detectors target. Callers that
    want a "Presidio everywhere" comparison can pass an override map.
  * Tests inject a fake engine via the ``engine=`` kwarg so the unit
    suite runs without the optional ``presidio_analyzer`` dependency.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from collections.abc import Sequence

from ._detector_common import (
    LLMFinding,
    normalize_person_lemma,
    prefilter,
)

if TYPE_CHECKING:
    from .config import PresidioConfig

logger = logging.getLogger(__name__)


# Default: PERSON only. We deliberately do NOT map Presidio's LOCATION
# to [STREET_ADDRESS] — Presidio's LOCATION recognizer fires on cities,
# countries, and any place-shaped capitalized token, which produces large
# false-positive counts against a [STREET_ADDRESS]-only ground truth
# (5 FP on EN, 30 FP on RU in trial runs). For an apples-to-apples
# comparison against OPF (which targets street addresses specifically),
# the honest default is "names only". Callers who want LOCATION coverage
# despite the noise can pass ``label_map=LOCATION_LABEL_MAP`` or override
# it via ``presidio.label_map`` in ``traciora.yaml``.
DEFAULT_LABEL_MAP: dict[str, str] = {
    "PERSON": "[PERSON_NAME]",
}


# PERSON + LOCATION. Reproduces the previous (noisier) behavior for
# callers who explicitly opt in.
LOCATION_LABEL_MAP: dict[str, str] = {
    "PERSON": "[PERSON_NAME]",
    "LOCATION": "[STREET_ADDRESS]",
}


# Wider map for callers that want to evaluate Presidio as a first-pass
# replacement — matches the regex first-pass placeholder taxonomy so a
# Presidio-only pipeline can plug in where the regex layer normally sits.
# Exported so eval scripts can reach it without rebuilding the dictionary.
WIDE_LABEL_MAP: dict[str, str] = {
    "PERSON": "[PERSON_NAME]",
    "LOCATION": "[STREET_ADDRESS]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "US_SSN": "[PERSONAL_ID]",
    "IP_ADDRESS": "[IP]",
    "CREDIT_CARD": "[CARD]",
    "IBAN_CODE": "[IBAN]",
    "URL": "[URL]",
    "DATE_TIME": "[DATE]",
    "US_DRIVER_LICENSE": "[PERSONAL_ID]",
    "US_PASSPORT": "[PERSONAL_ID]",
    "MEDICAL_LICENSE": "[SECRET]",
}


# Per-language spaCy model used to build Presidio's NlpEngine when the
# caller does not supply one. ``_lg`` packages give Presidio the word
# vectors its context-similarity heuristics rely on. Override per
# detector via ``nlp_model=`` or via ``presidio.nlp_model`` in YAML.
_DEFAULT_SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_lg",
    "de": "de_core_news_lg",
    "ru": "ru_core_news_lg",
    "es": "es_core_news_lg",
    "fr": "fr_core_news_lg",
    "it": "it_core_news_lg",
}


def _shim_protobuf_message_factory() -> None:
    """Silence a protobuf 5.x compatibility traceback printed by Presidio.

    Background: protobuf 4.21 deprecated ``MessageFactory.GetPrototype``
    in favor of ``message_factory.GetMessageClass``; protobuf 5.x removed
    the method entirely. Several Presidio transitive deps (opentelemetry,
    azure-core tracing) still call the old name at recognizer-registry
    construction time. The exception is caught upstream but the
    traceback is printed to stderr, producing five identical lines per
    run::

        AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

    Patching ``GetPrototype`` to delegate to ``GetMessageClass`` on the
    class restores the old API surface without pinning protobuf or
    suppressing real errors. Idempotent: skipped when the method already
    exists or when ``GetMessageClass`` is unavailable.
    """
    try:
        from google.protobuf import message_factory as _mf
    except ImportError:
        return
    factory_cls = getattr(_mf, "MessageFactory", None)
    if factory_cls is None or hasattr(factory_cls, "GetPrototype"):
        return
    get_message_class = getattr(_mf, "GetMessageClass", None)
    if get_message_class is None:
        return

    def _get_prototype(self, descriptor):  # noqa: ARG001 — match old signature
        return get_message_class(descriptor)

    factory_cls.GetPrototype = _get_prototype


class PresidioPIIDetector:
    """Presidio analyzer wrapped as a drop-in second-pass detector.

    One detector instance corresponds to a single ``language`` because
    Presidio's spaCy backbone is loaded per language. For trilingual
    ablations construct three instances and dispatch by query language
    at the call site.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        score_threshold: float = 0.5,
        label_map: dict[str, str] | None = None,
        entities: Sequence[str] | None = None,
        prefilter_enabled: bool = False,
        nlp_model: str | None = None,
        engine: object | None = None,
    ):
        self.language = language
        self.score_threshold = score_threshold
        self._label_map = dict(label_map or DEFAULT_LABEL_MAP)
        # Restrict Presidio to entity types we know how to map. Callers
        # passing a bare ``entities`` override take responsibility for
        # extending ``label_map`` accordingly.
        self._entities: list[str] = list(entities) if entities is not None else list(self._label_map)
        self.prefilter_enabled = prefilter_enabled
        self.nlp_model = nlp_model or _DEFAULT_SPACY_MODELS.get(language)

        # Parity counters with sibling detectors so the harness can render
        # a uniform "detector cost" report. Presidio cannot hallucinate by
        # construction (every span is a slice of the input), but the
        # counter is kept for symmetry.
        self.lm_calls = 0
        self.prefilter_skips = 0
        self.hallucinated_drops = 0
        self.parse_failures = 0

        if engine is not None:
            self._engine = engine
        else:
            # Patch the protobuf MessageFactory before importing
            # presidio_analyzer so the recognizer-registry init does not
            # spam stderr with the GetPrototype-removed traceback.
            _shim_protobuf_message_factory()
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import NlpEngineProvider
            except ImportError as exc:  # noqa: BLE001
                raise ImportError(
                    "PresidioPIIDetector requires the `presidio` extra. "
                    "Install with: pip install -e '.[presidio]' "
                    "and download the spaCy model for your language: "
                    "python -m spacy download en_core_web_lg",
                ) from exc

            if not self.nlp_model:
                raise ValueError(
                    f"No spaCy model registered for language={language!r}. "
                    f"Set ``nlp_model=`` (or ``presidio.nlp_model`` in "
                    f"traciora.yaml) to a spaCy model name like "
                    f"'<lang>_core_news_lg', or pass a configured "
                    f"AnalyzerEngine via ``engine=``.",
                )

            # Build an explicit NlpEngine for the requested language so
            # Presidio's spaCy backbone matches the input text. Without
            # this, ``AnalyzerEngine(supported_languages=[language])``
            # silently falls back to Presidio's default English config
            # and runs en_core_web_lg over Cyrillic / non-EN text — the
            # source of the spurious-FP runs reported earlier.
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [
                        {"lang_code": language, "model_name": self.nlp_model},
                    ],
                },
            )
            try:
                nlp_engine = provider.create_engine()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load spaCy model {self.nlp_model!r} for "
                    f"language={language!r}. Install it with: "
                    f"pip install https://github.com/explosion/spacy-models/"
                    f"releases/download/{self.nlp_model}-3.7.0/"
                    f"{self.nlp_model}-3.7.0-py3-none-any.whl",
                ) from exc
            self._engine = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[language],
            )

    @classmethod
    def from_config(cls, presidio_cfg: "PresidioConfig") -> "PresidioPIIDetector":
        """Build the detector from a ``PIIConfig.detector.presidio`` section."""
        label_map = presidio_cfg.label_map or None
        entities = presidio_cfg.entities or None
        nlp_model = presidio_cfg.nlp_model or None
        return cls(
            language=presidio_cfg.language,
            score_threshold=presidio_cfg.score_threshold,
            label_map=label_map,
            entities=entities,
            prefilter_enabled=presidio_cfg.prefilter,
            nlp_model=nlp_model,
        )

    # ------------------------------------------------------------------
    # Public API — matches the _SecondPassDetector protocol
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[LLMFinding]:
        """Return ``[PERSON_NAME]`` / ``[STREET_ADDRESS]`` spans in *text*.

        Empty list is a valid and frequent answer (no findings, prefilter
        skip, analyzer error). Errors log a warning and degrade to empty
        rather than propagating, matching sibling detectors so a single
        bad input never crashes the virtualization pipeline.
        """
        if not text:
            return []

        if self.prefilter_enabled and not prefilter(text):
            self.prefilter_skips += 1
            return []

        self.lm_calls += 1
        try:
            results = self._engine.analyze(
                text=text,
                language=self.language,
                entities=self._entities,
                score_threshold=self.score_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PresidioPIIDetector analyze failed; returning empty findings. error_type=%s exc=%s",
                type(exc).__name__,
                exc,
            )
            return []

        out: list[LLMFinding] = []
        for r in results or []:
            placeholder = self._label_map.get(getattr(r, "entity_type", ""))
            if placeholder is None:
                continue
            start = getattr(r, "start", -1)
            end = getattr(r, "end", -1)
            if start < 0 or end <= start or end > len(text):
                self.hallucinated_drops += 1
                continue
            span_text = text[start:end]
            if not span_text:
                continue

            if placeholder == "[PERSON_NAME]":
                lemma = normalize_person_lemma(span_text, span_text.lower())
            elif placeholder == "[STREET_ADDRESS]":
                lemma = " ".join(span_text.split()).lower()
            else:
                lemma = " ".join(span_text.split()).lower()

            out.append(
                LLMFinding(
                    detector="presidio",
                    start=start,
                    end=end,
                    placeholder=placeholder,
                    lemma=lemma,
                )
            )
        return out

    def close(self) -> None:
        """Release the analyzer engine. Idempotent."""
        self._engine = None
