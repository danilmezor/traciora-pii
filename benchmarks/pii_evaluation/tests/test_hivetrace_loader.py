"""The hivetrace pii-bench loader and its imported-eval wiring, with ``load_dataset`` stubbed."""
from __future__ import annotations

import pytest

from benchmarks.data_import import loaders
from benchmarks.pii_evaluation.run_imported_dataset_eval import DATASETS, _load_examples

_GOOD = "Игорь Петров живёт на ул. Ленина, 5, его телефон +7 900 123-45-67"
_BAD = "Пишите Анне Смирновой в Москву"

_ROWS = {
    "domain": [
        {
            "id": "chat_001",
            "domain": "L-CHAT",
            "text": _GOOD,
            "entities": [
                {"type": "NAME", "start": 0, "end": 12, "text": "Игорь Петров"},
                {"type": "ADDRESS", "start": 22, "end": 35, "text": "ул. Ленина, 5"},
                {"type": "PHONE", "start": 49, "end": 65, "text": "+7 900 123-45-67"},
            ],
        },
    ],
    "entity": [
        {
            "id": "ent_007",
            "domain": "E-NAME",
            "text": _BAD,
            "entities": [
                {"type": "NAME", "start": 7, "end": 20, "text": "Анне Смирновой"},
                {"type": "ADDRESS", "start": 24, "end": 30, "text": "Москву"},
            ],
        },
    ],
}


@pytest.fixture
def stub_hivetrace(monkeypatch):
    calls: list[tuple] = []

    def _load_dataset(repo, *, split, revision):
        calls.append((repo, split, revision))
        return list(_ROWS[split])

    monkeypatch.setattr(loaders, "load_dataset", _load_dataset)
    return calls


def test_loader_keeps_names_and_addresses_whose_text_matches_the_offsets(stub_hivetrace):
    examples = list(loaders.load_hivetrace_ru())

    assert stub_hivetrace == [
        ("hivetrace/pii-bench", "domain", loaders.HIVETRACE_REVISION),
        ("hivetrace/pii-bench", "entity", loaders.HIVETRACE_REVISION),
    ]
    assert [ex.id for ex in examples] == ["hivetrace-domain-chat_001", "hivetrace-entity-ent_007"]
    assert {(ex.locale, ex.source) for ex in examples} == {("ru", "hivetrace")}

    good, bad = examples
    assert good.text == _GOOD
    assert good.spans == [
        loaders.Span(label="NAME", value="Игорь Петров", start=0, end=12),
        loaders.Span(label="ADDRESS", value="ул. Ленина, 5", start=22, end=35),
    ]
    assert good.meta["n_dropped_spans"] == 0
    # "Анне Смирновой" is off by one character at its end offset; "Москву" matches.
    assert bad.spans == [loaders.Span(label="ADDRESS", value="Москву", start=24, end=30)]
    assert bad.meta["n_dropped_spans"] == 1


def test_loader_stops_at_the_limit(stub_hivetrace):
    assert [ex.id for ex in loaders.load_hivetrace_ru(limit=1)] == ["hivetrace-domain-chat_001"]


def test_imported_eval_loads_hivetrace_with_placeholders_projected(stub_hivetrace):
    examples = _load_examples("hivetrace_ru", 10, 2)

    spec = DATASETS["hivetrace_ru"]
    assert spec["locale"] == "ru"
    assert [ex.id for ex in examples] == ["hivetrace-domain-chat_001", "hivetrace-entity-ent_007"]
    assert [[spec["label_map"][sp.label] for sp in ex.spans] for ex in examples] == [
        ["[PERSON_NAME]", "[STREET_ADDRESS]"],
        ["[STREET_ADDRESS]"],
    ]


def test_imported_eval_builds_the_encoder_for_the_dataset_locale(monkeypatch):
    from traciora_pii import pii_virtualizer
    from benchmarks.pii_evaluation.run_imported_dataset_eval import _build_detectors

    calls: list[tuple] = []

    def _recording_build(config, **kwargs):
        calls.append((config.detector.second_pass, kwargs))
        return object()

    monkeypatch.setattr(pii_virtualizer, "build_second_pass_detector", _recording_build)
    spec = DATASETS["hivetrace_ru"]
    _build_detectors(spec, second_pass="encoder")
    _build_detectors(spec, encoder_model="de", second_pass="encoder")

    assert calls == [
        ("encoder", {"encoder_model_name": "ru"}),
        ("encoder", {"encoder_model_name": "de"}),
    ]
