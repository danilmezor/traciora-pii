"""The encoder second pass: label schemes and merging, windows, the forward pass, the real checkpoints."""
import builtins
import json
from pathlib import Path

import pytest

from traciora_pii.config import EncoderConfig, EncoderModelEntry
from traciora_pii.encoder_checkpoint import FAMILY_LABELS, cached_encoder_path
from traciora_pii.encoder_pii_detector import (
    EncoderInferenceError,
    EncoderPIIDetector,
    Fragment,
    group_fragments,
    window_bounds,
)


def _fragment(text: str, piece: str, label: str, after: int = 0) -> Fragment:
    start = text.index(piece, after)
    return Fragment(label, start, start + len(piece), 1.0)


def _texts(text: str, groups) -> list[tuple[str, str]]:
    return [(placeholder, text[start:end]) for placeholder, start, end in groups]


# ----------------------------------------------------------------------
# Rules, no model
# ----------------------------------------------------------------------


def test_window_bounds_cover_with_overlap():
    assert window_bounds(1300) == [(0, 510), (382, 892), (764, 1274), (1146, 1300)]
    assert window_bounds(0) == []
    assert window_bounds(510) == [(0, 510)]


def test_rubert_lone_city_is_not_masked():
    text = "Я живу в Москве уже год."
    assert group_fragments(text, [_fragment(text, "Москве", "CITY")], "rubert") == []


def test_rubert_street_house_flat_is_one_span():
    text = "Адрес: ул. Ленина, д. 5, кв. 12. Звоните."
    fragments = [_fragment(text, "ул.", "STREET"), _fragment(text, "Ленина", "STREET"),
                 _fragment(text, "5", "HOUSE"), _fragment(text, "12", "HOUSE")]
    assert _texts(text, group_fragments(text, fragments, "rubert")) == [
        ("[STREET_ADDRESS]", "ул. Ленина, д. 5, кв. 12")]


def test_rubert_name_parts_merge_but_not_across_words():
    text = "Звонил Иванов Иван Иванович вчера."
    fragments = [_fragment(text, "Иванов", "LAST_NAME"),
                 _fragment(text, "Иван", "FIRST_NAME", after=text.index(" Иван ")),
                 _fragment(text, "Иванович", "MIDDLE_NAME")]
    assert _texts(text, group_fragments(text, fragments, "rubert")) == [
        ("[PERSON_NAME]", "Иванов Иван Иванович")]

    text = "Иванов и Петров пришли."
    fragments = [_fragment(text, "Иванов", "LAST_NAME"), _fragment(text, "Петров", "LAST_NAME")]
    assert _texts(text, group_fragments(text, fragments, "rubert")) == [
        ("[PERSON_NAME]", "Иванов"), ("[PERSON_NAME]", "Петров")]


def test_horizon_address_merges_over_short_gap():
    text = "Neue Adresse: Reeperbahn 84 #1, bitte speichern."
    fragments = [_fragment(text, "Reeperbahn 84", "STREET_ADDRESS"),
                 _fragment(text, "1", "STREET_ADDRESS", after=text.index("#"))]
    assert _texts(text, group_fragments(text, fragments, "horizon")) == [
        ("[STREET_ADDRESS]", "Reeperbahn 84 #1")]


def test_rubert_address_merges_across_preposition():
    text = "Доставка: дом 191 по улице пер. Тамбовский, до обеда."
    fragments = [_fragment(text, "дом 191", "HOUSE"), _fragment(text, "пер. Тамбовский", "STREET")]
    assert _texts(text, group_fragments(text, fragments, "rubert")) == [
        ("[STREET_ADDRESS]", "дом 191 по улице пер. Тамбовский")]


def test_group_fragments_trims_whitespace():
    text = "Call  Maria Schneider  now."
    start = text.index(" Maria")
    end = text.index("now")
    assert _texts(text, group_fragments(text, [Fragment("PERSON", start, end, 0.9)], "horizon")) == [
        ("[PERSON_NAME]", "Maria Schneider")]


def test_group_fragments_unknown_family_raises():
    with pytest.raises(ValueError, match="gliner"):
        group_fragments("text", [], "gliner")


# ----------------------------------------------------------------------
# The forward pass on a tiny random BERT
# ----------------------------------------------------------------------

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    ".", ",", "и", "в", "на", "по", "д", "дом", "кв", "ул", "улице", "иван", "иванов", "иванович",
    "петров", "мария", "шнаидер", "ленина", "москве", "живет", "звонил", "вчера", "сегодня",
    "склад", "деталеи", "график", "работы", "адрес", "доставка", "заказ", "клиент", "менеджер",
    "утром", "вечером", "завтра", "5", "12", "191", "##ов", "##а", "##ич", "##ы", "##е", "##и", "да",
]
TINY_TEXT = "звонил иван петров вчера, адрес ул. ленина, д. 5, кв. 12."


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = tmp_path_factory.mktemp("tiny-rubert")
    labels = ["O"] + [f"{bio}-{label[2:]}" for label in sorted(FAMILY_LABELS["rubert"]) for bio in "BI"]
    config = transformers.BertConfig(
        vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=512,
        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)})
    torch.manual_seed(20261028)
    transformers.BertForTokenClassification(config).save_pretrained(path)
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    transformers.BertTokenizerFast(vocab_file=str(vocab_file)).save_pretrained(path)
    (path / "source.json").write_text(json.dumps({"repo": str(path), "revision": None}))
    return path


@pytest.fixture(scope="module")
def tiny_detector(tiny_checkpoint) -> EncoderPIIDetector:
    return EncoderPIIDetector(checkpoint=tiny_checkpoint, family="rubert")


class _FailingModel:
    def __call__(self, **inputs):
        raise RuntimeError("forward pass exploded")


def test_detect_empty_text_is_empty_without_forward(tiny_detector, monkeypatch):
    monkeypatch.setattr(tiny_detector, "_model", _FailingModel())
    assert tiny_detector.detect("") == []


def test_padding_does_not_change_labels(tiny_detector):
    ids = tiny_detector._tokenizer(TINY_TEXT * 35, add_special_tokens=False)["input_ids"]
    bounds = window_bounds(len(ids))
    assert len(bounds) == 2 and bounds[0][1] - bounds[0][0] > bounds[1][1] - bounds[1][0]
    batched = tiny_detector._window_probs(ids, bounds)
    for (start, end), together in zip(bounds, batched):
        (alone,) = tiny_detector._window_probs(ids, [(start, end)])
        assert together.shape == alone.shape == (end - start, together.shape[1])
        assert together.argmax(-1).tolist() == alone.argmax(-1).tolist()
        assert (together - alone).abs().max().item() < 1e-5


def test_seam_token_takes_the_later_window_once_strictly_farther_from_its_inner_edge(
        tiny_detector, monkeypatch):
    torch = pytest.importorskip("torch")
    labels = tiny_detector._labels
    window_labels = ["CITY", "COUNTRY", "DISTRICT"]

    def constant_window_probs(ids, bounds):
        probs = []
        for k, (start, end) in enumerate(bounds):
            window = torch.zeros((end - start, len(labels)))
            window[:, labels.index(f"I-{window_labels[k]}")] = 1.0
            probs.append(window)
        return probs

    monkeypatch.setattr(tiny_detector, "_window_probs", constant_window_probs)
    text = "склад " * 1000
    assert len(tiny_detector._tokenizer(text, add_special_tokens=False)["input_ids"]) == 1000
    assert window_bounds(1000) == [(0, 510), (382, 892), (764, 1000)]

    # Seam 1|2: window 1 is 509 - i from its right edge, window 2 is i - 382 from its left;
    # 445 stays in window 1 (64 vs 63), 446 moves to window 2 (63 vs 64).
    # Seam 2|3: window 3 ends the document, so only its left edge counts; 828 moves (64 vs 63).
    # The document's first and last tokens sit at distance 0 from a window edge, but document
    # ends are not edges: token 0 keeps window 1 and token 999 keeps window 3.
    def chars(first, last):
        return 6 * first, 6 * last + 5

    fragments = tiny_detector._fragments(text)
    assert [(f.label, f.start, f.end) for f in fragments] == [
        ("CITY", *chars(0, 445)), ("COUNTRY", *chars(446, 827)), ("DISTRICT", *chars(828, 999))]


def test_forward_failure_raises_inference_error(tiny_checkpoint, monkeypatch):
    detector = EncoderPIIDetector(checkpoint=tiny_checkpoint, family="rubert")
    monkeypatch.setattr(detector, "_model", _FailingModel())
    with pytest.raises(EncoderInferenceError, match="forward pass exploded"):
        detector.detect(TINY_TEXT)


def test_cuda_without_cuda_raises(tiny_checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        EncoderPIIDetector(checkpoint=tiny_checkpoint, family="rubert", device="cuda")


def test_from_config_unknown_model_lists_registry(tiny_checkpoint):
    entry = EncoderModelEntry(checkpoint=str(tiny_checkpoint), family="rubert")
    cfg = EncoderConfig(default_model="ru", models={"ru": entry, "tiny": entry})
    with pytest.raises(KeyError, match="ru, tiny"):
        EncoderPIIDetector.from_config(cfg, model_name="de")


def test_missing_extra_raises_with_install_hint(tiny_checkpoint, monkeypatch):
    real_import = builtins.__import__

    def without_transformers(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_transformers)
    with pytest.raises(ImportError, match=r"\[encoder\]"):
        EncoderPIIDetector(checkpoint=tiny_checkpoint, family="rubert")


# ----------------------------------------------------------------------
# Against the cached checkpoints: run with `pytest -m integration`
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def detectors(request) -> dict[str, EncoderPIIDetector]:
    if "integration" not in (request.config.option.markexpr or ""):
        pytest.skip("model-backed; run with -m integration")
    from traciora_pii.runtime import LANGUAGES, traciora_config

    configs = {language: traciora_config(language).detector.encoder for language in LANGUAGES}
    for cfg in configs.values():
        path = cached_encoder_path(cfg.models[cfg.default_model], cfg.cache_dir)
        if not (path / "source.json").is_file():
            pytest.skip(f"{path} is not in the cache; run prepare_second_pass")
    return {language: EncoderPIIDetector.from_config(cfg) for language, cfg in configs.items()}


@pytest.mark.integration
@pytest.mark.parametrize("language", ["en", "de", "ru"])
def test_self_check_sentences_find_a_name(detectors, language):
    from traciora_pii.runtime import SELF_CHECK_SENTENCES

    findings = detectors[language].detect(SELF_CHECK_SENTENCES[language])
    assert [f.placeholder for f in findings] == ["[PERSON_NAME]"]


@pytest.mark.integration
def test_name_across_window_cut_is_one_span(detectors):
    detector = detectors["ru"]
    sentence = "Сегодня обсуждали поставку деталей на склад и график работы смены. "
    name = "Мария Шнайдер"

    def name_tokens(text: str) -> list[int]:
        start = text.index(name)
        offsets = detector._tokenizer(text, add_special_tokens=False,
                                      return_offsets_mapping=True, verbose=False)["offset_mapping"]
        return [i for i, (a, _) in enumerate(offsets) if start <= a < start + len(name)]

    words = (sentence * 60).split()
    for count in range(len(words)):
        text = f"{' '.join(words[:count])} Позвоните: {name}. {sentence * 40}"
        if name_tokens(text)[-1] >= 510:
            break
    assert name_tokens(text)[0] < 510 <= name_tokens(text)[-1]

    names = [text[f.start:f.end] for f in detector.detect(text) if f.placeholder == "[PERSON_NAME]"]
    assert names == ["Мария Шнайдер"]
