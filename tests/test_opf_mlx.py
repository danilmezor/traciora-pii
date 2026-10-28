"""The MLX forward pass feeds opf's own aggregation, Viterbi and span conversion."""
import json
from pathlib import Path

import pytest

pytest.importorskip("opf")

from traciora_pii.opf_mlx import MLXRuntime, window_starts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "benchmarks/pii_evaluation/dataset/data/dataset_v1.jsonl"
RU_CHECKPOINT = Path.home() / ".cache/traciora_pii/models/opf/pii-anon--opf-ru-v2"


def test_windows_cover_text_with_overlap():
    starts = window_starts(1300, 512, 128)
    assert starts[:3] == [0, 384, 768]
    assert all(b - a == 384 for a, b in zip(starts, starts[1:]))
    assert starts[-1] + 512 >= 1300 > starts[-2] + 512


def test_single_window_for_short_text():
    assert window_starts(39, 512, 128) == [0]
    assert window_starts(512, 512, 128) == [0]


def test_decoder_returning_the_wrong_length_raises():
    import torch

    class ShortDecoder:
        def decode(self, scores):
            return [0] * (scores.shape[0] - 1)

    runtime = MLXRuntime.__new__(MLXRuntime)
    runtime.decoder = ShortDecoder()
    runtime.token_log_probs = lambda token_ids: torch.zeros((len(token_ids), 3))
    with pytest.raises(RuntimeError, match=r"2 labels for 3 tokens"):
        runtime.token_labels([1, 2, 3])


def test_empty_text_gives_no_spans():
    import tiktoken

    runtime = MLXRuntime.__new__(MLXRuntime)
    runtime.encoding = tiktoken.get_encoding("o200k_base")
    result = runtime.redact("")
    assert result.detected_spans == ()
    assert result.redacted_text == ""


# ----------------------------------------------------------------------
# Against the real ru fine-tune: run with `pytest -m integration`
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def ru_paths(request):
    if "integration" not in (request.config.option.markexpr or ""):
        pytest.skip("model-backed; run with -m integration")
    pytest.importorskip("mlx.core")
    from traciora_pii.opf_mlx_convert import converted_path, is_current

    mlx_path = converted_path(RU_CHECKPOINT, None)
    if not (RU_CHECKPOINT / "model.safetensors").is_file() or not (mlx_path / "source.json").is_file():
        pytest.skip("ru checkpoint and its bf16 MLX conversion are not in the cache")
    if not is_current(RU_CHECKPOINT, mlx_path, None):
        pytest.skip("the bf16 MLX conversion of ru is stale; run prepare_opf")
    if not DATASET.is_file():
        pytest.skip(f"{DATASET} is missing")
    return RU_CHECKPOINT, mlx_path


@pytest.fixture(scope="module")
def mlx_runtime(ru_paths):
    checkpoint, mlx_path = ru_paths
    return MLXRuntime(mlx_path, checkpoint)


@pytest.fixture(scope="module")
def torch_opf(ru_paths):
    from opf import OPF

    return OPF(model=str(ru_paths[0]), device="cpu")


def _ru_rows() -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    return sorted((r for r in rows if r.get("kind") == "query" and r.get("language") == "ru"
                   and r.get("pii_spans")), key=lambda r: r["id"])


def _spans(result):
    return [(s.start, s.end, s.label) for s in result.detected_spans]


@pytest.mark.integration
def test_mlx_spans_equal_torch_spans_on_ru_rows(mlx_runtime, torch_opf):
    rows = _ru_rows()[:20]
    assert len(rows) == 20
    found = 0
    for row in rows:
        text = "\n".join(row["turns"])
        expected = _spans(torch_opf.redact(text))
        assert _spans(mlx_runtime.redact(text)) == expected, row["id"]
        found += len(expected)
    assert found > 0


def _filler_words() -> list[str]:
    sentence = ("Сегодня на складе провели плановую проверку остатков, обновили цены в каталоге "
                "и подготовили отчёт о продажах за прошлую неделю для отдела закупок.")
    return sentence.split()


@pytest.mark.integration
def test_name_across_window_cut_is_one_span(mlx_runtime):
    encoding = mlx_runtime.encoding
    name = "Мария Шнайдер"
    words, prefix = _filler_words(), []
    while True:
        head = " ".join(prefix) + " Вчера позвонила "
        text = head + name + " и попросила перезвонить ей. " + " ".join(_filler_words() * 40)
        start_tok = len(encoding.encode(head))
        end_tok = len(encoding.encode(head + name))
        if start_tok < 512 < end_tok:
            break
        assert start_tok < 512, "no prefix puts the name across token 512"
        prefix.append(words[len(prefix) % len(words)])
    tokens = encoding.encode(text)
    text = encoding.decode(tokens[:900])
    assert len(encoding.encode(text)) >= 880

    people = [s for s in mlx_runtime.redact(text).detected_spans if s.label == "private_person"]
    assert [(s.start, s.end) for s in people] == [(len(head), len(head) + len(name))]


@pytest.mark.integration
def test_long_text_labels_agree(mlx_runtime, torch_opf):
    import torch
    import torch.nn.functional as F

    encoding = mlx_runtime.encoding
    tokens: list[int] = []
    for row in _ru_rows():
        tokens += encoding.encode("\n".join(row["turns"]) + "\n\n")
        if len(tokens) >= 2048:
            break
    tokens = tokens[:2048]

    runtime, decoder = torch_opf.get_prediction_components()
    with torch.inference_mode():
        ids = torch.tensor([tokens], dtype=torch.int32)
        logits = runtime.model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
        expected = decoder.decode(F.log_softmax(logits.float(), dim=-1)[0])

    labels = mlx_runtime.token_labels(tokens)
    agree = sum(a == b for a, b in zip(labels, expected))
    assert len(labels) == len(expected) == 2048
    assert agree >= 2047, f"{agree}/2048 token labels agree"
