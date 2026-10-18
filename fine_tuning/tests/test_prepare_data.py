"""End-to-end smoke test for prepare_data on a tiny synthetic bench."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fine_tuning.prepare_data import run_prepare


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _make_bench(bench_dir: Path) -> None:
    """Build a 3-row-per-split fixture in the data_generator output schema."""
    rows = [
        {
            "id": "ru-gen-00001",
            "locale": "ru",
            "kind": "positive",
            "category": "names",
            "template": "Иванов Иван Иванович, ИНН {inn}",
            "text": "Иванов Иван Иванович, ИНН 7707083893",
            "spans": [
                {"placeholder": "[PERSON_NAME]", "value": "Иванов Иван Иванович",
                 "start": 0, "end": 20},
                {"placeholder": "[RU_INN]", "value": "7707083893",
                 "start": 26, "end": 36},
            ],
        },
        {
            "id": "ru-gen-00002",
            "locale": "ru",
            "kind": "negative",
            "category": "hard_negative",
            "template": "",
            "text": "Артикул товара 7707083893 на складе",
            "spans": [],
        },
        {
            "id": "ru-gen-00003",
            "locale": "ru",
            "kind": "positive",
            "category": "names",
            "template": "...",
            "text": "Карта 4111 1111 1111 1111",
            "spans": [
                {"placeholder": "[CARD]", "value": "4111 1111 1111 1111",
                 "start": 6, "end": 25},
            ],
        },
    ]
    for split in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        _write_jsonl(bench_dir / split, rows)


def test_prepare_emits_opf_format(tmp_path):
    bench_dir = tmp_path / "traciora_pii-bench-ru"
    out_dir = tmp_path / "data" / "ru-test"
    _make_bench(bench_dir)

    report = run_prepare(
        locale="ru",
        bench_dir=bench_dir,
        out_dir=out_dir,
        slug="ru-test",
    )

    # OPF runner expects train/validation/test.jsonl + label_space.json.
    for f in ("train.jsonl", "validation.jsonl", "test.jsonl", "label_space.json"):
        assert (out_dir / f).exists(), f"missing artifact: {f}"

    # Validation came from dev.jsonl (renamed) — same row count.
    val_lines = (out_dir / "validation.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(val_lines) == 3

    # Span schema: dict[str, list[list[int]]]
    rec = json.loads(val_lines[0])
    assert isinstance(rec["spans"], dict)
    assert "private_person" in rec["spans"]
    assert rec["spans"]["private_person"] == [[0, 20]]
    assert "ru_inn" in rec["spans"]

    # The negative row should have an empty span map.
    neg = json.loads(val_lines[1])
    assert neg["spans"] == {}

    # [CARD] is in regex/ru.yaml but not in the RU OPF policy's label_map
    # (payment_card maps to [CARD] in the policy, so the placeholder *is*
    # mapped). Check that it survived.
    third = json.loads(val_lines[2])
    assert "payment_card" in third["spans"]

    # Label space JSON has runner-compatible format.
    label_space = json.loads((out_dir / "label_space.json").read_text(encoding="utf-8"))
    assert label_space["span_class_names"][0] == "O"
    assert "private_person" in label_space["span_class_names"]
    assert "ru_inn" in label_space["span_class_names"]


def test_prepare_drops_unmapped_placeholders(tmp_path):
    """A placeholder not in the locale's OPF label_map must be dropped to O."""
    bench_dir = tmp_path / "traciora_pii-bench-en"
    out_dir = tmp_path / "data" / "en-test"
    rows = [{
        "id": "en-gen-00001", "locale": "en", "kind": "positive",
        "category": "names", "template": "...",
        "text": "Alice Liddell on 2024-03-15",
        "spans": [
            {"placeholder": "[PERSON_NAME]", "value": "Alice Liddell",
             "start": 0, "end": 13},
            # DATE is intentionally a hard-negative in en_comprehensive.yaml.
            # Should be dropped during prepare.
            {"placeholder": "[DATE]", "value": "2024-03-15",
             "start": 17, "end": 27},
        ],
    }]
    for split in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        _write_jsonl(bench_dir / split, rows)

    report = run_prepare(
        locale="en",
        bench_dir=bench_dir,
        out_dir=out_dir,
        slug="en-test",
    )
    assert report.placeholder_dropped_counts.get("[DATE]", 0) >= 3
    rec = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "private_person" in rec["spans"]
    # private_date is intentionally not in the EN policy → must be absent.
    assert "private_date" not in rec["spans"]


def test_prepare_raises_on_missing_bench_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_prepare(
            locale="ru",
            bench_dir=tmp_path / "nope",
            out_dir=tmp_path / "out",
            slug="x",
        )
