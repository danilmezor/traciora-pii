"""Smoke test for the internal_synthetic loader inside the imported-eval harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.pii_evaluation.run_imported_dataset_eval import (
    DATASETS,
    _load_examples,
    _load_internal_synthetic,
)


def _write_fixture(root: Path, slug: str, manifest_extra: dict, test_rows: list[dict]) -> Path:
    data_dir = root / "fine_tuning" / "data" / slug
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "slug": slug,
        "locale": manifest_extra.get("locale", "ru"),
        "policy": manifest_extra.get("policy", "ru_comprehensive"),
        "placeholder_to_label": manifest_extra.get("placeholder_to_label", {
            "[PERSON_NAME]": "private_person",
            "[EMAIL]": "private_email",
        }),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (data_dir / "test.jsonl").open("w", encoding="utf-8") as fh:
        for row in test_rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    return data_dir


def test_internal_loader_inverts_label_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Build a fixture under a fake repo root.
    _write_fixture(
        tmp_path,
        slug="ru-test",
        manifest_extra={},
        test_rows=[
            {
                "text": "Иванов Иван написал test@example.com",
                "spans": {"private_person": [[0, 11]], "private_email": [[20, 36]]},
                "source_id": "ru-fix-001",
                "locale": "ru",
                "kind": "positive",
            },
            {
                "text": "Random non-PII text",
                "spans": {},
                "source_id": "ru-fix-002",
                "locale": "ru",
                "kind": "negative",
            },
        ],
    )
    # Re-route the loader to the fixture root.
    import benchmarks.pii_evaluation.run_imported_dataset_eval as harness

    fixture_path = tmp_path / "fine_tuning" / "data"

    def _stub_resolve(slug: str, spec: dict):
        data_dir = fixture_path / slug
        manifest_path = data_dir / "manifest.json"
        test_path = data_dir / "test.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        locale = manifest["locale"]
        p2l = manifest["placeholder_to_label"]
        label_map = {v: k for k, v in p2l.items()}
        spec["locale"] = locale
        spec["opf_policy"] = f"{locale}_comprehensive"
        spec["label_map"] = label_map
        spec["allowed_labels"] = frozenset(label_map)
        from benchmarks.data_import.loaders import NormalizedExample, Span
        out = []
        with test_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                spans = [
                    Span(
                        label=lbl,
                        value=row["text"][s:e],
                        start=s,
                        end=e,
                    )
                    for lbl, ranges in (row.get("spans") or {}).items()
                    for s, e in ranges
                ]
                out.append(NormalizedExample(
                    id=row["source_id"],
                    locale=row["locale"],
                    text=row["text"],
                    spans=spans,
                    source=f"internal_synthetic:{slug}",
                    meta={"kind": row.get("kind", "")},
                ))
        return iter(out)

    monkeypatch.setattr(harness, "_load_internal_synthetic", _stub_resolve)

    # Reset the dynamic spec to its initial state.
    DATASETS["internal_synthetic"]["locale"] = "_dynamic_"
    DATASETS["internal_synthetic"]["opf_policy"] = "_dynamic_"
    DATASETS["internal_synthetic"]["label_map"] = {}
    DATASETS["internal_synthetic"]["allowed_labels"] = frozenset()

    examples = _load_examples("internal_synthetic", raw_limit=0, target_rows=10, internal_slug="ru-test")
    assert len(examples) == 2

    spec = DATASETS["internal_synthetic"]
    # Spec was rewritten by the loader.
    assert spec["locale"] == "ru"
    assert spec["opf_policy"] == "ru_comprehensive"
    # OPF label → placeholder inversion.
    assert spec["label_map"] == {
        "private_person": "[PERSON_NAME]",
        "private_email": "[EMAIL]",
    }

    # Positive row carries gold spans tagged with OPF labels (scoring code
    # remaps via label_map at score time).
    pos = examples[0]
    span_labels = {sp.label for sp in pos.spans}
    assert span_labels == {"private_person", "private_email"}

    # Negative row carries no gold.
    neg = examples[1]
    assert neg.spans == []
    assert neg.meta["kind"] == "negative"


def test_internal_loader_requires_slug():
    """The harness must error clearly when --internal-slug is missing."""
    with pytest.raises(ValueError, match="--internal-slug is required"):
        list(_load_examples("internal_synthetic", raw_limit=0, target_rows=10))


def test_internal_loader_against_real_fixture():
    """Smoke test against the actual fine_tuning/data/ru-v1 fixture.

    Skips cleanly if the fixture isn't present (e.g. CI environment).
    """
    real_data = Path(__file__).resolve().parents[3] / "fine_tuning" / "data" / "ru-v1"
    if not (real_data / "manifest.json").exists():
        pytest.skip("ru-v1 fixture not present")

    examples = list(_load_examples(
        "internal_synthetic", raw_limit=0, target_rows=5, internal_slug="ru-v1",
    ))
    assert len(examples) == 5
    # All RU labels should be present in the spec after loading.
    assert DATASETS["internal_synthetic"]["locale"] == "ru"
    assert "private_person" in DATASETS["internal_synthetic"]["label_map"]
