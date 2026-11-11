"""Smoke tests for ``build_release``: end-to-end artifact generation.

Slot-template architecture: no scaffold sources, no spaCy. Tests exercise
the build against shipped seed bundles and verify byte-determinism via
the produced manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.data_generator.build_release import main as build_main


def test_build_release_smoke(tmp_path):
    """Builder writes train/dev/test/datasheet for each locale + manifest + release card.

    Uses --skip-policy-check because at small n the soft-tier recall is
    statistically noisy. The dedicated test_build_release_policy_contract_passes
    runs at higher n where the gate is reliable.
    """
    out = tmp_path / "rel"
    rc = build_main([
        "--out", str(out),
        "--locales", "ru,de,en",
        "--n", "30",
        "--skip-policy-check",
    ])
    assert rc == 0, "build_release returned non-zero"

    for locale in ("ru", "de", "en"):
        bench_dir = out / f"traciora_pii-bench-{locale}"
        assert bench_dir.is_dir(), f"missing dir for locale {locale}"
        for fname in ("train.jsonl", "dev.jsonl", "test.jsonl", "datasheet.md"):
            assert (bench_dir / fname).exists(), f"missing {fname} for {locale}"

    assert (out / "RELEASE_CARD.md").exists()
    manifest = json.loads((out / "MANIFEST.json").read_text())
    # 3 locales × 4 files each (train/dev/test/datasheet) + RELEASE_CARD.md.
    assert len(manifest["files"]) == 3 * 4 + 1


def test_build_release_determinism(tmp_path):
    """Two runs with the same args produce byte-identical artifacts."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    common = ["--locales", "ru,de,en", "--n", "20", "--skip-policy-check"]
    build_main(["--out", str(out_a)] + common)
    build_main(["--out", str(out_b)] + common)

    files_a = json.loads((out_a / "MANIFEST.json").read_text())["files"]
    files_b = json.loads((out_b / "MANIFEST.json").read_text())["files"]
    assert set(files_a.keys()) == set(files_b.keys())
    for rel in files_a:
        assert files_a[rel]["sha256"] == files_b[rel]["sha256"], (
            f"non-deterministic file: {rel}"
        )


def test_build_release_dry_run(tmp_path):
    """--dry-run computes everything but writes no files."""
    out = tmp_path / "dry"
    rc = build_main([
        "--out", str(out),
        "--locales", "ru",
        "--n", "10",
        "--dry-run",
        "--skip-policy-check",
    ])
    assert rc == 0
    if out.exists():
        assert not any(out.iterdir())


def test_build_release_policy_contract_passes(tmp_path):
    """Default build runs the policy check at production-like n and exits 0.

    Uses n=400 so soft-tier slots get enough samples for the 30% recall
    floor to be statistically reliable. At smaller n the gate is noisy and
    can fire by chance even when generator and policy are in sync.
    """
    out = tmp_path / "rel"
    rc = build_main([
        "--out", str(out),
        "--locales", "ru,de,en",
        "--n", "400",
    ])
    assert rc == 0, "policy-contract check failed during build"


def test_build_release_invalid_locale(tmp_path):
    """Unknown locale → SystemExit."""
    with pytest.raises(SystemExit):
        build_main([
            "--out", str(tmp_path / "x"),
            "--locales", "fr",
            "--n", "5",
        ])


def test_build_release_with_noise(tmp_path):
    """Noise flags propagate and span invariant survives.

    Skip the policy-contract check: noise can mangle keyword-anchor words
    (e.g. 'паспорт' → 'парспорт') that the RU regex layer relies on for
    context, which is the point of stress evaluation, not a contract bug.
    """
    out = tmp_path / "noisy"
    rc = build_main([
        "--out", str(out),
        "--locales", "ru",
        "--n", "30",
        "--noise-typo-rate", "0.05",
        "--noise-whitespace-rate", "0.05",
        "--skip-policy-check",
    ])
    assert rc == 0
    train = (out / "traciora_pii-bench-ru" / "train.jsonl").read_text(encoding="utf-8")
    for line in train.splitlines():
        row = json.loads(line)
        for sp in row["spans"]:
            assert row["text"][sp["start"]:sp["end"]] == sp["value"]


def test_build_release_writes_release_card(tmp_path):
    """RELEASE_CARD lists every locale with at least one row."""
    out = tmp_path / "rel"
    build_main(["--out", str(out), "--locales", "ru,de,en", "--n", "30", "--skip-policy-check"])
    card = (out / "RELEASE_CARD.md").read_text(encoding="utf-8")
    for locale in ("ru", "de", "en"):
        assert f"`{locale}`" in card, f"locale {locale} missing from release card"
