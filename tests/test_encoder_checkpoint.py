import json
from pathlib import Path

import pytest

from traciora_pii.config import EncoderModelEntry
from traciora_pii.encoder_checkpoint import (
    FAMILY_LABELS,
    cached_encoder_path,
    ensure_encoder_checkpoint,
    validate_encoder_checkpoint,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


def _entry(revision: str = REVISION, family: str = "horizon") -> EncoderModelEntry:
    return EncoderModelEntry(checkpoint="org/model", revision=revision, family=family)


def _write_checkpoint(path: Path, labels) -> None:
    path.mkdir(parents=True, exist_ok=True)
    id2label = {str(i): label for i, label in enumerate(["O", *sorted(labels)])}
    (path / "config.json").write_text(json.dumps({"id2label": id2label}))
    (path / "model.safetensors").write_bytes(b"")
    (path / "tokenizer.json").write_text("{}")


@pytest.fixture
def fake_download(monkeypatch):
    """Patches ``huggingface_hub.snapshot_download``; the module imports it lazily from there."""
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        _write_checkpoint(Path(kwargs["local_dir"]), FAMILY_LABELS["horizon"])
        return kwargs["local_dir"]

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return calls


def test_cached_path_separates_revisions(tmp_path):
    first = cached_encoder_path(_entry(), str(tmp_path))
    second = cached_encoder_path(_entry(OTHER_REVISION), str(tmp_path))
    assert first == tmp_path / "org--model--0123456789ab"
    assert second == tmp_path / "org--model--fedcba987654"


def test_ensure_downloads_once_with_pinned_revision_and_patterns(tmp_path, fake_download):
    entry = _entry()
    path = ensure_encoder_checkpoint(entry, str(tmp_path))
    assert path == cached_encoder_path(entry, str(tmp_path))
    assert fake_download == [{
        "repo_id": "org/model",
        "revision": REVISION,
        "local_dir": path,
        "allow_patterns": ["*.json", "*.safetensors"],
    }]
    assert json.loads((path / "source.json").read_text()) == {"repo": "org/model", "revision": REVISION}
    validate_encoder_checkpoint(path, entry)

    assert ensure_encoder_checkpoint(entry, str(tmp_path)) == path
    assert len(fake_download) == 1


def test_validate_rejects_missing_file_wrong_revision_and_missing_labels(tmp_path):
    entry = _entry()

    missing_tokenizer = tmp_path / "missing"
    _write_checkpoint(missing_tokenizer, FAMILY_LABELS["horizon"])
    (missing_tokenizer / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match=r"missing.*tokenizer\.json"):
        validate_encoder_checkpoint(missing_tokenizer, entry)

    wrong_revision = tmp_path / "revision"
    _write_checkpoint(wrong_revision, FAMILY_LABELS["horizon"])
    (wrong_revision / "source.json").write_text(json.dumps({"repo": "org/model", "revision": OTHER_REVISION}))
    with pytest.raises(ValueError, match=f"revision.*{OTHER_REVISION}"):
        validate_encoder_checkpoint(wrong_revision, entry)

    missing_label = tmp_path / "labels"
    _write_checkpoint(missing_label, FAMILY_LABELS["rubert"] - {"B-CITY"})
    with pytest.raises(ValueError, match=r"label.*B-CITY"):
        validate_encoder_checkpoint(missing_label, _entry(family="rubert"))


def test_local_path_passes_through_without_download(tmp_path, fake_download):
    local = tmp_path / "local-model"
    _write_checkpoint(local, FAMILY_LABELS["rubert"])
    entry = EncoderModelEntry(checkpoint=str(local), family="rubert")
    assert cached_encoder_path(entry, str(tmp_path / "cache")) == local
    assert ensure_encoder_checkpoint(entry, str(tmp_path / "cache")) == local
    assert fake_download == []
    validate_encoder_checkpoint(local, entry)
