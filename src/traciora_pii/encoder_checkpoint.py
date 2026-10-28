"""Encoder checkpoints: one cache directory per pinned revision, downloaded at
prepare time and validated before the detector opens them."""
import json
from pathlib import Path

from .config import EncoderConfig, EncoderModelEntry

FAMILY_LABELS: dict[str, frozenset[str]] = {
    "rubert": frozenset({
        "B-FIRST_NAME", "B-LAST_NAME", "B-MIDDLE_NAME", "B-STREET", "B-HOUSE",
        "B-CITY", "B-REGION", "B-DISTRICT", "B-COUNTRY",
    }),
    "horizon": frozenset({"B-PERSON", "B-STREET_ADDRESS"}),
}

REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json")


def registered_entry(encoder_cfg: EncoderConfig, model_name: str | None = None) -> EncoderModelEntry:
    """The entry for ``model_name`` (default ``default_model``); an unknown key lists the registry."""
    chosen = model_name or encoder_cfg.default_model
    if chosen not in encoder_cfg.models:
        available = ", ".join(sorted(encoder_cfg.models)) or "(none registered)"
        raise KeyError(
            f"Encoder model {chosen!r} is not registered in detector.encoder.models. "
            f"Available: {available}. Add an entry to traciora.yaml or pick one of the "
            f"existing keys.",
        )
    return encoder_cfg.models[chosen]


def cached_encoder_path(entry: EncoderModelEntry, cache_dir: str | None) -> Path:
    """A local checkpoint is its own path; a repo id gets one directory per revision."""
    if entry.is_local:
        return Path(entry.checkpoint).expanduser()
    root = Path(cache_dir or "~/.cache/traciora_pii/models/encoder").expanduser()
    return root / f"{entry.checkpoint.replace('/', '--')}--{entry.revision[:12]}"


def ensure_encoder_checkpoint(entry: EncoderModelEntry, cache_dir: str | None) -> Path:
    """Download the pinned revision once; call at deployment/startup, never in a request."""
    target = cached_encoder_path(entry, cache_dir)
    if entry.is_local:
        return target
    source = target / "source.json"
    if source.is_file() and json.loads(source.read_text()).get("revision") == entry.revision:
        return target
    import huggingface_hub
    huggingface_hub.snapshot_download(entry.checkpoint, revision=entry.revision, local_dir=target,
                                      allow_patterns=["*.json", "*.safetensors"])
    source.write_text(json.dumps({"repo": entry.checkpoint, "revision": entry.revision}))
    return target


def validate_encoder_checkpoint(path: Path, entry: EncoderModelEntry) -> None:
    """Reject an incomplete checkpoint, another revision, or a head without the family's labels."""
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            raise ValueError(f"Encoder checkpoint at {path} is missing {name}")
    source = path / "source.json"
    if not entry.is_local and not source.is_file():
        raise ValueError(f"Encoder checkpoint at {path} has no source.json: the download is "
                         f"incomplete or was never prepared; run prepare_second_pass")
    if source.is_file():
        revision = json.loads(source.read_text()).get("revision")
        if revision != entry.revision:
            raise ValueError(f"Encoder checkpoint at {path} holds revision {revision}, "
                             f"expected {entry.revision}")
    labels = set(json.loads((path / "config.json").read_text()).get("id2label", {}).values())
    missing = sorted(FAMILY_LABELS[entry.family] - labels)
    if missing:
        raise ValueError(f"Encoder checkpoint at {path} lacks {entry.family} label {missing[0]}")
