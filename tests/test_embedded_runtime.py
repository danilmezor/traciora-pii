"""Contracts for application-owned configuration and model lifetime."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.fernet import Fernet

from traciora_pii import PIIVaultStore, PIIVirtualizer, detect_pii, runtime, scrub
from traciora_pii._detector_common import LLMFinding
from traciora_pii.pii_leak_guard import PIILeakError, assert_model_safe
from traciora_pii.policies.loader import load_opf_policy
from traciora_pii.config import EncoderModelEntry
from traciora_pii.redaction import PIIRedactor

HORIZON = {"checkpoint": "Horizon-Labs/pii-redactor-small",
           "revision": "b9d834a12fc946a53551be95ba20ed9c11e9a553", "family": "horizon"}
ENCODER_PROFILES = {
    "ru": EncoderModelEntry(
        checkpoint="redmadrobot-rnd/rubert-base-pii-ner",
        revision="c802e8cd26f85d1cf920973ea6f83965a0618d63", family="rubert",
        description="Russian BERT token classifier, 21 PII types (Apache-2.0)."),
    "en": EncoderModelEntry(
        **HORIZON, description="Multilingual ModernBERT PII redactor, 29 types (Apache-2.0)."),
    "de": EncoderModelEntry(**HORIZON),
}


@pytest.mark.parametrize("language", runtime.LANGUAGES)
def test_installed_profiles_are_explicit(language, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PII_OPF_ENABLED", "true")
    monkeypatch.setenv("PII_OPF_LABEL_MODE", "obsolete")
    config = runtime.traciora_config(language)
    assert config.detector.regex_policy == language
    assert list(config.detector.opf.models) == [language]
    assert config.detector.opf.default_model == language
    encoder = config.detector.encoder
    assert list(encoder.models) == [language]
    assert encoder.default_model == language
    assert encoder.models[language] == ENCODER_PROFILES[language]
    assert not detect_pii("https://example.org/path", config=config)
    assert PIIRedactor.from_config(config).redact_text("https://example.org/path").text == "https://example.org/path"
    assert "@" not in PIIRedactor.from_config(config).redact_text("person@example.org").text
    # Logs have a distinct policy: chat URL exclusions do not hide sensitive URLs in logs.
    assert "12345678901234567890" not in scrub("id 12345678901234567890", config=config)


def test_content_callbacks_and_binary_guard(tmp_path):
    config = runtime.traciora_config("en")
    vault = PIIVaultStore(tmp_path / "vault.db", Fernet.generate_key())
    vault.open_session("one")
    def map_text(content, transform):
        return [{**b, "text": transform(b["text"])} if b["type"] == "text" else b
                for b in content]
    virtualizer = PIIVirtualizer(vault, config=config, map_text=map_text)
    content = [{"type": "text", "text": "a@example.org"},
               {"type": "image", "data": "a@example.org"}]
    masked = virtualizer.anonymize_content("one", content)
    assert masked[0]["text"].startswith("email_ref:")
    assert masked[1] == content[1]
    assert virtualizer.rehydrate_content("one", masked) == content
    assert_model_safe(masked, "one", vault, config=config,
                      skip_block=lambda b: isinstance(b, dict) and b.get("type") == "image")
    with pytest.raises(PIILeakError):
        assert_model_safe(content[0], "one", vault, config=config)
    vault.close()


def checkpoint(tmp_path, config):
    entry = config.detector.opf.models[config.detector.opf.default_model]
    path = tmp_path / entry.checkpoint.replace("/", "--")
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"span_class_names": ["O", *load_opf_policy(entry.policy).label_map]}))
    (path / "model.safetensors").touch()
    return path


def test_one_shared_model_and_serialized_inference(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    config = runtime.traciora_config("en").with_overrides({"detector": {
        "second_pass": "opf", "opf": {"cache_dir": str(tmp_path)}}})
    checkpoint(tmp_path, config)
    made = []
    class FakeOPF:
        def __init__(self, **kwargs):
            made.append(kwargs)
            self.active = 0
            self.closed = False
        def detect(self, text):
            self.active += 1
            assert self.active == 1
            time.sleep(.002)
            self.active -= 1
            return [LLMFinding("opf", 0, 4, "[PERSON_NAME]", "maria")]
        def close(self):
            self.closed = True
    monkeypatch.setattr("traciora_pii.opf_pii_detector.OPFPIIDetector", FakeOPF)
    try:
        with ThreadPoolExecutor(8) as pool:
            shared = list(pool.map(lambda _: runtime.get_shared_second_pass(config), range(16)))
            list(pool.map(lambda item: item.detect("text"), shared))
        assert len(made) == 1
        assert all(item is shared[0] for item in shared)
        with pytest.raises(RuntimeError, match="restart"):
            runtime.get_shared_second_pass(runtime.traciora_config("de").with_overrides(
                {"detector": {"second_pass": "opf"}}))
        runtime.close_shared_second_pass()
        assert shared[0]._detector.closed
    finally:
        runtime.close_shared_second_pass()


def test_missing_checkpoint_never_downloads_and_failure_is_cached(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    config = runtime.traciora_config("ru").with_overrides({"detector": {
        "second_pass": "opf", "opf": {"cache_dir": str(tmp_path)}}})
    monkeypatch.setattr("traciora_pii.opf_pii_detector._resolve_opf_checkpoint",
                        lambda *a, **k: pytest.fail("request attempted a download"))
    loads = []
    real_validate = runtime.validate_checkpoint
    monkeypatch.setattr(runtime, "validate_checkpoint",
                        lambda *a: (loads.append(a), real_validate(*a)))
    try:
        with pytest.raises(runtime.SecondPassUnavailable) as excinfo:
            runtime.get_shared_second_pass(config)
        message = str(excinfo.value)
        assert "configured but cannot start" in message
        assert "model=ru" in message and "device=cpu" in message
        assert "ValueError" in message and "Prepare the model before startup" in message
        assert isinstance(excinfo.value.__cause__, ValueError)
        with pytest.raises(runtime.SecondPassUnavailable):
            runtime.get_shared_second_pass(config)
        assert len(loads) == 1
    finally:
        runtime.close_shared_second_pass()


class SelfCheckOPF:
    findings: list = []
    seen: list = []

    def __init__(self, **kwargs):
        self.closed = False

    def detect(self, text):
        SelfCheckOPF.seen.append(text)
        return list(SelfCheckOPF.findings)

    def close(self):
        self.closed = True


def opf_config(tmp_path, language):
    config = runtime.traciora_config(language).with_overrides({"detector": {
        "second_pass": "opf", "opf": {"cache_dir": str(tmp_path), "device": "mps"}}})
    checkpoint(tmp_path, config)
    return config


def test_shared_second_pass_refuses_empty_self_check(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    config = opf_config(tmp_path, "ru")
    monkeypatch.setattr(SelfCheckOPF, "findings", [])
    monkeypatch.setattr(SelfCheckOPF, "seen", [])
    monkeypatch.setattr("traciora_pii.opf_pii_detector.OPFPIIDetector", SelfCheckOPF)
    try:
        with pytest.raises(runtime.SecondPassUnavailable) as excinfo:
            runtime.get_shared_second_pass(config)
        message = str(excinfo.value)
        assert "device=mps" in message and "model=ru" in message
        assert "OPF_MOE_TRITON=0" in message
        assert SelfCheckOPF.seen == [runtime.SELF_CHECK_SENTENCES["ru"]]
        # The refusal is remembered: a later turn raises again instead of masking regex-only.
        with pytest.raises(runtime.SecondPassUnavailable):
            runtime.get_shared_second_pass(config)
        assert len(SelfCheckOPF.seen) == 1
    finally:
        runtime.close_shared_second_pass()


def test_shared_second_pass_accepts_name_finding(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    config = opf_config(tmp_path, "de")
    monkeypatch.setattr(SelfCheckOPF, "findings",
                        [LLMFinding("opf", 30, 45, "[PERSON_NAME]", "maria schneider")])
    monkeypatch.setattr(SelfCheckOPF, "seen", [])
    monkeypatch.setattr("traciora_pii.opf_pii_detector.OPFPIIDetector", SelfCheckOPF)
    try:
        shared = runtime.get_shared_second_pass(config)
        assert isinstance(shared, runtime.SharedSecondPass)
        assert SelfCheckOPF.seen == [runtime.SELF_CHECK_SENTENCES["de"]]
    finally:
        runtime.close_shared_second_pass()


def test_self_check_inference_error_propagates(monkeypatch, tmp_path):
    from traciora_pii.opf_pii_detector import OPFInferenceError
    runtime.close_shared_second_pass()
    config = opf_config(tmp_path, "en")

    class Broken(SelfCheckOPF):
        def detect(self, text):
            raise OPFInferenceError("OPF inference failed on device=mps: ModuleNotFoundError: triton")

    monkeypatch.setattr("traciora_pii.opf_pii_detector.OPFPIIDetector", Broken)
    try:
        with pytest.raises(OPFInferenceError, match="triton"):
            runtime.get_shared_second_pass(config)
    finally:
        runtime.close_shared_second_pass()


def test_checkpoint_rejects_wrong_labels(tmp_path):
    config = runtime.traciora_config("ru")
    path = checkpoint(tmp_path, config)
    runtime.validate_checkpoint(path, "ru_comprehensive")
    with pytest.raises(ValueError, match="labels"):
        runtime.validate_checkpoint(path, "en_comprehensive")


@pytest.mark.parametrize("language", runtime.LANGUAGES)
def test_real_model_smoke(language):
    """Run deliberately with TRACIORA_PII_REAL_MODEL_TESTS=1; downloads several GB."""
    import os
    if os.getenv("TRACIORA_PII_REAL_MODEL_TESTS") != "1":
        pytest.skip("explicit opt-in required for model downloads")
    runtime.close_shared_second_pass()
    config = runtime.traciora_config(language).with_overrides({"detector": {"second_pass": "opf"}})
    try:
        runtime.prepare_second_pass(config)
        detector = runtime.get_shared_second_pass(config)
        assert detector is not None
        assert isinstance(detector.detect("Contact Alex Example for details."), list)
    finally:
        runtime.close_shared_second_pass()


def test_prepare_second_pass_converts_for_mlx(monkeypatch, tmp_path):
    from traciora_pii import opf_mlx_convert

    config = runtime.traciora_config("ru").with_overrides({"detector": {
        "second_pass": "opf", "opf": {"cache_dir": str(tmp_path), "device": "mlx", "mlx_bits": 8}}})
    source = checkpoint(tmp_path, config)
    (source / ".traciora_pii-opf-ready").touch()
    calls = []

    def fake_convert(src, dst, *, bits):
        calls.append((src, dst, bits))
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "model.safetensors").touch()
        (dst / "source.json").write_text(json.dumps(
            {"sha256": opf_mlx_convert._source_sha256(src), "bits": bits}))
        return dst

    monkeypatch.setattr(opf_mlx_convert, "convert_checkpoint", fake_convert)
    expected = source.with_name(source.name + "-mlx-8bit")
    assert runtime.prepare_second_pass(config) == expected
    assert calls == [(source, expected, 8)]
    assert runtime.prepare_second_pass(config) == expected
    assert len(calls) == 1
    (source / "model.safetensors").write_bytes(b"retrained")
    runtime.prepare_second_pass(config)
    assert len(calls) == 2


def test_shared_second_pass_passes_mlx_bits(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    config = runtime.traciora_config("ru").with_overrides({"detector": {
        "second_pass": "opf", "opf": {"cache_dir": str(tmp_path), "device": "mlx", "mlx_bits": 8}}})
    checkpoint(tmp_path, config)
    made = []

    class Recording(SelfCheckOPF):
        def __init__(self, **kwargs):
            made.append(kwargs)
            super().__init__()

    monkeypatch.setattr(SelfCheckOPF, "findings",
                        [LLMFinding("opf", 30, 45, "[PERSON_NAME]", "мария шнайдер")])
    monkeypatch.setattr("traciora_pii.opf_pii_detector.OPFPIIDetector", Recording)
    try:
        runtime.get_shared_second_pass(config)
        assert made[0]["device"] == "mlx"
        assert made[0]["mlx_bits"] == 8
    finally:
        runtime.close_shared_second_pass()


def encoder_config(tmp_path, language):
    return runtime.traciora_config(language).with_overrides({"detector": {
        "second_pass": "encoder", "encoder": {"cache_dir": str(tmp_path)}}})


def test_prepare_second_pass_dispatches(monkeypatch, tmp_path):
    assert runtime.prepare_second_pass(runtime.traciora_config("ru")) is None
    config = encoder_config(tmp_path, "ru")
    entry = config.detector.encoder.models["ru"]
    calls = []
    monkeypatch.setattr(runtime, "ensure_encoder_checkpoint",
                        lambda e, cache_dir: (calls.append(("ensure", e, cache_dir)), tmp_path / "ru")[1])
    monkeypatch.setattr(runtime, "validate_encoder_checkpoint",
                        lambda path, e: calls.append(("validate", path, e)))
    assert runtime.prepare_second_pass(config) == tmp_path / "ru"
    assert calls == [("ensure", entry, str(tmp_path)), ("validate", tmp_path / "ru", entry)]

    unknown = config.with_overrides({"detector": {"encoder": {"default_model": "xx"}}})
    with pytest.raises(KeyError, match=r"'xx' is not registered.*Available: ru\."):
        runtime.prepare_second_pass(unknown)


class SelfCheckEncoder:
    findings: list = []
    seen: list = []
    made: list = []

    def __init__(self):
        self.closed = False

    @classmethod
    def from_config(cls, encoder_cfg, *, model_name=None):
        cls.made.append((encoder_cfg, model_name))
        return cls()

    def detect(self, text):
        SelfCheckEncoder.seen.append(text)
        return list(SelfCheckEncoder.findings)

    def close(self):
        self.closed = True


def test_shared_encoder_never_downloads_and_self_checks(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    monkeypatch.setattr(runtime, "ensure_encoder_checkpoint",
                        lambda *a, **k: pytest.fail("request attempted a download"))
    monkeypatch.setattr(SelfCheckEncoder, "seen", [])
    monkeypatch.setattr(SelfCheckEncoder, "made", [])
    monkeypatch.setattr("traciora_pii.encoder_pii_detector.EncoderPIIDetector", SelfCheckEncoder)
    config = encoder_config(tmp_path, "de")
    try:
        monkeypatch.setattr(SelfCheckEncoder, "findings",
                            [LLMFinding("encoder", 23, 38, "[PERSON_NAME]", "maria schneider")])
        shared = runtime.get_shared_second_pass(config)
        assert isinstance(shared, runtime.SharedSecondPass)
        assert runtime.get_shared_second_pass(config) is shared
        assert SelfCheckEncoder.made == [(config.detector.encoder, None)]
        assert SelfCheckEncoder.seen == [runtime.SELF_CHECK_SENTENCES["de"]]
        runtime.close_shared_second_pass()
        assert shared._detector.closed

        monkeypatch.setattr(SelfCheckEncoder, "seen", [])
        monkeypatch.setattr(SelfCheckEncoder, "findings",
                            [LLMFinding("encoder", 0, 6, "[STREET_ADDRESS]", "bitte")])
        config = encoder_config(tmp_path, "ru")
        with pytest.raises(runtime.SecondPassUnavailable) as excinfo:
            runtime.get_shared_second_pass(config)
        assert "model=ru" in str(excinfo.value) and "device=cpu" in str(excinfo.value)
        assert SelfCheckEncoder.seen == [runtime.SELF_CHECK_SENTENCES["ru"]]
        # The refusal is remembered: no second load, no second self-check.
        with pytest.raises(runtime.SecondPassUnavailable):
            runtime.get_shared_second_pass(config)
        assert len(SelfCheckEncoder.seen) == 1
        assert len(SelfCheckEncoder.made) == 2
    finally:
        runtime.close_shared_second_pass()


def test_shared_encoder_refuses_language_without_self_check(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    monkeypatch.setattr("traciora_pii.encoder_pii_detector.EncoderPIIDetector", SelfCheckEncoder)
    config = encoder_config(tmp_path, "ru").with_overrides({"detector": {"encoder": {
        "default_model": "fr", "models": {"fr": ENCODER_PROFILES["en"].model_dump()}}}})
    try:
        with pytest.raises(runtime.SecondPassUnavailable, match="'fr'"):
            runtime.get_shared_second_pass(config)
    finally:
        runtime.close_shared_second_pass()


def test_shared_encoder_missing_checkpoint_is_unavailable(tmp_path):
    runtime.close_shared_second_pass()
    try:
        with pytest.raises(runtime.SecondPassUnavailable, match="Prepare the model") as excinfo:
            runtime.get_shared_second_pass(encoder_config(tmp_path, "en"))
        assert isinstance(excinfo.value.__cause__, ValueError)
    finally:
        runtime.close_shared_second_pass()


def test_switching_backend_requires_restart(monkeypatch, tmp_path):
    runtime.close_shared_second_pass()
    monkeypatch.setattr(SelfCheckOPF, "findings",
                        [LLMFinding("opf", 30, 45, "[PERSON_NAME]", "maria schneider")])
    monkeypatch.setattr("traciora_pii.opf_pii_detector.OPFPIIDetector", SelfCheckOPF)
    monkeypatch.setattr("traciora_pii.encoder_pii_detector.EncoderPIIDetector",
                        lambda *a, **k: pytest.fail("a second backend was loaded"))
    try:
        runtime.get_shared_second_pass(opf_config(tmp_path, "de"))
        with pytest.raises(RuntimeError, match="restart"):
            runtime.get_shared_second_pass(encoder_config(tmp_path, "de"))
    finally:
        runtime.close_shared_second_pass()


def test_invalid_mode_lists_all_modes():
    config = runtime.traciora_config("en").with_overrides({"detector": {"second_pass": "bogus"}})
    for call in (runtime.prepare_second_pass, runtime.get_shared_second_pass):
        with pytest.raises(ValueError) as excinfo:
            call(config)
        assert all(mode in str(excinfo.value) for mode in ("none", "opf", "encoder"))
