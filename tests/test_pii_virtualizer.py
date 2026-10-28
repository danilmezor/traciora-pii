"""Tests for PIIVirtualizer and StreamingRefRehydrator."""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_vault import PIIVaultStore, REF_TOKEN_PATTERN
from traciora_pii.pii_virtualizer import PIIVirtualizer, StreamingRefRehydrator


@pytest.fixture()
def vault(tmp_path):
    key = Fernet.generate_key().decode()
    db = tmp_path / "vault.db"
    v = PIIVaultStore(db_path=db, encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def virtualizer(vault):
    return PIIVirtualizer(vault)


@pytest.fixture()
def scope(vault):
    key = "customer:test_sess"
    vault.open_session(key)
    return key


class TestAnonymizeText:
    def test_phone_replaced_with_ref(self, virtualizer, scope):
        result = virtualizer.anonymize_text(scope, "Call me at +79128565950 please")
        assert "+79128565950" not in result
        assert "phone_ref:" in result

    def test_email_replaced_with_ref(self, virtualizer, scope):
        result = virtualizer.anonymize_text(scope, "Email me at user@example.com")
        assert "user@example.com" not in result
        assert "email_ref:" in result

    def test_multiple_pii_in_one_text(self, virtualizer, scope):
        result = virtualizer.anonymize_text(
            scope,
            "Phone: +79128565950, email: test@mail.ru"
        )
        assert "+79128565950" not in result
        assert "test@mail.ru" not in result
        assert "phone_ref:" in result
        assert "email_ref:" in result

    def test_empty_text_passthrough(self, virtualizer, scope):
        assert virtualizer.anonymize_text(scope, "") == ""
        assert virtualizer.anonymize_text(scope, None) is None

    def test_no_pii_text_unchanged(self, virtualizer, scope):
        text = "Hello, how are you?"
        assert virtualizer.anonymize_text(scope, text) == text


class TestIdempotency:
    def test_already_tokenized_text_unchanged(self, virtualizer, scope):
        text = "Call me at +79128565950"
        first_pass = virtualizer.anonymize_text(scope, text)
        second_pass = virtualizer.anonymize_text(scope, first_pass)
        assert first_pass == second_pass

    def test_deterministic_same_value(self, virtualizer, scope):
        t1 = virtualizer.anonymize_text(scope, "+79128565950")
        t2 = virtualizer.anonymize_text(scope, "+79128565950")
        assert t1 == t2


class TestRehydrateText:
    def test_roundtrip_phone(self, virtualizer, scope):
        original = "Call +79128565950 now"
        anonymized = virtualizer.anonymize_text(scope, original)
        rehydrated = virtualizer.rehydrate_text(scope, anonymized)
        assert rehydrated == original

    def test_roundtrip_email(self, virtualizer, scope):
        original = "Email test@example.com please"
        anonymized = virtualizer.anonymize_text(scope, original)
        rehydrated = virtualizer.rehydrate_text(scope, anonymized)
        assert rehydrated == original

    def test_unresolved_ref_left_as_is(self, virtualizer, scope):
        text = "Use phone_ref:ph_00000000 please"
        result = virtualizer.rehydrate_text(scope, text)
        assert "phone_ref:ph_00000000" in result


class TestStructured:
    def test_anonymize_dict(self, virtualizer, scope):
        payload = {"phone": "+79128565950", "name": "Олег"}
        result = virtualizer.anonymize_structured(scope, payload)
        assert "+79128565950" not in str(result)
        assert "phone_ref:" in str(result)
        # Name is preserved (not a detected PII type)
        assert result["name"] == "Олег"

    def test_rehydrate_dict(self, virtualizer, scope):
        payload = {"phone": "+79128565950"}
        anon = virtualizer.anonymize_structured(scope, payload)
        restored = virtualizer.rehydrate_structured(scope, anon)
        assert restored["phone"] == "+79128565950"

    def test_nested_list(self, virtualizer, scope):
        payload = [{"data": ["+79128565950"]}]
        anon = virtualizer.anonymize_structured(scope, payload)
        assert "+79128565950" not in str(anon)
        restored = virtualizer.rehydrate_structured(scope, anon)
        assert restored[0]["data"][0] == "+79128565950"


class TestStreamingRefRehydrator:
    def test_complete_ref_in_single_chunk(self, virtualizer, scope):
        # First create a ref token
        anon = virtualizer.anonymize_text(scope, "+79128565950")
        ref_token = anon.strip()

        rehydrator = StreamingRefRehydrator(virtualizer, scope)
        result = rehydrator.feed(f"Call {ref_token} now")
        result += rehydrator.flush()
        assert "+79128565950" in result

    def test_ref_split_across_chunks(self, virtualizer, scope):
        anon = virtualizer.anonymize_text(scope, "+79128565950")
        ref_token = anon.strip()

        rehydrator = StreamingRefRehydrator(virtualizer, scope)
        mid = len(ref_token) // 2
        chunk1 = f"Call {ref_token[:mid]}"
        chunk2 = f"{ref_token[mid:]} now"

        out1 = rehydrator.feed(chunk1)
        out2 = rehydrator.feed(chunk2)
        out3 = rehydrator.flush()
        full = out1 + out2 + out3
        assert "+79128565950" in full

    def test_no_ref_passes_through(self, virtualizer, scope):
        rehydrator = StreamingRefRehydrator(virtualizer, scope)
        out = rehydrator.feed("Hello world")
        out += rehydrator.flush()
        assert "Hello world" in out

    def test_flush_on_empty_buffer(self, virtualizer, scope):
        rehydrator = StreamingRefRehydrator(virtualizer, scope)
        assert rehydrator.flush() == ""


class TestLandlineDetection:
    def test_landline_phone_anonymized(self, virtualizer, scope):
        text = "Call +7 (3412) 60-21-20 for office"
        result = virtualizer.anonymize_text(scope, text)
        assert "+7 (3412) 60-21-20" not in result
        assert "phone_ref:" in result

    def test_moscow_landline(self, virtualizer, scope):
        text = "Moscow office: 8 (495) 123-45-67"
        result = virtualizer.anonymize_text(scope, text)
        assert "8 (495) 123-45-67" not in result


class TestInputNormalization:
    """Regression tests for zero-width + NFKC normalization at the
    virtualizer boundary. Added after the §3 evaluation surfaced leaks
    driven by U+200B inserted between phone digits.
    """

    def test_zero_width_inside_phone_gets_anonymized(self, virtualizer, scope):
        # `+7 912 8​56 5950` with U+200B between digits is the attack shape.
        # Before normalization, the U+200B breaks the `\d{3}` body match
        # and the phone leaks raw. After stripping, the ru_phone regex
        # catches it.
        text = "Reach me at +7 912 8​56 5950 today."
        result = virtualizer.anonymize_text(scope, text)
        assert "​" not in result, "zero-width char survived normalization"
        assert "+7 912" not in result, f"raw phone fragment survived: {result!r}"
        assert "phone_ref:" in result, f"phone not anonymized: {result!r}"

    def test_nfkc_collapses_compatibility_digits(self, virtualizer, scope):
        # Full-width digits (U+FF10..U+FF19) are NFKC-equivalent to ASCII
        # digits. Without NFKC, the regex misses; with NFKC, the canonical
        # form matches normally. Use US-shaped phone (matched by us_phone
        # in ru.yaml, which is the default policy).
        full_width = "Call ４１５-５５５-２６７１."  # 415-555-2671
        result = virtualizer.anonymize_text(scope, full_width)
        assert "phone_ref:" in result, f"NFKC did not enable detection: {result!r}"
        assert "415-555-2671" not in result


class TestVaultTransaction:
    def test_anonymize_text_commits_once(self, vault, virtualizer, scope):
        from test_pii_vault import _CommitCounter
        counter = _CommitCounter(vault._conn)
        vault._conn = counter
        text = " ".join(f"user{i}@example.com" for i in range(50))
        result = virtualizer.anonymize_text(scope, text)
        assert result.count("email_ref:") == 50
        assert counter.commits == 1

    def test_anonymize_text_failure_leaves_no_tokens(self, vault, virtualizer, scope, monkeypatch):
        real = vault.get_or_create_token
        calls = []

        def failing(*args, **kwargs):
            calls.append(args)
            if len(calls) == 3:
                raise RuntimeError("vault write failed")
            return real(*args, **kwargs)

        monkeypatch.setattr(vault, "get_or_create_token", failing)
        text = " ".join(f"user{i}@example.com" for i in range(5))
        with pytest.raises(RuntimeError, match="vault write failed"):
            virtualizer.anonymize_text(scope, text)
        assert vault.resolve_all_tokens(scope) == {}


def test_anonymize_text_propagates_second_pass_failure(vault, scope):
    from traciora_pii.opf_pii_detector import OPFInferenceError

    class _FailingSecondPass:
        def detect(self, text):
            raise OPFInferenceError("boom")

    virtualizer = PIIVirtualizer(vault, second_pass_detector=_FailingSecondPass())
    text = "Ivan Petrov, call +79128565950"
    returned = []
    with pytest.raises(OPFInferenceError, match="boom"):
        returned.append(virtualizer.anonymize_text(scope, text))
    assert returned == []


def test_build_second_pass_detector_encoder_mode(monkeypatch):
    from traciora_pii import runtime
    from traciora_pii.pii_virtualizer import build_second_pass_detector

    made = []

    class Stub:
        @classmethod
        def from_config(cls, encoder_cfg, *, model_name=None):
            made.append((encoder_cfg, model_name))
            return cls()

    monkeypatch.setattr("traciora_pii.encoder_pii_detector.EncoderPIIDetector", Stub)
    config = runtime.traciora_config("de").with_overrides({"detector": {"second_pass": "encoder"}})
    assert isinstance(build_second_pass_detector(config, encoder_model_name="de"), Stub)
    assert made == [(config.detector.encoder, "de")]


def test_build_second_pass_detector_invalid_mode_lists_all_modes():
    from traciora_pii.config import PIIConfig
    from traciora_pii.pii_virtualizer import build_second_pass_detector

    config = PIIConfig().with_overrides({"detector": {"second_pass": "bogus"}})
    with pytest.raises(ValueError, match="none, opf, encoder, llm, presidio"):
        build_second_pass_detector(config)
