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
    virtualizer boundary: a U+200B inserted between phone digits must not
    let the phone through.
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


def test_bare_id_must_belong_to_one_token_in_the_active_session(vault, virtualizer, scope, monkeypatch):
    monkeypatch.setattr(vault, "_generate_short_id", lambda prefix: "pe_41854987")
    name = vault.get_or_create_token(scope, "[PERSON_NAME]", "Alex")
    bare = name.split(":")[1]
    vault.open_session("another")
    assert virtualizer.rehydrate_text("another", bare) == bare
    assert virtualizer.rehydrate_structured(scope, {"brief": [bare]}) == {"brief": ["Alex"]}
    identifier = vault.get_or_create_token(scope, "[PERSONAL_ID]", "010203004")
    assert name != identifier
    assert virtualizer.rehydrate_text(scope, f"{name} {bare} {identifier}") == f"Alex {bare} 010203004"
    vault.get_or_create_token("another", "[PERSON_NAME]", "Sam")
    vault.close_session("another")
    assert virtualizer.rehydrate_text("another", bare) == bare


def test_phone_prediction_inside_a_task_uuid_preserves_the_id(vault, scope):
    from types import SimpleNamespace
    identifier = "cafeaabb-cafe-4abc-8349-6571045f3075"

    class Detector:
        def detect(self, text):
            start = text.index("8349-6571045")
            return [SimpleNamespace(start=start, end=start + 12, placeholder="[PHONE]", lemma=None)]

    virtualizer = PIIVirtualizer(vault, second_pass_detector=Detector())
    assert virtualizer.anonymize_structured(scope, {"task": identifier}) == {"task": identifier}


@pytest.mark.parametrize("template", ["{}@example.org", "https://example.org/a?q={}&case=123"])
@pytest.mark.parametrize("bare", [False, True])
def test_regex_masks_parts_around_an_existing_ref_without_wrapping_it(vault, scope, template, bare):
    token = vault.get_or_create_token(scope, "[PERSON_NAME]", "Dana Vole")
    virtualizer = PIIVirtualizer(vault, excluded_categories=[])
    masked = virtualizer.anonymize_text(scope, template.format(token.split(":")[1] if bare else token))
    assert token in masked
    assert "example.org" not in masked
    assert virtualizer.rehydrate_text(scope, masked) == template.format("Dana Vole")


def test_a_bare_reference_without_a_unique_owner_is_not_expanded(vault, scope):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(vault, "_generate_short_id", lambda _prefix: "pe_123abcde")
        vault.get_or_create_token(scope, "[PERSON_NAME]", "Dana Vole")
        vault.get_or_create_token(scope, "[PERSONAL_ID]", "Test identity")
    virtualizer = PIIVirtualizer(vault, excluded_categories=[])
    source = "https://example.org/a?q=pe_123abcde&case=123"
    masked = virtualizer.anonymize_text(scope, source)
    assert "pe_123abcde" in masked
    assert "example.org" not in masked
    assert virtualizer.rehydrate_text(scope, masked) == source


@pytest.mark.parametrize(("source", "category"), [
    ("jd_19870512@example.test", "email_ref"),
    ("https://example.test/id/ab_1234abcd", "url_ref"),
])
def test_an_unknown_bare_shape_does_not_split_a_regex_value(vault, scope, source, category):
    virtualizer = PIIVirtualizer(vault, excluded_categories=[])
    masked = virtualizer.anonymize_text(scope, source)
    assert masked.startswith(category + ":")
    assert REF_TOKEN_PATTERN.fullmatch(masked)
    assert virtualizer.rehydrate_text(scope, masked) == source


def test_the_known_value_matcher_can_mask_an_unknown_bare_shape(vault, scope):
    source = "jd_19870512"
    token = vault.get_or_create_token(scope, "[PERSONAL_ID]", source)
    assert PIIVirtualizer(vault).anonymize_text(scope, source) == token


def test_second_pass_does_not_exempt_an_unknown_bare_shape(vault, scope):
    from traciora_pii._detector_common import LLMFinding

    class Detector:
        def detect(self, text):
            return [LLMFinding("synthetic", 0, len(text), "[PERSONAL_ID]", None)]

    source = "jd_19870512"
    pii = PIIVirtualizer(vault, second_pass_detector=Detector())
    masked = pii.anonymize_text(scope, source)
    assert masked.startswith("personal_id_ref:")
    assert pii.rehydrate_text(scope, masked) == source


def test_the_exemption_and_input_have_the_same_normalization(vault, scope):
    name = "Аптека №5"
    vault.get_or_create_token(scope, "[PERSON_NAME]", "Аптека No5")
    pii = PIIVirtualizer(vault, exempt_values=lambda: {name})
    assert pii.anonymize_text(scope, name) == "Аптека No5"


def test_exempt_values_are_current_and_match_exact_whole_words(vault, scope):
    import re
    from traciora_pii._detector_common import LLMFinding

    class Detector:
        def detect(self, text):
            return [LLMFinding("synthetic", match.start(), match.end(), "[PERSON_NAME]", None)
                    for match in re.finditer(r"Dana Voles?|dana vole", text)]

    exempt = {"Dana Vole", "reception@example.org"}
    virtualizer = PIIVirtualizer(vault, second_pass_detector=Detector(), exempt_values=lambda: exempt)
    # An existing token is still valid when it comes from the model, but the known-value
    # phase must leave that same exact spelling visible when it is now a site name.
    vault.get_or_create_token(scope, "[STREET_ADDRESS]", "Dana Vole")
    vault.get_or_create_token(scope, "[PERSON_NAME]", "Vole")
    vault.get_or_create_token(scope, "[EMAIL]", "reception@example.org")
    source = "Dana Vole; reception@example.org; dana vole; Dana Voles"
    masked = virtualizer.anonymize_text(scope, source)
    assert masked.startswith("Dana Vole; reception@example.org;")
    assert "dana vole" not in masked and "Dana Voles" not in masked
    assert virtualizer.rehydrate_text(scope, masked) == source
    exempt.clear()
    assert "Dana Vole" not in virtualizer.anonymize_text(scope, "Dana Vole")


def test_structured_batch_keeps_the_ordered_detector_inputs(vault, scope):
    from traciora_pii._detector_common import LLMFinding
    from traciora_pii.runtime import SharedSecondPass

    class Detector:
        def __init__(self):
            self.batches, self.scalars = [], []

        def findings(self, text):
            for value, context in (("Dana Vole", "Meet"), ("Robin Lake", "person_name_ref:"),
                                   ("Station House", "Leave")):
                if value in text and context in text:
                    start = text.index(value)
                    return [LLMFinding("synthetic", start, start + len(value), "[PERSON_NAME]", None)]
            return []

        def detect(self, text):
            self.scalars.append(text)
            return self.findings(text)

        def detect_many(self, texts):
            self.batches.append(texts)
            return [self.findings(text) for text in texts]

    detector = Detector()
    virtualizer = PIIVirtualizer(vault, second_pass_detector=SharedSecondPass(detector),
                                exempt_values=lambda: {"Station House"})
    first, second, third = ("Meet Dana Vole", "Tell Dana Vole to call Robin Lake",
                            "Leave a note for Station House")
    payload = {"items": [first, second, second, third], "again": first, "empty": ""}
    masked = virtualizer.anonymize_structured(scope, payload)
    assert detector.batches == [[first, second, third]]
    # Leaf B recognizes Robin only when the name learned in A has already become a ref.
    assert all("Dana Vole" not in text and "Robin Lake" not in text for text in masked["items"])
    assert masked["items"][-1] == third
    assert any(text.startswith("Tell person_name_ref:") and text.endswith("call Robin Lake")
               for text in detector.scalars)
    assert len(detector.scalars) == 3
    assert virtualizer.rehydrate_structured(scope, masked) == payload

    vault.open_session("scalar")
    sequential = {"items": [virtualizer.anonymize_text("scalar", text) for text in payload["items"]],
                  "again": virtualizer.anonymize_text("scalar", first), "empty": ""}
    def normalized(text):
        return REF_TOKEN_PATTERN.sub("[PERSON]", text)

    assert list(map(normalized, masked["items"])) == list(map(normalized, sequential["items"]))
    assert virtualizer.rehydrate_structured("scalar", sequential) == payload


def test_structured_regex_only_still_matches_an_ordered_walk(vault, scope):
    virtualizer = PIIVirtualizer(vault)
    source = ["Dana at dana@example.org", "dana@example.org", ""]
    masked = virtualizer.anonymize_structured(scope, source)
    assert masked == [virtualizer.anonymize_text(scope, text) for text in source]
    assert virtualizer.rehydrate_structured(scope, masked) == source


@pytest.mark.parametrize("phone", ["030 55501234", "+7 495 555-01-23", "+1 (202) 555-0123"])
@pytest.mark.parametrize("where", ["before", "after", "between"])
def test_phone_next_to_uuid_round_trips(vault, scope, phone, where):
    from traciora_pii.config import PIIConfig

    identifier = "12345678-cafe-4abc-8312-123456789012"
    other = "20000000-abcd-4def-bccd-000000000001"
    before, after = {"before": ("", identifier), "after": (identifier, ""), "between": (identifier, other)}[where]
    source = before + phone + after
    config = PIIConfig().with_overrides({"detector": {"regex_policies": ["de", "ru", "en"]}})
    pii = PIIVirtualizer(vault, config=config)
    masked = pii.anonymize_text(scope, source)
    assert REF_TOKEN_PATTERN.sub("[PHONE]", masked) == before + "[PHONE]" + after
    assert pii.rehydrate_text(scope, masked) == source


@pytest.mark.parametrize(("before", "after"), [
    ("", "2025550123"), ("0305550123", ""), ("0305550123", "2025550123"),
])
def test_a_phone_prediction_masks_only_the_parts_outside_the_uuid(vault, scope, before, after):
    from traciora_pii._detector_common import LLMFinding

    identifier = "cafeaabb-cafe-4abc-bcde-abcdefabcdef"
    source = before + identifier + after

    class Detector:
        def detect(self, text):
            return [LLMFinding("opf", 0, len(text), "[PHONE]", text)]

    pii = PIIVirtualizer(vault, second_pass_detector=Detector())
    masked = pii.anonymize_text(scope, source)
    assert REF_TOKEN_PATTERN.sub("[PHONE]", masked) == ("[PHONE]" if before else "") + identifier + ("[PHONE]" if after else "")
    assert pii.rehydrate_text(scope, masked) == source


def test_known_phone_token_beside_uuid_preserves_both(vault, scope, monkeypatch):
    from traciora_pii._detector_common import LLMFinding

    phone = "+12025550123"
    identifier = "cafeaabb-cafe-4abc-8349-6571045f3075"
    monkeypatch.setattr(vault, "_generate_short_id", lambda prefix: "ph_123abcde")
    token = vault.get_or_create_token(scope, "[PHONE]", phone)

    class Detector:
        def detect(self, text):
            start = text.index("8349-6571045")
            return [LLMFinding("opf", start, start + 12, "[PHONE]", "")]

    pii = PIIVirtualizer(vault, second_pass_detector=Detector())
    safe = pii.anonymize_text(scope, phone + identifier)
    assert safe == token + identifier
    assert pii.rehydrate_text(scope, safe) == phone + identifier
