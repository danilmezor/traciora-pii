"""A value the session's vault already holds never reaches the model as text.

The detectors run on each text alone: a plate found because the user wrote
"plate" is missed in a tool result that does not say "plate", and a name the
second pass caught once can be missed the next time. ``anonymize_text``
therefore replaces every known value by its existing token first.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_leak_guard import assert_model_safe
from traciora_pii.pii_tool_broker import make_pii_brokered_tool
from traciora_pii.pii_vault import REF_TOKEN_PATTERN, PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer

PHONE = "+1 555 014 7392"
PLATE = "QZX-7315"


class FakeTool:
    def __init__(self, answer):
        self._answer = answer
        self.last_input = None

    def invoke(self, input_data, config=None, **kwargs):
        self.last_input = input_data
        return self._answer(input_data)


@pytest.fixture()
def key():
    return Fernet.generate_key().decode()


@pytest.fixture()
def vault(tmp_path, key):
    v = PIIVaultStore(db_path=tmp_path / "vault.db", encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def virtualizer(vault):
    return PIIVirtualizer(vault, excluded_categories=[])


@pytest.fixture()
def scope(vault):
    vault.open_session("chat:known")
    return "chat:known"


def _tokens(text):
    return REF_TOKEN_PATTERN.findall(text)


def test_tool_result_keeps_the_tokens_the_user_message_got(vault, virtualizer, scope):
    user = virtualizer.anonymize_text(scope, f"Remind me to call {PHONE} about the plate {PLATE}")
    phone_token, plate_token = _tokens(user)
    assert phone_token.startswith("phone_ref:") and plate_token.startswith("plate_ref:")

    tool = FakeTool(lambda args: f"Created “{args['title']}”")
    make_pii_brokered_tool(tool, scope, virtualizer)
    result = tool.invoke({"title": f"Call {phone_token} about {plate_token}"})

    assert tool.last_input["title"] == f"Call {PHONE} about {PLATE}"
    assert result == f"Created “Call {phone_token} about {plate_token}”"
    assert_model_safe([user, result], scope, vault)
    assert virtualizer.rehydrate_text(scope, result) == f"Created “Call {PHONE} about {PLATE}”"


def test_the_same_value_in_another_format_gets_the_same_token(virtualizer, scope):
    plate_token = _tokens(virtualizer.anonymize_text(scope, f"the plate {PLATE}"))[0]
    phone_token = _tokens(virtualizer.anonymize_text(scope, f"call {PHONE}"))[0]
    text = "Seen QZX 7315, qzx7315 and QZX - 7315; dial +15550147392 or +1-555-014-7392."
    assert virtualizer.anonymize_text(scope, text) == (
        f"Seen {plate_token}, {plate_token} and {plate_token}; "
        f"dial {phone_token} or {phone_token}."
    )


def test_a_value_inside_a_longer_number_or_word_is_left_alone(vault, virtualizer, scope):
    vault.get_or_create_token(scope, "[LICENSE_PLATE]", PLATE)
    vault.get_or_create_token(scope, "[PERSON_NAME]", "Marta")
    vault.get_or_create_token(scope, "[PERSONAL_ID]", "48213907")
    text = ("QZX-73150 1QZX-7315 QZX-7315.4 a.QZX-7315 Martas Martapol marta Anna-Marta "
            "248213907 48213907.5 48213907-2")
    assert virtualizer.anonymize_text(scope, text) == text


def test_values_with_their_own_punctuation_around_them_are_replaced(vault, virtualizer, scope):
    name = vault.get_or_create_token(scope, "[PERSON_NAME]", "Marta")
    plate = vault.get_or_create_token(scope, "[LICENSE_PLATE]", PLATE)
    text = f"Marta's car ({PLATE}) belongs to Marta. Name:Marta, plate:{PLATE}!"
    assert virtualizer.anonymize_text(scope, text) == (
        f"{name}'s car ({plate}) belongs to {name}. Name:{name}, plate:{plate}!"
    )


def test_short_values_are_never_replaced_from_the_vault(vault, virtualizer, scope):
    vault.get_or_create_token(scope, "[PERSON_NAME]", "Bob")
    vault.get_or_create_token(scope, "[PERSONAL_ID]", "42")
    vault.get_or_create_token(scope, "[PERSONAL_ID]", "48213")
    text = "Bob paid 42 for order 48213"
    assert virtualizer.anonymize_text(scope, text) == text


def test_the_longest_known_value_wins(vault, virtualizer, scope):
    short = vault.get_or_create_token(scope, "[PERSON_NAME]", "Marta Lind")
    full = vault.get_or_create_token(scope, "[PERSON_NAME]", "Marta Lindqvist")
    assert virtualizer.anonymize_text(scope, "Marta Lindqvist and Marta Lind") == (
        f"{full} and {short}"
    )


def test_an_existing_token_is_not_tokenized_again(vault, virtualizer, scope):
    email = virtualizer.anonymize_text(scope, "anna@example.com")
    # A vault value that equals the inside of a token must not cut it.
    vault.get_or_create_token(scope, "[SECRET]", email.split(":", 1)[1])
    text = f"Write to {email} today"
    assert virtualizer.anonymize_text(scope, text) == text
    assert virtualizer.anonymize_text(scope, virtualizer.anonymize_text(scope, text)) == text


def test_a_value_added_mid_session_is_known_on_the_next_call(tmp_path, key, vault, virtualizer, scope):
    text = f"Ticket for {PLATE} closed"
    assert virtualizer.anonymize_text(scope, text) == text  # no keyword, not known yet
    plate = _tokens(virtualizer.anonymize_text(scope, f"the plate {PLATE}"))[0]
    assert virtualizer.anonymize_text(scope, text) == f"Ticket for {plate} closed"

    # A token written by another process on the same vault file is seen too.
    other = PIIVaultStore(db_path=tmp_path / "vault.db", encryption_key=key)
    try:
        name = other.get_or_create_token(scope, "[PERSON_NAME]", "Marta")
    finally:
        other.close()
    assert virtualizer.anonymize_text(scope, f"{text} by Marta") == (
        f"Ticket for {plate} closed by {name}"
    )


def test_a_rolled_back_token_is_not_kept_as_known(vault, virtualizer, scope):
    with pytest.raises(RuntimeError):
        with vault.batch():
            vault.get_or_create_token(scope, "[PERSON_NAME]", "Marta")
            assert virtualizer.anonymize_text(scope, "Marta").startswith("person_name_ref:")
            raise RuntimeError("roll back")
    lena = vault.get_or_create_token(scope, "[PERSON_NAME]", "Lena")
    assert virtualizer.anonymize_text(scope, "Marta and Lena") == f"Marta and {lena}"


def test_two_sessions_do_not_share_values(vault, virtualizer, scope):
    vault.open_session("chat:other")
    virtualizer.anonymize_text(scope, f"the plate {PLATE}")
    text = f"Ticket for {PLATE} closed"
    assert virtualizer.anonymize_text("chat:other", text) == text


def test_a_closed_session_has_no_known_values(vault, virtualizer, scope):
    vault.get_or_create_token(scope, "[LICENSE_PLATE]", PLATE)
    vault.close_session(scope)
    assert virtualizer.anonymize_text(scope, f"Ticket for {PLATE}") == f"Ticket for {PLATE}"


def test_an_excluded_category_stays_visible(vault, scope):
    vault.get_or_create_token(scope, "[LICENSE_PLATE]", PLATE)
    name = vault.get_or_create_token(scope, "[PERSON_NAME]", "Marta")
    virtualizer = PIIVirtualizer(vault, excluded_categories=["plate"])
    assert virtualizer.anonymize_text(scope, f"Marta drives {PLATE}") == f"{name} drives {PLATE}"


def test_a_name_the_second_pass_caught_once_is_replaced_after(vault, scope):
    seen = []

    class SecondPass:
        def __init__(self):
            self.catch = True

        def detect(self, text):
            seen.append(text)
            if not self.catch or "Marta Lind" not in text:
                return []
            start = text.index("Marta Lind")
            return [SimpleNamespace(detector="fake", start=start, end=start + 10,
                                    placeholder="[PERSON_NAME]", lemma="Marta Lind")]

    second_pass = SecondPass()
    virtualizer = PIIVirtualizer(vault, second_pass_detector=second_pass, excluded_categories=[])
    first = virtualizer.anonymize_text(scope, f"Marta Lind called from {PHONE}")
    name, phone = _tokens(first)
    assert name.startswith("person_name_ref:")

    second_pass.catch = False
    later = virtualizer.anonymize_text(scope, f"Order for Marta Lind, phone {PHONE}")
    assert later == f"Order for {name}, phone {phone}"
    # The second pass still runs after the regex pass, on tokenized text.
    assert seen[-1] == later


def test_known_values_decrypt_only_what_is_new(vault, scope, monkeypatch):
    for i in range(5):
        vault.get_or_create_token(scope, "[EMAIL]", f"user{i}@example.com")
    decrypted = []
    real = vault._fernet.decrypt
    monkeypatch.setattr(vault._fernet, "decrypt", lambda c: decrypted.append(c) or real(c))

    first = vault.known_values(scope)
    assert len(first.values) == 5 and len(decrypted) == 5
    assert vault.known_values(scope) is first and len(decrypted) == 5

    vault.get_or_create_token(scope, "[EMAIL]", "user5@example.com")
    grown = vault.known_values(scope)
    assert grown is not first and len(grown.values) == 6 and len(decrypted) == 6
    assert grown.values[-1].raw_value == "user5@example.com"
