"""Tests for PII leak guard: blocks raw PII, passes refs, exception type."""
from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_leak_guard import PIILeakError, assert_model_safe
from traciora_pii.pii_vault import PIIVaultStore


@pytest.fixture()
def vault(tmp_path):
    key = Fernet.generate_key().decode()
    db = tmp_path / "vault.db"
    v = PIIVaultStore(db_path=db, encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def scope(vault):
    key = "customer:guard_test"
    vault.open_session(key)
    return key


class TestLeakGuard:
    def test_blocks_raw_pii_in_vault(self, vault, scope):
        vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        payload = [{"role": "user", "content": "Call +79128565950"}]
        with pytest.raises(PIILeakError) as exc_info:
            assert_model_safe(payload, scope, vault)
        assert exc_info.value.pii_type == "[PHONE]"

    def test_passes_ref_tokens(self, vault, scope):
        ref = vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        payload = [{"role": "user", "content": f"Call {ref}"}]
        # Should not raise
        assert_model_safe(payload, scope, vault)

    def test_passes_unknown_pii_not_in_vault(self, vault, scope):
        # PII that matches a pattern but is NOT in the vault should be OK
        payload = [{"role": "user", "content": "Call +79031234567"}]
        assert_model_safe(payload, scope, vault)

    def test_passes_empty_payload(self, vault, scope):
        assert_model_safe([], scope, vault)
        assert_model_safe(None, scope, vault)

    def test_passes_when_vault_empty(self, vault, scope):
        payload = [{"role": "user", "content": "Call +79128565950"}]
        assert_model_safe(payload, scope, vault)

    def test_disabled_by_env(self, vault, scope, monkeypatch):
        monkeypatch.setenv("PII_LEAK_GUARD_ENABLED", "false")
        vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        payload = [{"role": "user", "content": "Call +79128565950"}]
        # Should not raise when disabled
        assert_model_safe(payload, scope, vault)

    def test_nested_payload_detected(self, vault, scope):
        vault.get_or_create_token(scope, "[EMAIL]", "test@example.com")
        payload = {"messages": [{"content": "Email test@example.com"}]}
        with pytest.raises(PIILeakError):
            assert_model_safe(payload, scope, vault)

    def test_exception_has_preview(self, vault, scope):
        vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        payload = "Call +79128565950"
        with pytest.raises(PIILeakError) as exc_info:
            assert_model_safe(payload, scope, vault)
        assert "***" in exc_info.value.preview

    def test_catches_card_in_tool_exception(self, vault, scope):
        """Archetype 3: a stringified upstream exception with raw card must trigger the guard.

        Production threat model: a tool raises and the agent loop in
        `pipelines.py` stringifies the exception verbatim into the next
        ToolMessage. The leak guard must inspect that content and raise.
        """
        card = "4111 1111 1111 1111"
        vault.get_or_create_token(scope, "[CARD]", card)
        payload = [
            {"role": "human", "content": "Run failing payment lookup."},
            {"role": "tool", "content": f"error: upstream rejected card={card}"},
        ]
        with pytest.raises(PIILeakError) as exc_info:
            assert_model_safe(payload, scope, vault)
        assert exc_info.value.pii_type == "[CARD]"
