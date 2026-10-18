"""Tests for PIIVaultStore: CRUD, deterministic tokens, cross-session isolation, TTL, encryption."""
from __future__ import annotations

import os
import tempfile

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_vault import PIIVaultStore, REF_TOKEN_PATTERN


@pytest.fixture()
def vault(tmp_path):
    key = Fernet.generate_key().decode()
    db = tmp_path / "vault.db"
    v = PIIVaultStore(db_path=db, encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def session_key():
    return "manager:42:sess_abc123"


class TestSessionLifecycle:
    def test_open_session_returns_nonce(self, vault, session_key):
        nonce = vault.open_session(session_key)
        assert nonce and isinstance(nonce, str)

    def test_open_session_is_idempotent(self, vault, session_key):
        n1 = vault.open_session(session_key)
        n2 = vault.open_session(session_key)
        assert n1 == n2

    def test_close_and_reopen_gives_new_nonce(self, vault, session_key):
        n1 = vault.open_session(session_key)
        vault.close_session(session_key)
        n2 = vault.open_session(session_key)
        assert n2 != n1

    def test_resolve_fails_on_closed_session(self, vault, session_key):
        vault.open_session(session_key)
        ref = vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        vault.close_session(session_key)
        assert vault.resolve_token(session_key, ref) is None


class TestDeterministicTokens:
    def test_same_value_same_session_returns_same_token(self, vault, session_key):
        vault.open_session(session_key)
        t1 = vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        t2 = vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        assert t1 == t2

    def test_same_value_normalized_variants(self, vault, session_key):
        vault.open_session(session_key)
        t1 = vault.get_or_create_token(session_key, "[PHONE]", "+7 (912) 856-59-50")
        t2 = vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        assert t1 == t2

    def test_different_values_give_different_tokens(self, vault, session_key):
        vault.open_session(session_key)
        t1 = vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        t2 = vault.get_or_create_token(session_key, "[PHONE]", "+79031234567")
        assert t1 != t2

    def test_different_pii_types_give_different_tokens(self, vault, session_key):
        vault.open_session(session_key)
        t1 = vault.get_or_create_token(session_key, "[PHONE]", "value")
        t2 = vault.get_or_create_token(session_key, "[EMAIL]", "value")
        assert t1 != t2

    def test_token_format_matches_pattern(self, vault, session_key):
        vault.open_session(session_key)
        ref = vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        assert REF_TOKEN_PATTERN.fullmatch(ref), f"Token {ref!r} does not match pattern"


class TestCrossSessionIsolation:
    def test_different_sessions_give_different_tokens(self, vault):
        k1, k2 = "customer:sess1", "customer:sess2"
        vault.open_session(k1)
        vault.open_session(k2)
        t1 = vault.get_or_create_token(k1, "[PHONE]", "+79128565950")
        t2 = vault.get_or_create_token(k2, "[PHONE]", "+79128565950")
        assert t1 != t2

    def test_resolve_in_wrong_session_returns_none(self, vault):
        k1, k2 = "customer:sess1", "customer:sess2"
        vault.open_session(k1)
        vault.open_session(k2)
        ref = vault.get_or_create_token(k1, "[PHONE]", "+79128565950")
        assert vault.resolve_token(k2, ref) is None


class TestEncryptionRoundtrip:
    def test_resolve_returns_original_value(self, vault, session_key):
        vault.open_session(session_key)
        raw = "+79128565950"
        ref = vault.get_or_create_token(session_key, "[PHONE]", raw)
        resolved = vault.resolve_token(session_key, ref)
        assert resolved == raw

    def test_resolve_all_tokens(self, vault, session_key):
        vault.open_session(session_key)
        vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        vault.get_or_create_token(session_key, "[EMAIL]", "test@example.com")
        all_tokens = vault.resolve_all_tokens(session_key)
        assert len(all_tokens) == 2
        assert "+79128565950" in all_tokens.values()
        assert "test@example.com" in all_tokens.values()


class TestCleanup:
    def test_cleanup_expired_with_zero_ttl(self, vault, session_key):
        vault.open_session(session_key)
        vault.get_or_create_token(session_key, "[PHONE]", "+79128565950")
        count = vault.cleanup_expired(ttl_hours=0)
        assert count == 1

    def test_cleanup_does_not_expire_fresh(self, vault, session_key):
        vault.open_session(session_key)
        count = vault.cleanup_expired(ttl_hours=24)
        assert count == 0
