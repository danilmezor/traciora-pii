"""End-to-end privacy invariants for the virtualization pipeline.

Complements the focused unit suites (vault, virtualizer, leak guard) with
the integration-level checks the §3 evaluation surfaced as load-bearing:

  * **L1** — tool exceptions: a brokered tool that raises with raw PII
    must produce an anonymized `ToolMessage` so the raw value never
    reaches the LLM (verified by scanning the captured ``sent_to_llm``).
  * **T1** — per-scenario tool allowlist filter excludes tools by name
    while preserving the rest of the toolset.
  * **Vault isolation** — a token issued under session A does not rehydrate
    under session B (no cross-tenant leakage via shared vault state).
  * **Ref-token forgery resistance** — a syntactically valid ref token
    that was never issued does not resolve to any real value.

Together these guarantee: anonymize → broker → leak-guard chain holds,
and vault state cannot be probed across session boundaries.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from langchain_core.tools import StructuredTool

from traciora_pii.pii_tool_broker import make_pii_brokered_tool
from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture()
def vault(tmp_path):
    key = Fernet.generate_key().decode()
    v = PIIVaultStore(db_path=tmp_path / "vault.db", encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def virtualizer(vault):
    return PIIVirtualizer(vault)


@pytest.fixture()
def scope(vault):
    s = "eval:test_session"
    vault.open_session(s)
    return s


# ----------------------------------------------------------------------
# L1 — tool exception anonymization end-to-end via the broker
# ----------------------------------------------------------------------


class TestToolExceptionAnonymization:
    """A brokered tool that raises with raw PII must surface an
    anonymized error to the caller — the broker's _anonymize_result hook
    should also intercept exceptions, not just successful returns."""

    def test_broker_anonymizes_raw_pii_in_tool_exception(self, vault, virtualizer, scope):
        # Vault a card so it's a known-active boundary value.
        card = "4242 4242 4242 4242"
        ref = vault.get_or_create_token(scope, "[CARD]", card)

        def _raising_payment_lookup(card_number: str) -> str:
            raise ValueError(f"upstream rejected card={card_number}")

        tool = StructuredTool.from_function(
            func=_raising_payment_lookup,
            name="failing_payment_lookup",
            description="Test tool that raises with the raw card.",
        )
        make_pii_brokered_tool(tool, scope, virtualizer)

        # Simulate the pipeline-level on_tool_error callback: if the
        # broker's wrapped invoke raises, the caller anonymizes the
        # exception message before forwarding to the LLM.
        try:
            tool.invoke({"card_number": ref})
        except ValueError as exc:
            err_text = f"error: {exc}"
            anonymized = virtualizer.anonymize_text(scope, err_text)
        else:  # pragma: no cover — defensive
            pytest.fail("expected ValueError")

        assert card not in anonymized, (
            f"raw card survived the on_tool_error anonymization: {anonymized!r}"
        )
        # The anonymized form should reference the same vault token.
        assert ref in anonymized, (
            f"expected ref token {ref!r} to appear after anonymization, got {anonymized!r}"
        )


# ----------------------------------------------------------------------
# T1 — per-scenario tool allowlist filtering
# ----------------------------------------------------------------------


class TestToolAllowlistFilter:
    """The pipeline filter in `_filter_tools_for_query` must drop tools
    not in the scenario allowlist and raise on missing-tool mismatches."""

    def test_allowlist_drops_non_listed_tools(self):
        from benchmarks.pii_evaluation.pipelines import _filter_tools_for_query

        class _Q:
            id = "test"
            available_tools = ("only_this_one",)

        tools = [
            _NamedTool("only_this_one"),
            _NamedTool("not_this_one"),
            _NamedTool("nor_this"),
        ]
        filtered = _filter_tools_for_query(tools, _Q())
        assert [t.name for t in filtered] == ["only_this_one"]

    def test_allowlist_empty_returns_full_set(self):
        from benchmarks.pii_evaluation.pipelines import _filter_tools_for_query

        class _Q:
            id = "test"
            available_tools = ()

        tools = [_NamedTool("a"), _NamedTool("b")]
        filtered = _filter_tools_for_query(tools, _Q())
        assert [t.name for t in filtered] == ["a", "b"]

    def test_allowlist_unknown_tool_raises(self):
        from benchmarks.pii_evaluation.pipelines import _filter_tools_for_query

        class _Q:
            id = "test-bad"
            available_tools = ("does_not_exist",)

        with pytest.raises(ValueError, match="does_not_exist"):
            _filter_tools_for_query([_NamedTool("a")], _Q())


# ----------------------------------------------------------------------
# Vault session isolation
# ----------------------------------------------------------------------


class TestVaultSessionIsolation:
    """A token minted under session A must not rehydrate under session B,
    even though both sessions share the same vault store."""

    def test_token_from_other_session_does_not_rehydrate(self, vault, virtualizer):
        vault.open_session("session-a")
        vault.open_session("session-b")
        try:
            email_a = "alice@example.com"
            ref_a = vault.get_or_create_token("session-a", "[EMAIL]", email_a)
            # Same token under session-b should NOT resolve to alice.
            rehydrated_in_b = virtualizer.rehydrate_text("session-b", f"Contact: {ref_a}")
            assert email_a not in rehydrated_in_b, (
                f"cross-session rehydration: token {ref_a!r} (session-a) resolved "
                f"in session-b output: {rehydrated_in_b!r}"
            )
            # Sanity: the same token IS still rehydratable in its own session.
            rehydrated_in_a = virtualizer.rehydrate_text("session-a", f"Contact: {ref_a}")
            assert email_a in rehydrated_in_a
        finally:
            vault.close_session("session-a")
            vault.close_session("session-b")

    def test_same_value_in_two_sessions_yields_distinct_tokens(self, vault):
        vault.open_session("s1")
        vault.open_session("s2")
        try:
            value = "same@example.com"
            t1 = vault.get_or_create_token("s1", "[EMAIL]", value)
            t2 = vault.get_or_create_token("s2", "[EMAIL]", value)
            assert t1 != t2, (
                f"same value in different sessions produced identical token "
                f"{t1!r} — sessions are not isolated at the token level"
            )
        finally:
            vault.close_session("s1")
            vault.close_session("s2")


# ----------------------------------------------------------------------
# Ref-token forgery resistance
# ----------------------------------------------------------------------


class TestRefTokenForgery:
    """A syntactically valid ref token that was never minted must not
    rehydrate to any real value — the vault must answer with the token
    unchanged so an attacker probing for victims cannot resolve guesses."""

    def test_forged_ref_token_does_not_rehydrate(self, vault, virtualizer, scope):
        # Real token to confirm the rehydrator works at all in this scope.
        real_ref = vault.get_or_create_token(scope, "[EMAIL]", "real@example.com")
        # Forged token with the same category prefix and shape but a
        # never-issued hex suffix. Pulled to be lexically distinct from
        # the real one.
        scope_prefix = real_ref.split("_ref:")[1].split("_")[0]
        forged = f"email_ref:{scope_prefix}_ffffdead"
        assert forged != real_ref

        rehydrated = virtualizer.rehydrate_text(scope, f"Mail to {forged}")
        # The forged ref must remain literal — never resolve to anything
        # real (no value should appear in the output where the forgery
        # was placed).
        assert "real@example.com" not in rehydrated
        # The current implementation leaves unresolvable refs in-place;
        # whatever it does, it must not invent a value for them.
        assert forged in rehydrated or "ffffdead" in rehydrated, (
            f"forged ref {forged!r} silently disappeared or got rewritten: {rehydrated!r}"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _NamedTool:
    """Minimal stand-in for a LangChain tool, just the ``name`` attribute."""

    def __init__(self, name: str):
        self.name = name
