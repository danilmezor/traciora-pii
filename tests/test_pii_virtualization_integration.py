"""End-to-end integration test for PII virtualization.

Reproduces the exact trace failure scenario:
1. User asks about a client → gets disambiguation table with phones
2. User selects a phone → model calls tool with ref → broker resolves → tool finds customer
3. Verify no raw PII in any model-bound payload
4. Verify streamed output contains real phone numbers (rehydrated)
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_vault import PIIVaultStore, REF_TOKEN_PATTERN
from traciora_pii.pii_virtualizer import PIIVirtualizer, StreamingRefRehydrator
from traciora_pii.pii_tool_broker import make_pii_brokered_tool
from traciora_pii.pii_leak_guard import assert_model_safe, PIILeakError


@pytest.fixture()
def vault(tmp_path):
    key = Fernet.generate_key().decode()
    db = tmp_path / "vault.db"
    v = PIIVaultStore(db_path=db, encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def scope(vault):
    key = "manager:42:sess_integration"
    vault.open_session(key)
    return key


@pytest.fixture()
def virtualizer(vault):
    return PIIVirtualizer(vault)


class FakeTool:
    def __init__(self, name, fn):
        self.name = name
        self.description = f"Tool {name}"
        self._fn = fn
        self.last_input = None

    def invoke(self, input_data, config=None, **kwargs):
        self.last_input = input_data
        return self._fn(input_data)


class TestTraceFailureScenario:
    """Reproduce the d3d37ecc / bb7b6a3a trace failure."""

    def test_disambiguation_then_selection_flow(self, vault, virtualizer, scope):
        # Step 1: User asks about client
        user_msg1 = "покажи информацию о клиенте Занозин Олег"
        anon_msg1 = virtualizer.anonymize_text(scope, user_msg1)
        # Name "Занозин Олег" is NOT detected as PII → stays in text
        assert "Занозин Олег" in anon_msg1

        # Step 2: Tool returns disambiguation with phones
        tool_result = (
            "Найдено 2 клиента:\n"
            "1. Занозин Олег — +79128565950, Коммунаров\n"
            "2. Занозин Олег — +7 (3412) 60-21-20, Орджоникидзе"
        )
        anon_result = virtualizer.anonymize_text(scope, tool_result)
        # Both phones should be replaced with refs
        assert "+79128565950" not in anon_result
        assert "+7 (3412) 60-21-20" not in anon_result
        assert "phone_ref:" in anon_result

        # Step 3: Simulate model showing refs to user → user picks one
        # Model response (in ref form): user sees rehydrated output
        rehydrated_result = virtualizer.rehydrate_text(scope, anon_result)
        assert "+79128565950" in rehydrated_result
        assert "+7 (3412) 60-21-20" in rehydrated_result

        # Step 4: User sends the selected phone in next message
        user_msg2 = "+79128565950"
        anon_msg2 = virtualizer.anonymize_text(scope, user_msg2)
        # Should be the SAME ref token as created from tool result
        assert "phone_ref:" in anon_msg2
        assert "+79128565950" not in anon_msg2

        # Step 5: Verify model payload is clean
        messages = [
            {"role": "user", "content": anon_msg1},
            {"role": "assistant", "content": anon_result},
            {"role": "user", "content": anon_msg2},
        ]
        # Leak guard should pass — no raw PII
        assert_model_safe(messages, scope, vault)

        # Step 6: Model calls tool with the ref → broker resolves
        def search_customer(kwargs):
            phone = kwargs.get("customer_phone", "")
            if phone == "+79128565950":
                return "Customer: Занозин Олег, VIP segment"
            return "Not found"

        tool = FakeTool("search_customer_manager_lvl", search_customer)
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)

        # Extract the ref token from the anonymized user message
        ref_token = anon_msg2.strip()
        result = wrapped.invoke({"customer_phone": ref_token})
        # Tool received real phone
        assert tool.last_input["customer_phone"] == "+79128565950"
        # Result should NOT contain raw phone back to model
        assert "+79128565950" not in result
        assert "Занозин Олег" in result  # Name passes through


class TestStreamingRehydration:
    def test_streamed_output_rehydrates(self, vault, virtualizer, scope):
        # Create a token
        ref = vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        model_output = f"Информация о клиенте с телефоном {ref}:\nСегмент: VIP"

        # Simulate streaming in small chunks
        chunk_size = 10
        chunks = [model_output[i:i+chunk_size] for i in range(0, len(model_output), chunk_size)]

        rehydrator = StreamingRefRehydrator(virtualizer, scope)
        received: list[str] = []
        for chunk in chunks:
            out = rehydrator.feed(chunk)
            if out:
                received.append(out)
        remaining = rehydrator.flush()
        if remaining:
            received.append(remaining)

        full_output = "".join(received)
        assert "+79128565950" in full_output
        assert ref not in full_output


class TestReAnonymizationFromHistory:
    """Messages stored raw → re-anonymized on each turn → same refs."""

    def test_re_anonymization_deterministic(self, vault, virtualizer, scope):
        raw_history_msg = "Клиент: +79128565950, запись на 15:00"
        anon1 = virtualizer.anonymize_text(scope, raw_history_msg)
        anon2 = virtualizer.anonymize_text(scope, raw_history_msg)
        # Same session, same value → identical ref tokens
        assert anon1 == anon2
        assert "+79128565950" not in anon1
