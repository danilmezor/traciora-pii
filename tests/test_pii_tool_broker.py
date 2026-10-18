"""Tests for PII tool broker: ref resolution in args, re-tokenization of results."""
from __future__ import annotations

import json

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer
from traciora_pii.pii_tool_broker import make_pii_brokered_tool


class FakeTool:
    """Minimal mock that mimics a LangChain tool."""
    def __init__(self, name: str, return_value=None):
        self.name = name
        self.description = f"Fake tool {name}"
        self._return_value = return_value
        self.last_input = None

    def invoke(self, input_data, config=None, **kwargs):
        self.last_input = input_data
        if callable(self._return_value):
            return self._return_value(input_data)
        return self._return_value


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
    key = "customer:broker_test"
    vault.open_session(key)
    return key


class TestRefResolution:
    def test_string_arg_ref_resolved(self, vault, virtualizer, scope):
        ref = vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        tool = FakeTool("search_customer", return_value="found")
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        wrapped.invoke({"customer_phone": ref})
        assert tool.last_input["customer_phone"] == "+79128565950"

    def test_nested_dict_ref_resolved(self, vault, virtualizer, scope):
        ref = vault.get_or_create_token(scope, "[EMAIL]", "test@example.com")
        tool = FakeTool("update", return_value="ok")
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        wrapped.invoke({"contact": {"email": ref}})
        assert tool.last_input["contact"]["email"] == "test@example.com"

    def test_json_string_arg_resolved(self, vault, virtualizer, scope):
        ref = vault.get_or_create_token(scope, "[PHONE]", "+79128565950")
        json_str = json.dumps({"phone": ref})
        tool = FakeTool("filter_tool", return_value="ok")
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        wrapped.invoke({"filter_spec_json": json_str})
        parsed = json.loads(tool.last_input["filter_spec_json"])
        assert parsed["phone"] == "+79128565950"


class TestResultAnonymization:
    def test_string_result_anonymized(self, vault, virtualizer, scope):
        tool = FakeTool("lookup", return_value="Customer phone: +79128565950")
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        result = wrapped.invoke({"id": "123"})
        assert "+79128565950" not in result
        assert "phone_ref:" in result

    def test_none_result_passthrough(self, vault, virtualizer, scope):
        tool = FakeTool("void_tool", return_value=None)
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        assert wrapped.invoke({}) is None

    def test_dict_result_anonymized(self, vault, virtualizer, scope):
        tool = FakeTool("get_info", return_value={"phone": "+79128565950", "name": "Олег"})
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        result = wrapped.invoke({})
        assert "+79128565950" not in str(result)
        assert result["name"] == "Олег"


class TestToolMetadataPreserved:
    def test_name_preserved(self, vault, virtualizer, scope):
        tool = FakeTool("my_tool")
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        assert wrapped.name == "my_tool"

    def test_description_preserved(self, vault, virtualizer, scope):
        tool = FakeTool("my_tool")
        wrapped = make_pii_brokered_tool(tool, scope, virtualizer)
        assert wrapped.description == "Fake tool my_tool"


class TestTrailingPunctuationNormalization:
    """The broker collapses runs of trailing duplicate punctuation in
    rehydrated tool-arg strings. This protects downstream exact-match
    lookups from PII detectors that over-consume sentence punctuation
    when a slot value already ends with ``.`` and the template appends
    another ``.``.
    """

    def test_normalize_collapses_double_period(self):
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Liam C..") == "Liam C."

    def test_normalize_collapses_triple_period(self):
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Some Name...") == "Some Name."

    def test_normalize_preserves_single_trailing_period(self):
        """Legitimate initials-style names like ``Liam C.`` must stay intact."""
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Liam C.") == "Liam C."

    def test_normalize_preserves_internal_periods(self):
        """``Dr. Smith`` must keep its internal period."""
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Dr. Smith") == "Dr. Smith"

    def test_normalize_collapses_double_comma(self):
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Some Place,,") == "Some Place,"

    def test_normalize_strips_trailing_whitespace_after_dup(self):
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Liam C..  ") == "Liam C."

    def test_normalize_handles_empty_and_non_string(self):
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("") == ""
        assert _normalize_rehydrated_text(None) is None
        assert _normalize_rehydrated_text(42) == 42

    def test_normalize_no_op_on_clean_string(self):
        from traciora_pii.pii_tool_broker import _normalize_rehydrated_text
        assert _normalize_rehydrated_text("Alice Smith") == "Alice Smith"
