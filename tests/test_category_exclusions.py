"""Category policy applies across both detector passes and tool boundaries."""
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from traciora_pii.config import PIIConfig, load_config, reset_config
from traciora_pii.pii import detect_pii, reset_default_detectors_cache
from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer
from traciora_pii.redaction import PIIRedactor


@pytest.fixture
def configured(tmp_path, monkeypatch):
    path = tmp_path / 'traciora.yaml'
    path.write_text('detector:\n  regex_policies: [en, de, ru]\n  excluded_categories: [plate, url]\n')
    monkeypatch.setenv('TRACIORA_PII_CONFIG', str(path))
    reset_config()
    reset_default_detectors_cache()
    yield path
    reset_config()
    reset_default_detectors_cache()


@pytest.fixture
def vault():
    store = PIIVaultStore(':memory:', Fernet.generate_key())
    store.open_session('test')
    yield store
    store.close()


TEXT = 'Plate А123ВС77, link https://example.com/help and email test@example.com'


def test_yaml_validation(configured):
    assert load_config(configured).detector.excluded_categories == ['plate', 'url']
    with pytest.raises(ValidationError):
        PIIConfig.model_validate({'detector': {'excluded_categories': ['plates']}})


def test_regex_and_destructive_masking(configured):
    findings = detect_pii(TEXT)
    assert '[LICENSE_PLATE]' not in [f.placeholder for f in findings]
    assert '[URL]' not in [f.placeholder for f in findings]
    result = PIIRedactor().redact_text(TEXT).text
    assert 'А123ВС77' in result and 'https://example.com/help' in result
    assert '[EMAIL]' in result and 'test@example.com' not in result


def test_virtualization_and_existing_tokens(configured, vault):
    defaults = PIIVirtualizer(vault, excluded_categories=[])
    original_tokens = defaults.anonymize_text('test', TEXT)
    assert 'plate_ref:' in original_tokens and 'url_ref:' in original_tokens
    virtualizer = PIIVirtualizer(vault)
    result = virtualizer.anonymize_structured('test', {'items': [TEXT]})['items'][0]
    assert 'А123ВС77' in result and 'https://example.com/help' in result
    assert 'email_ref:' in result
    assert virtualizer.rehydrate_text('test', original_tokens) == TEXT


@pytest.mark.parametrize('backend', ['opf', 'llm', 'presidio'])
def test_second_pass_exclusions(configured, vault, backend):
    class Detector:
        def detect(self, text):
            return [SimpleNamespace(
                detector=backend, start=text.index(value), end=text.index(value) + len(value),
                placeholder=placeholder, lemma=value,
            ) for value, placeholder in [
                ('А123ВС77', '[LICENSE_PLATE]'),
                ('https://example.com/help', '[URL]'),
                ('Alice', '[PERSON_NAME]'),
            ]]
    virtualizer = PIIVirtualizer(vault, second_pass_detector=Detector())
    result = virtualizer.anonymize_text('test', TEXT + ' Alice')
    assert 'А123ВС77' in result and 'https://example.com/help' in result
    assert 'person_name_ref:' in result and 'email_ref:' in result
    assert all('А123ВС77' != value and 'https://example.com/help' != value
               for value in vault.resolve_all_tokens('test').values())


def test_other_categories_inside_url_still_masked(configured, vault):
    result = PIIVirtualizer(vault).anonymize_text('test', 'https://example.com/?email=test@example.com')
    assert result.startswith('https://example.com/?email=email_ref:')


def test_explicit_redactor_config_overrides_global(configured):
    config = PIIConfig.model_validate({'detector': {'excluded_categories': []}})
    result = PIIRedactor.from_config(config).redact_text(TEXT).text
    assert '[LICENSE_PLATE]' in result and '[URL]' in result


def test_leak_guard_respects_exclusions_for_existing_tokens(configured, vault):
    from traciora_pii.pii_leak_guard import PIILeakError, assert_model_safe
    vault.get_or_create_token('test', '[LICENSE_PLATE]', 'А123ВС77')
    vault.get_or_create_token('test', '[URL]', 'https://example.com/help')
    assert_model_safe(TEXT, 'test', vault)
    vault.get_or_create_token('test', '[EMAIL]', 'test@example.com')
    with pytest.raises(PIILeakError):
        assert_model_safe(TEXT, 'test', vault)


def test_broker_keeps_excluded_categories_in_tool_results(configured, vault):
    from traciora_pii.pii_tool_broker import make_pii_brokered_tool

    class Tool:
        def invoke(self, input_data, config=None, **kwargs):
            assert input_data['email'] == 'test@example.com'
            return {'result': TEXT}

    virtualizer = PIIVirtualizer(vault)
    token = virtualizer.anonymize_text('test', 'test@example.com')
    tool = make_pii_brokered_tool(Tool(), 'test', virtualizer)
    result = tool.invoke({'email': token})['result']
    assert 'А123ВС77' in result and 'https://example.com/help' in result
    assert 'email_ref:' in result and 'test@example.com' not in result
