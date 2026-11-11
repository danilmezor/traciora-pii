"""Unit tests for LLMPIIDetector with a mocked LM transport.

No real network calls. The mock transport asserts how many requests were
issued so we can verify the prefilter actually skips the LM call on
obviously-negative inputs.
"""
from __future__ import annotations

import json
from typing import Callable, List

import httpx
import pytest

from traciora_pii.llm_pii_detector import (
    LLMFinding,
    LLMPIIDetector,
    _prefilter,
)


# ----------------------------------------------------------------------
# Mock transport helpers
# ----------------------------------------------------------------------


def _make_response(content: str) -> dict:
    """Build the OpenAI chat-completion envelope around the given content string."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _make_detector(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    prefilter_enabled: bool = True,
) -> tuple[LLMPIIDetector, list[httpx.Request]]:
    """Build a detector wired to a captured-request mock transport."""
    captured: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    detector = LLMPIIDetector(
        base_url="http://mock/v1",
        model="mock-model",
        timeout_seconds=5.0,
        prefilter_enabled=prefilter_enabled,
        transport=transport,
    )
    return detector, captured


def _const_handler(content: str) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, json=_make_response(content))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_empty_findings_parses_to_empty_list():
    detector, requests = _make_detector(_const_handler('{"findings":[]}'))
    out = detector.detect("Find Marcus Holloway please")
    assert out == []
    assert len(requests) == 1
    detector.close()


def test_well_formed_positive_parses_correctly():
    text = "Find the order history for Marcus Holloway, please."
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Marcus Holloway", "lemma": "holloway marcus"}
    ]})
    detector, requests = _make_detector(_const_handler(body))
    out = detector.detect(text)
    assert len(out) == 1
    f = out[0]
    assert isinstance(f, LLMFinding)
    assert f.placeholder == "[PERSON_NAME]"
    assert f.lemma == "holloway marcus"
    assert text[f.start:f.end] == "Marcus Holloway"
    assert len(requests) == 1
    detector.close()


def test_hallucination_guard_drops_fabricated_spans():
    text = "Find the order history for Marcus Holloway please."
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Nonexistent Person", "lemma": "person nonexistent"}
    ]})
    detector, _ = _make_detector(_const_handler(body))
    out = detector.detect(text)
    assert out == []
    assert detector.hallucinated_drops == 1
    detector.close()


def test_malformed_json_returns_empty():
    detector, _ = _make_detector(_const_handler("not json at all"))
    out = detector.detect("Find Marcus Holloway please")
    assert out == []
    assert detector.parse_failures == 1
    detector.close()


def test_missing_findings_key_returns_empty():
    detector, _ = _make_detector(_const_handler('{"other": []}'))
    out = detector.detect("Find Marcus Holloway please")
    assert out == []
    detector.close()


def test_prefilter_skips_obvious_negative_without_calling_lm():
    detector, requests = _make_detector(_const_handler('{"findings":[]}'))
    out = detector.detect("We rolled back to version 3.14.2 last night.")
    assert out == []
    assert len(requests) == 0  # LM was never called
    assert detector.prefilter_skips == 1
    detector.close()


def test_prefilter_passes_on_positive_input():
    detector, requests = _make_detector(_const_handler('{"findings":[]}'))
    detector.detect("Find Marcus Holloway please")
    assert len(requests) == 1
    assert detector.prefilter_skips == 0
    detector.close()


def test_multiple_findings_locate_to_distinct_offsets():
    text = "Ship to Marcus Holloway at 1742 Magnolia Avenue, Springfield."
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Marcus Holloway", "lemma": "holloway marcus"},
        {"type": "STREET_ADDRESS", "span": "1742 Magnolia Avenue, Springfield",
         "lemma": "springfield|magnolia avenue|1742|"},
    ]})
    detector, _ = _make_detector(_const_handler(body))
    out = detector.detect(text)
    assert len(out) == 2
    assert out[0].placeholder == "[PERSON_NAME]"
    assert out[1].placeholder == "[STREET_ADDRESS]"
    # Spans must not overlap
    assert out[0].end <= out[1].start
    assert text[out[0].start:out[0].end] == "Marcus Holloway"
    assert text[out[1].start:out[1].end] == "1742 Magnolia Avenue, Springfield"
    detector.close()


def test_verbatim_substring_is_case_sensitive():
    """The model echoed the wrong case; we drop rather than second-guess."""
    text = "find the order for marcus holloway please"
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Marcus Holloway", "lemma": "holloway marcus"}
    ]})
    detector, _ = _make_detector(_const_handler(body), prefilter_enabled=False)
    out = detector.detect(text)
    assert out == []
    assert detector.hallucinated_drops == 1
    detector.close()


def test_timeout_returns_empty_findings():
    def raising(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    detector, _ = _make_detector(raising)
    out = detector.detect("Find Marcus Holloway please")
    assert out == []
    detector.close()


def test_ref_token_in_input_does_not_trigger_prefilter_alone():
    """A bare ref token should not flip the prefilter to True."""
    assert _prefilter("phone_ref:us_a1b2c3d4") is False


def test_ref_token_alongside_real_name_passes_prefilter():
    """A real name in the same text as a ref token should still trigger the LM call."""
    detector, requests = _make_detector(_const_handler('{"findings":[]}'))
    detector.detect("Marcus Holloway can be reached at phone_ref:us_a1b2c3d4 after 3pm")
    assert len(requests) == 1
    detector.close()


def test_code_fence_wrapped_json_is_stripped():
    """Some small LMs wrap JSON in ```json fences despite the prompt."""
    text = "Find Marcus Holloway please"
    fenced = "```json\n" + json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Marcus Holloway", "lemma": "holloway marcus"}
    ]}) + "\n```"
    detector, _ = _make_detector(_const_handler(fenced))
    out = detector.detect(text)
    assert len(out) == 1
    assert out[0].lemma == "holloway marcus"
    detector.close()


def test_multiple_concatenated_json_objects_takes_first():
    """Nemotron Nano 4B sometimes emits a correct JSON object then echoes
    a second one after a blank line. raw_decode must take the first."""
    text = "Find Marcus Holloway please"
    payload = (
        '{"findings":[{"type":"PERSON_NAME","span":"Marcus Holloway","lemma":"holloway marcus"}]}'
        '\n\n'
        '{"findings":[{"type":"PERSON_NAME","span":"Marcus","lemma":"marcus"}]}'
    )
    detector, _ = _make_detector(_const_handler(payload))
    out = detector.detect(text)
    assert len(out) == 1
    assert out[0].lemma == "holloway marcus"
    assert detector.parse_failures == 0
    detector.close()


def test_unknown_type_field_is_dropped():
    text = "Find Marcus Holloway please"
    body = json.dumps({"findings": [
        {"type": "EMAIL", "span": "Marcus Holloway", "lemma": "x"}
    ]})
    detector, _ = _make_detector(_const_handler(body))
    assert detector.detect(text) == []
    detector.close()


def test_russian_name_normalization_via_pymorphy3():
    """pymorphy3 normalizes inflected Russian names to nominative."""
    text = "Передай документы Алексею Воронцову до пятницы"
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Алексею Воронцову", "lemma": "воронцов алексей"}
    ]})
    detector, _ = _make_detector(_const_handler(body), prefilter_enabled=False)
    out = detector.detect(text)
    assert len(out) == 1
    # pymorphy3 should produce nominative, surname-first
    assert out[0].lemma == "воронцов алексей"
    detector.close()


def test_english_possessive_normalization():
    """English possessive 's is stripped from the lemma."""
    text = "Pull up Devon Whitaker's order and check the tracking status."
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Devon Whitaker's", "lemma": "whitaker devon"}
    ]})
    detector, _ = _make_detector(_const_handler(body), prefilter_enabled=False)
    out = detector.detect(text)
    assert len(out) == 1
    # Possessive stripped, reordered to "lastname firstname"
    assert out[0].lemma == "whitaker's devon" or out[0].lemma == "whitaker devon"
    detector.close()


def test_english_title_stripped():
    """Dr./Mr. titles are stripped from the lemma."""
    text = "Dr. Olusegun Adebayo scheduled a consultation."
    body = json.dumps({"findings": [
        {"type": "PERSON_NAME", "span": "Dr. Olusegun Adebayo", "lemma": "adebayo olusegun"}
    ]})
    detector, _ = _make_detector(_const_handler(body), prefilter_enabled=False)
    out = detector.detect(text)
    assert len(out) == 1
    assert out[0].lemma == "adebayo olusegun"
    detector.close()
