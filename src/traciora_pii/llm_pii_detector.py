"""LLM PII detector for person names and street addresses.

Second-pass detector that runs *after* the regex layer (`pii.py:detect_pii`)
on text that has already had every regex-detectable PII type replaced with
a vault ref token. By design the LM only ever sees what the regex layer
could not catch — names, addresses, and benign content — and never raw
phones, emails, IBANs, or any other regex-handled type. See §6.8 of the
PII virtualization paper for the experimental motivation.

The detector returns ``LLMFinding`` objects carrying both:
  - the surface span (offsets into the input text — for redaction), and
  - a canonical ``lemma`` (for vault keying via ``get_or_create_token``).

Reasoning is suppressed via assistant prefill (``<think></think>``) so
the detector behaves identically across reasoning and non-reasoning
small LMs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import LLMConfig

from ._detector_common import (
    LLMFinding,
    _TYPE_TO_PLACEHOLDER,
    _normalize_person_lemma,
    _prefilter,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You extract person names and street addresses from text. Nothing else.

For each finding, return three fields:
- "type": "PERSON_NAME" or "STREET_ADDRESS"
- "span": the exact substring from the input, character-for-character
- "lemma": a canonical form for matching the same entity across rephrasings
    PERSON_NAME lemma: "lastname firstname" in lowercase, nominative case
    STREET_ADDRESS lemma: "city|street|building|unit" in lowercase, no abbreviations

If the text contains no person names and no street addresses, return {"findings":[]}.

Never extract:
- Brand, product, or company names (Apple, Tesla, Google, Yandex, Газпром, Роскомнадзор, Сбер)
- Place names used as common nouns (Park Plaza, Main branch, Phoenix release)
- Role titles (CEO, CFO, manager, директор)
- Version numbers, ticket IDs, build numbers, room numbers
- IP addresses, URLs, phone numbers, emails, passport numbers (a different system handles those)
- Reference tokens of the form <type>_ref:xx_xxxxxxxx (e.g. phone_ref:us_a1b2c3d4, email_ref:em_9f8e7d6c) — these are placeholders inserted by an upstream system; treat them as opaque and ignore them entirely

Output exactly one JSON object on one line. No markdown. No commentary. No code fences.

Examples:

Input: Find the order history for Marcus Holloway, please.
Output: {"findings":[{"type":"PERSON_NAME","span":"Marcus Holloway","lemma":"holloway marcus"}]}

Input: Apple released a new firmware update last night and broke our integration.
Output: {"findings":[]}

Input: Ship the replacement to Marcus Holloway at 1742 Magnolia Avenue, Springfield, IL 62701.
Output: {"findings":[{"type":"PERSON_NAME","span":"Marcus Holloway","lemma":"holloway marcus"},{"type":"STREET_ADDRESS","span":"1742 Magnolia Avenue, Springfield, IL 62701","lemma":"springfield|magnolia avenue|1742|"}]}

Input: Marcus Holloway can be reached at phone_ref:us_a1b2c3d4 after 3pm.
Output: {"findings":[{"type":"PERSON_NAME","span":"Marcus Holloway","lemma":"holloway marcus"}]}

Input: Курьер выехал на улицу Некрасова, дом 17, квартира 56 к Алексею Воронцову.
Output: {"findings":[{"type":"PERSON_NAME","span":"Алексею Воронцову","lemma":"воронцов алексей"},{"type":"STREET_ADDRESS","span":"улицу Некрасова, дом 17, квартира 56","lemma":"|некрасова|17|56"}]}

Input: We rolled back to version 3.14.2 after the regression in 3.15.0.
Output: {"findings":[]}
"""

# Assistant prefill that suppresses reasoning on Nemotron-style models.
# Non-reasoning models treat this as an empty prefix and continue normally.
_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"



# ======================================================================
# Detector
# ======================================================================

class LLMPIIDetector:
    """OpenAI-compatible LLM detector for [PERSON_NAME] / [STREET_ADDRESS]."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "lm-studio",
        timeout_seconds: float = 15.0,
        prefilter_enabled: bool = True,
        max_tokens: int = 256,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.prefilter_enabled = prefilter_enabled
        self.max_tokens = max_tokens

        # Tracking counters for the experiment metrics. The detector itself
        # never reads these — they exist for the harness to inspect.
        self.lm_calls = 0
        self.prefilter_skips = 0
        self.hallucinated_drops = 0
        self.parse_failures = 0

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=transport,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    @classmethod
    def from_config(cls, llm_cfg: "LLMConfig") -> "LLMPIIDetector":
        """Build from a ``PIIConfig.detector.llm`` section.

        The API key is resolved at call time from the env-var name in
        ``llm_cfg.api_key_env`` (defaults to ``PII_EVAL_API_KEY`` so a
        single OpenRouter key serves both the eval LLM and the
        detector LLM). When no key is set, an ``lm-studio`` placeholder
        is used because LM Studio / vLLM accept any non-empty bearer.
        """
        if not llm_cfg.model:
            raise ValueError(
                "detector.llm.model is unset. Pass --detector-llm-model "
                "<openrouter-name> on the CLI (or set PII_LLM_MODEL) before "
                "enabling detector.second_pass: llm. See traciora.yaml for "
                "examples of small free-tier OpenRouter models."
            )
        api_key = os.getenv(llm_cfg.api_key_env, "lm-studio")
        return cls(
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            api_key=api_key,
            timeout_seconds=llm_cfg.timeout_seconds,
            prefilter_enabled=llm_cfg.prefilter,
            max_tokens=llm_cfg.max_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[LLMFinding]:
        """Return [PERSON_NAME] and [STREET_ADDRESS] spans found in *text*.

        Empty list is a valid and frequent answer — the prefilter, the
        LM, and the verbatim-substring guard all return empty when the
        input contains no names or addresses. Empty is never a retry
        trigger; it is a final answer.
        """
        if not text:
            return []

        if self.prefilter_enabled and not _prefilter(text):
            self.prefilter_skips += 1
            return []

        messages = self._build_messages(text)
        try:
            payload_text = self._call_lm(messages)
        except Exception as exc:  # noqa: BLE001 — runtime LM failure must not crash the redaction pipeline
            logger.warning(
                "LLMPIIDetector LM call failed; returning empty findings. error_type=%s exc=%s",
                type(exc).__name__,
                exc,
            )
            return []

        raw_findings = self._parse_response(payload_text)
        return self._materialize(text, raw_findings)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            self._client.close()
        except httpx.HTTPError as exc:
            logger.warning("LLMPIIDetector client close failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _build_messages(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": _ASSISTANT_PREFILL},
        ]

    # Retry budget for transient HTTP errors (429 rate-limit and 5xx). Free-tier
    # OpenRouter models throttle aggressively when an ablation pass slams the
    # endpoint with hundreds of sequential requests — without retries the
    # detector silently degrades to "empty findings" and corrupts F1 scoring.
    _RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
    _MAX_RETRIES = 4

    def _call_lm(self, messages: list[dict]) -> str:
        """Single OpenAI-compatible chat completion. Returns the content string.

        Transient failures (429, 5xx, connection errors) are retried with
        exponential backoff up to ``_MAX_RETRIES`` times. Non-retryable
        errors (4xx other than 429) raise immediately and the caller's
        outer ``except`` falls back to empty findings.
        """
        # Reasoning suppression. `enabled: false` forces reasoning-mode
        # models (qwen3.x, glm-4.x, glm-5.1, deepseek-r1) to emit the
        # answer directly in `content` instead of consuming the token
        # budget on hidden CoT. `openai/gpt-oss-*` rejects that flag with
        # HTTP 400 ("Reasoning is mandatory ... cannot be disabled"), so
        # for that family we keep only `exclude: true` and let the server
        # reason internally without echoing it back.
        reasoning = {"exclude": True}
        if not self.model.startswith("openai/gpt-oss"):
            reasoning["enabled"] = False
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "reasoning": reasoning,
        }
        self.lm_calls += 1
        last_exc: BaseException | None = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response = self._client.post("/chat/completions", json=body)
                status = response.status_code
                if status in self._RETRY_STATUS and attempt < self._MAX_RETRIES:
                    # Honor Retry-After if present (OpenRouter sets it on 429).
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = 2.0 * (2 ** attempt)
                    else:
                        delay = 2.0 * (2 ** attempt)
                    time.sleep(min(delay, 30.0))
                    continue
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                # Reasoning-mode fallback: some OpenRouter models (qwen3.5,
                # deepseek-r1, …) put the assistant turn in a separate
                # ``reasoning`` field and leave ``content`` empty when the
                # turn ran out of tokens mid-reasoning. The actual JSON
                # object is usually embedded in the reasoning text, so try
                # the alternate locations rather than dropping the answer.
                for alt in (message.get("reasoning"),
                            message.get("reasoning_content")):
                    if isinstance(alt, str) and alt:
                        return alt
                details = message.get("reasoning_details")
                if isinstance(details, list) and details:
                    head = details[0]
                    if isinstance(head, dict):
                        txt = head.get("text")
                        if isinstance(txt, str) and txt:
                            return txt
                return ""
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    time.sleep(2.0 * (2 ** attempt))
                    continue
                raise
        # All retries exhausted on a retry-able status; raise to caller.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"LLMPIIDetector: retries exhausted (status was retryable across "
            f"{self._MAX_RETRIES + 1} attempts)"
        )

    def _parse_response(self, payload_text: str) -> list[dict]:
        """Parse the LM's JSON response into a list of raw finding dicts.

        Strict: any malformed JSON, missing key, or wrong-type value
        produces an empty list. The detector never tries to "recover"
        from a malformed response — that would mask precision issues.
        """
        if not payload_text:
            return []

        text = payload_text.strip()

        # Skip past reasoning-block prefixes and markdown fences. Small
        # models in the wild emit:
        #   - bare JSON                              (Nemotron, Llama)
        #   - ```json\n{...}\n```                   (Mistral, sometimes Llama)
        #   - <think>...</think>\n```json\n{...}    (Mistral Ministral reasoning)
        #   - "Reasoning. ... Constructing: {...}"  (Qwen reasoning-mode fallback)
        # Strategy: scan from the start of ``text`` to the first ``{`` and
        # try ``raw_decode`` there. raw_decode consumes the first complete
        # JSON object and ignores trailing content, which also handles the
        # Nemotron Nano 4B "echo the JSON twice" failure mode.
        json_start = text.find("{")
        if json_start < 0:
            self.parse_failures += 1
            logger.warning(
                "LLMPIIDetector response has no '{' character; returning empty. preview=%r",
                text[:120],
            )
            return []
        try:
            obj, _consumed = json.JSONDecoder().raw_decode(text[json_start:])
        except json.JSONDecodeError:
            self.parse_failures += 1
            logger.warning(
                "LLMPIIDetector failed to parse JSON response; returning empty. preview=%r",
                text[json_start:json_start + 120],
            )
            return []

        if not isinstance(obj, dict):
            self.parse_failures += 1
            return []
        findings = obj.get("findings")
        if not isinstance(findings, list):
            return []

        clean: list[dict] = []
        for item in findings:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            span = item.get("span")
            lemma = item.get("lemma")
            if not isinstance(t, str) or not isinstance(span, str) or not isinstance(lemma, str):
                continue
            if t not in _TYPE_TO_PLACEHOLDER:
                continue
            if not span or not lemma:
                continue
            clean.append({"type": t, "span": span, "lemma": lemma})
        return clean

    def _materialize(self, text: str, raw: list[dict]) -> list[LLMFinding]:
        """Map parsed dicts → LLMFinding via verbatim substring lookup.

        Hallucination guard: if ``span`` is not a verbatim substring of
        ``text``, the finding is dropped (and counted in
        ``hallucinated_drops``). This is the structural defense against
        the model fabricating plausible-looking names that do not
        actually appear in the input.
        """
        out: list[LLMFinding] = []
        # Track per-text consumed positions so two findings with the
        # same span string locate to different occurrences.
        cursor_by_span: dict[str, int] = {}
        for item in raw:
            span_str = item["span"]
            search_from = cursor_by_span.get(span_str, 0)
            idx = text.find(span_str, search_from)
            if idx < 0:
                self.hallucinated_drops += 1
                logger.info(
                    "LLMPIIDetector dropped hallucinated span: %r not in text",
                    span_str[:80],
                )
                continue
            cursor_by_span[span_str] = idx + len(span_str)
            out.append(
                LLMFinding(
                    detector="llm",
                    start=idx,
                    end=idx + len(span_str),
                    placeholder=_TYPE_TO_PLACEHOLDER[item["type"]],
                    lemma=(
                        _normalize_person_lemma(span_str, item["lemma"])
                        if item["type"] == "PERSON_NAME"
                        else item["lemma"]
                    ),
                )
            )
        return out
