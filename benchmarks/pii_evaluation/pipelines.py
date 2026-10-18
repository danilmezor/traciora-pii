"""Three pipeline configurations for the PII evaluation experiment.

Each pipeline runs the same dataset query through a real LangChain
function-calling loop. The configurations differ only in:

  - run_baseline:        no protection (raw values everywhere)
  - run_destructive:     production PIIRedactor applied to user input only
                         (matches the legacy `_anonymize_text` path in
                         `deep_agent_system.py` when virtualization is off)
  - run_virtualization:  vault + virtualizer + brokered tools + leak guard
                         (matches the production virtualization path)

The agent loop in `_run_loop` is a minimal handwritten LangChain
function-calling loop. It is intentionally agnostic to PII handling: each
configuration provides a `transform_user_turn` function that anonymizes
user input *once* at message-construction time, plus an optional
`on_turn_start` callback for the per-turn leak guard. This mirrors
production exactly:

  - Production anonymizes user messages at build time (see
    `_anonymize_text` and its call sites in `deep_agent_system.py` lines
    646, 666, 673, 683), not on every model iteration.
  - Production runs `assert_model_safe` once per user turn (line 1384),
    then on failure re-anonymizes once and passes through (no retry loop).
  - Production wraps tools with `make_pii_brokered_tool` once at agent
    construction (line 1319) and relies on the broker to keep tool results
    clean across all internal iterations of the LangChain agent.

We deliberately do not re-anonymize tool results in the loop. The broker
handles that under virtualization, and production does not redact tool
results in the destructive path.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from traciora_pii.pii_leak_guard import (
    PIILeakError,
    assert_model_safe,
)
from traciora_pii.pii_tool_broker import make_pii_brokered_tool
from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer
from traciora_pii.redaction import PIIRedactor

from benchmarks.pii_evaluation.dataset import EvalQuery
from benchmarks.pii_evaluation.tools import (
    SYSTEM_PROMPT,
    InvocationLog,
    ToolInvocation,
    make_tools,
)


MAX_AGENT_ITERATIONS = 4


def _filter_tools_for_query(tools: Sequence[BaseTool], query: EvalQuery) -> Sequence[BaseTool]:
    """Apply the scenario's `available_tools` allowlist, if any.

    Returns the input list unchanged when the query has no allowlist
    (the default for most scenarios). When an allowlist is set, the
    returned list is filtered to only those tools and a hard error is
    raised if the allowlist names a tool that doesn't exist — that's a
    dataset/wiring bug worth surfacing loudly.
    """
    allow = getattr(query, "available_tools", ()) or ()
    if not allow:
        return tools
    allowed = set(allow)
    filtered = [t for t in tools if t.name in allowed]
    missing = allowed - {t.name for t in filtered}
    if missing:
        raise ValueError(
            f"query {query.id!r} declares available_tools={sorted(allowed)} "
            f"but tool(s) {sorted(missing)} are not in the toolset"
        )
    return filtered


@dataclass
class RunResult:
    """Captured output of a single pipeline run on a single query.

    `transcript` is the raw message history (for human debugging).
    `sent_to_llm` is the *exact* sequence of messages handed to the LLM after
    any pipeline-specific rewriting (anonymization or redaction). The model
    PII exposure metric scans `sent_to_llm`, not `transcript`, because the
    latter would include raw values that the LLM never actually saw.
    """
    query_id: str
    config: str                                 # "baseline" | "destructive" | "virtualization"
    final_response: str
    transcript: List[Dict[str, Any]]            # raw history (debugging only)
    sent_to_llm: List[Dict[str, Any]]           # post-hook messages actually sent to the LLM
    tool_invocations: List[ToolInvocation]
    elapsed_seconds: float
    leak_guard_triggers: int = 0
    error: Optional[str] = None


# ----------------------------------------------------------------------
# Transcript serialization
# ----------------------------------------------------------------------

def _message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Serialize a LangChain message into a JSON-friendly dict.

    We capture only the fields we need for metric scoring: role, content,
    and (for assistant messages) the tool_calls list. The full LangChain
    object is lossy on round-trip, but we never deserialize — we only
    re-scan the content with detect_pii() for the model-exposure metric.
    """
    role = type(message).__name__.replace("Message", "").lower()
    record: Dict[str, Any] = {"role": role, "content": message.content}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        record["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in tool_calls
        ]
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = message.tool_call_id
    return record


# ----------------------------------------------------------------------
# Core agent loop — shared by all three configurations
# ----------------------------------------------------------------------

def _run_loop(
    *,
    user_turns: Sequence[str],
    tools: Sequence[BaseTool],
    llm: Any,
    transform_user_turn: Optional[Callable[[str], str]] = None,
    on_turn_start: Optional[Callable[[List[BaseMessage]], List[BaseMessage]]] = None,
    on_tool_error: Optional[Callable[[str], str]] = None,
    transcript: List[Dict[str, Any]],
    sent_to_llm: List[Dict[str, Any]],
) -> str:
    """Multi-turn LangChain function-calling loop, agnostic to PII handling.

    The loop holds a single growing `history` list of LangChain messages
    and never rewrites it on a per-iteration basis. PII handling enters
    through two extension points, mirroring production:

      `transform_user_turn` is applied to each user turn's text *once*,
      at the moment that turn is appended to history. This is the
      anonymize-at-construction step that production performs in
      `_anonymize_text` (see `deep_agent_system.py` lines 646, 666, 673,
      683). After this point the user message lives in history in its
      anonymized form for every subsequent iteration.

      `on_turn_start` runs once per user turn, immediately after appending
      the new user message and before the first model call of that turn.
      The virtualization config uses it to run `assert_model_safe` exactly
      once, mirroring `deep_agent_system.py` line 1384, and to recover
      from a leak by re-anonymizing the offending messages without a
      retry loop.

    Tool results are appended to history as the broker returns them. We
    do not post-process tool results in the loop because the production
    broker handles that under virtualization (and production never
    redacts tool results under destructive mode).

    `transcript` accumulates the full history for human inspection.
    `sent_to_llm` accumulates exactly the messages that cross the trust
    boundary into the model on every iteration; the model-exposure
    metric scans this list.
    """
    llm_with_tools = llm.bind_tools(list(tools))
    tool_lookup: Dict[str, BaseTool] = {t.name: t for t in tools}

    history: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    transcript.append(_message_to_dict(history[0]))

    final_response = ""

    for user_turn in user_turns:
        text = transform_user_turn(user_turn) if transform_user_turn else user_turn
        history.append(HumanMessage(content=text))
        transcript.append(_message_to_dict(history[-1]))

        if on_turn_start is not None:
            history = on_turn_start(history)
            # `on_turn_start` may have replaced the history list; re-record
            # the user message in the transcript only if it changed.

        for _ in range(MAX_AGENT_ITERATIONS):
            for m in history:
                sent_to_llm.append(_message_to_dict(m))
            response = llm_with_tools.invoke(history)
            history.append(response)
            transcript.append(_message_to_dict(response))

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_response = response.content if isinstance(response.content, str) else str(response.content)
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args") or {}
                tool_obj = tool_lookup.get(tool_name)
                if tool_obj is None:
                    result_text = f"error: unknown tool {tool_name!r}"
                else:
                    try:
                        raw_result = tool_obj.invoke(tool_args)
                    except Exception as exc:  # noqa: BLE001
                        # Tool exceptions bypass the broker's _anonymize_result
                        # hook (it only runs on successful returns). Apply the
                        # caller-supplied sanitizer here so raw PII embedded in
                        # the exception message never reaches the LLM.
                        err_text = f"error: {exc}"
                        if on_tool_error is not None:
                            err_text = on_tool_error(err_text)
                        raw_result = err_text
                    result_text = raw_result if isinstance(raw_result, str) else str(raw_result)
                tool_message = ToolMessage(content=result_text, tool_call_id=tool_call.get("id", ""))
                history.append(tool_message)
                transcript.append(_message_to_dict(tool_message))

    return final_response


# ----------------------------------------------------------------------
# Pipeline 1: baseline (no protection)
# ----------------------------------------------------------------------

def run_baseline(
    *,
    query: EvalQuery,
    llm: Any,
) -> RunResult:
    """No protection. Raw values flow through every layer."""
    log = InvocationLog()
    tools = _filter_tools_for_query(make_tools(log), query)
    transcript: List[Dict[str, Any]] = []
    sent_to_llm: List[Dict[str, Any]] = []
    started = time.perf_counter()
    error: Optional[str] = None
    final_response = ""

    try:
        final_response = _run_loop(
            user_turns=query.turns,
            tools=tools,
            llm=llm,
            transcript=transcript,
            sent_to_llm=sent_to_llm,
        )
    except Exception as exc:  # noqa: BLE001 — capture any per-query failure into the result record so the eval batch continues
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return RunResult(
        query_id=query.id,
        config="baseline",
        final_response=final_response,
        transcript=transcript,
        sent_to_llm=sent_to_llm,
        tool_invocations=list(log.invocations),
        elapsed_seconds=elapsed,
        error=error,
    )


# ----------------------------------------------------------------------
# Pipeline 2: destructive redaction
# ----------------------------------------------------------------------

def run_destructive(
    *,
    query: EvalQuery,
    llm: Any,
    redactor: PIIRedactor,
) -> RunResult:
    """Production legacy destructive path: PIIRedactor on user input only.

    Mirrors `_anonymize_text` in `deep_agent_system.py` when virtualization
    is disabled — it calls `_mask_for_external_llm` which delegates to
    `get_pii_redactor().redact_text(...).text`. Production applies this
    transformation only to user-input messages at construction time
    (lines 646, 666, 673, 683); tool results are NOT redacted because the
    tools execute against literal placeholder strings and naturally return
    "not found" or equivalent failure responses.
    """
    log = InvocationLog()
    tools = _filter_tools_for_query(make_tools(log), query)
    transcript: List[Dict[str, Any]] = []
    sent_to_llm: List[Dict[str, Any]] = []
    started = time.perf_counter()
    error: Optional[str] = None
    final_response = ""

    def transform_user_turn(text: str) -> str:
        return redactor.redact_text(text).text

    try:
        final_response = _run_loop(
            user_turns=query.turns,
            tools=tools,
            llm=llm,
            transform_user_turn=transform_user_turn,
            transcript=transcript,
            sent_to_llm=sent_to_llm,
        )
    except Exception as exc:  # noqa: BLE001 — capture any per-query failure into the result record so the eval batch continues
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return RunResult(
        query_id=query.id,
        config="destructive",
        final_response=final_response,
        transcript=transcript,
        sent_to_llm=sent_to_llm,
        tool_invocations=list(log.invocations),
        elapsed_seconds=elapsed,
        error=error,
    )


# ----------------------------------------------------------------------
# Pipeline 2b: Presidio-as-framework (industry baseline)
# ----------------------------------------------------------------------

def run_destructive_presidio(
    *,
    query: EvalQuery,
    llm: Any,
    analyzers: Dict[str, Any],   # {lang_code: presidio_analyzer.AnalyzerEngine}
    anonymizer: Any,             # presidio_anonymizer.AnonymizerEngine
    language: str = "en",
) -> RunResult:
    """Industry-baseline pipeline: Presidio Analyzer detects PII, Presidio
    Anonymizer replaces it with native tags (``<PERSON>``, ``<EMAIL_ADDRESS>``,
    ...) before the LLM sees user input. Same destructive-failure profile
    as ``run_destructive`` — tool args carry anonymized strings; the
    runtime has no rehydration step.

    This exists so §3 has a real framework competitor and the framework
    comparison isn't a strawman ("our destructive vs. their destructive").
    Detector + anonymizer are Presidio's defaults — their tool, their
    behavior. The pipeline picks the analyzer matching the query's
    ``language`` from the ``analyzers`` dict; falls back to ``en`` (or
    the first available entry) with a logged warning if the language
    isn't registered. Anonymizer is language-agnostic and stays single.
    """
    log = InvocationLog()
    tools = _filter_tools_for_query(make_tools(log), query)
    transcript: List[Dict[str, Any]] = []
    sent_to_llm: List[Dict[str, Any]] = []
    started = time.perf_counter()
    error: Optional[str] = None
    final_response = ""

    if language in analyzers:
        analyzer = analyzers[language]
        effective_language = language
    elif "en" in analyzers:
        logger.warning(
            "presidio analyzer not registered for language=%r; falling back to 'en'. "
            "This means Presidio runs the English spaCy model over non-EN text — "
            "the result understates Presidio's real performance.",
            language,
        )
        analyzer = analyzers["en"]
        effective_language = "en"
    elif analyzers:
        fallback_lang = next(iter(analyzers))
        logger.warning(
            "presidio analyzer not registered for language=%r and no 'en' fallback; "
            "using %r.",
            language, fallback_lang,
        )
        analyzer = analyzers[fallback_lang]
        effective_language = fallback_lang
    else:
        raise ValueError(
            "run_destructive_presidio requires at least one analyzer in `analyzers`; "
            "got an empty dict."
        )

    def transform_user_turn(text: str) -> str:
        if not text:
            return text
        try:
            results = analyzer.analyze(text=text, language=effective_language)
        except Exception as exc:  # noqa: BLE001
            # If Presidio fails on a row, we leave the text raw — that's
            # the realistic failure mode for an industry baseline (a
            # production deployment would log and pass through, exposing
            # PII).
            logger.warning(
                "presidio analyzer failed; passing raw. error_type=%s",
                type(exc).__name__,
            )
            return text
        if not results:
            return text
        try:
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "presidio anonymizer failed; passing raw. error_type=%s",
                type(exc).__name__,
            )
            return text
        return anonymized.text

    try:
        final_response = _run_loop(
            user_turns=query.turns,
            tools=tools,
            llm=llm,
            transform_user_turn=transform_user_turn,
            transcript=transcript,
            sent_to_llm=sent_to_llm,
        )
    except Exception as exc:  # noqa: BLE001 — capture any per-query failure into the result record so the eval batch continues
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return RunResult(
        query_id=query.id,
        config="destructive_presidio",
        final_response=final_response,
        transcript=transcript,
        sent_to_llm=sent_to_llm,
        tool_invocations=list(log.invocations),
        elapsed_seconds=elapsed,
        error=error,
    )


# ----------------------------------------------------------------------
# Pipeline 3: PII virtualization
# ----------------------------------------------------------------------

def run_virtualization(
    *,
    query: EvalQuery,
    llm: Any,
    vault: PIIVaultStore,
    virtualizer: PIIVirtualizer,
) -> RunResult:
    """Full PII virtualization, matching the production code path.

    Mirrors `deep_agent_system.py` exactly:
      - One vault session per conversation (`vault.open_session(scope)` /
        `vault.close_session(scope)`, lines 157, 1453).
      - Tools wrapped with `make_pii_brokered_tool(t, scope, virtualizer)`
        once at construction (line 1319).
      - User input anonymized at message-construction time via
        `virtualizer.anonymize_text(scope, text)` (lines 646–683).
      - `assert_model_safe(messages, scope, vault)` runs once per turn,
        immediately before the model call (line 1384). On failure, every
        message is re-anonymized once and passed through; no retry loop
        (lines 1386–1391).
      - Final response rehydrated via `virtualizer.rehydrate_text(scope,
        text)` for the user-facing output (line 1423).
    """
    log = InvocationLog()
    tools = _filter_tools_for_query(make_tools(log), query)
    transcript: List[Dict[str, Any]] = []
    sent_to_llm: List[Dict[str, Any]] = []
    leak_triggers = 0
    started = time.perf_counter()
    error: Optional[str] = None
    final_response = ""

    scope = f"eval:{query.id}"
    vault.open_session(scope)

    # Broker every tool. The broker mutates each tool object in place; using
    # a fresh list per query (from make_tools) prevents cross-config bleed.
    for tool_obj in tools:
        make_pii_brokered_tool(tool_obj, scope, virtualizer)

    def transform_user_turn(text: str) -> str:
        return virtualizer.anonymize_text(scope, text)

    def on_tool_error(text: str) -> str:
        return virtualizer.anonymize_text(scope, text)

    def on_turn_start(history: List[BaseMessage]) -> List[BaseMessage]:
        """Run the leak guard once per turn, mirroring production line 1384."""
        nonlocal leak_triggers
        payload = [
            {"role": type(m).__name__, "content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in history
        ]
        try:
            assert_model_safe(payload, scope, vault)
            return history
        except PIILeakError:
            leak_triggers += 1
            # Production: re-anonymize every message once, then pass through.
            # No retry, no recheck (deep_agent_system.py lines 1386–1391).
            rewritten: List[BaseMessage] = []
            for m in history:
                content = m.content if isinstance(m.content, str) else str(m.content)
                anon = virtualizer.anonymize_text(scope, content)
                if isinstance(m, SystemMessage):
                    rewritten.append(SystemMessage(content=anon))
                elif isinstance(m, HumanMessage):
                    rewritten.append(HumanMessage(content=anon))
                elif isinstance(m, ToolMessage):
                    rewritten.append(ToolMessage(content=anon, tool_call_id=m.tool_call_id))
                elif isinstance(m, AIMessage):
                    rewritten.append(
                        AIMessage(content=anon, tool_calls=getattr(m, "tool_calls", None) or [])
                    )
                else:
                    rewritten.append(m)
            return rewritten

    try:
        final_response = _run_loop(
            user_turns=query.turns,
            tools=tools,
            llm=llm,
            transform_user_turn=transform_user_turn,
            on_turn_start=on_turn_start,
            on_tool_error=on_tool_error,
            transcript=transcript,
            sent_to_llm=sent_to_llm,
        )
        # Rehydrate the final response so the user-facing answer contains real values
        # (deep_agent_system.py line 1423).
        final_response = virtualizer.rehydrate_text(scope, final_response)
    except Exception as exc:  # noqa: BLE001 — capture any per-query failure into the result record so the eval batch continues
        error = f"{type(exc).__name__}: {exc}"
    finally:
        vault.close_session(scope)

    elapsed = time.perf_counter() - started
    return RunResult(
        query_id=query.id,
        config="virtualization",
        final_response=final_response,
        transcript=transcript,
        sent_to_llm=sent_to_llm,
        tool_invocations=list(log.invocations),
        elapsed_seconds=elapsed,
        leak_guard_triggers=leak_triggers,
        error=error,
    )
