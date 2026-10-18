"""PII tool broker — wraps LangChain tools so that ref tokens in arguments
are resolved to raw values before invocation and raw PII in results is
re-tokenized before returning to the model.

Two integration modes:
1. ``broker_raw_tool(tool_obj, ...)`` — patches the ``.invoke()`` method on
   a *raw* catalog tool (before it goes through ``make_skill_guarded_tool``).
   The skill guard's inner ``tool_obj.invoke(kwargs)`` call then hits the
   brokered version.
2. ``make_pii_brokered_tool(tool_obj, ...)`` — wraps an already-guarded
   ``@lc_tool`` function by patching its ``_run`` method, which is safe for
   LangChain ``BaseTool`` subclasses.
"""
from __future__ import annotations

import json
import re
from typing import Any
from collections.abc import Callable

from .pii_virtualizer import PIIVirtualizer
import logging

logger = logging.getLogger(__name__)


# Collapse trailing punctuation runs ("Liam C.." → "Liam C.") introduced
# when a PII detector over-consumes adjacent sentence punctuation. The
# corrupted span is stored in the vault verbatim, so downstream exact-match
# tool lookups fail silently without this normalizer at the trust boundary.
_TRAILING_DUP_PERIOD = re.compile(r"\.{2,}\s*$")
_TRAILING_DUP_COMMA = re.compile(r",{2,}\s*$")


def _normalize_rehydrated_text(value: str) -> str:
    """Collapse consecutive trailing ``.`` or ``,`` runs at end-of-string."""
    if not isinstance(value, str) or not value:
        return value
    value = _TRAILING_DUP_PERIOD.sub(".", value)
    value = _TRAILING_DUP_COMMA.sub(",", value)
    return value


def _try_json_rehydrate(value: str, session_scope_key: str, virtualizer: PIIVirtualizer) -> str:
    """If *value* looks like a JSON string, parse → rehydrate → re-serialize."""
    stripped = value.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return _normalize_rehydrated_text(virtualizer.rehydrate_text(session_scope_key, value))
    try:
        parsed = json.loads(stripped)
        rehydrated = virtualizer.rehydrate_structured(session_scope_key, parsed)
        return json.dumps(rehydrated, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return _normalize_rehydrated_text(virtualizer.rehydrate_text(session_scope_key, value))


def _resolve_input(input_data: Any, session_scope_key: str, virtualizer: PIIVirtualizer) -> Any:
    """Resolve ref tokens in tool input arguments."""
    if input_data is None:
        return input_data
    if isinstance(input_data, str):
        return _try_json_rehydrate(input_data, session_scope_key, virtualizer)
    if isinstance(input_data, dict):
        resolved = {}
        for k, v in input_data.items():
            if isinstance(v, str):
                resolved[k] = _try_json_rehydrate(v, session_scope_key, virtualizer)
            else:
                resolved[k] = virtualizer.rehydrate_structured(session_scope_key, v)
        return resolved
    return virtualizer.rehydrate_structured(session_scope_key, input_data)


def _anonymize_result(result: Any, session_scope_key: str, virtualizer: PIIVirtualizer) -> Any:
    """Anonymize raw PII in a tool result."""
    if result is None:
        return result
    if isinstance(result, str):
        return virtualizer.anonymize_text(session_scope_key, result)
    return virtualizer.anonymize_structured(session_scope_key, result)


def make_pii_brokered_tool(
    tool_obj: Any,
    session_scope_key: str,
    virtualizer: PIIVirtualizer,
    status_callback: Callable | None = None,
) -> Any:
    """Wrap a tool (raw catalog or already-guarded) with PII brokering.

    Patches the tool's ``_run`` method if it's a LangChain ``BaseTool``
    (``StructuredTool``), preserving the ``config`` and ``run_manager``
    keyword-only args that LangChain passes internally.  Falls back to
    patching ``invoke`` for plain callable tools.

    Returns the same object (mutated in-place).
    """
    # Preferred path: patch _run (works for @tool-decorated StructuredTool)
    original_run = getattr(tool_obj, "_run", None)
    if callable(original_run):
        def brokered_run(*args, config=None, run_manager=None, **kwargs):
            # Rehydrate only the user-facing kwargs (not config/run_manager)
            resolved_kwargs = _resolve_input(kwargs, session_scope_key, virtualizer) if kwargs else kwargs
            resolved_args = tuple(
                _normalize_rehydrated_text(virtualizer.rehydrate_text(session_scope_key, a)) if isinstance(a, str)
                else _resolve_input(a, session_scope_key, virtualizer) if isinstance(a, dict)
                else a
                for a in args
            )
            result = original_run(*resolved_args, config=config, run_manager=run_manager, **resolved_kwargs)
            return _anonymize_result(result, session_scope_key, virtualizer)

        tool_obj._run = brokered_run
        return tool_obj

    # Fallback: patch invoke (for plain callable tools)
    original_invoke = tool_obj.invoke

    def brokered_invoke(input_data: Any, config: Any = None, **kwargs) -> Any:
        resolved_input = _resolve_input(input_data, session_scope_key, virtualizer)
        result = original_invoke(resolved_input, config=config, **kwargs)
        return _anonymize_result(result, session_scope_key, virtualizer)

    tool_obj.invoke = brokered_invoke
    return tool_obj
