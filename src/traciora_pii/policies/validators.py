"""Named-validator registry for regex-policy patterns.

A validator is a post-match callable that decides whether a regex hit is
kept (returns ``True``) or discarded (``False``). Used to apply structural
checks regex cannot express — e.g. the Luhn checksum on credit-card spans.

Validators are referenced by name from policy YAML files (``validator: luhn``)
and resolved to callables here at policy-compile time. Keeping callables out
of YAML preserves the "config is data" property.

Adding a new validator: define the function and call :func:`register`.
"""
from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable[[str], bool]] = {}


def register(name: str, fn: Callable[[str], bool]) -> None:
    """Register a validator under ``name``. Raises if already registered."""
    if name in _REGISTRY:
        raise ValueError(f"validator {name!r} is already registered")
    _REGISTRY[name] = fn


def get(name: str) -> Callable[[str], bool]:
    """Resolve a registered name. Raises :class:`KeyError` if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown validator: {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------

def luhn(value: str) -> bool:
    """Return ``True`` iff ``value``'s digits satisfy the Luhn checksum.

    Used by the credit-card pattern: real cards are 13–19 digits and
    satisfy Luhn (defining property), while random sequences satisfy it
    only ~1 in 10 by chance. This filters the dominant source of card
    false positives — long internal IDs, OGRNs, other numeric runs.
    """
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


register("luhn", luhn)
