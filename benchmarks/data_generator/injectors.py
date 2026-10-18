"""Slot-template renderer: fills templates from seeds with synthetic PII.

A template is a complete sentence with named slots like ``{PERSON_NAME}``
or ``{RU_INN}``. The engine picks a category (weighted), picks a template
inside that category (weighted), generates one synthetic value per slot
via the locale's PII helper, and emits the rendered text plus character-
level gold spans for every slot.

The slot-template design replaces the previous scaffold→append-PII flow:
each template was hand-crafted to host exactly the PII it declares, so
the resulting prose reads naturally instead of having identifiers tacked
onto unrelated sentences. See ``seeds/<locale>/templates.yaml``.

Span invariant ``text[start:end] == value`` holds for every emitted span
and is asserted at render time.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .faker_providers import RuPII
from .seeds import CategoryConfig, SeedBundle, Template


# Slot syntax in templates: ``{PLACEHOLDER_NAME}``. Same regex as in
# seeds.py so validation and rendering can't disagree.
_SLOT_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


# Slot name → method on the locale's PII helper. Adding a new slot to the
# taxonomy means: (a) declaring it in ``_KNOWN_SLOTS`` in ``seeds.py``,
# (b) implementing the matching method on the locale helper, (c) listing
# it here. The smoke test fails fast on any missing entry.
_SLOT_METHODS: Dict[str, str] = {
    # Universal
    "PERSON_NAME": "person_name",
    "STREET_ADDRESS": "street_address",
    "EMAIL": "email",
    "DATE": "date",
    "CARD": "card",
    "IBAN": "iban",
    "IP": "ip",
    "URL": "url",
    "SECRET": "secret_token",
    "LICENSE_PLATE": "license_plate",
    # Russian-specific
    "RU_INN": "inn_any",
    "RU_SNILS": "snils",
    "PASSPORTNUM": "passport",
    "RU_OGRN": "ogrn",
    "RU_KPP": "kpp",
    "RU_BANK_ACCOUNT": "bank_account",
    "PHONE": "phone",
    "VIN": "vin",
    "RU_DRV_LICENSE": "drv_license",
    "RU_STS": "sts",
    "RU_OSAGO": "osago",
    # German-specific (helpers added once de_DE is refactored)
    "DE_STEUER_ID": "steuer_id",
    "DE_PERSONALAUSWEIS": "personalausweis",
    "DE_PLZ_CITY": "plz_city",
    # English-specific
    "US_SSN": "us_ssn",
}


# Slot name → emitted placeholder when it differs from the default
# ``f"[{slot}]"``. Used to collapse multiple structurally-colliding
# government-ID slots onto a single ``[PERSONAL_ID]`` output: the
# templates keep their typed slot keys (``{PASSPORTNUM}``,
# ``{RU_DRV_LICENSE}``, ``{US_SSN}``, ``{DE_PERSONALAUSWEIS}``) so each
# still calls its own checksum-correct faker method, but the gold span
# we emit is a single generic placeholder. See policies/opf/*.yaml for
# the rationale.
_SLOT_PLACEHOLDER_OVERRIDES: Dict[str, str] = {
    "PASSPORTNUM": "[PERSONAL_ID]",
    "RU_DRV_LICENSE": "[PERSONAL_ID]",
    "US_SSN": "[PERSONAL_ID]",
    "DE_PERSONALAUSWEIS": "[PERSONAL_ID]",
}


def _emit_placeholder(slot: str) -> str:
    """Return the placeholder string to emit for a slot key."""
    return _SLOT_PLACEHOLDER_OVERRIDES.get(slot, f"[{slot}]")


@dataclass(frozen=True)
class Span:
    """A single gold-labeled PII span in the rendered text."""

    placeholder: str
    value: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span: {self.start}..{self.end}")


@dataclass
class SlotFillEngine:
    """Renders slot-templates by drawing synthetic values from the locale's PII helper.

    Construct one per locale per pipeline run. The engine owns a
    ``random.Random`` instance and the locale's PII helper (which itself
    holds the same RNG so all randomness is deterministic from a single seed).
    """

    seeds: SeedBundle
    rng: random.Random
    pii: object = field(init=False)

    def __post_init__(self) -> None:
        self.pii = self._build_pii()

    def _build_pii(self):
        """Pick the right PII helper for the locale.

        Imports are local so adding a new locale doesn't force every caller
        of this module to import every helper.
        """
        if self.seeds.locale == "ru":
            return RuPII(seeds=self.seeds, rng=self.rng)
        if self.seeds.locale == "de":
            from .faker_providers import DePII  # type: ignore[attr-defined]
            return DePII(seeds=self.seeds, rng=self.rng)
        if self.seeds.locale == "en":
            from .faker_providers import EnPII  # type: ignore[attr-defined]
            return EnPII(seeds=self.seeds, rng=self.rng)
        raise ValueError(f"no PII helper for locale {self.seeds.locale!r}")

    # ------------------------------------------------------------------

    def sample_category(self) -> Tuple[str, CategoryConfig]:
        """Pick a category by its declared weight."""
        cats = list(self.seeds.templates.categories.items())
        weights = [cfg.weight for _, cfg in cats]
        return self.rng.choices(cats, weights=weights, k=1)[0]

    def sample_template(self, category: CategoryConfig) -> Template:
        weights = [t.weight for t in category.templates]
        return self.rng.choices(category.templates, weights=weights, k=1)[0]

    def render(self, template: Template) -> Tuple[str, List[Span]]:
        """Fill every slot in ``template.text`` and return ``(text, spans)``.

        Span invariant ``text[start:end] == value`` is asserted before return.
        """
        out_parts: List[str] = []
        spans: List[Span] = []
        cursor = 0
        out_offset = 0

        for match in _SLOT_RE.finditer(template.text):
            prefix = template.text[cursor:match.start()]
            out_parts.append(prefix)
            out_offset += len(prefix)

            slot = match.group(1)
            value = self._generate_for_slot(slot)
            out_parts.append(value)
            span = Span(
                placeholder=_emit_placeholder(slot),
                value=value,
                start=out_offset,
                end=out_offset + len(value),
            )
            spans.append(span)
            out_offset += len(value)
            cursor = match.end()

        out_parts.append(template.text[cursor:])
        text = "".join(out_parts)

        # Invariant — cheap and load-bearing.
        for sp in spans:
            assert text[sp.start:sp.end] == sp.value, (
                f"slot-fill span mismatch: {text[sp.start:sp.end]!r} != {sp.value!r}"
            )
        return text, spans

    def _generate_for_slot(self, slot: str) -> str:
        method_name = _SLOT_METHODS.get(slot)
        if method_name is None:
            raise ValueError(f"no generator registered for slot {{{slot}}}")
        method = getattr(self.pii, method_name, None)
        if method is None:
            raise ValueError(
                f"locale {self.seeds.locale!r} has no method {method_name!r} "
                f"for slot {{{slot}}}"
            )
        return method()
