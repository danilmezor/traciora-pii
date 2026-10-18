"""Pydantic schemas for traciora_pii policy files.

Two layers, loaded independently:

* :class:`RegexPolicy` declares the placeholder taxonomy + first-pass regex
  patterns. Always required. Source of truth for ``placeholder ->
  vault_category`` mapping.

* :class:`OPFPolicy` describes a fine-tuned OPF model — its routing pattern
  and how its model labels map onto placeholders declared by a regex policy.
  Optional.

The closed :class:`VaultCategory` enum keeps the placeholder taxonomy
disciplined: adding a new category is a deliberate code change, not an
accidental YAML typo.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VaultCategory(str, Enum):
    """Closed set of vault categories. Adding a new value is a code change.

    A vault category is the canonical "kind of secret" the vault organizes
    by. Multiple placeholders can map to the same category.
    """

    PHONE = "phone"
    EMAIL = "email"
    CARD = "card"
    IP = "ip"
    URL = "url"
    # Generic government-issued personal identifier. Replaces the
    # previously typed SSN / PASSPORT / DRV_LICENSE / PERSONALAUSWEIS
    # categories that mapped to structurally identical regex shapes —
    # those typed labels were unreachable at the model layer.
    PERSONAL_ID = "personal_id"
    INN = "inn"
    SNILS = "snils"
    OGRN = "ogrn"
    BANK_ACCT = "bank_acct"
    KPP = "kpp"
    DATE = "date"
    PLATE = "plate"
    IBAN = "iban"
    SECRET = "secret"
    PERSON_NAME = "person_name"
    STREET_ADDRESS = "street_address"
    VIN = "vin"
    STS = "sts"
    OSAGO = "osago"
    STEUER_ID = "steuer_id"


_PLACEHOLDER_RE = re.compile(r"^\[[A-Z0-9_]+\]$")


class Placeholder(BaseModel):
    """One entry in the placeholder taxonomy.

    The placeholder string is what the vault sees and what redacted output
    contains. Multiple regex patterns may emit the same placeholder; OPF
    label maps reference placeholders by string.
    """

    model_config = ConfigDict(extra="forbid")

    placeholder: str = Field(
        ...,
        description="Bracketed UPPER_SNAKE token, e.g. '[PHONE]'",
    )
    vault_category: VaultCategory
    description: str | None = None

    @field_validator("placeholder")
    @classmethod
    def _check_placeholder_format(cls, v: str) -> str:
        if not _PLACEHOLDER_RE.match(v):
            raise ValueError(
                f"placeholder must match /^\\[[A-Z0-9_]+\\]$/, got {v!r}"
            )
        return v


class PIIPattern(BaseModel):
    """One regex rule.

    The pattern emits ``placeholder``. The placeholder must already be
    declared in the surrounding :class:`RegexPolicy`'s ``placeholders``.

    ``validator`` is the name of a callable in the validator registry
    (e.g. ``"luhn"``); resolved at load time. Keeps callables out of YAML.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    placeholder: str
    pattern: str
    keyword_anchors: list[str] = Field(default_factory=list)
    lookaround_chars: int = 40
    validator: str | None = None
    description: str | None = None

    @field_validator("pattern")
    @classmethod
    def _check_compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"pattern is not a valid regex: {e}") from e
        return v

    @field_validator("placeholder")
    @classmethod
    def _check_placeholder_format(cls, v: str) -> str:
        if not _PLACEHOLDER_RE.match(v):
            raise ValueError(
                f"placeholder must match /^\\[[A-Z0-9_]+\\]$/, got {v!r}"
            )
        return v


class VocabNormalization(BaseModel):
    """Lemmatizer / language hints used by the regex+LLM path.

    ``cyrillic_aware`` toggles Cyrillic-script handling (e.g. routing
    tokens through pymorphy3 instead of treating them as opaque strings).
    """

    model_config = ConfigDict(extra="forbid")

    cyrillic_aware: bool = False
    lemmatizer: str | None = None  # "pymorphy3" today; future: "spacy", etc.


class RegexPolicy(BaseModel):
    """Placeholder taxonomy + first-pass regex layer.

    Pattern ordering matters: when two patterns match overlapping spans,
    the first-listed wins after merging. List keyword-anchored patterns
    before unconditional ones so contextual matches outrank generic
    numeric patterns.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    locale: str
    description: str | None = None
    vocab_normalization: VocabNormalization = Field(default_factory=VocabNormalization)
    placeholders: list[Placeholder]
    patterns: list[PIIPattern]

    @model_validator(mode="after")
    def _check_consistency(self) -> "RegexPolicy":
        ph_to_cat: dict[str, VaultCategory] = {}
        for p in self.placeholders:
            if p.placeholder in ph_to_cat:
                raise ValueError(
                    f"duplicate placeholder declaration: {p.placeholder!r}"
                )
            ph_to_cat[p.placeholder] = p.vault_category

        seen_names: set[str] = set()
        for pat in self.patterns:
            if pat.name in seen_names:
                raise ValueError(f"duplicate pattern name: {pat.name!r}")
            seen_names.add(pat.name)
            if pat.placeholder not in ph_to_cat:
                raise ValueError(
                    f"pattern {pat.name!r} emits placeholder "
                    f"{pat.placeholder!r} not declared in `placeholders`"
                )
        return self

    @property
    def placeholder_to_category(self) -> dict[str, VaultCategory]:
        return {p.placeholder: p.vault_category for p in self.placeholders}


class OPFPolicy(BaseModel):
    """OPF model-label -> placeholder map for one fine-tuned model.

    The placeholders referenced here must exist in the regex policy that
    is loaded alongside this one. Cross-validation happens at compose
    time (see :func:`validate_opf_against_regex`).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    locale: str
    description: str | None = None
    label_map: dict[str, str]

    @field_validator("label_map")
    @classmethod
    def _check_label_map_placeholders(cls, v: dict[str, str]) -> dict[str, str]:
        for label, placeholder in v.items():
            if not _PLACEHOLDER_RE.match(placeholder):
                raise ValueError(
                    f"label_map[{label!r}] = {placeholder!r} is not a valid "
                    f"placeholder (expected /^\\[[A-Z0-9_]+\\]$/)"
                )
        return v


def validate_opf_against_regex(opf: OPFPolicy, regex: RegexPolicy) -> None:
    """Raise :class:`ValueError` if any placeholder in ``opf.label_map``
    is missing from ``regex.placeholders``.

    Call this once at compose time after loading both policies. The OPF
    layer is purely additive: it can never introduce a new placeholder
    without a corresponding taxonomy entry in the regex policy.
    """
    known = set(regex.placeholder_to_category)
    unknown = sorted({ph for ph in opf.label_map.values() if ph not in known})
    if unknown:
        raise ValueError(
            f"OPF policy {opf.name!r} references placeholders absent from "
            f"regex policy {regex.name!r}: {unknown}. Declare them in the "
            f"regex policy's `placeholders` list (no `pattern` needed for "
            f"OPF-only entries)."
        )
