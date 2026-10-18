"""Per-locale OPF label space, sourced from `traciora_pii/policies/opf/`.

The fine-tune's output head is sized to one logit per row in the
emitted ``span_class_names`` tuple, so this file is the single source of
truth for the (label_map_keys + ``O``) ontology each locale's checkpoint
will learn.

The ``label_map`` keys in ``policies/opf/<locale>_comprehensive.yaml``
are the OPF labels the runtime *expects* the fine-tune to emit. We read
those keys directly so the trainer can never drift from the runtime
policy: if a label is added to the YAML, the next ``prepare`` run picks
it up automatically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from traciora_pii.policies.loader import load_opf_policy

LOCALES: Tuple[str, ...] = ("en", "de", "ru")

# Each locale's OPF policy filename in src/traciora_pii/policies/opf/.
_POLICY_NAME = {
    "en": "en_comprehensive",
    "de": "de_comprehensive",
    "ru": "ru_comprehensive",
}

LABEL_SPACE_VERSION = "traciora_pii_v1"


@dataclass(frozen=True)
class LabelSpace:
    """OPF label ontology for one locale's fine-tune.

    ``span_class_names`` is the ordered tuple ``opf._train.runner`` writes
    into the checkpoint config (``O`` first, then label_map keys
    alphabetised for stability).

    ``placeholder_to_label`` is the inversion of the policy YAML's
    ``label_map``, used by ``prepare_data`` to map generated-bench
    placeholders onto OPF training labels. Two placeholders may map to
    the same OPF label (e.g. ``[RU_BANK_ACCOUNT]`` appears under both
    ``account_number`` and ``ru_bank_account`` in the RU policy); we
    keep the first declaration encountered to honour the YAML order.
    """

    locale: str
    policy_name: str
    span_class_names: Tuple[str, ...]
    placeholder_to_label: Dict[str, str]


def _build(locale: str) -> LabelSpace:
    policy_name = _POLICY_NAME[locale]
    policy = load_opf_policy(policy_name)

    labels = sorted(policy.label_map.keys())
    span_class_names = ("O", *labels)

    placeholder_to_label: Dict[str, str] = {}
    for opf_label, placeholder in policy.label_map.items():
        placeholder_to_label.setdefault(placeholder, opf_label)

    return LabelSpace(
        locale=locale,
        policy_name=policy_name,
        span_class_names=span_class_names,
        placeholder_to_label=placeholder_to_label,
    )


def load_label_space(locale: str) -> LabelSpace:
    if locale not in LOCALES:
        raise ValueError(f"unknown locale {locale!r}; expected one of {LOCALES}")
    return _build(locale)


def write_label_space_json(path: Path, *, locale: str) -> Path:
    """Write the OPF-runner-compatible label_space.json for one locale."""
    space = load_label_space(locale)
    payload = {
        "category_version": f"{LABEL_SPACE_VERSION}_{locale}",
        "span_class_names": list(space.span_class_names),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
