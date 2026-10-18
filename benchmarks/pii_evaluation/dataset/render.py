"""Render scenario YAMLs into a parallel trilingual EvalQuery dataset.

Workflow
--------

For each scenario × locale ∈ {en, de, ru}:

  1. Build a ``random.Random`` seeded by ``hash(scenario_id, locale)``
     so the output is byte-stable across machines and re-renders.
  2. Load the locale's ``SeedBundle`` (per-locale name/address/identifier
     pools) via ``benchmarks.data_generator.load_seed_bundle``.
  3. Construct the appropriate per-locale PII helper (``EnPII`` / ``DePII``
     / ``RuPII``) — these emit checksum-valid IDs (Luhn cards, ISO 7064
     Steuer-ID, FNS INN, PFR SNILS, ISO 3779 VINs).
  4. Resolve scenario PII slots (e.g. ``{PHONE}``, ``{EMAIL}``,
     ``{NATIONAL_ID}``) by calling the corresponding helper method.
  5. Resolve ``{{intent_*}}`` phrase fragments from the locale's
     ``scenario_phrases.yaml`` so the surface phrasing is natural per
     locale while the scenario shape stays language-agnostic.
  6. Resolve any ``record_key`` references against ``fixtures.TEMPLATES``,
     materializing the canned record and pulling the requested
     ``from_record_field`` for cross-turn-taint scenarios.
  7. Substitute slots into the turn templates and validate the span
     invariant: every ``PIIGroundTruth.value`` must appear verbatim in
     the corresponding turn's text.
  8. Emit one ``EvalQuery`` plus the list of ``Fixture`` records that
     scenario relies on.

The renderer is pure — given the same inputs it produces byte-identical
output, which lets ``--freeze`` write a JSONL artifact whose SHA-256 is
stable across CI runs.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import yaml

from benchmarks.data_generator import load_seed_bundle
from benchmarks.data_generator.faker_providers import DePII, EnPII, RuPII

from benchmarks.pii_evaluation.dataset.fixtures import (
    DEFAULT_NATIONAL_ID_TYPE_BY_LOCALE,
    TEMPLATES,
)


LOCALES: Tuple[str, ...] = ("en", "de", "ru")

PII_HELPER_BY_LOCALE = {"en": EnPII, "de": DePII, "ru": RuPII}

# Mapping from scenario YAML slot names to (helper-method, placeholder).
# The ``placeholder`` is the canonical PII tag the detector is expected to
# emit for that slot, used when the scenario YAML doesn't override it.
SLOT_TO_METHOD: Dict[str, Tuple[str, str]] = {
    "PERSON_NAME": ("person_name", "[PERSON_NAME]"),
    "PHONE": ("phone", "[PHONE]"),
    "EMAIL": ("email", "[EMAIL]"),
    "STREET_ADDRESS": ("street_address", "[STREET_ADDRESS]"),
    "CARD": ("card", "[CARD]"),
    "IBAN": ("iban", "[IBAN]"),
    "IP": ("ip", "[IP]"),
    "URL": ("url", "[URL]"),
    "DATE": ("date", "[DATE]"),
    "VIN": ("vin", "[VIN]"),
    "LICENSE_PLATE": ("license_plate", "[LICENSE_PLATE]"),
    # Locale-specific national identifiers — resolved per-locale by
    # _resolve_national_id below; placeholder follows from id_type.
}

# Resolution table for the generic ``{NATIONAL_ID}`` slot per locale.
NATIONAL_ID_METHOD_BY_TYPE: Dict[str, Tuple[str, str]] = {
    # us_ssn and ru_passport collapse to [PERSONAL_ID] (structurally
    # indistinguishable from other 9-digit / XX YY ZZZZZZ IDs).
    "us_ssn": ("us_ssn", "[PERSONAL_ID]"),
    "ru_passport": ("passport", "[PERSONAL_ID]"),
    # Tax IDs and SNILS retain distinct placeholders — distinguishable
    # shapes (11-digit grouped / 11-digit with dashes).
    "de_steuer_id": ("steuer_id", "[DE_STEUER_ID]"),
    "ru_snils": ("snils", "[RU_SNILS]"),
    "ru_inn": ("inn_legal", "[RU_INN]"),
}


@dataclass(frozen=True)
class Fixture:
    """One synthetic record to register with the tool store at runtime.

    ``store`` selects the underlying record store (``customers``,
    ``appointments``, ``tickets``, ``contractors``, ``payments``,
    ``background_checks``, ``steuer_ids``, ``snils``, ``passports``).
    """
    store: str
    record_id: str
    record_locale: str
    payload: Mapping[str, Any]


_COMMON_FIXTURE_PII_FIELDS: Mapping[str, str] = {
    "name": "[PERSON_NAME]",
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "address": "[STREET_ADDRESS]",
    "contact_name": "[PERSON_NAME]",
    "contact_phone": "[PHONE]",
    "card": "[CARD]",
    "ssn": "[PERSONAL_ID]",
    "steuer_id": "[DE_STEUER_ID]",
    "snils": "[RU_SNILS]",
    "series_number": "[PERSONAL_ID]",
    "vin": "[VIN]",
    "license_plate": "[LICENSE_PLATE]",
}


def _tax_id_placeholder(locale: str) -> str:
    if locale == "ru":
        return "[RU_INN]"
    if locale == "de":
        return "[DE_STEUER_ID]"
    return "[TAX_ID]"


def _fixture_boundary_values(record: Mapping[str, Any], locale: str) -> List[Dict[str, str]]:
    """Extract PII-bearing fields from a materialized fixture record."""
    out: List[Dict[str, str]] = []
    record_id = str(record.get("record_id") or "")
    for field_name, raw_value in record.items():
        if raw_value is None:
            continue
        if field_name == "tax_id":
            placeholder = _tax_id_placeholder(locale)
        else:
            placeholder = _COMMON_FIXTURE_PII_FIELDS.get(field_name)
        if not placeholder:
            continue
        value = str(raw_value)
        if not value:
            continue
        out.append(
            {
                "placeholder": placeholder,
                "value": value,
                "source": "fixture_result",
                "record_id": record_id,
            }
        )
    return out


@dataclass
class _RenderContext:
    """Per-scenario × locale rendering state, threaded through resolvers."""
    scenario_id: str
    locale: str
    rng: random.Random
    pii: Any
    phrases: Mapping[str, List[str]]
    # Per-locale PII helpers for code-switched id_types. Maps locale →
    # PII helper. Always includes the rendering locale's helper as
    # ``id_helpers[locale]``; cross-locale helpers are added on demand.
    id_helpers: Dict[str, Any] = field(default_factory=dict)
    # Slot values selected for this rendering pass. Repeated mention of
    # the same slot in the YAML resolves to the same value, so a phone
    # mentioned in turn 0 and again in turn 1 stays consistent.
    slot_cache: MutableMapping[str, str] = field(default_factory=dict)
    # Records materialized so far, keyed by ``record_key`` string.
    record_cache: MutableMapping[str, Tuple[Fixture, Mapping[str, Any]]] = field(default_factory=dict)
    # Loaded per-locale seed bundle. Available for narrow overrides that
    # need to build a divergent PII helper (e.g. fixture-address
    # divergence for scn-mt-205) without re-loading from disk.
    seeds: Optional[Mapping[str, Any]] = None


# Map id_type → the locale whose PII helper implements its generator.
# Used by code-switched scenarios where the id_type does not match the
# rendering locale's default helper class.
_ID_TYPE_HOME_LOCALE: Dict[str, str] = {
    "us_ssn": "en",
    "de_steuer_id": "de",
    "ru_snils": "ru",
    "ru_passport": "ru",
    "ru_inn": "ru",
}


def _seeded_rng(scenario_id: str, locale: str) -> random.Random:
    """Stable RNG keyed on (scenario_id, locale).

    Hash the joined string with SHA-256 and take the leading 16 hex
    digits as the seed. SHA-256 is byte-stable across machines (unlike
    Python's hash() which is salted), so the rendered dataset is
    reproducible from any checkout.
    """
    digest = hashlib.sha256(f"{scenario_id}|{locale}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _build_pii(seeds: Any, rng: random.Random, locale: str) -> Any:
    helper_cls = PII_HELPER_BY_LOCALE[locale]
    return helper_cls(seeds=seeds, rng=rng)


def _load_scenario_phrases(locale: str) -> Mapping[str, List[str]]:
    """Load ``benchmarks/data_generator/seeds/<locale>/scenario_phrases.yaml``.

    File is keyed ``intent_<name>: [variant, variant, ...]``. Returning
    an empty mapping when the file is missing keeps the renderer working
    on locales that don't have phrases yet (the renderer will then fail
    loudly only for scenarios that actually reference an intent).
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "data_generator"
        / "seeds"
        / locale
        / "scenario_phrases.yaml"
    )
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"{path} key {k!r} must be a list of phrase variants")
        out[str(k)] = [str(item) for item in v]
    return out


def _resolve_intent(name: str, ctx: _RenderContext) -> str:
    variants = ctx.phrases.get(name)
    if not variants:
        raise KeyError(
            f"scenario {ctx.scenario_id!r} references intent {name!r} "
            f"but locale {ctx.locale!r} has no phrase for it. Add it to "
            f"benchmarks/data_generator/seeds/{ctx.locale}/scenario_phrases.yaml."
        )
    # Stable selection: deterministic across re-renders.
    return ctx.rng.choice(variants)


def _resolve_id_type(scenario_meta: Mapping[str, Any], locale: str) -> str:
    """Resolve the active national-id type for a scenario × locale.

    Precedence:
      1. ``scenario.id_type_per_locale[locale]`` if set
      2. ``scenario.id_type`` if set
      3. The locale's default from ``DEFAULT_NATIONAL_ID_TYPE_BY_LOCALE``
    """
    per_locale = scenario_meta.get("id_type_per_locale") or {}
    if isinstance(per_locale, Mapping) and per_locale.get(locale):
        return str(per_locale[locale])
    if scenario_meta.get("id_type"):
        return str(scenario_meta["id_type"])
    return DEFAULT_NATIONAL_ID_TYPE_BY_LOCALE[locale]


def _resolve_pii_slot(
    slot: str, ctx: _RenderContext, scenario_meta: Mapping[str, Any],
) -> str:
    """Materialize a PII slot value once per scenario render.

    The same ``{PHONE}`` referenced in turn 0 and turn 2 must yield the
    same literal so that ground-truth scoring matches. Cache by slot name
    on the context.

    Resolution paths:
      - ``scenario.literal_slots_per_locale[locale][slot]`` — opaque
        per-locale literal (e.g. distinct TICKET_IDs across locales so
        the user-visible string stays clean while fixture keys still
        uniquify).
      - ``scenario.literal_slots[slot]`` — shared literal across all
        locales. Supports ``{LOCALE}`` and ``{SCENARIO_ID}`` substitution.
      - ``slot == "NATIONAL_ID"`` — locale-specific national-id helper.
      - Otherwise look up ``SLOT_TO_METHOD`` and call the locale's PII
        helper.
    """
    if slot in ctx.slot_cache:
        return ctx.slot_cache[slot]

    literals_per_locale = (
        scenario_meta.get("literal_slots_per_locale") or {}
    ).get(ctx.locale, {}) or {}
    if slot in literals_per_locale:
        value = str(literals_per_locale[slot])
        ctx.slot_cache[slot] = value
        return value

    literals = scenario_meta.get("literal_slots") or {}
    if slot in literals:
        value = (
            str(literals[slot])
            .replace("{LOCALE}", ctx.locale)
            .replace("{SCENARIO_ID}", ctx.scenario_id)
        )
        ctx.slot_cache[slot] = value
        return value

    if slot == "NATIONAL_ID":
        id_type = _resolve_id_type(scenario_meta, ctx.locale)
        method, _ph = NATIONAL_ID_METHOD_BY_TYPE.get(id_type, ("", ""))
        if not method:
            raise KeyError(
                f"scenario {ctx.scenario_id!r}: unknown id_type {id_type!r}",
            )
        # Code-switched scenarios may force a non-locale-default id_type
        # (e.g. ru_snils inside an EN turn). Generate the value using the
        # PII helper of the id type's home locale, since EnPII does not
        # implement snils() and DePII does not implement us_ssn().
        helper = ctx.pii
        home_locale = _ID_TYPE_HOME_LOCALE.get(id_type, ctx.locale)
        if home_locale != ctx.locale:
            helper = ctx.id_helpers.get(home_locale)
            if helper is None:
                raise RuntimeError(
                    f"scenario {ctx.scenario_id!r}: code-switched id_type "
                    f"{id_type!r} requires a {home_locale!r} PII helper "
                    "but none was provided",
                )
        value = getattr(helper, method)()
        ctx.slot_cache[slot] = value
        return value

    spec = SLOT_TO_METHOD.get(slot)
    if not spec:
        raise KeyError(
            f"scenario {ctx.scenario_id!r} uses unknown PII slot {slot!r}",
        )
    method_name, _placeholder = spec
    if not hasattr(ctx.pii, method_name):
        raise AttributeError(
            f"locale {ctx.locale!r} PII helper has no method {method_name!r} "
            f"required by slot {slot!r} in scenario {ctx.scenario_id!r}",
        )
    value = getattr(ctx.pii, method_name)()
    ctx.slot_cache[slot] = value
    return value


def _placeholder_for_slot(slot: str, scenario_meta: Mapping[str, Any], locale: str) -> str:
    """Return the canonical PII placeholder for a slot, or "" for non-PII.

    Non-PII slots (literals like TICKET_ID, ORDER_ID) are stripped from
    the ground-truth span list by returning the empty string here; the
    caller skips them when building ``pii_spans``.
    """
    if slot == "NATIONAL_ID":
        id_type = _resolve_id_type(scenario_meta, locale)
        _method, ph = NATIONAL_ID_METHOD_BY_TYPE[id_type]
        return ph
    spec = SLOT_TO_METHOD.get(slot)
    if not spec:
        return ""
    return spec[1]


def _expand_record_key(record_key: str, scenario_meta: Mapping[str, Any], locale: str) -> str:
    """Expand ``{ID_TYPE}`` (and similar) inside a record_key string.

    Scenarios that use ``record_key: "national_id:{ID_TYPE}"`` need the
    per-locale id_type substituted before fixture lookup.
    """
    if "{ID_TYPE}" in record_key:
        record_key = record_key.replace("{ID_TYPE}", _resolve_id_type(scenario_meta, locale))
    return record_key


def _apply_fixture_overrides(record_key: str, ctx: _RenderContext, payload: Dict[str, Any]) -> None:
    """Patch the materialized record payload for scenario-specific needs.

    Currently a no-op kept as a structured extension point — scn-mt-205's
    fixture/turn divergence is handled at the scenario-YAML level (turn 2
    uses {STREET_ADDRESS} instead of {INJECTED_ADDRESS}, which advances
    the RNG and gives a value distinct from the fixture's draw). Future
    scenarios with similar semantics-collision issues can branch here.
    """
    return


def _materialize_record(
    record_key: str, ctx: _RenderContext,
) -> Tuple[Fixture, Mapping[str, Any]]:
    """Materialize a fixture record, caching per scenario × locale."""
    if record_key in ctx.record_cache:
        return ctx.record_cache[record_key]
    template = TEMPLATES.get(record_key)
    if template is None:
        raise KeyError(
            f"scenario {ctx.scenario_id!r} references unknown record_key {record_key!r}; "
            f"add it to benchmarks/pii_evaluation/dataset/fixtures.TEMPLATES",
        )
    payload = dict(template.fields(ctx.pii, ctx.rng, ctx.locale))
    _apply_fixture_overrides(record_key, ctx, payload)
    record_id = f"{template.store}_{ctx.scenario_id}_{ctx.locale}"
    # Stash the record_id inside the payload so <record:KEY.record_id>
    # arguments resolve to the canonical id without a separate accessor.
    payload["record_id"] = record_id
    fixture = Fixture(
        store=template.store,
        record_id=record_id,
        record_locale=ctx.locale,
        payload=payload,
    )
    ctx.record_cache[record_key] = (fixture, payload)
    return fixture, payload


def _national_id_record_field(id_type: str) -> str:
    return {
        "us_ssn": "ssn",
        "de_steuer_id": "steuer_id",
        "ru_snils": "snils",
        "ru_passport": "series_number",
        "ru_inn": "tax_id",
    }.get(id_type, "")


def _align_record_to_expected_args(
    *,
    tool_name: str,
    resolved_args: Mapping[str, str],
    record: MutableMapping[str, Any],
) -> None:
    """Mutate a fixture payload so its lookup key matches gold args."""
    if tool_name == "customer_lookup_by_phone" and "phone" in resolved_args:
        record["phone"] = resolved_args["phone"]
    elif tool_name == "customer_lookup_by_email" and "email" in resolved_args:
        record["email"] = resolved_args["email"]
    elif tool_name == "customer_lookup_by_name" and "name" in resolved_args:
        record["name"] = resolved_args["name"]
    elif tool_name in {"appointment_cancel_by_phone", "appointment_reschedule_by_phone"} and "phone" in resolved_args:
        record["phone"] = resolved_args["phone"]
    elif tool_name == "payment_lookup_by_card" and "card_number" in resolved_args:
        record["card"] = resolved_args["card_number"]
        record["card_last4"] = re.sub(r"\D", "", resolved_args["card_number"])[-4:]
    elif tool_name == "verify_national_id":
        field_name = _national_id_record_field((resolved_args.get("id_type") or "").strip().lower())
        if field_name and "value" in resolved_args:
            record[field_name] = resolved_args["value"]
    elif tool_name == "invoice_lookup_by_tax_id" and "tax_id" in resolved_args:
        record["tax_id"] = resolved_args["tax_id"]
    elif tool_name == "update_ticket" and "ticket_id" in resolved_args:
        record["ticket_id"] = resolved_args["ticket_id"]


_INTENT_PATTERN = re.compile(r"\{\{intent_([a-z0-9_]+)\}\}")
_SLOT_PATTERN = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_INJECTED_PATTERN = re.compile(r"\{INJECTED_([A-Z_]+)\}")


def _substitute_template(
    template_text: str,
    ctx: _RenderContext,
    scenario_meta: Mapping[str, Any],
    injected_lookup: Mapping[str, str],
    injected_placeholder_lookup: Mapping[str, str] = MappingProxyType({}),
) -> Tuple[str, List[Tuple[str, str]]]:
    """Substitute intents, slots, and injected placeholders in one turn.

    Returns the rendered text plus the list of (placeholder, value) pairs
    for every PII slot that was substituted, so the caller can build
    ``PIIGroundTruth`` entries with the canonical placeholder.
    """
    out = template_text
    out = _INTENT_PATTERN.sub(lambda m: _resolve_intent(f"intent_{m.group(1)}", ctx), out)

    placeholders: List[Tuple[str, str]] = []

    def _slot_sub(match: "re.Match[str]") -> str:
        slot = match.group(1)
        if slot.startswith("INJECTED_"):
            key = slot[len("INJECTED_"):]
            value = injected_lookup.get(key)
            if value is None:
                raise KeyError(
                    f"scenario {ctx.scenario_id!r}: turn references "
                    f"{{{slot}}} but no matching injected_pii entry exists",
                )
            # Prefer the placeholder declared by the matching injected_pii
            # entry — that's the authoritative ground-truth tag. Fall back
            # to slot-based inference only if no entry matches.
            placeholder = injected_placeholder_lookup.get(key, "")
            if not placeholder:
                placeholder = _placeholder_for_slot(key, scenario_meta, ctx.locale) \
                    if key in SLOT_TO_METHOD or key == "NATIONAL_ID" else f"[{key}]"
            if placeholder:
                placeholders.append((placeholder, value))
            return value
        value = _resolve_pii_slot(slot, ctx, scenario_meta)
        placeholder = _placeholder_for_slot(slot, scenario_meta, ctx.locale)
        if placeholder:
            placeholders.append((placeholder, value))
        return value

    out = _SLOT_PATTERN.sub(_slot_sub, out)
    return out, placeholders


def _resolve_argument_value(
    arg_value: str,
    ctx: _RenderContext,
    scenario_meta: Mapping[str, Any],
    injected_lookup: Mapping[str, str],
    record_key_lookup: Mapping[str, Mapping[str, Any]],
) -> str:
    """Resolve an expected_tool_calls.arguments value template.

    Supported forms:
      - ``"literal"`` → unchanged
      - ``"{PHONE}"`` → cached PII slot value
      - ``"{INJECTED_EMAIL}"`` → injected PII value for that scenario
      - ``"{ID_TYPE}"`` → locale-default national-id type (or
        ``scenario.id_type`` if set)
      - ``"{LOCALE}"`` → the rendering locale (``"en"`` / ``"de"`` / ``"ru"``)
      - ``"<record:KEY.field>"`` → looks up ``KEY``'s materialized record
        and returns ``record.field`` (e.g. ``<record:customer_phone.email>``)
    """
    if arg_value.startswith("<record:") and arg_value.endswith(">"):
        body = arg_value[len("<record:"):-1]
        if "." not in body:
            raise ValueError(f"bad <record:...> arg: {arg_value!r}")
        key, field_name = body.split(".", 1)
        record = record_key_lookup.get(key)
        if record is None:
            raise KeyError(
                f"scenario {ctx.scenario_id!r}: <record:{key}.*> referenced "
                f"before record_key {key!r} materialized",
            )
        return str(record.get(field_name, ""))
    if arg_value.startswith("{") and arg_value.endswith("}"):
        slot = arg_value[1:-1]
        if slot == "ID_TYPE":
            return _resolve_id_type(scenario_meta, ctx.locale)
        if slot == "LOCALE":
            return ctx.locale
        if slot.startswith("INJECTED_"):
            key = slot[len("INJECTED_"):]
            value = injected_lookup.get(key)
            if value is None:
                raise KeyError(
                    f"scenario {ctx.scenario_id!r}: argument references "
                    f"{{{slot}}} but no matching injected_pii entry exists",
                )
            return value
        return _resolve_pii_slot(slot, ctx, scenario_meta)
    return arg_value


# ----------------------------------------------------------------------
# Top-level rendering
# ----------------------------------------------------------------------

# Imported lazily inside _render_one to avoid circular import; the parent
# package re-exports these dataclasses.
def _datatypes():
    from benchmarks.pii_evaluation.dataset import (  # noqa: WPS433 - intentional late import
        BoundaryPIIValue,
        EvalQuery,
        ExpectedToolCall,
        InjectedPII,
        PIIGroundTruth,
    )
    return EvalQuery, ExpectedToolCall, InjectedPII, PIIGroundTruth, BoundaryPIIValue


def _seed_bundle_cache_factory():
    cache: Dict[str, Any] = {}

    def loader(locale: str) -> Any:
        if locale not in cache:
            cache[locale] = load_seed_bundle(locale)
        return cache[locale]

    return loader


def _phrases_cache_factory():
    cache: Dict[str, Mapping[str, List[str]]] = {}

    def loader(locale: str) -> Mapping[str, List[str]]:
        if locale not in cache:
            cache[locale] = _load_scenario_phrases(locale)
        return cache[locale]

    return loader


def _setup_render_context(
    scenario: Mapping[str, Any],
    locale: str,
    seed_loader,
    phrases_loader,
) -> "_RenderContext":
    """Build the per-scenario × locale render context.

    Constructs the main and cross-locale PII helpers off a deterministic
    scenario-keyed RNG so the rendered dataset is byte-stable across
    runs.
    """
    scenario_id = str(scenario["scenario_id"])
    rng = _seeded_rng(scenario_id, locale)
    seeds = seed_loader(locale)
    pii = _build_pii(seeds, rng, locale)
    phrases = phrases_loader(locale)
    # Pre-build cross-locale PII helpers in case the scenario uses
    # code-switched id_types. Each helper gets its own scenario-keyed
    # sub-RNG so the cross-locale id is also deterministic.
    id_helpers: Dict[str, Any] = {locale: pii}
    for other_locale in LOCALES:
        if other_locale == locale:
            continue
        sub_rng = _seeded_rng(scenario_id, f"id-helper:{other_locale}")
        sub_seeds = seed_loader(other_locale)
        id_helpers[other_locale] = _build_pii(sub_seeds, sub_rng, other_locale)
    return _RenderContext(
        scenario_id=scenario_id,
        locale=locale,
        rng=rng,
        pii=pii,
        phrases=phrases,
        id_helpers=id_helpers,
        seeds=seeds,
    )


def _materialize_expected_records(
    scenario: Mapping[str, Any],
    ctx: "_RenderContext",
) -> Tuple[Dict[str, Mapping[str, Any]], List[Fixture], List[str]]:
    """Resolve and materialize the records named by expected_tool_calls.

    Returns ``(record_key_lookup, fixtures, resolved_record_keys)``. The
    ``resolved_record_keys`` list is parallel to ``expected_tool_calls``
    (one entry per call, empty string when the call has no record_key)
    so subsequent phases can look up by call index without mutating
    the input scenario dict (the same scenario is rendered for all
    three locales).
    """
    record_key_lookup: Dict[str, Mapping[str, Any]] = {}
    fixtures: List[Fixture] = []
    resolved_record_keys: List[str] = []
    for call_spec in scenario.get("expected_tool_calls") or ():
        record_key_raw = call_spec.get("record_key")
        if not record_key_raw:
            resolved_record_keys.append("")
            continue
        record_key = _expand_record_key(str(record_key_raw), scenario, ctx.locale)
        resolved_record_keys.append(record_key)
        if record_key not in record_key_lookup:
            fixture, payload = _materialize_record(record_key, ctx)
            record_key_lookup[record_key] = payload
            fixtures.append(fixture)
    return record_key_lookup, fixtures, resolved_record_keys


def _resolve_injected_pii(
    scenario: Mapping[str, Any],
    ctx: "_RenderContext",
    record_key_lookup: Mapping[str, Mapping[str, Any]],
    resolved_record_keys: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """Resolve injected_pii specs by pulling the named field off the producing record.

    Returns three parallel structures: ``injected_lookup`` (slot name →
    value, for template substitution), ``injected_placeholder_lookup``
    (slot name → placeholder), and ``injected_resolved`` (full records
    used downstream for boundary collection + invariant assertions).
    """
    expected_calls_raw = list(scenario.get("expected_tool_calls") or ())
    injected_lookup: Dict[str, str] = {}
    injected_placeholder_lookup: Dict[str, str] = {}
    injected_resolved: List[Dict[str, Any]] = []
    for inj in scenario.get("injected_pii") or ():
        producing_call_idx = int(inj.get("producing_call_index", inj.get("turn_index", 0)))
        if producing_call_idx >= len(expected_calls_raw):
            raise IndexError(
                f"scenario {ctx.scenario_id!r}: injected_pii.producing_call_index "
                f"{producing_call_idx} exceeds expected_tool_calls length "
                f"{len(expected_calls_raw)}",
            )
        producing_record_key = resolved_record_keys[producing_call_idx]
        if not producing_record_key:
            raise ValueError(
                f"scenario {ctx.scenario_id!r}: injected_pii references call "
                f"{producing_call_idx} which has no record_key",
            )
        record_payload = record_key_lookup[producing_record_key]
        from_field = str(inj["from_record_field"])
        value = str(record_payload.get(from_field, ""))
        if not value:
            raise ValueError(
                f"scenario {ctx.scenario_id!r}: record {producing_record_key!r} "
                f"has no value for field {from_field!r}",
            )
        slot_name = str(inj.get("slot_name", from_field.upper()))
        injected_lookup[slot_name] = value
        injected_placeholder_lookup[slot_name] = str(inj["placeholder"])
        injected_resolved.append(
            {
                "turn_index": int(inj["turn_index"]),
                "placeholder": str(inj["placeholder"]),
                "value": value,
                "must_appear_in_turn": int(inj["must_appear_in_turn"]),
            }
        )
    return injected_lookup, injected_placeholder_lookup, injected_resolved


def _pick_turn_templates(scenario: Mapping[str, Any], locale: str) -> List[str]:
    """Select the per-locale turn-template list.

    Scenarios declare turn templates one of two ways:

      ``turns:``               — one list, all locales share it; intent
                                 fragments translate per locale.
      ``turns_per_locale:``    — explicit per-locale list, for
                                 scaffolding that doesn't survive
                                 single-template + intent-fragment
                                 rendering across DE/RU.

    Exactly one form per scenario; mixing both raises ``ValueError``.
    All locales must produce the same turn count.
    """
    scenario_id = str(scenario["scenario_id"])
    turns_per_locale = scenario.get("turns_per_locale")
    if turns_per_locale and scenario.get("turns"):
        raise ValueError(
            f"scenario {scenario_id!r}: define either `turns` or `turns_per_locale`, not both",
        )
    if turns_per_locale:
        if not isinstance(turns_per_locale, Mapping):
            raise ValueError(
                f"scenario {scenario_id!r}: turns_per_locale must be a mapping",
            )
        if locale not in turns_per_locale:
            raise KeyError(
                f"scenario {scenario_id!r}: turns_per_locale missing locale "
                f"{locale!r} (have {sorted(turns_per_locale)})",
            )
        turn_counts = {loc: len(list(seq)) for loc, seq in turns_per_locale.items()}
        if len(set(turn_counts.values())) != 1:
            raise ValueError(
                f"scenario {scenario_id!r}: turns_per_locale turn counts differ "
                f"across locales: {turn_counts}",
            )
        turn_templates = list(turns_per_locale[locale])
    else:
        turn_templates = list(scenario.get("turns") or ())
    if not turn_templates:
        raise ValueError(f"scenario {scenario_id!r} has no turns")
    return turn_templates


def _render_turns_and_spans(
    scenario: Mapping[str, Any],
    ctx: "_RenderContext",
    injected_lookup: Mapping[str, str],
    injected_placeholder_lookup: Mapping[str, str],
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Render user turns and collect ground-truth + boundary spans from slot substitutions."""
    turn_templates = _pick_turn_templates(scenario, ctx.locale)
    rendered_turns: List[str] = []
    pii_spans_raw: List[Dict[str, Any]] = []
    boundary_pii_raw: List[Dict[str, str]] = []
    for turn_idx, template in enumerate(turn_templates):
        text, slot_pairs = _substitute_template(
            str(template), ctx, scenario, injected_lookup,
            injected_placeholder_lookup=injected_placeholder_lookup,
        )
        rendered_turns.append(text)
        for placeholder, value in slot_pairs:
            pii_spans_raw.append(
                {"placeholder": placeholder, "value": value, "turn": turn_idx},
            )
            boundary_pii_raw.append(
                {"placeholder": placeholder, "value": value, "source": "user_turn"},
            )
    return rendered_turns, pii_spans_raw, boundary_pii_raw


def _append_literal_pii_spans(
    scenario: Mapping[str, Any],
    locale: str,
    pii_spans_raw: List[Dict[str, Any]],
    boundary_pii_raw: List[Dict[str, str]],
) -> None:
    """Extend pii_spans_raw / boundary_pii_raw with YAML-declared literal spans.

    Used by hard-negative / security-stress / IBAN / IP scenarios that
    embed raw values into turn templates rather than slot placeholders.

      ``literal_pii_spans:``            applies to every locale
      ``literal_pii_spans_per_locale:`` picks the locale-matching list
                                        (used when the literal varies
                                        per locale, e.g. zero-width
                                        split phones in different
                                        countries' formats).
    """
    scenario_id = str(scenario["scenario_id"])
    literal_per_locale = scenario.get("literal_pii_spans_per_locale")
    if literal_per_locale and scenario.get("literal_pii_spans"):
        raise ValueError(
            f"scenario {scenario_id!r}: define either `literal_pii_spans` or "
            "`literal_pii_spans_per_locale`, not both",
        )
    if literal_per_locale:
        if locale not in literal_per_locale:
            raise KeyError(
                f"scenario {scenario_id!r}: literal_pii_spans_per_locale missing locale {locale!r}",
            )
        literal_iter = literal_per_locale[locale] or ()
    else:
        literal_iter = scenario.get("literal_pii_spans") or ()
    for literal in literal_iter:
        pii_spans_raw.append(
            {
                "placeholder": str(literal["placeholder"]),
                "value": str(literal["value"]),
                "turn": int(literal.get("turn", 0)),
            }
        )
        boundary_pii_raw.append(
            {
                "placeholder": str(literal["placeholder"]),
                "value": str(literal["value"]),
                "source": "user_turn",
            }
        )


def _build_expected_tool_calls(
    scenario: Mapping[str, Any],
    ctx: "_RenderContext",
    record_key_lookup: MutableMapping[str, Mapping[str, Any]],
    resolved_record_keys: List[str],
    injected_lookup: Mapping[str, str],
    ExpectedToolCall: Any,
) -> List[Any]:
    """Build the ExpectedToolCall list and align fixture records with their tool args.

    Note: ``_align_record_to_expected_args`` mutates the payloads in
    ``record_key_lookup`` so that the fixture record carries the
    exact values the tool is expected to be called with. This must
    run before fixture boundary collection.
    """
    expected_calls_raw = list(scenario.get("expected_tool_calls") or ())
    expected_tool_calls: List[Any] = []
    for call_idx, call_spec in enumerate(expected_calls_raw):
        args_in = dict(call_spec.get("arguments") or {})
        record_key = resolved_record_keys[call_idx]
        record = record_key_lookup.get(record_key) if record_key else None
        # Explicit literal record id wins over the materialized one,
        # used by no_pii_control scenarios that reference pre-populated
        # static records (e.g. orders).
        record_id = str(call_spec.get("expected_record_id") or "")
        if not record_id and record:
            tpl = TEMPLATES[record_key]
            record_id = f"{tpl.store}_{ctx.scenario_id}_{ctx.locale}"
        resolved_args: Dict[str, str] = {}
        for arg_key, arg_val in args_in.items():
            resolved_args[arg_key] = _resolve_argument_value(
                str(arg_val), ctx, scenario, injected_lookup, record_key_lookup,
            )
        if record is not None:
            _align_record_to_expected_args(
                tool_name=str(call_spec["tool_name"]),
                resolved_args=resolved_args,
                record=record,
            )
        expected_tool_calls.append(
            ExpectedToolCall(
                turn_index=int(call_spec.get("turn_index", 0)),
                tool_name=str(call_spec["tool_name"]),
                arguments=resolved_args,
                expected_record_id=record_id,
            )
        )
    return expected_tool_calls


def _assert_render_invariants(
    scenario_id: str,
    locale: str,
    rendered_turns: List[str],
    pii_spans_raw: List[Dict[str, Any]],
    injected_resolved: List[Dict[str, Any]],
) -> None:
    """Verify ground-truth and injected-PII values appear in their target turns."""
    for span in pii_spans_raw:
        turn_text = rendered_turns[span["turn"]]
        if span["value"] not in turn_text:
            raise ValueError(
                f"scenario {scenario_id!r}/{locale}: ground-truth value "
                f"{span['value']!r} not present in turn {span['turn']} text "
                f"{turn_text!r}",
            )
    for inj in injected_resolved:
        appear_turn_text = rendered_turns[inj["must_appear_in_turn"]]
        if inj["value"] not in appear_turn_text:
            raise ValueError(
                f"scenario {scenario_id!r}/{locale}: injected_pii value "
                f"{inj['value']!r} expected in turn "
                f"{inj['must_appear_in_turn']} but not present in "
                f"{appear_turn_text!r}",
            )


def _collect_post_call_boundary_pii(
    injected_resolved: List[Dict[str, Any]],
    record_key_lookup: Mapping[str, Mapping[str, Any]],
    locale: str,
    boundary_pii_raw: List[Dict[str, str]],
) -> None:
    """Extend boundary_pii_raw with injected-PII and fixture-record values.

    Called after ``_build_expected_tool_calls`` so the fixture boundary
    values reflect any record-payload alignment that happened there.
    """
    for inj in injected_resolved:
        boundary_pii_raw.append(
            {
                "placeholder": inj["placeholder"],
                "value": inj["value"],
                "source": "injected_pii",
            }
        )
    for record in record_key_lookup.values():
        boundary_pii_raw.extend(_fixture_boundary_values(record, locale))


def _assemble_eval_query(
    scenario: Mapping[str, Any],
    ctx: "_RenderContext",
    rendered_turns: List[str],
    pii_spans_raw: List[Dict[str, Any]],
    expected_tool_calls: List[Any],
    injected_resolved: List[Dict[str, Any]],
    boundary_pii_raw: List[Dict[str, str]],
    EvalQuery: Any,
    PIIGroundTruth: Any,
    InjectedPII: Any,
    BoundaryPIIValue: Any,
) -> Any:
    """Assemble the final EvalQuery from the per-phase outputs."""
    return EvalQuery(
        id=f"{ctx.scenario_id}-{ctx.locale}",
        scenario_id=ctx.scenario_id,
        language=ctx.locale,
        bucket=str(scenario["bucket"]),
        workflow=str(scenario.get("workflow", "")),
        turns=tuple(rendered_turns),
        pii_spans=tuple(
            PIIGroundTruth(
                placeholder=s["placeholder"], value=s["value"], turn=s["turn"],
            )
            for s in pii_spans_raw
        ),
        expected_tool_calls=tuple(expected_tool_calls),
        injected_pii=tuple(
            InjectedPII(
                turn_index=inj["turn_index"],
                placeholder=inj["placeholder"],
                value=inj["value"],
                must_appear_in_turn=inj["must_appear_in_turn"],
            )
            for inj in injected_resolved
        ),
        boundary_pii_values=tuple(
            BoundaryPIIValue(
                placeholder=item["placeholder"],
                value=item["value"],
                source=item["source"],
                record_id=item.get("record_id"),
            )
            for item in boundary_pii_raw
        ),
        available_tools=tuple(scenario.get("available_tools", ()) or ()),
    )


def _render_one(
    scenario: Mapping[str, Any],
    locale: str,
    *,
    seed_loader,
    phrases_loader,
) -> Tuple[Any, List[Fixture]]:
    """Render one scenario × locale into an EvalQuery + fixtures list.

    The orchestrator runs each phase in order; see the called helpers
    for per-phase contracts. The phase ordering is load-bearing:
    ``_build_expected_tool_calls`` mutates fixture payloads via
    ``_align_record_to_expected_args``, and the subsequent
    ``_collect_post_call_boundary_pii`` step reads those payloads.
    """
    EvalQuery, ExpectedToolCall, InjectedPII, PIIGroundTruth, BoundaryPIIValue = _datatypes()

    ctx = _setup_render_context(scenario, locale, seed_loader, phrases_loader)

    record_key_lookup, fixtures, resolved_record_keys = _materialize_expected_records(
        scenario, ctx,
    )
    injected_lookup, injected_placeholder_lookup, injected_resolved = _resolve_injected_pii(
        scenario, ctx, record_key_lookup, resolved_record_keys,
    )

    rendered_turns, pii_spans_raw, boundary_pii_raw = _render_turns_and_spans(
        scenario, ctx, injected_lookup, injected_placeholder_lookup,
    )
    _append_literal_pii_spans(scenario, ctx.locale, pii_spans_raw, boundary_pii_raw)

    expected_tool_calls = _build_expected_tool_calls(
        scenario, ctx, record_key_lookup, resolved_record_keys,
        injected_lookup, ExpectedToolCall,
    )

    _assert_render_invariants(
        ctx.scenario_id, ctx.locale, rendered_turns, pii_spans_raw, injected_resolved,
    )
    _collect_post_call_boundary_pii(
        injected_resolved, record_key_lookup, ctx.locale, boundary_pii_raw,
    )

    query = _assemble_eval_query(
        scenario, ctx, rendered_turns, pii_spans_raw, expected_tool_calls,
        injected_resolved, boundary_pii_raw,
        EvalQuery, PIIGroundTruth, InjectedPII, BoundaryPIIValue,
    )
    return query, fixtures


def _scenario_files(scenarios_dir: Path) -> List[Path]:
    return sorted(p for p in scenarios_dir.glob("*.yaml") if p.is_file())


def _load_scenarios(scenarios_dir: Path) -> List[Mapping[str, Any]]:
    """Load all scenarios from ``scenarios_dir/*.yaml`` in deterministic order."""
    out: List[Mapping[str, Any]] = []
    for path in _scenario_files(scenarios_dir):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(data, list):
            raise ValueError(f"{path} top-level must be a list of scenario dicts")
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError(f"{path} contains non-dict entry: {entry!r}")
            out.append(entry)
    # Deduplicate scenario_ids
    seen = set()
    for entry in out:
        sid = entry.get("scenario_id")
        if sid in seen:
            raise ValueError(f"duplicate scenario_id {sid!r}")
        seen.add(sid)
    return out


def render_all(
    scenarios_dir: Optional[Path] = None,
    locales: Tuple[str, ...] = LOCALES,
) -> Tuple[List[Any], List[Fixture]]:
    """Render every scenario × locale into a flat list of EvalQuery + Fixture.

    The output order is deterministic: scenarios in source-file order,
    locales in ``LOCALES`` order. Re-rendering yields byte-identical
    output, which is what makes ``--freeze`` stable.
    """
    if scenarios_dir is None:
        scenarios_dir = Path(__file__).resolve().parent / "scenarios"
    scenarios = _load_scenarios(scenarios_dir)
    seed_loader = _seed_bundle_cache_factory()
    phrases_loader = _phrases_cache_factory()

    queries: List[Any] = []
    fixtures: List[Fixture] = []
    for scenario in scenarios:
        for locale in locales:
            query, scenario_fixtures = _render_one(
                scenario, locale,
                seed_loader=seed_loader,
                phrases_loader=phrases_loader,
            )
            queries.append(query)
            fixtures.extend(scenario_fixtures)
    return queries, fixtures


# ----------------------------------------------------------------------
# Frozen JSONL artifact
# ----------------------------------------------------------------------

def _query_to_json(query: Any) -> Dict[str, Any]:
    return {
        "id": query.id,
        "scenario_id": query.scenario_id,
        "language": query.language,
        "bucket": query.bucket,
        "workflow": query.workflow,
        "turns": list(query.turns),
        "pii_spans": [
            {"placeholder": s.placeholder, "value": s.value, "turn": s.turn}
            for s in query.pii_spans
        ],
        "expected_tool_calls": [
            {
                "turn_index": c.turn_index,
                "tool_name": c.tool_name,
                "arguments": dict(c.arguments),
                "expected_record_id": c.expected_record_id,
            }
            for c in query.expected_tool_calls
        ],
        "injected_pii": [
            {
                "turn_index": i.turn_index,
                "placeholder": i.placeholder,
                "value": i.value,
                "must_appear_in_turn": i.must_appear_in_turn,
            }
            for i in query.injected_pii
        ],
        "boundary_pii_values": [
            {
                "placeholder": b.placeholder,
                "value": b.value,
                "source": b.source,
                "record_id": b.record_id,
            }
            for b in query.boundary_pii_values
        ],
        "available_tools": list(query.available_tools),
    }


def _fixture_to_json(fixture: Fixture) -> Dict[str, Any]:
    return {
        "store": fixture.store,
        "record_id": fixture.record_id,
        "record_locale": fixture.record_locale,
        "payload": dict(fixture.payload),
    }


def serialize(queries: List[Any], fixtures: List[Fixture]) -> str:
    """Serialize the rendered dataset as a single JSONL string.

    Each line is a JSON object with a ``kind`` discriminator (``query`` or
    ``fixture``). Queries come first (sorted by id), fixtures after
    (sorted by record_id), so the file is human-scrollable as well as
    machine-parseable.
    """
    lines: List[str] = []
    for q in sorted(queries, key=lambda q: q.id):
        rec = {"kind": "query", **_query_to_json(q)}
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    for f in sorted(fixtures, key=lambda f: f.record_id):
        rec = {"kind": "fixture", **_fixture_to_json(f)}
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


def freeze(
    output_dir: Optional[Path] = None,
    scenarios_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Render and write ``data/dataset_v1.jsonl`` + ``.sha256``."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    queries, fixtures = render_all(scenarios_dir=scenarios_dir)
    payload = serialize(queries, fixtures)
    out_jsonl = output_dir / "dataset_v1.jsonl"
    out_sha = output_dir / "dataset_v1.sha256"
    out_jsonl.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    out_sha.write_text(digest + "\n", encoding="utf-8")
    return out_jsonl, digest


def load_frozen(
    jsonl_path: Optional[Path] = None,
) -> Tuple[List[Any], List[Fixture]]:
    """Load a previously frozen JSONL into EvalQuery + Fixture lists."""
    EvalQuery, ExpectedToolCall, InjectedPII, PIIGroundTruth, BoundaryPIIValue = _datatypes()
    if jsonl_path is None:
        jsonl_path = Path(__file__).resolve().parent / "data" / "dataset_v1.jsonl"
    if not jsonl_path.exists():
        return [], []
    queries: List[Any] = []
    fixtures: List[Fixture] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        kind = rec.pop("kind")
        if kind == "query":
            queries.append(
                EvalQuery(
                    id=rec["id"],
                    scenario_id=rec.get("scenario_id", ""),
                    language=rec["language"],
                    bucket=rec["bucket"],
                    workflow=rec.get("workflow", ""),
                    turns=tuple(rec["turns"]),
                    pii_spans=tuple(
                        PIIGroundTruth(
                            placeholder=s["placeholder"], value=s["value"], turn=s["turn"],
                        )
                        for s in rec.get("pii_spans", [])
                    ),
                    expected_tool_calls=tuple(
                        ExpectedToolCall(
                            turn_index=c["turn_index"],
                            tool_name=c["tool_name"],
                            arguments=dict(c.get("arguments") or {}),
                            expected_record_id=c.get("expected_record_id", ""),
                        )
                        for c in rec.get("expected_tool_calls", [])
                    ),
                    injected_pii=tuple(
                        InjectedPII(
                            turn_index=i["turn_index"],
                            placeholder=i["placeholder"],
                            value=i["value"],
                            must_appear_in_turn=i["must_appear_in_turn"],
                        )
                        for i in rec.get("injected_pii", [])
                    ),
                    boundary_pii_values=tuple(
                        BoundaryPIIValue(
                            placeholder=b["placeholder"],
                            value=b["value"],
                            source=b["source"],
                            record_id=b.get("record_id"),
                        )
                        for b in rec.get("boundary_pii_values", [])
                    ),
                    available_tools=tuple(rec.get("available_tools") or ()),
                )
            )
        elif kind == "fixture":
            fixtures.append(
                Fixture(
                    store=rec["store"],
                    record_id=rec["record_id"],
                    record_locale=rec.get("record_locale", ""),
                    payload=dict(rec["payload"]),
                )
            )
        else:
            raise ValueError(f"unknown record kind in JSONL: {kind!r}")
    return queries, fixtures


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Render all scenarios and write data/dataset_v1.jsonl + sha256.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-render and verify the SHA matches the committed sha256 file.",
    )
    args = parser.parse_args()

    if args.check:
        queries, fixtures = render_all()
        payload = serialize(queries, fixtures)
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        sha_path = Path(__file__).resolve().parent / "data" / "dataset_v1.sha256"
        if not sha_path.exists():
            print(f"no committed sha at {sha_path}; run --freeze first")
            return 2
        expected = sha_path.read_text(encoding="utf-8").strip()
        if actual == expected:
            print(f"OK: {len(queries)} queries / {len(fixtures)} fixtures, sha={actual}")
            return 0
        print(f"MISMATCH: expected {expected}, got {actual}")
        return 1

    out_jsonl, digest = freeze()
    print(f"wrote {out_jsonl} (sha256={digest})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
