"""Canned per-locale record templates for the rendered dataset.

When the renderer expands a scenario × locale, it picks from a per-locale
catalog of *record templates* whose PII fields are filled by the
``data_generator`` checksum-valid PII helpers. Each filled record is
registered with the synthetic CRM tool store at experiment startup so
that:

  - ``customer_lookup_by_phone`` returns the customer record the scenario
    asked it to return,
  - ``update_contact`` mutates that record's email/phone in place,
  - ``verify_national_id`` finds the right id record per id_type.

Template keys are referenced from scenario YAMLs as ``record_key``. The
renderer uses the ``store`` field on the template to decide which tool
record store the materialized record belongs to.

Per-scenario record ids are generated as ``<store>_<scenario_id>_<locale>``
so they are globally unique and join the EvalQuery to its fixture without
requiring extra metadata in the scenario file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple


@dataclass(frozen=True)
class RecordTemplate:
    """One record template entry, keyed by ``record_key`` in scenario YAMLs.

    ``store`` selects the tool record store ("customers", "appointments",
    "tickets", "contractors", "national_id:us_ssn", ...).

    ``fields`` is a callable that, given a per-locale PII helper and an RNG,
    returns the record body (without ``record_id``; the renderer injects
    that). Splitting the helper out lets a single template definition work
    across all three locales — the helper resolves names/phones/etc.
    correctly per locale, while the template controls the *shape* of the
    record.
    """
    store: str
    fields: Callable[[Any, Any, str], Mapping[str, Any]]


def _customer_template(
    pii: Any, rng: Any, locale: str,
) -> Mapping[str, Any]:
    return {
        "name": pii.person_name(),
        "phone": pii.phone(),
        "email": pii.email(),
        "address": pii.street_address(),
        "tier": rng.choice(("gold", "silver", "platinum")),
    }


def _appointment_template(
    pii: Any, rng: Any, locale: str,
) -> Mapping[str, Any]:
    services_by_locale = {
        "en": ("consultation", "follow-up", "diagnostic", "intake"),
        "de": ("Beratung", "Nachuntersuchung", "Diagnose", "Aufnahme"),
        "ru": ("консультация", "повторный приём", "диагностика", "приём"),
    }
    return {
        "phone": pii.phone(),
        "slot": f"2026-{rng.randint(4, 9):02d}-{rng.randint(1, 28):02d} {rng.randint(9, 17):02d}:00",
        "service": rng.choice(services_by_locale.get(locale, services_by_locale["en"])),
    }


def _contractor_template(
    pii: Any, rng: Any, locale: str,
) -> Mapping[str, Any]:
    org_names_by_locale = {
        "en": ("ExampleCorp", "Acme Holdings", "Beacon Systems", "OakRiver Ltd"),
        "de": ("ExampleCorp GmbH", "Beacon Systeme GmbH", "OakRiver AG", "Akme Holding"),
        "ru": ("ООО Ромашка", "ЗАО Лютик", "ООО Маяк", "АО Восход"),
    }
    return {
        "name": rng.choice(org_names_by_locale.get(locale, org_names_by_locale["en"])),
        "tax_id": pii.inn_legal() if locale == "ru" else (
            "".join(str(rng.randint(0, 9)) for _ in range(9)) if locale == "en"
            else "".join(str(rng.randint(0, 9)) for _ in range(9))
        ),
        "balance_rub": rng.randint(0, 5_000_000),
        "contact_name": pii.person_name(),
        "contact_phone": pii.phone(),
    }


def _payment_template(
    pii: Any, rng: Any, locale: str,
) -> Mapping[str, Any]:
    card = pii.card()
    return {
        "card": card,
        "card_last4": card.replace(" ", "")[-4:],
        "amount": round(rng.uniform(10.0, 999.0), 2),
        "merchant": "ExampleCorp",
        "disputable": True,
    }


def _ticket_template(
    pii: Any, rng: Any, locale: str,
) -> Mapping[str, Any]:
    subjects_by_locale = {
        "en": ("Refund inquiry", "Address change", "Billing dispute", "Account access"),
        "de": ("Rückerstattung", "Adressänderung", "Rechnungsstreit", "Kontozugriff"),
        "ru": ("Возврат средств", "Смена адреса", "Спор по счёту", "Доступ к аккаунту"),
    }
    return {
        "ticket_id": f"TCK-{rng.randint(1000, 9999)}",
        "subject": rng.choice(subjects_by_locale.get(locale, subjects_by_locale["en"])),
        "status": "open",
        "owner": rng.choice(("support", "billing", "operations")),
    }


def _us_ssn_template(pii: Any, rng: Any, locale: str) -> Mapping[str, Any]:
    return {"ssn": pii.us_ssn(), "status": "clear", "checked_at": "2026-03-15"}


def _de_steuer_id_template(pii: Any, rng: Any, locale: str) -> Mapping[str, Any]:
    return {"steuer_id": pii.steuer_id(), "status": "active", "issued": "2019-04-22"}


def _ru_snils_template(pii: Any, rng: Any, locale: str) -> Mapping[str, Any]:
    return {"snils": pii.snils(), "registered": True}


def _ru_passport_template(pii: Any, rng: Any, locale: str) -> Mapping[str, Any]:
    return {"series_number": pii.passport(), "valid": True, "issued": "2018-06-01"}


# Aliases for the canonical customer record. The distinct keys exist so
# scenario YAMLs can read self-documentingly (``customer_phone`` vs
# ``customer_email``); the rendered record is identical.
_CUSTOMER_RECORD = RecordTemplate(store="customers", fields=_customer_template)
_CUSTOMER_ALIASES = ("customer", "customer_phone", "customer_email", "customer_name", "contact_update")

# Public catalog of templates keyed by ``record_key`` strings used in YAMLs.
TEMPLATES: Dict[str, RecordTemplate] = {
    **{alias: _CUSTOMER_RECORD for alias in _CUSTOMER_ALIASES},
    "appointment": RecordTemplate(store="appointments", fields=_appointment_template),
    "contractor": RecordTemplate(store="contractors", fields=_contractor_template),
    "payment": RecordTemplate(store="payments", fields=_payment_template),
    "ticket": RecordTemplate(store="tickets", fields=_ticket_template),
    "national_id:us_ssn": RecordTemplate(store="background_checks", fields=_us_ssn_template),
    "national_id:de_steuer_id": RecordTemplate(store="steuer_ids", fields=_de_steuer_id_template),
    "national_id:ru_snils": RecordTemplate(store="snils", fields=_ru_snils_template),
    "national_id:ru_passport": RecordTemplate(store="passports", fields=_ru_passport_template),
}


# Per-locale id types preferred for the *generic* national-id slot. Used by
# the renderer when a scenario asks for ``id_type: locale_default``.
DEFAULT_NATIONAL_ID_TYPE_BY_LOCALE: Mapping[str, str] = {
    "en": "us_ssn",
    "de": "de_steuer_id",
    "ru": "ru_snils",
}
