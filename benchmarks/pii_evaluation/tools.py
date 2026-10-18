"""Synthetic CRM tools for the PII evaluation experiment.

These are real LangChain @tool functions, indistinguishable in shape from
production tools — the broker can wrap them, the LLM can call them via
function calling. Each tool maintains a small in-memory record store keyed
to the dataset's expected_record_id values; on lookup miss it returns the
literal string "not found", which is exactly the failure mode the paper
attributes to destructive redaction.

Records returned by the tools deliberately embed *other* PII fields (a phone
lookup yields a record containing the customer's email, address, etc.).
That is what makes the multi-turn bucket meaningful: the model sees PII in
tool results and is then asked about it on the next turn, which is exactly
where the leak guard has to do its job.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool, tool


# ----------------------------------------------------------------------
# Synthetic record store
# ----------------------------------------------------------------------

# Customers, keyed by record_id. Each record carries multiple PII fields.
_CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "cust_001": {
        "record_id": "cust_001",
        "name": "Jordan Lee",
        "phone": "+1 555 014 2233",
        "email": "jordan.lee@example.com",
        "address": "742 Evergreen Terrace, Springfield",
        "tier": "gold",
    },
    "cust_002": {
        "record_id": "cust_002",
        "name": "Sam Patel",
        "phone": "+1 555 018 7654",
        "email": "sam.patel@example.com",
        "address": "1600 Amphitheatre Pkwy, Mountain View",
        "tier": "silver",
        "card_last4": "4242",
    },
    "cust_003": {
        "record_id": "cust_003",
        "name": "Alice Wong",
        "phone": "+1 555 014 1010",
        "email": "alice@example.com",
        "address": "350 5th Ave, New York",
        "tier": "gold",
    },
    "cust_004": {
        "record_id": "cust_004",
        "name": "Robert Smith",
        "phone": "+1 555 014 2020",
        "email": "bob.smith@example.org",
        "address": "1 Infinite Loop, Cupertino",
        "tier": "platinum",
    },
    "cust_007": {
        "record_id": "cust_007",
        "name": "Maria Hernandez",
        "phone": "+1 555 014 9988",
        "email": "maria.h@example.com",
        "address": "500 Terry A Francois Blvd, San Francisco",
        "tier": "silver",
    },
    "cust_008": {
        "record_id": "cust_008",
        "name": "Carol Jensen",
        "phone": "+1 555 014 3030",
        "email": "carol@example.net",
        "address": "10 Downing St, London",
        "tier": "gold",
    },
    "cust_101": {
        "record_id": "cust_101",
        "name": "Иван Петров",
        "phone": "+7 495 014 2233",
        "email": "ivan.petrov@example.ru",
        "address": "Тверская 1, Москва",
        "tier": "gold",
    },
    "cust_102": {
        "record_id": "cust_102",
        "name": "Ольга Соколова",
        "phone": "+7 495 018 7654",
        "email": "olga.sokolova@example.ru",
        "address": "Невский пр. 5, Санкт-Петербург",
        "tier": "silver",
    },
    "cust_107": {
        "record_id": "cust_107",
        "name": "Иван Кузнецов",
        "phone": "+7 495 014 4040",
        "email": "ivan@example.ru",
        "address": "Арбат 12, Москва",
        "tier": "gold",
    },
    "cust_108": {
        "record_id": "cust_108",
        "name": "Дмитрий Орлов",
        "phone": "+7 495 014 9988",
        "email": "dmitry.orlov@example.ru",
        "address": "Маяковского 3, Казань",
        "tier": "platinum",
    },
    "cust_011": {
        "record_id": "cust_011",
        "name": "Diane Park",
        "phone": "+1 555 014 5151",
        "email": "support@example.io",
        "address": "20 W 34th St, New York",
        "tier": "platinum",
    },
    "cust_113": {
        "record_id": "cust_113",
        "name": "Алексей Морозов",
        # Landline (city code 3412 = Izhevsk) — exercises the ru_landline detector
        "phone": "+7 (3412) 60-21-20",
        "email": "a.morozov@example.ru",
        "address": "Пушкинская 270, Ижевск",
        "tier": "silver",
    },
}

_APPOINTMENTS: Dict[str, Dict[str, Any]] = {
    "appt_002": {"record_id": "appt_002", "phone": "+1 555 018 7654", "slot": "2026-04-12 14:00", "service": "consultation"},
    "appt_102": {"record_id": "appt_102", "phone": "+7 495 018 7654", "slot": "2026-04-12 14:00", "service": "консультация"},
}

_BACKGROUND_CHECKS: Dict[str, Dict[str, Any]] = {
    "bg_005": {"record_id": "bg_005", "ssn": "123-45-6789", "status": "clear", "checked_at": "2026-03-15"},
}

# Identity records keyed by (id_type, normalized_value) for the unified
# verify_national_id tool. Populated lazily below from the per-type
# stores so the source of truth remains the legacy structure (preserves
# back-compat for any test that imports background_check_by_ssn etc.).
_NATIONAL_ID_RECORDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "us_ssn": {},
    "de_steuer_id": {},
    "ru_snils": {},
    "ru_passport": {},
}

# DE Steuer-ID corpus — small fixed set used by scenarios that exercise the
# DE branch of verify_national_id. Values are checksum-valid (ISO 7064
# MOD 11,10) so the regex policy accepts them; their digits are arbitrary
# beyond that constraint.
_STEUER_ID_RECORDS: Dict[str, Dict[str, Any]] = {
    "steuer_205": {"record_id": "steuer_205", "steuer_id": "65929970489", "status": "active", "issued": "2019-04-22"},
    "steuer_206": {"record_id": "steuer_206", "steuer_id": "23145679880", "status": "active", "issued": "2020-11-30"},
}

_TICKETS: Dict[str, Dict[str, Any]] = {
    "ticket_701": {"record_id": "ticket_701", "ticket_id": "TCK-7001", "subject": "Refund inquiry", "status": "open", "owner": "support"},
    "ticket_702": {"record_id": "ticket_702", "ticket_id": "TCK-7042", "subject": "Address change", "status": "open", "owner": "support"},
    "ticket_703": {"record_id": "ticket_703", "ticket_id": "TCK-7099", "subject": "Billing dispute", "status": "open", "owner": "billing"},
}

_PAYMENTS: Dict[str, Dict[str, Any]] = {
    "pay_006": {"record_id": "pay_006", "card_last4": "1111", "amount": 199.00, "merchant": "ExampleCorp", "disputable": True},
}

_CONTRACTORS: Dict[str, Dict[str, Any]] = {
    "contractor_103": {
        "record_id": "contractor_103",
        "name": "ООО Ромашка",
        "tax_id": "7707083893",
        "balance_rub": 124500,
        "contact_name": "Сергей Иванов",
        "contact_phone": "+7 495 014 5050",
    },
    "contractor_104": {
        "record_id": "contractor_104",
        "name": "ЗАО Лютик",
        "tax_id": "5024153080",
        "balance_rub": 0,
        "contact_name": "Анна Петрова",
        "contact_phone": "+7 495 014 6060",
    },
}

_PASSPORTS: Dict[str, Dict[str, Any]] = {
    "passport_105": {"record_id": "passport_105", "series_number": "4509 123456", "valid": True, "issued": "2018-06-01"},
}

_SNILS_RECORDS: Dict[str, Dict[str, Any]] = {
    "snils_106": {"record_id": "snils_106", "snils": "112-233-445 95", "registered": True},
    "snils_111": {"record_id": "snils_111", "snils": "555-666-777 88", "registered": True},
}

_ORDERS: Dict[str, Dict[str, Any]] = {
    "order_401": {"record_id": "order_401", "order_id": "ORD-1001", "status": "shipped", "total": 49.99},
    "order_402": {"record_id": "order_402", "order_id": "ORD-1042", "status": "processing", "total": 129.50},
    "order_403": {"record_id": "order_403", "order_id": "ORD-1057", "status": "delivered", "total": 12.00},
    "order_404": {"record_id": "order_404", "order_id": "ORD-1099", "status": "cancelled", "total": 0.00},
    "order_405": {"record_id": "order_405", "order_id": "ORD-1112", "status": "shipped", "total": 88.00},
    "order_406": {"record_id": "order_406", "order_id": "ORD-2001", "status": "delivered", "total": 245.00},
    "order_407": {"record_id": "order_407", "order_id": "ORD-2042", "status": "processing", "total": 19.99},
    "order_408": {"record_id": "order_408", "order_id": "ORD-2099", "status": "shipped", "total": 67.50},
}


# ----------------------------------------------------------------------
# Indexes — built lazily, keyed on normalized lookup values
# ----------------------------------------------------------------------

def _normalize_phone(value: str) -> str:
    """Canonicalize a phone number for index lookup.

    Strips all non-digit characters, then maps the result onto a single
    canonical form per region:
      - 11 digits starting with 7 or 8 → '7XXXXXXXXXX' (Russia: '8' and
        '+7' both refer to the same number)
      - 11 digits starting with 1     → unchanged (United States)
      - 10 digits                     → '1XXXXXXXXXX' (bare US format)
    Anything else is returned digits-only as a best-effort fallback.
    This lets the dataset express phone numbers in the formats users
    actually type — '(555) 014-2233', '555-014-2233', '+1 555 014 2233',
    '+7 495 014 2233', '8 495 014 2233' — and still resolve to the same
    record.
    """
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 11 and digits[0] in ("7", "8"):
        return "7" + digits[1:]
    if len(digits) == 11 and digits[0] == "1":
        return digits
    if len(digits) == 10:
        return "1" + digits
    return digits


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _normalize_order_id(value: str) -> str:
    return (value or "").strip().upper().replace(" ", "")


# Indexes — initialized after all the static record-store bodies have
# been declared. Mutable so register_fixtures can extend them at runtime
# when the rendered dataset's per-scenario records are loaded.

_PHONE_INDEX: Dict[str, Dict[str, Any]] = {}
_EMAIL_INDEX: Dict[str, Dict[str, Any]] = {}
_NAME_INDEX: Dict[str, Dict[str, Any]] = {}
_APPT_PHONE_INDEX: Dict[str, Dict[str, Any]] = {}
_SSN_INDEX: Dict[str, Dict[str, Any]] = {}
_CARD_FULL_INDEX: Dict[str, Dict[str, Any]] = {}
_TAX_ID_INDEX: Dict[str, Dict[str, Any]] = {}
_PASSPORT_INDEX: Dict[str, Dict[str, Any]] = {}
_SNILS_INDEX: Dict[str, Dict[str, Any]] = {}
_STEUER_ID_INDEX: Dict[str, Dict[str, Any]] = {}
_ORDER_INDEX: Dict[str, Dict[str, Any]] = {}
_TICKET_INDEX: Dict[str, Dict[str, Any]] = {}


def _rebuild_indexes() -> None:
    """Recompute every lookup index from the underlying record stores.

    Called once at module import and again whenever ``register_fixtures``
    extends the stores at runtime. Index dicts are mutated in place so
    references held elsewhere (e.g. the closure inside ``make_tools``)
    keep seeing the latest data.
    """
    _PHONE_INDEX.clear()
    _PHONE_INDEX.update({_normalize_phone(c["phone"]): c for c in _CUSTOMERS.values() if c.get("phone")})
    _EMAIL_INDEX.clear()
    _EMAIL_INDEX.update({_normalize_email(c["email"]): c for c in _CUSTOMERS.values() if c.get("email")})
    _NAME_INDEX.clear()
    _NAME_INDEX.update({str(c["name"]).strip().lower(): c for c in _CUSTOMERS.values() if c.get("name")})
    _APPT_PHONE_INDEX.clear()
    _APPT_PHONE_INDEX.update({_normalize_phone(a["phone"]): a for a in _APPOINTMENTS.values() if a.get("phone")})
    _SSN_INDEX.clear()
    _SSN_INDEX.update({_normalize_digits(b["ssn"]): b for b in _BACKGROUND_CHECKS.values() if b.get("ssn")})
    _CARD_FULL_INDEX.clear()
    _CARD_FULL_INDEX.update({_normalize_digits("4111111111111111"): _PAYMENTS["pay_006"]})
    for p in _PAYMENTS.values():
        if p.get("card"):
            _CARD_FULL_INDEX[_normalize_digits(p["card"])] = p
    _TAX_ID_INDEX.clear()
    _TAX_ID_INDEX.update({_normalize_digits(c["tax_id"]): c for c in _CONTRACTORS.values() if c.get("tax_id")})
    _PASSPORT_INDEX.clear()
    _PASSPORT_INDEX.update({_normalize_digits(p["series_number"]): p for p in _PASSPORTS.values() if p.get("series_number")})
    _SNILS_INDEX.clear()
    _SNILS_INDEX.update({_normalize_digits(s["snils"]): s for s in _SNILS_RECORDS.values() if s.get("snils")})
    _STEUER_ID_INDEX.clear()
    _STEUER_ID_INDEX.update({_normalize_digits(s["steuer_id"]): s for s in _STEUER_ID_RECORDS.values() if s.get("steuer_id")})
    _ORDER_INDEX.clear()
    _ORDER_INDEX.update({_normalize_order_id(o["order_id"]): o for o in _ORDERS.values() if o.get("order_id")})
    _TICKET_INDEX.clear()
    _TICKET_INDEX.update({_normalize_order_id(t["ticket_id"]): t for t in _TICKETS.values() if t.get("ticket_id")})


# Per-id-type lookups for verify_national_id. Bound to the index dicts
# above (mutable in place) so register_fixtures takes effect immediately.
_NATIONAL_ID_INDEX: Dict[str, Dict[str, Dict[str, Any]]] = {
    "us_ssn": _SSN_INDEX,
    "de_steuer_id": _STEUER_ID_INDEX,
    "ru_snils": _SNILS_INDEX,
    "ru_passport": _PASSPORT_INDEX,
}


# ----------------------------------------------------------------------
# Runtime fixture registration — used by the rendered dataset
# ----------------------------------------------------------------------

# Map from ``Fixture.store`` strings to the underlying dict and index keys
# the renderer needs to populate.
_STORE_TO_DICT: Dict[str, Dict[str, Dict[str, Any]]] = {
    "customers": _CUSTOMERS,
    "appointments": _APPOINTMENTS,
    "tickets": _TICKETS,
    "contractors": _CONTRACTORS,
    "payments": _PAYMENTS,
    "background_checks": _BACKGROUND_CHECKS,
    "steuer_ids": _STEUER_ID_RECORDS,
    "snils": _SNILS_RECORDS,
    "passports": _PASSPORTS,
}


def register_fixtures(fixtures: List[Any]) -> None:
    """Insert rendered-dataset fixtures into the synthetic CRM stores.

    Each fixture has a ``store`` selector (matching the keys of
    ``_STORE_TO_DICT``) and a ``payload`` dict. The fixture's
    ``record_id`` is injected into the payload so tools that JSON-encode
    the record carry the canonical id back to the metric harness.

    Calling ``register_fixtures(...)`` is idempotent: re-registering a
    record with the same id replaces the previous payload. Indexes are
    rebuilt after each batch.
    """
    for fx in fixtures:
        store = _STORE_TO_DICT.get(fx.store)
        if store is None:
            raise KeyError(f"unknown fixture store {fx.store!r}")
        record = dict(fx.payload)
        record["record_id"] = fx.record_id
        store[fx.record_id] = record
    _rebuild_indexes()


_rebuild_indexes()


# ----------------------------------------------------------------------
# Invocation log — capture every tool call so metrics.py can audit
# ----------------------------------------------------------------------

@dataclass
class ToolInvocation:
    tool_name: str
    raw_args: Dict[str, Any]      # exactly what the tool received from the broker / LLM
    result: str                   # serialized result (JSON or "not found")
    record_id: Optional[str]      # parsed from result, if any


@dataclass
class InvocationLog:
    """Mutable shared log threaded through every tool in a single run."""
    invocations: List[ToolInvocation] = field(default_factory=list)

    def record(self, tool_name: str, raw_args: Dict[str, Any], result: str) -> None:
        record_id: Optional[str] = None
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                record_id = parsed.get("record_id")
        except (json.JSONDecodeError, TypeError):
            record_id = None
        self.invocations.append(
            ToolInvocation(
                tool_name=tool_name,
                raw_args=dict(raw_args),
                result=result,
                record_id=record_id,
            )
        )


# ----------------------------------------------------------------------
# Tool factory — fresh @tool objects per run so the broker can mutate them
# in-place without leaking state across configurations
# ----------------------------------------------------------------------

def _lookup_or_not_found(index: Dict[str, Any], key: str) -> str:
    record = index.get(key)
    return json.dumps(record, ensure_ascii=False) if record else "not found"


def make_tools(log: InvocationLog) -> List[BaseTool]:
    """Build a fresh list of @tool-decorated functions bound to *log*.

    A new list per pipeline run is required because the PII tool broker
    mutates tool objects in place; reusing the same list across configurations
    would smuggle a brokered tool into the destructive baseline.
    """

    @tool
    def customer_lookup_by_phone(phone: str) -> str:
        """Look up a customer by phone number. Returns the customer record as JSON, or 'not found'."""
        result = _lookup_or_not_found(_PHONE_INDEX, _normalize_phone(phone))
        log.record("customer_lookup_by_phone", {"phone": phone}, result)
        return result

    @tool
    def customer_lookup_by_email(email: str) -> str:
        """Look up a customer by email address. Returns the customer record as JSON, or 'not found'."""
        result = _lookup_or_not_found(_EMAIL_INDEX, _normalize_email(email))
        log.record("customer_lookup_by_email", {"email": email}, result)
        return result

    @tool
    def appointment_cancel_by_phone(phone: str, reason: str = "customer request") -> str:
        """Cancel a scheduled appointment by the customer's phone number. Returns the cancellation record."""
        result = _lookup_or_not_found(_APPT_PHONE_INDEX, _normalize_phone(phone))
        log.record("appointment_cancel_by_phone", {"phone": phone, "reason": reason}, result)
        return result

    @tool
    def background_check_by_ssn(ssn: str) -> str:
        """Run a background-check refresh for an SSN. Returns the check record."""
        result = _lookup_or_not_found(_SSN_INDEX, _normalize_digits(ssn))
        log.record("background_check_by_ssn", {"ssn": ssn}, result)
        return result

    @tool
    def payment_lookup_by_card(card_number: str) -> str:
        """Look up a payment record by full card number. Returns the payment record."""
        digits = _normalize_digits(card_number)
        result = _lookup_or_not_found(_CARD_FULL_INDEX, digits)
        log.record("payment_lookup_by_card", {"card_number": card_number}, result)
        return result

    @tool
    def failing_payment_lookup(card_number: str) -> str:
        """Payment lookup that raises with the raw card number on miss.

        Models a real upstream that echoes the rejected card in its
        error body. The agent loop catches this in ``pipelines.py`` and
        the raw value lands in the next ToolMessage — exercising the
        tool-exception leakage archetype 3.
        """
        digits = _normalize_digits(card_number)
        record = _CARD_FULL_INDEX.get(digits)
        if record is None:
            log.record(
                "failing_payment_lookup",
                {"card_number": card_number},
                f"error: upstream rejected card={card_number}",
            )
            raise ValueError(f"upstream rejected card={card_number}")
        result = json.dumps(record, ensure_ascii=False)
        log.record("failing_payment_lookup", {"card_number": card_number}, result)
        return result

    @tool
    def invoice_lookup_by_tax_id(tax_id: str) -> str:
        """Look up open invoices for a contractor by tax id (INN). Returns the contractor record."""
        result = _lookup_or_not_found(_TAX_ID_INDEX, _normalize_digits(tax_id))
        log.record("invoice_lookup_by_tax_id", {"tax_id": tax_id}, result)
        return result

    @tool
    def passport_validity_check(series_number: str) -> str:
        """Check the validity of a Russian passport given its series and number. Returns the passport record."""
        result = _lookup_or_not_found(_PASSPORT_INDEX, _normalize_digits(series_number))
        log.record("passport_validity_check", {"series_number": series_number}, result)
        return result

    @tool
    def snils_record_lookup(snils: str) -> str:
        """Look up a SNILS record. Returns the SNILS registration record."""
        result = _lookup_or_not_found(_SNILS_INDEX, _normalize_digits(snils))
        log.record("snils_record_lookup", {"snils": snils}, result)
        return result

    @tool
    def order_lookup_by_reference(order_id: str) -> str:
        """Look up an order by its reference identifier. Returns the order record."""
        result = _lookup_or_not_found(_ORDER_INDEX, _normalize_order_id(order_id))
        log.record("order_lookup_by_reference", {"order_id": order_id}, result)
        return result

    @tool
    def customer_lookup_by_name(name: str) -> str:
        """Look up a customer by full name. Returns the customer record as JSON, or 'not found'."""
        result = _lookup_or_not_found(_NAME_INDEX, (name or "").strip().lower())
        log.record("customer_lookup_by_name", {"name": name}, result)
        return result

    @tool
    def appointment_reschedule_by_phone(phone: str, new_slot: str) -> str:
        """Reschedule a customer's appointment by phone, to the given new slot.

        Returns the updated appointment record as JSON, or 'not found'.
        """
        appt = _APPT_PHONE_INDEX.get(_normalize_phone(phone))
        if appt is None:
            result = "not found"
        else:
            updated = dict(appt)
            updated["slot"] = new_slot
            updated["status"] = "rescheduled"
            result = json.dumps(updated, ensure_ascii=False)
        log.record(
            "appointment_reschedule_by_phone",
            {"phone": phone, "new_slot": new_slot},
            result,
        )
        return result

    @tool
    def verify_national_id(id_type: str, value: str) -> str:
        """Verify a national identifier of the given type. Returns the record JSON.

        ``id_type`` must be one of ``us_ssn``, ``de_steuer_id``, ``ru_snils``,
        ``ru_passport``. ``value`` is the identifier as the user typed it
        (any common formatting accepted; comparison is digits-only).
        """
        index = _NATIONAL_ID_INDEX.get((id_type or "").strip().lower())
        if index is None:
            result = f"error: unsupported id_type {id_type!r}"
        else:
            result = _lookup_or_not_found(index, _normalize_digits(value))
        log.record("verify_national_id", {"id_type": id_type, "value": value}, result)
        return result

    @tool
    def update_ticket(ticket_id: str, fields: Optional[Dict[str, str]] = None) -> str:
        """Update fields on an existing ticket. Returns the updated ticket as JSON.

        ``fields`` is a flat string-to-string mapping of fields to update
        (for example ``{"status": "resolved", "owner": "billing"}``).
        """
        record = _TICKET_INDEX.get(_normalize_order_id(ticket_id))
        if record is None:
            result = "not found"
        else:
            updated = dict(record)
            for k, v in (fields or {}).items():
                updated[k] = v
            result = json.dumps(updated, ensure_ascii=False)
        log.record("update_ticket", {"ticket_id": ticket_id, "fields": dict(fields or {})}, result)
        return result

    @tool
    def update_contact(record_id: str, email: str = "", phone: str = "", address: str = "") -> str:
        """Update the contact email, phone, or address on a customer record.

        Returns the updated customer record as JSON, or 'not found'. This
        is the primary cross-turn-taint sink: scenarios feed in a fresh
        email/phone/address the model picked up from a prior tool result
        (or a user-supplied update), and the argument-fidelity metric
        scores whether the model passed it through verbatim
        (post-normalization).
        """
        record = _CUSTOMERS.get((record_id or "").strip())
        if record is None:
            result = "not found"
        else:
            updated = dict(record)
            if email:
                updated["email"] = email
            if phone:
                updated["phone"] = phone
            if address:
                updated["address"] = address
            result = json.dumps(updated, ensure_ascii=False)
        log.record(
            "update_contact",
            {"record_id": record_id, "email": email, "phone": phone, "address": address},
            result,
        )
        return result

    return [
        customer_lookup_by_phone,
        customer_lookup_by_email,
        customer_lookup_by_name,
        appointment_cancel_by_phone,
        appointment_reschedule_by_phone,
        background_check_by_ssn,
        payment_lookup_by_card,
        failing_payment_lookup,
        invoice_lookup_by_tax_id,
        passport_validity_check,
        snils_record_lookup,
        verify_national_id,
        update_ticket,
        update_contact,
        order_lookup_by_reference,
    ]


SYSTEM_PROMPT = """You are a customer support assistant for a generic CRM. \
You have access to tools for looking up customers (by phone, email, or \
name), invoices (by tax id), appointments (cancel or reschedule by phone), \
payments (by card), national identifiers (verify_national_id with \
id_type ∈ {us_ssn, de_steuer_id, ru_snils, ru_passport}), tickets, \
contact updates, and orders. When a user asks about a customer, order, \
invoice, appointment, payment, identity record, ticket, or contact \
update, call the most appropriate tool.

For national identifiers, prefer ``verify_national_id`` with the right \
``id_type`` over the legacy single-type tools (``background_check_by_ssn``, \
``passport_validity_check``, ``snils_record_lookup``); the legacy tools \
remain available for back-compat.

When the user explicitly references a "failing payment endpoint", "failing \
payment lookup", or asks to test a tool error path, call \
``failing_payment_lookup`` instead of ``payment_lookup_by_card``. This tool \
may raise on rejection — that behaviour is intentional.

CRITICAL: When you call a tool, copy argument values **verbatim** from the \
user message or from prior tool results. Do not parse, edit, normalize, or \
shorten them. If a value looks like a reference token of the form \
``<type>_ref:<prefix>_<hex>`` (for example ``phone_ref:ph_a1b2c3d4`` or \
``email_ref:em_9f8e7d6c``), pass the entire token string including the \
``<type>_ref:`` prefix. Never strip the prefix or pass only the suffix.

After the tool returns, answer the user's question concisely based on the \
tool result.\
"""
