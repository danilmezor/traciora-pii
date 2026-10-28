"""Regression tests for the label-collision regex fixes (Track A + Track D).

Track A: us_phone lookbehind so the 10-digit phone body doesn't match
inside a longer digit run (12-digit INNs were tagging as PHONE on
wolframko).

Track D: Luhn validator on the CARD detector — already wired, but we
need a regression test so an obviously-invalid 16-digit string can't
silently start matching if the validator wiring gets touched later.
"""
from __future__ import annotations

import pytest

from traciora_pii.pii import detect_pii
from traciora_pii.policies.loader import compile_regex_policy, load_regex_policy


@pytest.fixture(scope="module")
def en_detectors():
    return compile_regex_policy(load_regex_policy("en"))


@pytest.fixture(scope="module")
def ru_detectors():
    return compile_regex_policy(load_regex_policy("ru"))


# ----------------------------------------------------------------------
# Track A — us_phone must not match inside a longer digit run
# ----------------------------------------------------------------------


@pytest.mark.parametrize("locale_detectors", ["en_detectors", "ru_detectors"])
def test_us_phone_does_not_match_substring_of_long_digit_run(
    locale_detectors, request,
):
    """Inn-shaped 12-digit string `841054176183` previously matched as
    PHONE because the 10-digit body matched a substring. With the
    (?<!\\d)/(?!\\d) lookbehind+lookahead this should now produce zero
    PHONE findings."""
    detectors = request.getfixturevalue(locale_detectors)
    findings = detect_pii(
        "Налоговый учётный номер 841054176183 указан в реквизитах.",
        detectors=detectors,
    )
    phone_findings = [f for f in findings if f.placeholder == "[PHONE]"]
    assert phone_findings == [], (
        f"PHONE regex must not match inside a long digit run; got: "
        f"{[(f.placeholder, f.start, f.end) for f in phone_findings]}"
    )


def test_us_phone_still_matches_a_real_phone(en_detectors):
    """The lookbehind tightening must not break legitimate phone matches."""
    findings = detect_pii("Call me at +1 (415) 555-2671 today.", detectors=en_detectors)
    phone_findings = [f for f in findings if f.placeholder == "[PHONE]"]
    assert len(phone_findings) == 1
    assert phone_findings[0].placeholder == "[PHONE]"


def test_us_phone_matches_dashed_form(en_detectors):
    findings = detect_pii("Phone: 415-555-2671", detectors=en_detectors)
    phone_findings = [f for f in findings if f.placeholder == "[PHONE]"]
    assert len(phone_findings) == 1


# ----------------------------------------------------------------------
# Track D — Luhn validator filters non-card 16-digit strings
# ----------------------------------------------------------------------


def test_luhn_rejects_non_luhn_sixteen_digit_string(en_detectors):
    """`1234567890123456` is shape-valid (16 digits) but Luhn sum = 49.
    With the validator wired, this must NOT produce a CARD finding."""
    findings = detect_pii(
        "Reference number: 1234567890123456 for your records.",
        detectors=en_detectors,
    )
    card_findings = [f for f in findings if f.placeholder == "[CARD]"]
    assert card_findings == [], (
        f"Luhn validator should suppress non-Luhn 16-digit strings; got: "
        f"{[(f.placeholder, f.start, f.end) for f in card_findings]}"
    )


def test_luhn_accepts_real_card_number(en_detectors):
    """Stripe's test card 4242 4242 4242 4242 is Luhn-valid."""
    findings = detect_pii(
        "Charge card 4242 4242 4242 4242 expiring 12/26.",
        detectors=en_detectors,
    )
    card_findings = [f for f in findings if f.placeholder == "[CARD]"]
    assert len(card_findings) == 1


def test_luhn_rejects_long_internal_id_that_isnt_a_card(en_detectors):
    """A 17-digit numeric run from internal-ID territory should not Luhn-pass."""
    findings = detect_pii(
        "Order ID 12345678901234567 — please reference in correspondence.",
        detectors=en_detectors,
    )
    card_findings = [f for f in findings if f.placeholder == "[CARD]"]
    assert card_findings == []


# ----------------------------------------------------------------------
# VIN regex — ISO 3779 (17 chars, I/O/Q excluded)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("locale_detectors", ["en_detectors", "ru_detectors"])
def test_vin_regex_catches_valid_17_char_vin(locale_detectors, request):
    """A real VIN like ``MVJSR87Z35L93ACF0`` (the one that leaked through
    the §3 smoke) must be caught as ``[VIN]`` across all locales."""
    detectors = request.getfixturevalue(locale_detectors)
    findings = detect_pii(
        "Vehicle registered under VIN MVJSR87Z35L93ACF0 is due for service.",
        detectors=detectors,
    )
    vin_findings = [f for f in findings if f.placeholder == "[VIN]"]
    assert len(vin_findings) == 1
    assert vin_findings[0].placeholder == "[VIN]"


@pytest.mark.parametrize("locale_detectors", ["en_detectors", "ru_detectors"])
def test_vin_regex_rejects_strings_with_excluded_letters(locale_detectors, request):
    """VINs by spec cannot contain I, O, or Q. A 17-char string containing
    these letters must NOT match the VIN regex."""
    detectors = request.getfixturevalue(locale_detectors)
    findings = detect_pii(
        "Order tracking code IOQ7H2K9PNB5M3CXR is your reference.",
        detectors=detectors,
    )
    vin_findings = [f for f in findings if f.placeholder == "[VIN]"]
    assert vin_findings == []


@pytest.mark.parametrize("locale_detectors", ["en_detectors", "ru_detectors"])
def test_vin_regex_rejects_wrong_length(locale_detectors, request):
    """VINs are exactly 17 characters. Anything shorter or longer at a
    word boundary must NOT match."""
    detectors = request.getfixturevalue(locale_detectors)
    # 16 chars (one short)
    findings = detect_pii("Code: MVJSR87Z35L93ACF apply soon.", detectors=detectors)
    assert [f for f in findings if f.placeholder == "[VIN]"] == []
    # 18 chars (one over)
    findings = detect_pii("Code: MVJSR87Z35L93ACF01 apply soon.", detectors=detectors)
    assert [f for f in findings if f.placeholder == "[VIN]"] == []


# ----------------------------------------------------------------------
# Cross-language regex union — typed identifiers caught regardless of
# the surrounding prose language. Exercises the multi-policy loader
# added after the §3 evaluation surfaced cross-language leaks
# (e.g. a DE Steuer-ID embedded inside an English query).
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def multi_detectors():
    from traciora_pii.policies.loader import compile_multi_regex_policy
    return compile_multi_regex_policy(["en", "de", "ru"])


def test_multi_policy_catches_de_steuer_id_in_english_query(multi_detectors):
    """`scn-sec-742-en` payload: a DE Steuer-ID inside English prose."""
    text = "Your tax ID 43 876 501 915 was processed. Steuer-ID confirmed."
    findings = detect_pii(text, detectors=multi_detectors)
    placeholders = [f.placeholder for f in findings]
    assert "[DE_STEUER_ID]" in placeholders, f"got {placeholders!r}"


def test_multi_policy_catches_us_ssn_in_german_query(multi_detectors):
    text = "Bitte SSN 123-45-6789 angeben."
    findings = detect_pii(text, detectors=multi_detectors)
    placeholders = [f.placeholder for f in findings]
    assert "[PERSONAL_ID]" in placeholders, f"got {placeholders!r}"


def test_multi_policy_catches_ru_snils_in_english_query(multi_detectors):
    text = "Provide СНИЛС 123-456-789 01 if available."
    findings = detect_pii(text, detectors=multi_detectors)
    placeholders = [f.placeholder for f in findings]
    assert "[RU_SNILS]" in placeholders, f"got {placeholders!r}"


# ----------------------------------------------------------------------
# License plate regex coverage — six US shapes + three DE shapes, all
# keyword-anchored. Added after the §3 evaluation surfaced an unrecognized
# Utah-style plate (AKD7052 UT) as the dominant residual leak archetype.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Original §3 leak — Utah suffix-state
        "The vehicle pickup uses plate AKD7052 UT.",
        # Six EN shapes the generator emits
        "Security should verify license plate AWT1292 OH before release.",
        "Security should verify license plate AZ PER4295 before release.",
        "Security should verify license plate AZ plate NDT5952 before release.",
        "Security should verify license plate AZ tag LXA 5627 before release.",
        "Security should verify license plate CA-ZAT3588 before release.",
        "Security should verify license plate CA TEC-6935 before release.",
    ],
)
def test_us_plate_detected_under_plate_keyword(multi_detectors, text):
    findings = detect_pii(text, detectors=multi_detectors)
    placeholders = [f.placeholder for f in findings]
    assert "[LICENSE_PLATE]" in placeholders, f"got {placeholders!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Die Sicherheitsstelle soll das Kennzeichen BI-DS 4953 prüfen.",
        "Die Sicherheitsstelle soll das Kennzeichen BO RR2337 prüfen.",
        "Der Auftrag ist mit Kennzeichen BI-PQ-3441 verknüpft.",
    ],
)
def test_de_plate_detected_under_kennzeichen_keyword(multi_detectors, text):
    findings = detect_pii(text, detectors=multi_detectors)
    placeholders = [f.placeholder for f in findings]
    assert "[LICENSE_PLATE]" in placeholders, f"got {placeholders!r}"


def test_plate_regex_keyword_anchored_no_false_positive(multi_detectors):
    """An `ABC1234` shape with no plate/tag keyword nearby must NOT match."""
    findings = detect_pii(
        "The reference code ABC1234 should be archived.",
        detectors=multi_detectors,
    )
    assert "[LICENSE_PLATE]" not in [f.placeholder for f in findings]


# ----------------------------------------------------------------------
# DATE — dates of birth only: a 1900–2015 year on word boundaries
# ----------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["en", "de", "ru"])
@pytest.mark.parametrize("value", ["1985-03-12", "12.03.1985", "03/06/1951"])
def test_date_of_birth_is_masked(policy, value):
    text = f"Born {value}, please verify."
    findings = detect_pii(text, detectors=compile_regex_policy(load_regex_policy(policy)))
    dates = [text[f.start:f.end] for f in findings if f.placeholder == "[DATE]"]
    assert dates == [value]


@pytest.mark.parametrize("policy", ["en", "de", "ru"])
@pytest.mark.parametrize(
    "text",
    [
        "Move the appointment to 2026-08-20 14:00.",
        "Build 5.2.0.1234 was deployed.",
        "Version 2.0.1234 is live.",
        "Order ORD-2026-0815 shipped.",
        "Invoice dated 31.12.2024 is attached.",
        "Ticket 1.2.1985.4 is open.",
    ],
)
def test_non_birth_dates_are_not_masked(policy, text):
    findings = detect_pii(text, detectors=compile_regex_policy(load_regex_policy(policy)))
    assert [text[f.start:f.end] for f in findings if f.placeholder == "[DATE]"] == []
