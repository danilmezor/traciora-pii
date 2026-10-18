"""Locale-specific PII helpers, driven by SeedBundle pools.

Algorithms (checksum math, format rules) live in this module. *Pools*
(name lists, city/street tables, plate letters, OSAGO series, ...) live
in ``seeds/<locale>/*.yaml`` and reach the helpers via ``SeedBundle``.

Each helper class (``RuPII``, ``DePII``, ``EnPII``) takes a ``SeedBundle``
and a ``random.Random`` and exposes methods that mirror the placeholder
slot names: ``person_name()``, ``inn_individual()``, ``us_ssn()``, etc.

Why hand-rolled rather than just Faker:

* Faker's ``ru_RU.inn``, ``ru_RU.snils`` etc. exist but checksum correctness
  varies by Faker version; we need guaranteed-valid IDs for honest evaluation.
* Steuer-ID and Personalausweis aren't in Faker's de_DE provider at all.
* Carrying our own implementation keeps the benchmark independent of Faker
  upgrades that could silently change the output distribution.

References for the checksum algorithms remain at function scope so they
travel with the code.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .seeds import SeedBundle


# ======================================================================
# Shared helpers (locale-neutral generators)
# ======================================================================


_BODY_CHARSETS = {
    "alphanum_mixed": string.ascii_letters + string.digits,
    "upper_alphanum": string.ascii_uppercase + string.digits,
    "digits": string.digits,
}


def _random_secret(seeds: SeedBundle, rng: random.Random) -> str:
    """Synthetic secret token in one of the shapes declared in ``seeds.shared.secrets``.

    Adding a new shape is a YAML edit (add an entry to
    ``seeds/shared/secrets.yaml``); no code change required as long as the
    shape uses one of the registered charsets.
    """
    shapes = seeds.shared.secrets.shapes
    shape = rng.choice(shapes)
    if shape.parts:
        # Compound shape (e.g. xoxb-NNNNNNNNNN-NNNNNNNNNNNN-XXXX...)
        slot_values = {}
        for slot, spec in shape.parts.items():
            charset = _BODY_CHARSETS[spec["charset"]]
            slot_values[slot] = "".join(rng.choices(charset, k=spec["length"]))
        return shape.pattern.format(**slot_values)
    # Simple shape: one body fills {body}.
    if shape.body_charset is None or shape.body_length is None:
        raise ValueError(f"secret shape {shape.name!r} missing body_charset/length")
    body = "".join(rng.choices(_BODY_CHARSETS[shape.body_charset], k=shape.body_length))
    return shape.pattern.format(body=body)


def _random_url(seeds: SeedBundle, rng: random.Random) -> str:
    """Synthetic URL on a documentation-reserved domain."""
    paths = ("api/v1/users", "billing/invoice", "search", "docs/getting-started",
             "files/report.pdf", "track/order", "reset-password", "profile/settings")
    domain = rng.choice(seeds.shared.domains.reserved)
    return f"https://{domain}/{rng.choice(paths)}"


def _random_ip(seeds: SeedBundle, rng: random.Random) -> str:
    """IPv4 in a TEST-NET block (RFC 5737)."""
    block = rng.choice(seeds.shared.ip_blocks.test_nets)
    lo, hi = seeds.shared.ip_blocks.host_octet_range
    return f"{block.prefix}.{rng.randint(lo, hi)}"


def _format_person_name_variant(given: str, surname: str, rng: random.Random) -> str:
    """Return a two-part person-name variant.

    Variants cover full given/surname order flips plus the natural
    given-then-initial form: ``Name Surname``, ``Surname Name``,
    ``Name S.``. The reverse ``S. Name`` was dropped because it reads
    as inverted in EN/DE/RU CRM contexts.
    """
    surname_initial = surname[0].upper()
    return rng.choice((
        f"{given} {surname}",
        f"{surname} {given}",
        f"{given} {surname_initial}.",
    ))


def _luhn_check_digit(digits: List[int]) -> int:
    """Luhn check digit, appended to ``digits`` to form a Luhn-valid number.

    Verification walks the FINAL number from the right doubling every second
    digit starting with the second-from-right. Here ``digits`` is the
    leading body without the check; the LAST body digit sits at position
    2-from-right (doubled), so we double at even ``i`` of ``reversed(digits)``.
    """
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            doubled = d * 2
            s += doubled if doubled < 10 else doubled - 9
        else:
            s += d
    return (10 - (s % 10)) % 10


def _random_card(rng: random.Random) -> str:
    """16-digit PAN with valid Luhn checksum.

    Uses Visa's documentation BIN ``4000 0000`` so the value never collides
    with a live BIN range. ISO/IEC 7812-1 layout.
    """
    head = [4, 0, 0, 0] + [rng.randint(0, 9) for _ in range(11)]
    head.append(_luhn_check_digit(head))
    s = "".join(str(d) for d in head)
    return f"{s[:4]} {s[4:8]} {s[8:12]} {s[12:]}"


def _iban_with_country(rng: random.Random, country: str, bban_len: int) -> str:
    """Generic mod-97 IBAN builder for any country/BBAN-length pair."""
    bban = "".join(str(rng.randint(0, 9)) for _ in range(bban_len))
    c1 = ord(country[0]) - 55
    c2 = ord(country[1]) - 55
    numeric = bban + str(c1) + str(c2) + "00"
    check = 98 - (int(numeric) % 97)
    return f"{country}{check:02d}{bban}"


# ISO 7064 MOD 11,10 (Steuer-ID, Personalausweis check digit). Kept module-
# level so DE helpers can call it directly.
def _mod_11_10_check(digits: List[int]) -> int:
    product = 10
    for d in digits:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (s * 2) % 11
    return (11 - product) % 10


# ======================================================================
# Locale-neutral base
# ======================================================================


@dataclass(frozen=True)
class LocalePII:
    """Shared locale-agnostic PII surface; subclassed per locale.

    Methods here produce values whose shape is independent of locale
    (cards, IPs, generic IBANs, VINs, secret tokens, URLs) or whose
    locale variation already lives entirely in the seed pools
    (``street_address`` reads city / street / format pools from the
    locale's ``SeedBundle``). Locale subclasses add ID / phone / email
    / person-name methods specific to their jurisdiction and may
    override an inherited method when the locale convention differs
    (``DePII.iban`` is country-fixed; see below).
    """

    seeds: SeedBundle
    rng: random.Random

    # ---------- Generic / shared ----------

    def secret_token(self) -> str:
        return _random_secret(self.seeds, self.rng)

    def url(self) -> str:
        return _random_url(self.seeds, self.rng)

    def card(self) -> str:
        return _random_card(self.rng)

    def ip(self) -> str:
        return _random_ip(self.seeds, self.rng)

    def vin(self) -> str:
        return _random_vin(self.rng)

    def iban(self) -> str:
        # International IBAN — mod-97 with a varied country mix.
        country, length = self.rng.choice([("GB", 18), ("FR", 23), ("ES", 20), ("DE", 20)])
        return _iban_with_country(self.rng, country, length)

    def street_address(self) -> str:
        addr = self.seeds.addresses
        cs = self.rng.choice(addr.city_streets)
        city = cs.city
        street = self.rng.choice(cs.streets)
        house = self.rng.randint(*addr.house_range)
        apt = self.rng.randint(*addr.apt_range)
        korp = self.rng.randint(*addr.korp_range)
        fmt = self.rng.choice(addr.formats)
        return fmt.format(city=city, street=street, house=house, apt=apt, korp=korp)


# ======================================================================
# Russian
# ======================================================================


_INN_WEIGHTS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_12_A = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_12_B = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_SNILS_WEIGHTS = (9, 8, 7, 6, 5, 4, 3, 2, 1)

# VIN: ISO 3779 / GOST R 51980-2008. Allowed alphabet excludes I, O, Q.
_VIN_ALPHABET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
_VIN_TRANSLIT = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
for _d in range(10):
    _VIN_TRANSLIT[str(_d)] = _d
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def _random_vin(rng: random.Random) -> str:
    """ISO 3779 VIN with valid check digit."""
    chars = [rng.choice(_VIN_ALPHABET) for _ in range(17)]
    s = sum(_VIN_TRANSLIT[c] * w for c, w in zip(chars, _VIN_WEIGHTS))
    check = s % 11
    chars[8] = "X" if check == 10 else str(check)
    return "".join(chars)


class RuPII(LocalePII):
    """Structured-PII helper for the ru_RU locale, seed-driven."""

    # ---------- IDs ----------

    def inn_individual(self) -> str:
        digits = [self.rng.randint(0, 9) for _ in range(10)]
        c1 = sum(d * w for d, w in zip(digits, _INN_WEIGHTS_12_A)) % 11 % 10
        digits.append(c1)
        c2 = sum(d * w for d, w in zip(digits, _INN_WEIGHTS_12_B)) % 11 % 10
        digits.append(c2)
        return "".join(str(d) for d in digits)

    def inn_legal(self) -> str:
        digits = [self.rng.randint(0, 9) for _ in range(9)]
        c = sum(d * w for d, w in zip(digits, _INN_WEIGHTS_10)) % 11 % 10
        digits.append(c)
        return "".join(str(d) for d in digits)

    def inn_any(self) -> str:
        """Mix of legal-entity (10-digit) and individual (12-digit) INNs.

        Even mix because both forms appear in real production traffic at
        comparable rates depending on the workload.
        """
        return self.inn_legal() if self.rng.random() < 0.5 else self.inn_individual()

    def snils(self) -> str:
        n = self.rng.randint(1_001_999, 999_999_999)
        digits = [int(d) for d in f"{n:09d}"]
        s = sum(d * w for d, w in zip(digits, _SNILS_WEIGHTS))
        if s < 100:
            check = s
        elif s in (100, 101):
            check = 0
        else:
            r = s % 101
            check = 0 if r in (100, 101) else r
        return f"{digits[0]}{digits[1]}{digits[2]}-{digits[3]}{digits[4]}{digits[5]}-{digits[6]}{digits[7]}{digits[8]} {check:02d}"

    def passport(self) -> str:
        region = self.rng.randint(1, 89)
        year = self.rng.randint(95, 99) if self.rng.random() < 0.2 else self.rng.randint(0, 24)
        number = self.rng.randint(100000, 999999)
        return f"{region:02d} {year:02d} {number}"

    def ogrn(self) -> str:
        head = "".join(str(self.rng.randint(0, 9)) for _ in range(12))
        check = int(head) % 11 % 10
        return head + str(check)

    def kpp(self) -> str:
        region = self.rng.randint(1, 89)
        office = self.rng.randint(1, 99)
        reason = self.rng.choice(["01", "02", "03", "43", "44", "45"])
        serial = self.rng.randint(1, 999)
        return f"{region:02d}{office:02d}{reason}{serial:03d}"

    def bank_account(self) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(20))

    def license_plate(self) -> str:
        ids = self.seeds.identifiers
        plate_letters = ids.model_extra["plate_letters"]
        plate_regions = ids.model_extra["plate_regions"]
        rus_p = ids.model_extra.get("plate_rus_suffix_probability", 0.5)
        l1 = self.rng.choice(plate_letters)
        l2l3 = "".join(self.rng.choices(plate_letters, k=2))
        digits = self.rng.randint(0, 999)
        region = self.rng.choice(plate_regions)
        suffix = "RUS" if self.rng.random() < rus_p else ""
        return f"{l1}{digits:03d}{l2l3}{region}{suffix}"

    def vin(self) -> str:
        return _random_vin(self.rng)

    def drv_license(self) -> str:
        region = self.rng.randint(1, 89)
        year = self.rng.randint(15, 24)
        serial = self.rng.randint(100000, 999999)
        return f"{region:02d} {year:02d} {serial}"

    def sts(self) -> str:
        plate_letters = self.seeds.identifiers.model_extra["plate_letters"]
        head = "".join(str(self.rng.randint(0, 9)) for _ in range(4))
        tail = self.rng.choices(string.digits + plate_letters, k=6)
        return f"{head} {''.join(tail)}"

    def osago(self) -> str:
        series = self.rng.choice(self.seeds.identifiers.model_extra["osago_series"])
        number = self.rng.randint(0, 9_999_999_999)
        return f"{series} {number:010d}"

    # ---------- Contacts ----------

    def phone(self) -> str:
        ids = self.seeds.identifiers
        op = self.rng.choice(ids.model_extra["mobile_prefixes"])
        rest = self.rng.randint(0, 9_999_999)
        s = f"{rest:07d}"
        fmt = self.rng.choice(ids.model_extra.get("phone_formats",
                                                   ["+7 {op} {a}-{b}-{c}"]))
        return fmt.format(op=op, a=s[:3], b=s[3:5], c=s[5:])

    def email(self) -> str:
        # Latin transliterations of Russian first/last names; emails in
        # production traffic are almost always ASCII even for RU subjects.
        firsts = ("alexey", "ivan", "marina", "olga", "dmitry", "ekaterina",
                  "pavel", "yulia", "ruslan", "tatiana", "sergey", "anna")
        lasts = ("sokolov", "ivanov", "petrov", "smirnov", "kuznetsov",
                 "popov", "lebedev", "kozlov", "novikov", "morozov")
        domains = list(self.seeds.shared.domains.reserved) + ["example.ru"]
        u = f"{self.rng.choice(firsts)}.{self.rng.choice(lasts)}{self.rng.randint(0, 99)}"
        return f"{u}@{self.rng.choice(domains)}"

    # ---------- Names + addresses ----------

    def person_name(self) -> str:
        n = self.seeds.names
        if self.rng.random() < 0.5 and n.first_names_male:
            given = self.rng.choice(n.first_names_male)
            surname = self.rng.choice(n.last_names)
        else:
            given = self.rng.choice(n.first_names_female)
            surname = self.rng.choice(n.last_names)
            for inflection in n.feminine_inflections:
                if surname.endswith(inflection.suffix):
                    surname = surname[:-len(inflection.suffix)] + inflection.becomes
                    break
        return _format_person_name_variant(given, surname, self.rng)

    # ---------- Locale-conventional date ----------

    def date(self) -> str:
        year = self.rng.randint(1940, 2005)
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        yy = year % 100
        # Locale-conventional day-first ordering with format variation.
        choices = [
            (35, f"{day:02d}.{month:02d}.{year}"),
            (20, f"{day:02d}/{month:02d}/{year}"),
            (10, f"{day:02d}.{month:02d}.{yy:02d}"),
            (10, f"{day:02d}/{month:02d}/{yy:02d}"),
            (15, f"{year}-{month:02d}-{day:02d}"),
            (10, f"{day:02d}-{month:02d}-{year}"),
        ]
        return self.rng.choices(
            [c[1] for c in choices], weights=[c[0] for c in choices], k=1
        )[0]


# ======================================================================
# German
# ======================================================================


class DePII(LocalePII):
    """Structured-PII helper for the de_DE locale, seed-driven."""

    # ---------- IDs ----------

    def steuer_id(self) -> str:
        """German Steuer-ID with ISO 7064 MOD 11,10 check digit.

        The first ten digits follow the public IdNr distribution rule:
        one digit appears twice, one digit does not appear, and all other
        digits appear once.
        """
        omit = self.rng.randint(0, 9)
        dup_candidates = [d for d in range(10) if d != omit]
        duplicate = self.rng.choice(dup_candidates)
        body = [d for d in range(10) if d != omit] + [duplicate]
        self.rng.shuffle(body)
        check = _mod_11_10_check(body)
        raw = "".join(str(d) for d in body + [check])
        fmt = self.rng.choice(self.seeds.identifiers.model_extra.get(
            "steuer_id_formats",
            ["{a}{b}{c}{d}"],
        ))
        return fmt.format(a=raw[:2], b=raw[2:5], c=raw[5:8], d=raw[8:])

    def personalausweis(self) -> str:
        ids = self.seeds.identifiers
        first_letters = ids.model_extra.get(
            "personalausweis_first_letters",
            "CFGHJKLMNPRTVWXYZ",
        )
        alphabet = ids.model_extra.get(
            "personalausweis_alphabet",
            string.ascii_uppercase + string.digits,
        )
        return self.rng.choice(first_letters) + "".join(
            self.rng.choices(alphabet, k=9),
        )

    def license_plate(self) -> str:
        ids = self.seeds.identifiers
        city_code = self.rng.choice(ids.model_extra.get("plate_city_codes", ["B", "M", "K"]))
        letters = "".join(self.rng.choices(string.ascii_uppercase, k=self.rng.randint(1, 2)))
        digits = self.rng.randint(1, 9999)
        fmt = self.rng.choice(ids.model_extra.get(
            "plate_formats",
            ["{city_code}-{letters} {digits}"],
        ))
        return fmt.format(city_code=city_code, letters=letters, digits=digits)

    def vin(self) -> str:
        return _random_vin(self.rng)

    # ---------- Contacts ----------

    def phone(self) -> str:
        ids = self.seeds.identifiers
        area = str(self.rng.choice(ids.model_extra.get("phone_area_codes", ["30", "40", "89"])))
        subscriber = "".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.randint(6, 8)))
        fmt = self.rng.choice(ids.model_extra.get(
            "phone_formats",
            ["+49 {area} {subscriber}", "0{area} {subscriber}"],
        ))
        return fmt.format(area=area, subscriber=subscriber)

    def email(self) -> str:
        n = self.seeds.names
        first = self._email_part(self.rng.choice(n.first_names_male + n.first_names_female))
        last = self._email_part(self.rng.choice(n.last_names))
        domains = list(self.seeds.shared.domains.reserved) + ["example.de"]
        separator = self.rng.choice([".", "-", ""])
        suffix = self.rng.randint(0, 99)
        return f"{first}{separator}{last}{suffix}@{self.rng.choice(domains)}"

    # ---------- Names + addresses ----------

    def person_name(self) -> str:
        n = self.seeds.names
        given = self.rng.choice(n.first_names_male + n.first_names_female)
        surname = self.rng.choice(n.last_names)
        return _format_person_name_variant(given, surname, self.rng)

    def plz_city(self) -> str:
        addr = self.seeds.addresses
        entries = addr.plz_city_streets or []
        if not entries:
            city = self.rng.choice(addr.city_streets).city
            return f"{self.rng.randint(10000, 99999)} {city}"
        entry = self.rng.choice(entries)
        return f"{entry['plz']} {entry['city']}"

    # ---------- Locale overrides ----------

    def iban(self) -> str:
        return _iban_with_country(self.rng, "DE", 18)

    def date(self) -> str:
        year = self.rng.randint(1940, 2005)
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        choices = [
            (40, f"{day:02d}.{month:02d}.{year}"),
            (15, f"{day:02d}/{month:02d}/{year}"),
            (20, f"{year}-{month:02d}-{day:02d}"),
            (10, f"{day:02d}-{month:02d}-{year}"),
        ]
        return self.rng.choices(
            [c[1] for c in choices], weights=[c[0] for c in choices], k=1
        )[0]

    def passport(self) -> str:
        # German Reisepass: 1 letter from a restricted alphabet + 9 alphanumeric.
        # Same character class as Personalausweis; disambiguation in the regex
        # policy is by keyword anchor (reisepass / passport / passnummer).
        head_alpha = "CFGHJKLMNPRTVWXYZ"
        tail_alpha = string.ascii_uppercase + string.digits
        return self.rng.choice(head_alpha) + "".join(
            self.rng.choices(tail_alpha, k=9)
        )

    @staticmethod
    def _email_part(value: str) -> str:
        table = str.maketrans({
            "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
            "Ä": "ae", "Ö": "oe", "Ü": "ue",
        })
        return "".join(
            ch for ch in value.translate(table).lower()
            if ch in string.ascii_lowercase
        )


# ======================================================================
# English / US-centric
# ======================================================================

class EnPII(LocalePII):
    """Structured-PII helper for the en_US locale, seed-driven."""

    # ---------- IDs ----------

    def us_ssn(self) -> str:
        ids = self.seeds.identifiers
        areas = ids.model_extra.get("ssn_area_numbers")
        area = int(self.rng.choice(areas)) if areas else self.rng.randint(1, 665)
        group = self.rng.randint(1, 99)
        serial = self.rng.randint(1, 9999)
        # Emit both the canonical XXX-XX-XXXX form and the dashless 9-digit form
        # roughly evenly. The dashless variant matches the format ai4privacy
        # uses for SOCIALNUM and is detected by the keyword-anchored
        # us_ssn_dashless pattern in the en/ru regex policies.
        if self.rng.random() < 0.5:
            return f"{area:03d}-{group:02d}-{serial:04d}"
        return f"{area:03d}{group:02d}{serial:04d}"

    def license_plate(self) -> str:
        ids = self.seeds.identifiers
        states = ids.model_extra.get("plate_states", ["CA", "NY", "TX", "FL"])
        letters = ids.model_extra.get("plate_letters", string.ascii_uppercase)
        serial_letters = "".join(self.rng.choices(letters, k=3))
        serial_digits = self.rng.randint(1000, 9999)
        serial = f"{serial_letters}{serial_digits}"
        fmt = self.rng.choice(ids.model_extra.get("plate_formats", ["{state} {serial}"]))
        return fmt.format(
            state=self.rng.choice(states),
            serial=serial,
            letters=serial_letters,
            digits=f"{serial_digits:04d}",
        )

    def vin(self) -> str:
        return _random_vin(self.rng)

    # ---------- Contacts ----------

    def phone(self) -> str:
        ids = self.seeds.identifiers
        area_codes = ids.model_extra.get("phone_area_codes", [212, 213, 312, 415])
        area = int(self.rng.choice(area_codes))
        exchange = self.rng.randint(200, 999)
        line = self.rng.randint(0, 9999)
        fmt = self.rng.choice(ids.model_extra.get(
            "phone_formats",
            ["({area}) {exchange}-{line}"],
        ))
        return fmt.format(area=area, exchange=exchange, line=f"{line:04d}")

    def email(self) -> str:
        n = self.seeds.names
        first = self.rng.choice(n.first_names_male + n.first_names_female).lower()
        last = self.rng.choice(n.last_names).lower()
        domains = list(self.seeds.shared.domains.reserved) + ["example.com"]
        separator = self.rng.choice([".", "_", ""])
        suffix = self.rng.randint(0, 99)
        return f"{first}{separator}{last}{suffix}@{self.rng.choice(domains)}"

    # ---------- Names + addresses ----------

    def person_name(self) -> str:
        n = self.seeds.names
        given = self.rng.choice(n.first_names_male + n.first_names_female)
        surname = self.rng.choice(n.last_names)
        return _format_person_name_variant(given, surname, self.rng)

    # ---------- Locale-conventional date ----------

    def date(self) -> str:
        year = self.rng.randint(1940, 2005)
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        yy = year % 100
        # US convention: month-first.
        choices = [
            (30, f"{month:02d}/{day:02d}/{year}"),
            (20, f"{month:02d}.{day:02d}.{year}"),
            (10, f"{month:02d}/{day:02d}/{yy:02d}"),
            (5, f"{month:02d}.{day:02d}.{yy:02d}"),
            (25, f"{year}-{month:02d}-{day:02d}"),
            (10, f"{month:02d}-{day:02d}-{year}"),
        ]
        return self.rng.choices(
            [c[1] for c in choices], weights=[c[0] for c in choices], k=1
        )[0]

    def passport(self) -> str:
        # US passport: 9 digits. Optional letter-prefix variant for older books.
        if self.rng.random() < 0.15:
            return self.rng.choice(string.ascii_uppercase) + "".join(
                str(self.rng.randint(0, 9)) for _ in range(8)
            )
        return "".join(str(self.rng.randint(0, 9)) for _ in range(9))
