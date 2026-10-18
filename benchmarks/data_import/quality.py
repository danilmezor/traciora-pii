"""Per-label quality scanner for imported PII datasets.

Goal: surface mislabeling rates that would cap eval recall regardless of
detector quality. Five categories of check:

1. **Anchor presence.** Does the surrounding text (±40 chars) contain a
   keyword that disambiguates the labeled type? E.g., a `PASSPORTNUM`
   span with no `passport` / `Reisepass` / `паспорт` nearby is suspect.
2. **Format compliance.** Where the entity type has a strict format
   (SSN dashed/dashless, СНИЛС, ИНН, Luhn-validated card, ISO date),
   does the value satisfy it?
3. **Junk values.** Empty, single-character, or all-punctuation values.
4. **Label leakage.** The exact same string labeled with two different
   types across the dataset.
5. **Format collisions.** Two different labels producing values of the
   same shape in the same row (forces context-only disambiguation).

Scope: read-only, single pass per dataset, no transforms applied.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from .loaders import NormalizedExample, Span


# Anchor windows: ±N characters of context around the span we treat as
# "near." 40 chars matches what the framework's keyword-anchored regexes
# use, and is what wolframko's tightest contexts (single-sentence rows)
# typically span.
ANCHOR_RADIUS = 40

# A value is "junk" if it is shorter than this OR has no alphanumerics.
MIN_VALUE_LEN = 2


# --- Anchor patterns ---------------------------------------------------
# Each entry: (label, locale) -> compiled regex. Matched case-insensitive.
# An empty / missing entry means "no anchor expected for this label" (e.g.,
# names are inherently contextual; no keyword anchors them).

_AI4PRIVACY_EN_ANCHORS: Dict[str, re.Pattern] = {
    "TITLE":            re.compile(r"\b(mr|mrs|ms|dr|prof|master|mister|miss|sir|madam|honorable)\b\.?", re.I),
    "AGE":              re.compile(r"\b(age|aged|years?\s*old|year[\s-]old|y\.?o\.?)\b", re.I),
    "GENDER":           re.compile(r"\b(gender|identif(?:ies|y)\s+as)\b", re.I),
    "SEX":              re.compile(r"\b(sex|biological\s+sex|male|female|nonbinary|non-binary)\b", re.I),
    "TELEPHONENUM":     re.compile(r"\b(phone|tel(?:ephone)?|call|mobile|cell|contact|number|reach)\b\.?", re.I),
    "STREET":           re.compile(r"\b(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|circle|cir|place|pl|highway|hwy|address|residing|residence)\b\.?", re.I),
    "CITY":             re.compile(r"\b(city|town|township|village|locality|municipality)\b", re.I),
    "ZIPCODE":          re.compile(r"\b(zip|postal|postcode|zip code|post code)\b", re.I),
    "BUILDINGNUM":      None,  # contextual on STREET adjacency
    "DATE":             re.compile(r"\b(date|on|day|year|month|born|d\.?o\.?b\.?|birthday|expire|effective|valid)\b", re.I),
    "PASSPORTNUM":      re.compile(r"\b(passport|travel\s*document|passport\s*no|passport\s*number|passport-no)\b\.?", re.I),
    "DRIVERLICENSENUM": re.compile(r"\b(driver|driving|license|licence|dl|chauffeur)\b", re.I),
    "IDCARDNUM":        re.compile(r"\b(id|identification|identity|id\s*card|id\s*number|national\s*id)\b\.?", re.I),
    "SOCIALNUM":        re.compile(r"\b(ssn|social\s*security|social-security|social-security-number|sosec)\b", re.I),
    "TAXNUM":           re.compile(r"\b(tax|tax\s*id|tax\s*number|ein|itin|tin|taxpayer|tax\s*reference|tax-ref)\b\.?", re.I),
    "EMAIL":            None,   # format check is sufficient
    "CREDITCARDNUMBER": None,   # Luhn check is sufficient
    "GIVENNAME":        None,   # inherently contextual
    "SURNAME":          None,
}

_AI4PRIVACY_DE_ANCHORS: Dict[str, re.Pattern] = {
    "TITLE":            re.compile(r"\b(herr|frau|dr|prof|sehr\s+geehrte|fräulein)\b\.?", re.I),
    "AGE":              re.compile(r"\b(alt|jahre|jahre\s*alt|jahrgang|geburtsjahr)\b", re.I),
    "GENDER":           re.compile(r"\b(geschlecht|gender)\b", re.I),
    "SEX":              re.compile(r"\b(geschlecht|männlich|weiblich|divers|sex)\b", re.I),
    "TELEPHONENUM":     re.compile(r"\b(telefon|mobil|anruf|kontakt|tel|nummer|rufnummer|fax|festnetz)\b\.?", re.I),
    "STREET":           re.compile(r"(?:\b(straße|strasse|str|weg|allee|platz|gasse|ufer|ring|chaussee|adresse|wohnhaft|residiert)\b\.?|straße)", re.I),
    "CITY":             re.compile(r"\b(stadt|ort|gemeinde|dorf|kreis|bezirk)\b", re.I),
    "ZIPCODE":          re.compile(r"\b(plz|postleitzahl|postal\s*code)\b", re.I),
    "BUILDINGNUM":      None,
    "DATE":             re.compile(r"\b(datum|am|tag|jahr|monat|geboren|geburtstag|stichtag|wirksam|gültig)\b", re.I),
    "PASSPORTNUM":      re.compile(r"\b(reisepass|passnummer|reisepass-nr|reisedokument|pass-nr|pass\s*nr|passport)\b\.?", re.I),
    "DRIVERLICENSENUM": re.compile(r"\b(führerschein|fahrerlaubnis|führerscheinnummer|fahrerlaubnisnummer)\b\.?", re.I),
    "IDCARDNUM":        re.compile(r"\b(personalausweis|ausweisnummer|ausweis\s*nr|ausweis-nr|perso|ausweis|identifikationskarte)\b\.?", re.I),
    "SOCIALNUM":        re.compile(r"\b(sozialversicherung|sv-nr|sozialversicherungsnummer|sv\s*nr|rentenversicherung)\b\.?", re.I),
    "TAXNUM":           re.compile(r"\b(steuer|steuer-id|idnr|steuerliche\s*identifikationsnummer|steuernummer|steuer\s*nr|finanzamt)\b\.?", re.I),
    "EMAIL":            None,
    "CREDITCARDNUMBER": None,
    "GIVENNAME":        None,
    "SURNAME":          None,
}

_WOLFRAMKO_RU_ANCHORS: Dict[str, re.Pattern] = {
    "TELEPHONENUM":     re.compile(r"(?:телефон|тел\.|мобильн|номер\s+телефона|связь|контактн|звон|перезвон)", re.I),
    "EMAIL":            None,
    "CREDITCARDNUMBER": None,
    "STREET":           re.compile(r"(?:улиц|ул\.|проспект|пр-?т|пр\.|шоссе|бульвар|переулок|пер\.|набережн|набр|наб\.|площад|пл\.|тупик|проезд|переулок)", re.I),
    "CITY":             re.compile(r"(?:город|г\.|посёл|пос\.|село|с\.|деревн|д\.|станиц|ст\.|край|область|обл\.)", re.I),
    "BUILDINGNUM":      None,
    "ZIPCODE":          re.compile(r"(?:индекс|почтовый\s*индекс|zip)", re.I),
    "DATEOFBIRTH":      re.compile(r"(?:дата\s+рождения|род(?:ил|ился|илась)|г\.р\.|год\s+рождения|день\s+рождения|born)", re.I),
    "IDCARDNUM":        re.compile(r"(?:паспорт|серия|выдан|удостоверени(?:е|я)\s+личности)", re.I),
    "DRIVERLICENSENUM": re.compile(r"(?:водительск(?:ое|ий)|водительск\w*\s+(?:удостоверени|документ|прав)|вод\.\s*удост|\bв[оу]\b\.?\s*\w*|\bправ[аы]\b|категори[яи]\s*[abcdABCD])", re.I),
    "SOCIALNUM":        re.compile(r"(?:снилс|страховой\s+номер|пенсионн)", re.I),
    "TAXNUM":           re.compile(r"(?:инн|налогов|налоговый\s+номер)", re.I),
    "ACCOUNTNUM":       re.compile(r"(?:счёт|счет|р/с|расчётный|банковск\w*\s+счёт|номер\s+счета|iban)", re.I),
    "USERNAME":         re.compile(r"(?:логин|никнейм|ник|пользовател|аккаунт|account|username)", re.I),
    "PASSWORD":         re.compile(r"(?:пароль|password|временный\s+пароль|временн\w*\s+парол)", re.I),
    "GIVENNAME":        None,
    "SURNAME":          None,
}

_NEMOTRON_EN_ANCHORS: Dict[str, re.Pattern] = {
    "first_name":             None,
    "last_name":              None,
    "email":                  None,
    "credit_debit_card":      None,
    "date":                   re.compile(r"\b(date|on|as of|effective|expires?|expiration|issued|signed)\b", re.I),
    "date_of_birth":          re.compile(r"\b(date of birth|birthdate|born|dob|d\.?o\.?b\.?)\b", re.I),
    "phone_number":           re.compile(r"\b(phone|telephone|mobile|cell|call|contact|reached|callback)\b", re.I),
    "ssn":                    re.compile(r"\b(ssn|social\s*security|social-security-number|social security number)\b", re.I),
    "tax_id":                 re.compile(r"\b(tax|taxpayer|tin|ein|itin|ssn|social\s*security)\b", re.I),
    "street_address":         re.compile(r"\b(address|residing|resident|lives?|located|property|mailing|ship|delivery)\b", re.I),
    "url":                    re.compile(r"\b(url|link|website|visit|access|http|https)\b", re.I),
    "ipv4":                   re.compile(r"\b(ip|ipv4|server|vpn|network|address)\b", re.I),
    "license_plate":          re.compile(r"\b(plate|license plate|vehicle|car|truck|fleet)\b", re.I),
    "vehicle_identifier":     re.compile(r"\b(vin|vehicle identifier|vehicle id)\b", re.I),
    "password":               re.compile(r"\b(password|credential|login|vpn)\b", re.I),
    "api_key":                re.compile(r"\b(api key|apikey|token|secret|credential)\b", re.I),
}

ANCHOR_TABLES: Dict[str, Dict[str, Optional[re.Pattern]]] = {
    "ai4privacy_en": _AI4PRIVACY_EN_ANCHORS,
    "ai4privacy_de": _AI4PRIVACY_DE_ANCHORS,
    "nemotron_en": _NEMOTRON_EN_ANCHORS,
    "wolframko_ru":  _WOLFRAMKO_RU_ANCHORS,
}


# --- Format validators -------------------------------------------------

def _luhn_ok(value: str) -> bool:
    digs = [int(c) for c in value if c.isdigit()]
    if len(digs) < 13 or len(digs) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digs)):
        if i % 2 == 0:
            total += d
        else:
            total += d * 2 if d * 2 < 10 else d * 2 - 9
    return total % 10 == 0


_EMAIL_RE = re.compile(r"^[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+$")
_DATE_RE = re.compile(
    r"^("
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"      # 01.01.2001 etc
    r"|\d{4}[./-]\d{1,2}[./-]\d{1,2}"       # 2001-01-01
    r"|[A-Za-zА-Яа-я]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}"  # March 15, 1985
    r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-zА-Яа-я]+\s+\d{2,4}"   # 15th March 1985
    r"|[A-Za-zА-Яа-я]+/\d{1,4}"              # Mai/04 / July/40
    r")$"
)
_RU_INN_10_RE = re.compile(r"^\d{10}$")
_RU_INN_12_RE = re.compile(r"^\d{12}$")
_RU_SNILS_RE = re.compile(r"^\d{3}[-\s]\d{3}[-\s]\d{3}\s\d{2}$")
_RU_PASSPORT_RE = re.compile(r"^\d{2}\s?\d{2}\s*(?:№|N|#)?\s*\d{6}$")
_US_SSN_RE = re.compile(r"^(?!000|666|9\d\d)(\d{3}-\d{2}-\d{4}|\d{9})$")
_DE_STEUERID_RE = re.compile(r"^\d{2}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{3}$")
_DE_PERSONALAUSWEIS_RE = re.compile(r"^[CFGHJKLMNPRTVWXYZ][0-9A-Z]{9}$", re.I)
_HTTP_URL_RE = re.compile(r"^https?://\S+$")
_US_PHONE_RE = re.compile(r"^(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}$")
_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-\s]{2,12}$", re.I)


def _ru_inn_check(value: str) -> bool:
    """Accept either 10-digit (legal entity) or 12-digit (individual) ИНН with checksum."""
    if not value.isdigit():
        return False
    if len(value) == 10:
        weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        digits = [int(c) for c in value]
        return digits[-1] == sum(d * w for d, w in zip(digits[:-1], weights)) % 11 % 10
    if len(value) == 12:
        a = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        b = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        digits = [int(c) for c in value]
        c1 = sum(d * w for d, w in zip(digits[:10], a)) % 11 % 10
        c2 = sum(d * w for d, w in zip(digits[:11], b)) % 11 % 10
        return digits[10] == c1 and digits[11] == c2
    return False


def _ru_snils_check(value: str) -> bool:
    body = value.replace(" ", "").replace("-", "")
    if len(body) != 11 or not body.isdigit():
        return False
    digits = [int(c) for c in body[:9]]
    expected = int(body[9:])
    weights = (9, 8, 7, 6, 5, 4, 3, 2, 1)
    sm = sum(d * w for d, w in zip(digits, weights))
    if sm < 100:
        check = sm
    elif sm in (100, 101):
        check = 0
    else:
        r = sm % 101
        check = 0 if r in (100, 101) else r
    return check == expected


# Format validators per (dataset, label). Returns "ok" or a short reason
# tag if the value clearly does not match the claimed type. None means
# "no validator for this label."
FormatValidator = Callable[[str], Optional[str]]


def _email_ok(v: str) -> Optional[str]:
    return None if _EMAIL_RE.match(v) else "not_email_format"


def _date_ok(v: str) -> Optional[str]:
    return None if _DATE_RE.match(v.strip()) else "not_date_format"


def _card_ok(v: str) -> Optional[str]:
    return None if _luhn_ok(v) else "luhn_fail"


def _ru_inn_ok(v: str) -> Optional[str]:
    if not (_RU_INN_10_RE.match(v) or _RU_INN_12_RE.match(v)):
        return "not_inn_shape"
    return None if _ru_inn_check(v) else "inn_checksum_fail"


def _ru_snils_ok(v: str) -> Optional[str]:
    if not _RU_SNILS_RE.match(v):
        return "not_snils_shape"
    return None if _ru_snils_check(v) else "snils_checksum_fail"


def _ru_passport_ok(v: str) -> Optional[str]:
    return None if _RU_PASSPORT_RE.match(v) else "not_ru_passport_shape"


def _us_ssn_ok(v: str) -> Optional[str]:
    return None if _US_SSN_RE.match(v) else "not_ssn_shape"


def _de_steuerid_ok(v: str) -> Optional[str]:
    return None if _DE_STEUERID_RE.match(v) else "not_steuerid_shape"


def _de_personalausweis_ok(v: str) -> Optional[str]:
    return None if _DE_PERSONALAUSWEIS_RE.match(v) else "not_personalausweis_shape"


def _http_url_ok(v: str) -> Optional[str]:
    return None if _HTTP_URL_RE.match(v.strip()) else "unsupported_url_scheme"


def _us_phone_ok(v: str) -> Optional[str]:
    return None if _US_PHONE_RE.match(v.strip()) else "not_us_phone_shape"


def _vin_shape_ok(v: str) -> Optional[str]:
    return None if _VIN_RE.match(v.strip().upper()) else "not_vin_shape"


def _plate_shape_ok(v: str) -> Optional[str]:
    return None if _PLATE_RE.match(v.strip()) else "not_plate_shape"


VALIDATORS: Dict[str, Dict[str, FormatValidator]] = {
    "ai4privacy_en": {
        "EMAIL":            _email_ok,
        "DATE":             _date_ok,
        "CREDITCARDNUMBER": _card_ok,
        "SOCIALNUM":        _us_ssn_ok,
    },
    "ai4privacy_de": {
        "EMAIL":            _email_ok,
        "DATE":             _date_ok,
        "CREDITCARDNUMBER": _card_ok,
        "TAXNUM":           _de_steuerid_ok,
        "IDCARDNUM":        _de_personalausweis_ok,
    },
    "nemotron_en": {
        "email":                  _email_ok,
        "date":                   _date_ok,
        "date_of_birth":          _date_ok,
        "credit_debit_card":      _card_ok,
        "ssn":                    _us_ssn_ok,
        "tax_id":                 _us_ssn_ok,
        "phone_number":           _us_phone_ok,
        "url":                    _http_url_ok,
        "vehicle_identifier":     _vin_shape_ok,
        "license_plate":          _plate_shape_ok,
    },
    "wolframko_ru": {
        "EMAIL":            _email_ok,
        "DATEOFBIRTH":      _date_ok,
        "CREDITCARDNUMBER": _card_ok,
        "TAXNUM":           _ru_inn_ok,
        "SOCIALNUM":        _ru_snils_ok,
        "IDCARDNUM":        _ru_passport_ok,
        "DRIVERLICENSENUM": _ru_passport_ok,  # same XX XX XXXXXX shape
    },
}


# --- Quality report data classes ---------------------------------------


@dataclass
class LabelQuality:
    spans: int = 0
    spans_with_anchor: int = 0
    spans_no_anchor: int = 0
    spans_format_ok: int = 0
    spans_format_fail: Counter = field(default_factory=Counter)
    spans_junk_value: int = 0
    sample_no_anchor: list = field(default_factory=list)
    sample_format_fail: list = field(default_factory=list)
    value_lengths: Counter = field(default_factory=Counter)


@dataclass
class DatasetQuality:
    dataset: str
    rows_scanned: int = 0
    rows_with_quality_issue: int = 0
    by_label: Dict[str, LabelQuality] = field(default_factory=lambda: defaultdict(LabelQuality))
    same_value_multi_label: list = field(default_factory=list)  # leakage examples
    same_row_format_collisions: Counter = field(default_factory=Counter)


# --- Scanner -----------------------------------------------------------


def _value_is_junk(v: str) -> bool:
    if len(v) < MIN_VALUE_LEN:
        return True
    if not any(ch.isalnum() for ch in v):
        return True
    return False


def scan(stream: Iterable[NormalizedExample], dataset_slug: str) -> DatasetQuality:
    anchors = ANCHOR_TABLES[dataset_slug]
    validators = VALIDATORS.get(dataset_slug, {})

    out = DatasetQuality(dataset=dataset_slug)
    # value -> set of labels it appeared under (for label leakage detection)
    value_to_labels: Dict[str, set] = defaultdict(set)
    # for keeping leakage report bounded
    leakage_examples_capacity = 20

    for ex in stream:
        out.rows_scanned += 1
        text = ex.text
        row_has_issue = False

        for sp in ex.spans:
            lbl = sp.label
            lq = out.by_label[lbl]
            lq.spans += 1
            lq.value_lengths[len(sp.value)] += 1
            value_to_labels[sp.value].add(lbl)

            # 1. Junk
            if _value_is_junk(sp.value):
                lq.spans_junk_value += 1
                row_has_issue = True

            # 2. Anchor presence
            anchor_re = anchors.get(lbl)
            if anchor_re is not None:
                window_start = max(0, sp.start - ANCHOR_RADIUS)
                window_end = min(len(text), sp.end + ANCHOR_RADIUS)
                window = text[window_start:window_end]
                if anchor_re.search(window):
                    lq.spans_with_anchor += 1
                else:
                    lq.spans_no_anchor += 1
                    row_has_issue = True
                    if len(lq.sample_no_anchor) < 5:
                        lq.sample_no_anchor.append({
                            "value": sp.value,
                            "context": window.replace("\n", " ").replace("\r", " "),
                            "row_id": ex.id,
                        })
            # else: no anchor expected for this label, skip the check

            # 3. Format
            v_check = validators.get(lbl)
            if v_check is not None:
                reason = v_check(sp.value)
                if reason is None:
                    lq.spans_format_ok += 1
                else:
                    lq.spans_format_fail[reason] += 1
                    row_has_issue = True
                    if len(lq.sample_format_fail) < 5:
                        lq.sample_format_fail.append({
                            "value": sp.value,
                            "reason": reason,
                            "row_id": ex.id,
                        })

        # 4. Same-row format collisions: any two labels emit values with
        # identical (length, charclass) within this row?
        if len(ex.spans) >= 2:
            shapes = defaultdict(list)
            for sp in ex.spans:
                shape = (
                    len(sp.value),
                    "".join(sorted({
                        "U" if ch.isupper() else "u" if ch.islower() else "d" if ch.isdigit() else "p"
                        for ch in sp.value
                    })),
                )
                shapes[shape].append(sp.label)
            for shape, lbls in shapes.items():
                uniq = set(lbls)
                if len(uniq) >= 2:
                    pair = tuple(sorted(uniq))
                    out.same_row_format_collisions[pair] += 1

        if row_has_issue:
            out.rows_with_quality_issue += 1

    # 5. Label leakage: same string under multiple labels.
    for value, lbls in value_to_labels.items():
        if len(lbls) >= 2 and len(out.same_value_multi_label) < leakage_examples_capacity:
            out.same_value_multi_label.append({"value": value, "labels": sorted(lbls)})

    return out


def render(q: DatasetQuality) -> str:
    n = q.rows_scanned or 1
    lines = [
        f"# Quality scan - {q.dataset}",
        "",
        f"- Rows scanned: **{q.rows_scanned:,}**",
        f"- Rows with at least one quality issue: **{q.rows_with_quality_issue:,}** "
        f"({100 * q.rows_with_quality_issue / n:.1f}%)",
        "",
        "## Per-label quality",
        "",
        "| Label | Spans | Anchor rate | Format pass | Format fails | Junk |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for lbl in sorted(q.by_label):
        lq = q.by_label[lbl]
        anchor_total = lq.spans_with_anchor + lq.spans_no_anchor
        anchor_rate = (
            f"{100 * lq.spans_with_anchor / anchor_total:.1f}%"
            if anchor_total else "n/a"
        )
        fmt_total = lq.spans_format_ok + sum(lq.spans_format_fail.values())
        fmt_pass = (
            f"{100 * lq.spans_format_ok / fmt_total:.1f}%"
            if fmt_total else "n/a"
        )
        fmt_fail_summary = ", ".join(
            f"`{r}`={c}" for r, c in lq.spans_format_fail.most_common(3)
        ) or "—"
        lines.append(
            f"| `{lbl}` | {lq.spans} | {anchor_rate} | {fmt_pass} | "
            f"{fmt_fail_summary} | {lq.spans_junk_value} |"
        )
    lines.append("")

    # Anchor-fail samples
    lines.append("## Sample anchor failures (first 5 per label)")
    lines.append("")
    for lbl in sorted(q.by_label):
        lq = q.by_label[lbl]
        if not lq.sample_no_anchor:
            continue
        lines.append(f"### `{lbl}`  ({lq.spans_no_anchor} / {lq.spans_with_anchor + lq.spans_no_anchor} no-anchor)")
        for ex in lq.sample_no_anchor:
            ctx = ex["context"]
            if len(ctx) > 160:
                ctx = ctx[:157] + "..."
            lines.append(f"- `{ex['row_id']}` value=`{ex['value']}`  context: `{ctx}`")
        lines.append("")

    # Format-fail samples
    lines.append("## Sample format failures")
    lines.append("")
    for lbl in sorted(q.by_label):
        lq = q.by_label[lbl]
        if not lq.sample_format_fail:
            continue
        total_fail = sum(lq.spans_format_fail.values())
        lines.append(f"### `{lbl}`  ({total_fail} fails)")
        for ex in lq.sample_format_fail:
            lines.append(f"- `{ex['row_id']}` value=`{ex['value']}`  reason=`{ex['reason']}`")
        lines.append("")

    # Cross-label format collisions
    if q.same_row_format_collisions:
        lines.append("## Same-row format collisions")
        lines.append("")
        lines.append("Two labels in the same row carrying values of identical (length, charclass).")
        lines.append("Forces context-only disambiguation.")
        lines.append("")
        lines.append("| Label pair | Same-row collisions |")
        lines.append("|---|---:|")
        for pair, cnt in q.same_row_format_collisions.most_common(20):
            lines.append(f"| `{pair[0]}` ↔ `{pair[1]}` | {cnt} |")
        lines.append("")

    # Label leakage
    if q.same_value_multi_label:
        lines.append("## Label leakage (same value, multiple labels)")
        lines.append("")
        lines.append("| Value | Labels seen |")
        lines.append("|---|---|")
        for ent in q.same_value_multi_label:
            v = ent["value"]
            if len(v) > 60:
                v = v[:57] + "..."
            lines.append(f"| `{v}` | {', '.join(f'`{l}`' for l in ent['labels'])} |")
        lines.append("")

    return "\n".join(lines) + "\n"
