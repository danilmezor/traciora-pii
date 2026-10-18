"""Schema-level transforms over a stream of NormalizedExample.

Source datasets label first/last names separately (``GIVENNAME``,
``SURNAME``). The traciora_pii framework labels names with a single
``PERSON_NAME`` token. ``combine_names`` reconciles the two: it merges
adjacent ``GIVENNAME`` + ``SURNAME`` spans (in either order) into one
``PERSON_NAME`` span and drops rows where the merge is ambiguous.

Drop rule (strict): a row is dropped if **any** ``GIVENNAME`` or
``SURNAME`` span on it cannot be paired with an adjacent counterpart.
That covers
- rows with only one of the two labels,
- rows where a name span is separated from its partner by other text,
- rows with chained name spans that don't form clean pairs.

Rationale: keeping partial rows would silently introduce label noise
into the eval, which is exactly what the audit is supposed to prevent.
The transform also returns a small dataclass of counters so the caller
can report attrition rates.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .loaders import NormalizedExample, Span


PERSON_NAME = "PERSON_NAME"
NAME_LABEL_PAIRS = frozenset({
    frozenset({"GIVENNAME", "SURNAME"}),
    frozenset({"first_name", "last_name"}),
})
NAME_LABELS = frozenset(label for pair in NAME_LABEL_PAIRS for label in pair)

# Cap on the gap between GIVENNAME and SURNAME for them to count as
# adjacent. Imported datasets usually use a single whitespace char between
# name components, so ``<= 3`` covers normal space, NBSP, narrow NBSP,
# and the occasional double space without admitting punctuation.
_MAX_NAME_GAP = 3


@dataclass
class CombineStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_no_name_spans: int = 0  # rows that just had no GN/SN to begin with - kept, not dropped
    rows_dropped_only_one_kind: int = 0  # had GN but no SN (or vice versa)
    rows_dropped_unpaired: int = 0       # had both, but at least one couldn't be paired
    pairs_merged: int = 0


def _is_adjacent(left: Span, right: Span, text: str) -> bool:
    """True if ``right`` starts at or just after ``left`` ends with only whitespace between."""
    if right.start < left.end:
        return False  # overlap, not adjacent
    gap = text[left.end : right.start]
    if len(gap) > _MAX_NAME_GAP:
        return False
    return gap == "" or gap.isspace()


def _try_combine_one(ex: NormalizedExample) -> tuple[NormalizedExample | None, str]:
    """Apply the merge to one row.

    Returns ``(new_example, reason)`` where ``reason`` is one of
    ``"kept"``, ``"no_names"``, ``"only_one_kind"``, ``"unpaired"``.
    """
    name_spans = sorted(
        (s for s in ex.spans if s.label in NAME_LABELS),
        key=lambda s: (s.start, s.end),
    )
    other_spans = [s for s in ex.spans if s.label not in NAME_LABELS]

    if not name_spans:
        return ex, "no_names"

    labels_present = frozenset(s.label for s in name_spans)
    if labels_present not in NAME_LABEL_PAIRS:
        return None, "only_one_kind"

    # Greedy left-to-right pairing. We require the entire name_spans list
    # to consume cleanly into (GN, SN) or (SN, GN) adjacent pairs. Any
    # leftover span = drop the row.
    merged: list[Span] = []
    i = 0
    while i < len(name_spans):
        if i + 1 >= len(name_spans):
            return None, "unpaired"
        a, b = name_spans[i], name_spans[i + 1]
        if frozenset({a.label, b.label}) not in NAME_LABEL_PAIRS:
            return None, "unpaired"
        if not _is_adjacent(a, b, ex.text):
            return None, "unpaired"
        # The merged span covers (a.start, b.end) and uses the actual
        # text slice as the value. Order in source text is preserved -
        # we don't rewrite "Smith John" to "John Smith"; we just collapse
        # the two spans into one PERSON_NAME annotation.
        merged.append(
            Span(
                label=PERSON_NAME,
                value=ex.text[a.start : b.end],
                start=a.start,
                end=b.end,
            )
        )
        i += 2

    new_spans = sorted(other_spans + merged, key=lambda s: (s.start, s.end))
    return (
        NormalizedExample(
            id=ex.id,
            locale=ex.locale,
            text=ex.text,
            spans=new_spans,
            source=ex.source,
            meta={**ex.meta, "name_pairs_merged": len(merged)},
        ),
        "kept",
    )


def combine_names(
    stream: Iterable[NormalizedExample],
) -> tuple[Iterator[NormalizedExample], CombineStats]:
    """Stream-friendly wrapper. Returns (filtered_stream, stats).

    ``stats`` is mutated as the stream is consumed; read it after the
    consumer drains the iterator.
    """
    stats = CombineStats()

    def _gen() -> Iterator[NormalizedExample]:
        for ex in stream:
            stats.rows_in += 1
            new_ex, reason = _try_combine_one(ex)
            if reason == "kept":
                stats.rows_out += 1
                stats.pairs_merged += new_ex.meta.get("name_pairs_merged", 0)
                yield new_ex
            elif reason == "no_names":
                # No GN/SN spans at all - the rule about "drop if only a
                # name or surname" doesn't apply. Pass through.
                stats.rows_out += 1
                stats.rows_dropped_no_name_spans += 0  # explicit: not a drop
                yield ex
            elif reason == "only_one_kind":
                stats.rows_dropped_only_one_kind += 1
            else:  # "unpaired"
                stats.rows_dropped_unpaired += 1

    return _gen(), stats


ADDRESS_COMPONENT_LABELS = frozenset({"STREET", "BUILDINGNUM", "CITY", "ZIPCODE"})
STREET_ADDRESS = "STREET_ADDRESS"


# ======================================================================
# Nemotron EN quality gate
# ======================================================================


_NEMOTRON_DROP_LABELS = frozenset({
    "account_number",
    "bank_routing_number",
    "swift_bic",
    "city",
    "state",
    "county",
    "country",
    "postcode",
})

_EMAIL_RE = re.compile(r"^[A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9.\-]+$")
_US_SSN_RE = re.compile(r"^(?!000|666|9\d\d)(\d{3}-\d{2}-\d{4}|\d{9})$")
_US_PHONE_RE = re.compile(r"^(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}$")
_DATE_RE = re.compile(
    r"^(?:"
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+[A-Za-z]+\s+\d{4}"
    r")$"
)
_IPV4_RE = re.compile(r"^(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}$")
_HTTP_URL_RE = re.compile(r"^https?://\S+$")
_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-\s]{2,12}$", re.I)

_VIN_TRANSLIT = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
for _d in range(10):
    _VIN_TRANSLIT[str(_d)] = _d
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def _luhn_ok(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            total += d
        else:
            doubled = d * 2
            total += doubled if doubled < 10 else doubled - 9
    return total % 10 == 0


def _vin_ok(value: str) -> bool:
    value = value.strip().upper()
    if not _VIN_RE.match(value):
        return False
    check = sum(_VIN_TRANSLIT[c] * w for c, w in zip(value, _VIN_WEIGHTS)) % 11
    expected = "X" if check == 10 else str(check)
    return value[8] == expected


def _nemotron_format_reason(span: Span) -> str | None:
    value = span.value.strip()
    label = span.label
    if label in _NEMOTRON_DROP_LABELS:
        return f"drop_label:{label}"
    if label == "tax_id":
        return "drop_label:tax_id_ssn_shape" if _US_SSN_RE.match(value) else "drop_label:tax_id"
    if label in {"date", "date_of_birth"}:
        return None if _DATE_RE.match(value) else "malformed_date"
    if label == "credit_debit_card":
        return None if _luhn_ok(value) else "card_luhn_fail"
    if label == "email":
        return None if _EMAIL_RE.match(value) else "malformed_email"
    if label == "phone_number":
        return None if _US_PHONE_RE.match(value) else "malformed_phone"
    if label == "ssn":
        return None if _US_SSN_RE.match(value) else "malformed_ssn"
    if label == "url":
        return None if _HTTP_URL_RE.match(value) else "unsupported_url_scheme"
    if label == "ipv4":
        return None if _IPV4_RE.match(value) else "malformed_ipv4"
    if label == "vehicle_identifier":
        return None if _vin_ok(value) else "malformed_vin"
    if label == "license_plate":
        return None if _PLATE_RE.match(value) else "malformed_license_plate"
    return None


@dataclass
class NemotronQualityGateStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)
    labels_rewritten: Counter[str] = field(default_factory=Counter)


@dataclass
class ProjectAllowedStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_no_allowed_spans: int = 0
    spans_dropped: int = 0
    label_dropped_counts: Counter[str] = field(default_factory=Counter)


def quality_gate_nemotron(
    stream: Iterable[NormalizedExample],
) -> tuple[Iterator[NormalizedExample], NemotronQualityGateStats]:
    """Drop Nemotron rows that cannot support a fair traciora_pii eval.

    The gate removes explicitly unsupported labels, rejects malformed strict
    identifiers (dates, cards, SSNs, phones, URLs, VINs), and drops
    ``tax_id`` spans rather than relabeling them as SSNs. A row is
    dropped if any span fails the gate.
    """
    stats = NemotronQualityGateStats()

    def _gen() -> Iterator[NormalizedExample]:
        for ex in stream:
            stats.rows_in += 1
            new_spans: list[Span] = []
            reasons: set[str] = set()
            for sp in ex.spans:
                reason = _nemotron_format_reason(sp)
                if reason is not None:
                    reasons.add(reason)
                    continue
                new_spans.append(sp)
            if reasons:
                stats.rows_dropped += 1
                for reason in reasons:
                    stats.reason_counts[reason] += 1
                continue
            stats.rows_out += 1
            yield NormalizedExample(
                id=ex.id,
                locale=ex.locale,
                text=ex.text,
                spans=new_spans,
                source=ex.source,
                meta={**ex.meta, "nemotron_quality_gate": True},
            )

    return _gen(), stats


def project_allowed_labels(
    stream: Iterable[NormalizedExample],
    *,
    allowlist: frozenset[str],
) -> tuple[Iterator[NormalizedExample], ProjectAllowedStats]:
    """Keep only spans that map to the framework's placeholder taxonomy.

    Unlike ``keep_only_allowed_labels``, this is a span projection rather
    than a row-level drop. It is useful for broad multilingual corpora
    where documents often contain unsupported labels alongside supported
    labels; those unsupported spans are outside the benchmark target, but
    their surrounding text remains valuable eval context.
    """
    stats = ProjectAllowedStats()

    def _gen() -> Iterator[NormalizedExample]:
        for ex in stream:
            stats.rows_in += 1
            kept: list[Span] = []
            for sp in ex.spans:
                if sp.label in allowlist:
                    kept.append(sp)
                else:
                    stats.spans_dropped += 1
                    stats.label_dropped_counts[sp.label] += 1
            if not kept:
                stats.rows_dropped_no_allowed_spans += 1
                continue
            stats.rows_out += 1
            yield NormalizedExample(
                id=ex.id,
                locale=ex.locale,
                text=ex.text,
                spans=kept,
                source=ex.source,
                meta={
                    **ex.meta,
                    "allowed_label_projection": True,
                    "projected_spans_dropped": len(ex.spans) - len(kept),
                },
            )

    return _gen(), stats


@dataclass
class CombineAddressStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_scattered: int = 0   # address components separated by non-address spans
    rows_no_address: int = 0          # passed through unchanged
    addresses_merged: int = 0


def combine_addresses(
    stream: Iterable[NormalizedExample],
) -> tuple[Iterator[NormalizedExample], CombineAddressStats]:
    """Merge contiguous STREET / BUILDINGNUM / CITY / ZIPCODE spans into one
    STREET_ADDRESS span.

    The detector emits one ``[STREET_ADDRESS]`` span covering the full
    address; the source datasets emit four separate component spans. This
    transform reconciles the two so span-overlap matching in the eval is
    fair (one detector hit vs one gold span, not 4-to-1).

    Drop rule: a row is dropped iff its address-component spans split
    into more than one *cluster*. A cluster is a maximal run of
    address-component spans, in document order, with no non-address PII
    span starting inside the run. Single-cluster rows are merged; rows
    with no address components pass through unchanged.

    The merged span covers ``[min(start)..max(end)]`` over the cluster,
    with ``value = text[start:end]`` so the standard
    ``text[start:end] == value`` invariant still holds.
    """
    stats = CombineAddressStats()

    def _gen() -> Iterator[NormalizedExample]:
        for ex in stream:
            stats.rows_in += 1
            new_ex, reason = _try_combine_addresses_one(ex)
            if reason == "scattered":
                stats.rows_dropped_scattered += 1
                continue
            stats.rows_out += 1
            if reason == "no_address":
                stats.rows_no_address += 1
            else:
                stats.addresses_merged += 1
            yield new_ex

    return _gen(), stats


def _try_combine_addresses_one(ex: NormalizedExample) -> tuple[NormalizedExample, str]:
    spans_sorted = sorted(ex.spans, key=lambda s: (s.start, s.end))
    addr_spans = [s for s in spans_sorted if s.label in ADDRESS_COMPONENT_LABELS]
    other_spans = [s for s in spans_sorted if s.label not in ADDRESS_COMPONENT_LABELS]

    if not addr_spans:
        return ex, "no_address"

    # Cluster detection: walk all spans in document order; a non-address span
    # appearing between two address spans splits the cluster. We collect
    # address spans into clusters by scanning the sorted full span list and
    # bumping the cluster index every time we see a non-address span between
    # already-accumulated address spans.
    clusters: list[list[Span]] = [[]]
    have_addr_in_current = False
    for s in spans_sorted:
        if s.label in ADDRESS_COMPONENT_LABELS:
            clusters[-1].append(s)
            have_addr_in_current = True
        else:
            # A non-address span only splits if we already have address
            # spans in the current cluster; otherwise it's a leading
            # non-address span and the cluster stays "fresh".
            if have_addr_in_current:
                clusters.append([])
                have_addr_in_current = False

    clusters = [c for c in clusters if c]
    if len(clusters) > 1:
        return ex, "scattered"

    cluster = clusters[0]
    start = cluster[0].start
    end = cluster[-1].end
    merged = Span(label=STREET_ADDRESS, value=ex.text[start:end], start=start, end=end)

    new_spans = sorted(other_spans + [merged], key=lambda s: (s.start, s.end))
    return (
        NormalizedExample(
            id=ex.id,
            locale=ex.locale,
            text=ex.text,
            spans=new_spans,
            source=ex.source,
            meta={**ex.meta, "address_components_merged": len(cluster)},
        ),
        "merged",
    )


@dataclass
class KeepOnlyAllowedStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    # Number of rows whose drop was triggered by a given out-of-allowlist
    # label. A row carrying both TITLE and AGE counts once for each, so the
    # report can show which unmappable labels are most prevalent.
    label_trigger_counts: Counter[str] = field(default_factory=Counter)


def keep_only_allowed_labels(
    stream: Iterable[NormalizedExample],
    *,
    allowlist: frozenset[str],
) -> tuple[Iterator[NormalizedExample], KeepOnlyAllowedStats]:
    """Drop rows whose spans include any label not in ``allowlist``.

    Rationale: this is the inversion of the previous blocklist filter. With
    a fixed allowlist of labels that map cleanly to the framework's
    placeholder taxonomy, any row carrying a span outside the allowlist
    has at least one annotated entity the framework cannot detect - keeping
    it would inflate false-negative counts against ground truth the
    detector can't emit.

    A row that has *no* spans at all is still passed through. The decision
    is made on the labels actually present, so empty-span rows aren't
    forced through this filter.
    """
    stats = KeepOnlyAllowedStats()

    def _gen() -> Iterator[NormalizedExample]:
        for ex in stream:
            stats.rows_in += 1
            disallowed = {s.label for s in ex.spans} - allowlist
            if disallowed:
                stats.rows_dropped += 1
                for label in disallowed:
                    stats.label_trigger_counts[label] += 1
                continue
            stats.rows_out += 1
            yield ex

    return _gen(), stats
