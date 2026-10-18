"""Regression test: rendered DE / RU turns must not leak English fragments.

Background: an earlier multi_pii / hard_negative authoring pass left
hardcoded English tails ("needs a callback", "was shipped yesterday")
in YAML turn templates. Those leaked into DE and RU rendered turns
because the renderer substitutes intent fragments + slot placeholders
but doesn't translate free text.

This test scans every DE and RU rendered turn for an *unambiguous*
English-only token list. The list is curated to avoid false positives
from URLs, emails, ORD-XXXX, ISO timestamps, and the like — a hit on
this list is almost certainly a real leakage bug.

Add new tokens conservatively. False negatives are tolerable; false
positives that block legitimate rendering are not.
"""
from __future__ import annotations

import re
from typing import Iterable

import pytest

from benchmarks.pii_evaluation.dataset.render import render_all


# Words that are clearly English-only (no cross-language collision with
# German or Russian common words). Short prepositions / articles like
# "the" / "a" / "in" / "to" / "an" / "was" / "for" are deliberately
# omitted because they collide with DE / RU lexicon ("Das Meeting ist
# IN Raum 412", "die Überweisung AN IBAN", "WAS sind die...") and
# would flood the test with false positives.
#
# This list is the high-confidence cohort: hits here are almost
# certainly real English leakage. False negatives are tolerated
# (a longer English fragment that uses only short prepositions
# wouldn't be caught — but the test still catches realistic mistakes
# because almost every English sentence contains one of these
# content-words).
ENGLISH_LEAKAGE_TOKENS: tuple = (
    "needs",
    "callback",
    "shipped",
    "closed",
    "deployed",
    "appeared",
    "yesterday",
    "morning",
    "sharp",
    "investigate",
    "customer",
    "customers",
    "account",
    "pull",
    "find",
    "look",
    "show",
    "please",
    "thanks",
    "their",
    "them",
    "they",
    "those",
    "these",
    "rolled",
    "logged",
    "clicked",
    "owns",
    "born",
    "reached",
    "disputed",
    "migrated",
    "overnight",
)

# Patterns to strip out before scanning so URL/email/timestamp/UUID
# tokens don't trip the word-boundary regex (e.g. ``smith.was.shipped@x.com``
# would otherwise match "was" / "shipped"). The order matters: longer
# patterns first.
_REDACT_PATTERNS = (
    re.compile(r"https?://\S+"),                                 # URLs
    re.compile(r"\b\w[\w.\-+_]*@[\w.\-]+\.[a-z]{2,}\b", re.I),    # emails
    re.compile(r"\b[A-Z]{2,4}-\d{3,8}\b"),                       # ORD-XXXX, TCK-7042
    re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?\b"),    # ISO timestamps
    re.compile(r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b"),  # UUIDs
    re.compile(r"\bVIN\b"),                                      # acronym used in all 3 langs
    re.compile(r"\bIBAN\b"),                                     # acronym used in all 3 langs
    re.compile(r"\bTCK-\d+\b"),                                  # ticket ids
    re.compile(r"\bUTC\b"),                                      # used in all 3 langs
    re.compile(r"\bzero-width\b", re.I),                         # technical term, used inline in RU/DE
    re.compile(r"\bphone_ref\b|\bemail_ref\b"),                  # ref-token forgeries
    re.compile(r"\bstaging\b"),                                  # English loan-word in DE/RU dev jargon
    re.compile(r"\btrace[-_]?id\b", re.I),                       # English loan-term in DE/RU
)


def _strip_neutrals(text: str) -> str:
    out = text
    for pattern in _REDACT_PATTERNS:
        out = pattern.sub(" ", out)
    return out


def _english_words_in(text: str, vocab: Iterable[str]) -> list:
    cleaned = _strip_neutrals(text)
    found = []
    for word in vocab:
        if re.search(rf"\b{re.escape(word)}\b", cleaned, re.IGNORECASE):
            found.append(word)
    return found


@pytest.fixture(scope="module")
def rendered_queries():
    queries, _fixtures = render_all()
    return queries


def test_no_english_leakage_in_de_or_ru(rendered_queries):
    failures = []
    for q in rendered_queries:
        if q.language == "en":
            continue
        for turn_idx, turn in enumerate(q.turns):
            hits = _english_words_in(turn, ENGLISH_LEAKAGE_TOKENS)
            if hits:
                failures.append(
                    f"  {q.id} (lang={q.language}, turn={turn_idx}): "
                    f"english leak {hits!r} in {turn!r}",
                )
    assert not failures, (
        "English fragments leaked into non-EN rendered turns:\n"
        + "\n".join(failures)
        + "\n\nFix by:\n"
        "  - using `turns_per_locale` instead of `turns` in the scenario YAML, OR\n"
        "  - moving free-text tails into `intent_*` fragments under "
        "benchmarks/data_generator/seeds/<locale>/scenario_phrases.yaml.\n"
        "If a hit is a genuine technical term shared across locales (e.g. an "
        "acronym), add it to _REDACT_PATTERNS in this test."
    )


def test_no_trailing_whitespace_in_turns(rendered_queries):
    """Empty intent_please_lookup variants used to leak trailing spaces."""
    failures = []
    for q in rendered_queries:
        for turn_idx, turn in enumerate(q.turns):
            if turn != turn.rstrip():
                failures.append(
                    f"  {q.id} (lang={q.language}, turn={turn_idx}): "
                    f"trailing whitespace in {turn!r}",
                )
    assert not failures, (
        "Rendered turns with trailing whitespace:\n" + "\n".join(failures)
    )


def test_no_double_spaces_in_turns(rendered_queries):
    """Rendered turns should never contain consecutive spaces."""
    failures = []
    for q in rendered_queries:
        for turn_idx, turn in enumerate(q.turns):
            if "  " in turn:
                failures.append(
                    f"  {q.id} (lang={q.language}, turn={turn_idx}): "
                    f"double space in {turn!r}",
                )
    assert not failures, (
        "Rendered turns with consecutive spaces:\n" + "\n".join(failures)
    )
