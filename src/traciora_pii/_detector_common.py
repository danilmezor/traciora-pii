"""Shared types and deterministic helpers for second-pass PII detectors."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ======================================================================
# Public types
# ======================================================================


@dataclass(frozen=True)
class LLMFinding:
    """A single name or address span emitted by the LLM.

    ``start`` and ``end`` are character offsets into the text the
    detector was called with. ``lemma`` is the canonical form to be
    handed to ``PIIVaultStore.get_or_create_token`` as the value
    argument, so that referring expressions across turns collapse to
    the same vault token.
    """
    detector: str            # "llm" or "opf"
    start: int
    end: int
    placeholder: str         # "[PERSON_NAME]" | "[STREET_ADDRESS]"
    lemma: str               # canonical form for vault keying


# Mapping from the LM's "type" field to the production placeholder syntax.
_TYPE_TO_PLACEHOLDER = {
    "PERSON_NAME": "[PERSON_NAME]",
    "STREET_ADDRESS": "[STREET_ADDRESS]",
}


# ======================================================================
# Name lemma normalization (pymorphy3 for RU, regex for EN)
# ======================================================================

# Lazy-init pymorphy3 analyzer — imported once on first use, ~200 ms cold.
_morph_analyzer = None
_morph_available = None


def _get_morph():
    global _morph_analyzer, _morph_available
    if _morph_available is None:
        try:
            import pymorphy3
            _morph_analyzer = pymorphy3.MorphAnalyzer()
            _morph_available = True
        except ImportError:
            _morph_available = False
            logger.info("pymorphy3 not installed; Russian name normalization disabled")
    return _morph_analyzer


_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_TITLE_RE = re.compile(r"^(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.?\s+", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"'s$")


def _normalize_person_lemma(span_text: str, lm_lemma: str) -> str:
    """Deterministic canonical form for a person name span.

    Russian (Cyrillic): split into words, run pymorphy3 on each,
    take normal_form (nominative), sort surname-first, lowercase, join.

    English (Latin): strip possessive 's, strip titles
    (Dr./Mr./Mrs./Ms./Prof.), lowercase, reorder "lastname firstname".

    Fallback: return lm_lemma unchanged (the LM's best guess).
    """
    text = span_text.strip()
    if not text:
        return lm_lemma

    # Russian path — pymorphy3
    if _CYRILLIC_RE.search(text):
        morph = _get_morph()
        if morph is None:
            return lm_lemma
        words = text.split()
        surnames = []
        non_surnames = []
        for word in words:
            parsed = morph.parse(word)[0]
            nf = parsed.normal_form
            if "Surn" in str(parsed.tag):
                surnames.append(nf)
            else:
                non_surnames.append(nf)
        # Canonical order: surnames first, then given names
        parts = surnames + non_surnames
        return " ".join(parts).lower()

    # English path — regex only
    text = _TITLE_RE.sub("", text)
    text = _POSSESSIVE_RE.sub("", text)
    words = text.split()
    if len(words) >= 2:
        # Reorder to "lastname firstname [middle ...]"
        parts = [words[-1]] + words[:-1]
    else:
        parts = words
    return " ".join(parts).lower()


# ======================================================================
# Prefilter — cheap pre-LM gate
# ======================================================================

# Tokens that look like person names but never are. Lowercased for matching.
_BRAND_DENYLIST_EN = frozenset({
    "apple", "tesla", "google", "microsoft", "amazon", "meta", "facebook",
    "twitter", "openai", "anthropic", "cloudflare", "aws", "azure", "github",
    "linux", "windows", "intel", "nvidia", "ibm", "oracle", "salesforce",
    "stripe", "shopify", "slack", "zoom", "netflix", "spotify", "uber", "lyft",
    "phoenix", "madison", "park", "plaza", "main", "broadway", "central",
})
_BRAND_DENYLIST_RU = frozenset({
    "яндекс", "газпром", "роскомнадзор", "сбер", "сбербанк", "тинькофф",
    "вконтакте", "одноклассники", "мегафон", "билайн", "мтс", "ростелеком",
    "лукойл", "роснефть", "новатэк",
})

# Address keywords. Presence of any one of these is enough to flip the
# prefilter to True regardless of capitalized-token analysis.
_ADDRESS_KEYWORDS_EN = (
    r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\.?|"
    r"boulevard|blvd\.?|court|ct\.?|place|pl\.?|square|sq\.?|"
    r"highway|hwy\.?|parkway|pkwy\.?|suite|ste\.?|apt\.?|apartment|"
    r"floor|building|bldg\.?|unit|po\s*box)\b"
)
_ADDRESS_KEYWORDS_RU = (
    r"(?:\bул\.?|улица|улицу|улице|проспект|пр-?т|пр\.|"
    r"переулок|пер\.|шоссе|бульвар|площадь|пл\.|"
    r"\bдом\b|\bд\.|\bкв\.|квартира|корпус|корп\.|строение|стр\.|"
    r"\bг\.|город)"
)

_ADDRESS_RE_EN = re.compile(_ADDRESS_KEYWORDS_EN, re.IGNORECASE)
_ADDRESS_RE_RU = re.compile(_ADDRESS_KEYWORDS_RU, re.IGNORECASE)

# Capitalized token finder. We check non-sentence-initial position by
# requiring at least one preceding non-space character on the same line.
_CAPITAL_EN = re.compile(r"(?<=\S\s)([A-Z][a-z]{2,})")
_CAPITAL_RU = re.compile(r"(?<=\S\s)([А-ЯЁ][а-яё]{2,})")

# Reference-token shape, so the prefilter does not count them as capitalized words.
_REF_TOKEN_RE = re.compile(r"\b[a-z_]+_ref:[a-z]{2}_[a-f0-9]{8}\b")


def _prefilter(text: str) -> bool:
    """Return True iff the text plausibly contains a name or address.

    Cheap regex check intended to skip the LM call entirely on obviously-
    negative inputs (version numbers, ticket IDs, technical noise).
    Returns True on the slightest doubt — false positives cost an LM
    call, false negatives cost a missed name. Tunable.
    """
    if not text or len(text.strip()) < 3:
        return False

    # Strip ref tokens before any analysis so they cannot trigger the
    # capitalized-token branch.
    stripped = _REF_TOKEN_RE.sub(" ", text)

    # Address keyword present anywhere -> definite pass.
    if _ADDRESS_RE_EN.search(stripped) or _ADDRESS_RE_RU.search(stripped):
        return True

    # Capitalized non-sentence-initial token outside the brand denylist
    # -> probable name. Sentence-initial tokens are excluded by the
    # lookbehind requiring a preceding non-space character.
    for match in _CAPITAL_EN.finditer(stripped):
        if match.group(1).lower() not in _BRAND_DENYLIST_EN:
            return True
    for match in _CAPITAL_RU.finditer(stripped):
        if match.group(1).lower() not in _BRAND_DENYLIST_RU:
            return True

    return False


# ======================================================================
# Public aliases — shared with sibling detectors (e.g. OPFPIIDetector)
# ======================================================================

# The canonical pymorphy3-based Russian lemma normalizer and the regex
# prefilter are both detector-agnostic: any second-pass PII detector that
# populates the vault needs the same normalization (§4.3 determinism) and
# benefits from the same cheap-negative gate. Re-exported without the
# underscore so other detectors import from a stable surface.
normalize_person_lemma = _normalize_person_lemma
prefilter = _prefilter
TYPE_TO_PLACEHOLDER = _TYPE_TO_PLACEHOLDER


