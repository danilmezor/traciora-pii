"""Adversarial noise: typos, OCR confusables, whitespace jitter, code-switching.

Default behaviour is **identity**: ``NoiseConfig`` ships with all rates at
zero. Users opt in by setting individual rates (e.g. ``typo_rate=0.05``).

Span invariance is the load-bearing contract:

* ``target="scaffold"`` (default) — perturb only characters *outside* PII
  zones. Span ``start``/``end`` offsets are recomputed against the
  perturbed text so ``text[start:end] == value`` holds for every span.
* ``target="pii"`` — perturb the PII value itself, then update the span's
  ``value`` *and* ``end`` to match. The placeholder still labels the
  *intended* entity type, which is exactly what we want from a stress
  evaluation.
* ``target="both"`` — applies the scaffold-zone pass first, then the PII
  pass.

OCR confusable pairs come from ``seeds/shared/ocr_confusables.yaml``;
keyboard adjacency tables stay in code (they're a property of the
keyboard layout, not configuration).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .injectors import Span
from .seeds import SeedBundle


# ======================================================================
# Constants — keyboard adjacency and code-switch tags (locale physics)
# ======================================================================

# Latin keyboard adjacency. Used for EN and DE (DE QWERTZ shares enough
# geometry that QWERTY adjacency is a reasonable proxy at the ~5% rate
# this module operates at).
_LATIN_NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

# Cyrillic ЙЦУКЕН adjacency.
_CYRILLIC_NEIGHBORS = {
    "а": "вфыпрол", "б": "ьюти", "в": "ыапрфа", "г": "нпрши",
    "д": "вылж", "е": "куы", "з": "щэхъ", "и": "тмс", "й": "цфя",
    "к": "уеп", "л": "джо", "м": "ис", "н": "грт",
    "о": "лщр", "п": "ркаи", "р": "паон", "с": "чми",
    "т": "иьрн", "у": "цкев", "ф": "йяв", "х": "ъзэ",
    "ц": "йукф", "ч": "сяьм", "ш": "лгщ", "щ": "шгз",
    "ы": "вае", "ь": "тбч", "э": "хжъ", "ю": "бть", "я": "чфй",
}

# Code-switch tokens — single foreign-script words inserted at word
# boundaries. Kept short so the host sentence remains comprehensible.
_CODESWITCH_INSERTS = {
    "ru": ("note: ", "see ", "ok, ", "fyi: "),
    "de": ("zwischen ", "siehe ", "btw, ", "i.e. "),
    "en": ("кстати, ", "siehe ", "и далее ", "btw "),
}


# ======================================================================
# Config + Applier
# ======================================================================


@dataclass
class NoiseConfig:
    """Per-pipeline noise configuration. Default is identity (no noise)."""

    typo_rate: float = 0.0
    ocr_confusable_rate: float = 0.0
    whitespace_jitter_rate: float = 0.0
    code_switch_rate: float = 0.0
    target: str = "scaffold"  # "scaffold" | "pii" | "both"
    locale: str = "en"

    def __post_init__(self) -> None:
        if self.target not in ("scaffold", "pii", "both"):
            raise ValueError(f"target must be scaffold|pii|both, got {self.target!r}")
        for name in ("typo_rate", "ocr_confusable_rate",
                     "whitespace_jitter_rate", "code_switch_rate"):
            r = getattr(self, name)
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {r!r}")


@dataclass
class NoiseApplier:
    """Applies a ``NoiseConfig`` to ``(text, spans)`` while preserving the
    span invariant ``text[start:end] == value``.

    OCR confusable pairs are taken from ``seeds.shared.ocr_confusables``
    when ``seeds`` is provided; without seeds, OCR perturbation is skipped
    even if its rate is non-zero.
    """

    config: NoiseConfig
    seeds: Optional[SeedBundle] = None
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def apply(self, text: str, spans: Sequence[Span]) -> Tuple[str, List[Span]]:
        if self._is_identity():
            return text, list(spans)

        spans_list = sorted(spans, key=lambda s: s.start)

        if self.config.target in ("scaffold", "both"):
            text, spans_list = self._apply_scaffold(text, spans_list)

        if self.config.target in ("pii", "both"):
            text, spans_list = self._apply_pii(text, spans_list)

        for sp in spans_list:
            assert text[sp.start:sp.end] == sp.value, (
                f"noise broke span invariant: {text[sp.start:sp.end]!r} != {sp.value!r}"
            )
        return text, spans_list

    # ------------------------------------------------------------------

    def _is_identity(self) -> bool:
        c = self.config
        return (c.typo_rate == 0 and c.ocr_confusable_rate == 0
                and c.whitespace_jitter_rate == 0 and c.code_switch_rate == 0)

    def _ocr_pairs(self) -> List[Tuple[str, str]]:
        """Read OCR pairs from seeds; multi-char patterns first so longer
        substrings match before single chars at the same cursor position.
        Returns an empty list when seeds is None.
        """
        if self.seeds is None:
            return []
        oc = self.seeds.shared.ocr_confusables
        return [(p.a, p.b) for p in oc.multi_char] + [(p.a, p.b) for p in oc.single_char]

    def _apply_scaffold(self, text: str, spans: List[Span]) -> Tuple[str, List[Span]]:
        """Perturb scaffold-zone characters; PII zones are copied verbatim."""
        c = self.config
        ocr_pairs = self._ocr_pairs()
        out: List[str] = []
        span_idx = 0
        new_starts: List[int] = []  # parallel to spans

        i = 0
        while i < len(text):
            # Inside a PII zone? Copy verbatim.
            while span_idx < len(spans) and i >= spans[span_idx].end:
                span_idx += 1
            if span_idx < len(spans) and spans[span_idx].start <= i < spans[span_idx].end:
                if i == spans[span_idx].start:
                    new_starts.append(len(out))
                out.append(text[i])
                i += 1
                continue

            ch = text[i]

            if ch == " " and self.rng.random() < c.whitespace_jitter_rate:
                if self.rng.random() < 0.5:
                    i += 1
                    continue
                else:
                    out.append("  ")
                    i += 1
                    continue

            if ch == " " and self.rng.random() < c.code_switch_rate:
                tag = self.rng.choice(_CODESWITCH_INSERTS.get(c.locale, ("",)))
                out.append(" " + tag)
                i += 1
                continue

            if self.rng.random() < c.typo_rate:
                neighbors_map = (
                    _CYRILLIC_NEIGHBORS if c.locale == "ru" else _LATIN_NEIGHBORS
                )
                lower = ch.lower()
                if lower in neighbors_map:
                    repl = self.rng.choice(neighbors_map[lower])
                    out.append(repl.upper() if ch.isupper() else repl)
                    i += 1
                    continue

            if ocr_pairs and self.rng.random() < c.ocr_confusable_rate:
                replaced = False
                for a, b in ocr_pairs:
                    if text[i:i + len(a)] == a:
                        out.append(b)
                        i += len(a)
                        replaced = True
                        break
                    if text[i:i + len(b)] == b:
                        out.append(a)
                        i += len(b)
                        replaced = True
                        break
                if replaced:
                    continue

            out.append(ch)
            i += 1

        new_text = "".join(out)

        # Re-anchor span offsets: each span's value was preserved verbatim,
        # so we can find each in order in the new text.
        cursor = 0
        new_spans: List[Span] = []
        for sp in spans:
            idx = new_text.find(sp.value, cursor)
            if idx < 0:
                idx = sp.start
            new_spans.append(Span(placeholder=sp.placeholder, value=sp.value,
                                  start=idx, end=idx + len(sp.value)))
            cursor = idx + len(sp.value)
        return new_text, new_spans

    def _apply_pii(self, text: str, spans: List[Span]) -> Tuple[str, List[Span]]:
        """Perturb PII zone contents; update span.value/end."""
        c = self.config
        ocr_pairs = self._ocr_pairs()
        new_spans: List[Span] = []
        for sp in spans:
            value = sp.value
            mutated_chars: List[str] = []
            for ch in value:
                if ch.isspace():
                    mutated_chars.append(ch)
                    continue
                if self.rng.random() < c.typo_rate and ch.isalpha():
                    neighbors_map = (
                        _CYRILLIC_NEIGHBORS if c.locale == "ru" else _LATIN_NEIGHBORS
                    )
                    lower = ch.lower()
                    if lower in neighbors_map:
                        repl = self.rng.choice(neighbors_map[lower])
                        mutated_chars.append(repl.upper() if ch.isupper() else repl)
                        continue
                if ocr_pairs and self.rng.random() < c.ocr_confusable_rate:
                    swapped = None
                    for a, b in ocr_pairs:
                        if len(a) == 1 and ch == a:
                            swapped = b
                            break
                        if len(b) == 1 and ch == b:
                            swapped = a
                            break
                    if swapped is not None:
                        mutated_chars.append(swapped)
                        continue
                mutated_chars.append(ch)
            new_spans.append(Span(
                placeholder=sp.placeholder,
                value="".join(mutated_chars),
                start=sp.start,
                end=sp.start + len(mutated_chars),
            ))
        return _rebase_pii_text(text, spans, new_spans)


def _rebase_pii_text(
    original_text: str, originals: Sequence[Span], mutated: Sequence[Span],
) -> Tuple[str, List[Span]]:
    """Splice mutated PII values back into the text and re-anchor offsets."""
    out_chars = list(original_text)
    # Apply mutations in reverse so later splices don't disturb earlier ones.
    for orig, mut in zip(reversed(list(originals)), reversed(list(mutated))):
        out_chars[orig.start:orig.end] = list(mut.value)

    new_text = "".join(out_chars)
    rebased: List[Span] = []
    delta = 0
    for orig, mut in zip(originals, mutated):
        new_start = orig.start + delta
        new_end = new_start + len(mut.value)
        assert new_text[new_start:new_end] == mut.value, (
            f"rebase mismatch: {new_text[new_start:new_end]!r} != {mut.value!r}"
        )
        rebased.append(Span(placeholder=mut.placeholder, value=mut.value,
                            start=new_start, end=new_end))
        delta += len(mut.value) - (orig.end - orig.start)
    return new_text, rebased
