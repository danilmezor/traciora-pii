"""Distribution statistics for generated benchmark releases.

Used by both:
  * Per-locale ``datasheet.md`` (placeholder + category breakdown).
  * Top-level ``RELEASE_CARD.md`` (cross-locale summary).

The output is deterministic Markdown so two runs against the same
metadata produce byte-identical files.
"""
from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .pipeline import GeneratedExample


# ======================================================================
# Distribution record
# ======================================================================


@dataclass
class Distribution:
    """Aggregate statistics about a list of generated examples."""

    locale: str
    n_total: int = 0
    n_positive: int = 0
    n_negative: int = 0

    by_category: Dict[str, int] = field(default_factory=dict)
    by_placeholder: Dict[str, int] = field(default_factory=dict)
    by_kind: Dict[str, int] = field(default_factory=dict)

    n_with_prepended_filler: int = 0
    n_with_appended_filler: int = 0

    text_len_p50: int = 0
    text_len_p95: int = 0
    text_len_max: int = 0

    spans_per_positive_p50: float = 0.0
    spans_per_positive_max: int = 0

    @property
    def negative_ratio(self) -> float:
        return self.n_negative / self.n_total if self.n_total else 0.0

    @property
    def filler_prepend_rate(self) -> float:
        return self.n_with_prepended_filler / self.n_positive if self.n_positive else 0.0

    @property
    def filler_append_rate(self) -> float:
        return self.n_with_appended_filler / self.n_positive if self.n_positive else 0.0


def compute_distribution(
    examples: Sequence[GeneratedExample], locale: str,
) -> Distribution:
    """Compute every aggregate stat for the per-locale datasheet."""
    by_cat: collections.Counter = collections.Counter()
    by_ph: collections.Counter = collections.Counter()
    by_kind: collections.Counter = collections.Counter()

    text_lens: List[int] = []
    spans_per_pos: List[int] = []
    n_pre = 0
    n_app = 0
    n_pos = 0
    n_neg = 0

    for ex in examples:
        text_lens.append(len(ex.text))
        by_cat[ex.category] += 1
        by_kind[ex.kind] += 1
        for sp in ex.spans:
            by_ph[sp.placeholder] += 1
        if ex.kind == "positive":
            n_pos += 1
            spans_per_pos.append(len(ex.spans))
            if ex.fillers_prepended:
                n_pre += 1
            if ex.fillers_appended:
                n_app += 1
        else:
            n_neg += 1

    def _pct(values: List[int], q: float) -> int:
        if not values:
            return 0
        s = sorted(values)
        idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[idx]

    return Distribution(
        locale=locale,
        n_total=len(examples),
        n_positive=n_pos,
        n_negative=n_neg,
        by_category=dict(by_cat),
        by_placeholder=dict(by_ph),
        by_kind=dict(by_kind),
        n_with_prepended_filler=n_pre,
        n_with_appended_filler=n_app,
        text_len_p50=_pct(text_lens, 0.50),
        text_len_p95=_pct(text_lens, 0.95),
        text_len_max=max(text_lens) if text_lens else 0,
        spans_per_positive_p50=(
            statistics.median(spans_per_pos) if spans_per_pos else 0.0
        ),
        spans_per_positive_max=max(spans_per_pos) if spans_per_pos else 0,
    )


# ======================================================================
# Markdown rendering
# ======================================================================


def render_distribution_md(d: Distribution) -> str:
    """Render a per-locale Markdown distribution section.

    Designed to slot into the bottom of the per-locale ``datasheet.md`` so
    the audit-time view of "what's in this artifact" is the same view
    reviewers see in the paper appendix.
    """
    cat_rows = "\n".join(
        f"| `{cat}` | {n} | {n / d.n_total * 100:.1f}% |"
        for cat, n in sorted(d.by_category.items(), key=lambda kv: -kv[1])
    )
    ph_rows = "\n".join(
        f"| `{ph}` | {n} |"
        for ph, n in sorted(d.by_placeholder.items(), key=lambda kv: -kv[1])
    ) or "| (no spans) | 0 |"

    return (
        f"## Distribution\n"
        f"\n"
        f"### Mix\n"
        f"\n"
        f"| Kind | Count | Share |\n"
        f"|---|---|---|\n"
        f"| Positive | {d.n_positive} | {d.n_positive / d.n_total * 100:.1f}% |\n"
        f"| Negative (hard) | {d.n_negative} | {d.negative_ratio * 100:.1f}% |\n"
        f"\n"
        f"### Category breakdown\n"
        f"\n"
        f"| Category | Count | Share of total |\n"
        f"|---|---|---|\n"
        f"{cat_rows}\n"
        f"\n"
        f"### Placeholder coverage (positives only)\n"
        f"\n"
        f"| Placeholder | Span count |\n"
        f"|---|---|\n"
        f"{ph_rows}\n"
        f"\n"
        f"### Filler usage (positives only)\n"
        f"\n"
        f"| Side | Examples carrying a filler | Rate |\n"
        f"|---|---|---|\n"
        f"| Prepended | {d.n_with_prepended_filler} | {d.filler_prepend_rate * 100:.1f}% |\n"
        f"| Appended  | {d.n_with_appended_filler} | {d.filler_append_rate * 100:.1f}% |\n"
        f"\n"
        f"### Length and density\n"
        f"\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Text length p50 | {d.text_len_p50} chars |\n"
        f"| Text length p95 | {d.text_len_p95} chars |\n"
        f"| Text length max | {d.text_len_max} chars |\n"
        f"| Spans per positive p50 | {d.spans_per_positive_p50:.1f} |\n"
        f"| Spans per positive max | {d.spans_per_positive_max} |\n"
    )


def render_release_card(distributions: Dict[str, Distribution], *, version: str,
                        license: str, git_sha: str) -> str:
    """Top-level release card summarising every locale in one table."""
    if not distributions:
        return "# (empty release)\n"

    locales = sorted(distributions.keys())
    rows = []
    for loc in locales:
        d = distributions[loc]
        rows.append(
            f"| `{loc}` | {d.n_total} | {d.n_positive} | {d.n_negative} | "
            f"{len(d.by_category)} | {len(d.by_placeholder)} | "
            f"{d.text_len_p50}/{d.text_len_p95}/{d.text_len_max} |"
        )
    table = "\n".join(rows)

    return (
        f"# traciora_pii-bench {version}\n"
        f"\n"
        f"Hybrid synthetic PII benchmark, slot-template architecture.\n"
        f"Generator commit: `{git_sha or 'unknown'}` · License: `{license}`\n"
        f"\n"
        f"## Per-locale summary\n"
        f"\n"
        f"| Locale | Total | Positives | Negatives | Categories | Placeholders | Text length p50/p95/max |\n"
        f"|---|---|---|---|---|---|---|\n"
        f"{table}\n"
        f"\n"
        f"See each locale's `datasheet.md` for the full distribution breakdown,\n"
        f"and `MANIFEST.json` for sha256 of every released file.\n"
    )
