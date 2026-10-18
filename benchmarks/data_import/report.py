"""Render an :class:`Analysis` as deterministic Markdown."""
from __future__ import annotations

from .analyze import Analysis
from .loaders import NormalizedExample


_SUMMARY_KEYS = ("min", "p25", "p50", "p75", "p95", "max", "mean")


def _fmt_summary(s: dict[str, float]) -> str:
    cells = " | ".join(f"{s[k]:.1f}" for k in _SUMMARY_KEYS)
    return f"| {cells} |"


def _summary_table(title: str, summary: dict[str, float]) -> str:
    head = "| " + " | ".join(_SUMMARY_KEYS) + " |"
    sep = "|" + "|".join(["---"] * len(_SUMMARY_KEYS)) + "|"
    return f"### {title}\n\n{head}\n{sep}\n{_fmt_summary(summary)}\n"


def _label_table(an: Analysis) -> str:
    total = sum(an.label_counts.values()) or 1
    rows = ["| Label | Count | % of spans | Example value |", "|---|---:|---:|---|"]
    for label, n in an.label_counts.most_common():
        pct = 100.0 * n / total
        ex = (an.label_examples.get(label) or "").replace("|", "\\|").replace("\n", " ")
        if len(ex) > 60:
            ex = ex[:57] + "..."
        rows.append(f"| `{label}` | {n} | {pct:.2f} | {ex} |")
    return "\n".join(rows)


def _quality_table(an: Analysis) -> str:
    q = an.quality
    n = an.n_rows or 1

    def pct(x: int) -> str:
        return f"{100.0 * x / n:.2f}%"

    rows = [
        "| Check | Rows | % of rows | Notes |",
        "|---|---:|---:|---|",
        f"| Rows with **zero spans** | {q.rows_with_no_spans} | {pct(q.rows_with_no_spans)} | Unusable for recall measurement |",
        f"| Rows with **value/text mismatch** | {q.rows_with_value_mismatch} | {pct(q.rows_with_value_mismatch)} | `text[start:end] != value` ({q.total_value_mismatches} bad spans) |",
        f"| Rows with **out-of-bounds spans** | {q.rows_with_out_of_bounds_span} | {pct(q.rows_with_out_of_bounds_span)} | Offsets exceed text length ({q.total_out_of_bounds} bad spans) |",
        f"| Rows with **overlapping spans** | {q.rows_with_overlapping_spans} | {pct(q.rows_with_overlapping_spans)} | {q.total_overlap_pairs} overlap pairs total |",
        f"| Rows with **dropped spans at load** | {q.rows_with_dropped_spans} | {pct(q.rows_with_dropped_spans)} | Span dict missing required fields |",
        f"| Rows with **non-NFC text** | {q.rows_with_non_nfc_text} | {pct(q.rows_with_non_nfc_text)} | Inconsistent unicode normalization |",
        f"| Rows with **control characters** | {q.rows_with_control_chars} | {pct(q.rows_with_control_chars)} | Non-whitespace Cc category |",
    ]
    return "\n".join(rows)


def _format_example(ex: NormalizedExample, *, max_text: int = 400) -> str:
    text = ex.text
    if len(text) > max_text:
        text = text[: max_text - 3] + "..."
    text_md = text.replace("`", "\\`").replace("\n", "\\n")
    span_lines = []
    for sp in ex.spans[:20]:
        val = sp.value.replace("|", "\\|").replace("\n", " ")
        if len(val) > 50:
            val = val[:47] + "..."
        span_lines.append(f"  - `{sp.label}` [{sp.start}:{sp.end}] = `{val}`")
    if len(ex.spans) > 20:
        span_lines.append(f"  - ... and {len(ex.spans) - 20} more spans")

    return (
        f"**id**: `{ex.id}`  •  **spans**: {len(ex.spans)}\n\n"
        f"```\n{text_md}\n```\n\n"
        + "\n".join(span_lines)
    )


def render(an: Analysis) -> str:
    parts: list[str] = []
    parts.append(f"# {an.source} ({an.locale}) - dataset audit\n")
    parts.append(f"Rows analyzed: **{an.n_rows:,}**\n")
    parts.append(f"Total spans: **{an.quality.total_spans:,}**\n")

    parts.append("## Length distributions\n")
    parts.append(_summary_table("Text length (characters)", an.text_len_chars))
    parts.append(_summary_table("Text length (whitespace tokens)", an.text_len_words))
    parts.append(_summary_table("Spans per row", an.spans_per_row))
    parts.append(_summary_table("Span length (characters)", an.span_len_chars))

    parts.append("## Label distribution\n")
    parts.append(_label_table(an))
    parts.append("")

    parts.append("## Span-level quality checks\n")
    parts.append(_quality_table(an))
    parts.append("")

    if an.flagged:
        parts.append("## Flagged examples (one per failure mode)\n")
        for tag in sorted(an.flagged):
            parts.append(f"### `{tag}`\n")
            parts.append(_format_example(an.flagged[tag]))
            parts.append("")

    if an.samples:
        parts.append("## Random samples\n")
        for ex in an.samples:
            parts.append(_format_example(ex))
            parts.append("")

    return "\n".join(parts) + "\n"
