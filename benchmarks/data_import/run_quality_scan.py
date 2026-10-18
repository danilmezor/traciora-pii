"""Run the quality scanner on imported eval-candidate datasets and write reports.

    PYTHONPATH=src python -m benchmarks.data_import.run_quality_scan --rows 10000

Outputs ``reports/quality_<slug>.md`` for each dataset and a top-level
``reports/quality_summary.md`` table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .loaders import load_nemotron_pii, load_wolframko_ru
from .quality import DatasetQuality, render, scan


def _load(slug: str, limit: int):
    if slug == "nemotron_en":
        return load_nemotron_pii(limit=limit)
    if slug == "wolframko_ru":
        return load_wolframko_ru(limit=limit)
    raise ValueError(slug)


def _summary_row(q: DatasetQuality) -> str:
    n = q.rows_scanned or 1
    issue_rate = 100 * q.rows_with_quality_issue / n
    # Aggregate anchor + format pass rates across labels
    anchor_t = anchor_y = fmt_t = fmt_y = 0
    junk = 0
    for lq in q.by_label.values():
        anchor_t += lq.spans_with_anchor + lq.spans_no_anchor
        anchor_y += lq.spans_with_anchor
        fmt_t += lq.spans_format_ok + sum(lq.spans_format_fail.values())
        fmt_y += lq.spans_format_ok
        junk += lq.spans_junk_value
    anchor_rate = f"{100 * anchor_y / anchor_t:.1f}%" if anchor_t else "n/a"
    fmt_rate = f"{100 * fmt_y / fmt_t:.1f}%" if fmt_t else "n/a"
    return (
        f"| `{q.dataset}` | {q.rows_scanned} | {issue_rate:.1f}% | "
        f"{anchor_rate} ({anchor_y}/{anchor_t}) | "
        f"{fmt_rate} ({fmt_y}/{fmt_t}) | {junk} | "
        f"{len(q.same_value_multi_label)} | "
        f"{len(q.same_row_format_collisions)} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "reports",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# Quality summary across datasets",
        "",
        "| Dataset | Rows | Rows with issue | Anchor rate | Format pass | Junk values | Leaked values | Collision pairs |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]

    for slug in ("nemotron_en", "wolframko_ru"):
        print(f"[quality] {slug}: scanning {args.rows} rows...", file=sys.stderr)
        t0 = time.perf_counter()
        q = scan(_load(slug, args.rows), slug)
        elapsed = time.perf_counter() - t0
        print(f"[quality] {slug}: scanned {q.rows_scanned} rows in {elapsed:.1f}s", file=sys.stderr)

        out = args.out_dir / f"quality_{slug}.md"
        out.write_text(render(q), encoding="utf-8")
        summary_lines.append(_summary_row(q))
        print(f"[quality] {slug}: report -> {out}", file=sys.stderr)

    summary_path = args.out_dir / "quality_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[quality] summary -> {summary_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
