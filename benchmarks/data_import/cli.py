"""CLI for fetching, transforming, and analyzing third-party PII datasets.

Examples
--------
Plain audit (no transforms)::

    python -m benchmarks.data_import.cli --dataset all --limit 5000

Combine source first-name + surname labels into PERSON_NAME::

    python -m benchmarks.data_import.cli --dataset all --transform combine_names

Quality-gate Nemotron, combine names, then project to supported labels::

    python -m benchmarks.data_import.cli --dataset nemotron_en \
        --transform quality_gate_nemotron,combine_names,project_allowed

Per-dataset allowlists are encoded in ``DATASETS`` below. The allowlist is
the set of dataset labels that map onto the framework's placeholder
taxonomy (see ``data_import/reports/label_mapping.md``); any row carrying
a span with a label outside the allowlist is dropped, because keeping it
would inflate FN counts against gold the detector cannot emit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

from .analyze import analyze
from .loaders import NormalizedExample, load_nemotron_pii, load_wolframko_ru
from .report import render
from .transforms import (
    CombineAddressStats,
    CombineStats,
    KeepOnlyAllowedStats,
    NemotronQualityGateStats,
    ProjectAllowedStats,
    combine_addresses,
    combine_names,
    keep_only_allowed_labels,
    project_allowed_labels,
    quality_gate_nemotron,
)


# Per-dataset config. ``allowed_labels`` is the set of dataset labels that
# map cleanly onto a traciora_pii placeholder. ``keep_only_allowed`` uses
# it as a strict row filter; ``project_allowed`` uses it as a span projection.
#
# These sets are derived from ``data_import/reports/label_mapping.md`` and
# reflect the framework state after the DATE rename + PASSPORTNUM additions
# + dashless SSN support.
DATASETS = {
    "nemotron_en": {
        "loader": lambda limit: load_nemotron_pii(limit=limit),
        "source": "nemotron",
        "locale": "en",
        "allowed_labels": frozenset({
            # Source labels before transforms, plus normalized labels emitted
            # by quality_gate_nemotron / combine_names.
            "first_name", "last_name", "PERSON_NAME",
            "email",
            "phone_number",
            "credit_debit_card",
            "ssn",
            "date", "date_of_birth",
            "street_address",
            "url",
            "ipv4",
            "license_plate",
            "vehicle_identifier",        # -> [VIN]
            "password", "api_key",
        }),
    },
    # ai4privacy en/de were audited and disqualified for primary eval. Keep
    # loader support in loaders.py for reproducibility, but exclude them from
    # the default import/eval surface.
    "wolframko_ru": {
        "loader": lambda limit: load_wolframko_ru(limit=limit),
        "source": "wolframko",
        "locale": "ru",
        "allowed_labels": frozenset({
            "GIVENNAME", "SURNAME", "PERSON_NAME",
            "DATEOFBIRTH",              # -> [DATE]
            "EMAIL",
            "TELEPHONENUM",             # -> [PHONE]
            "CREDITCARDNUMBER",
            "TAXNUM",                   # -> [RU_INN]
            "SOCIALNUM",                # -> [RU_SNILS]
            "IDCARDNUM",                # -> [PASSPORTNUM] (RU internal passport)
            "DRIVERLICENSENUM",         # -> [RU_DRV_LICENSE]
            "STREET", "BUILDINGNUM", "CITY", "ZIPCODE", "STREET_ADDRESS",
        }),
    },
}

KNOWN_TRANSFORMS = (
    "quality_gate_nemotron",
    "combine_names",
    "combine_addresses",
    "project_allowed",
    "keep_only_allowed",
)


def _apply_transforms(
    stream: Iterable[NormalizedExample],
    transforms: list[str],
    spec: dict,
) -> tuple[Iterator[NormalizedExample], list[tuple[str, object]]]:
    """Compose the requested transforms in order. Returns (stream, [(name, stats), ...])."""
    applied: list[tuple[str, object]] = []
    current: Iterable[NormalizedExample] = stream
    for name in transforms:
        if name == "combine_names":
            current, stats = combine_names(current)
        elif name == "combine_addresses":
            current, stats = combine_addresses(current)
        elif name == "quality_gate_nemotron":
            current, stats = quality_gate_nemotron(current)
        elif name == "project_allowed":
            current, stats = project_allowed_labels(
                current, allowlist=spec["allowed_labels"]
            )
        elif name == "keep_only_allowed":
            current, stats = keep_only_allowed_labels(
                current, allowlist=spec["allowed_labels"]
            )
        else:
            raise ValueError(f"unknown transform: {name!r}")
        applied.append((name, stats))
    return iter(current), applied


def _stats_section(name: str, stats: object, spec: dict) -> str:
    if isinstance(stats, CombineStats):
        survival = (100.0 * stats.rows_out / stats.rows_in) if stats.rows_in else 0.0
        return (
            f"## `combine_names` - attrition\n\n"
            f"- Rows in: **{stats.rows_in:,}**\n"
            f"- Rows out: **{stats.rows_out:,}** ({survival:.1f}% survival)\n"
            f"- Dropped, only-one-kind (had GIVENNAME xor SURNAME): **{stats.rows_dropped_only_one_kind:,}**\n"
            f"- Dropped, unpaired/separated names: **{stats.rows_dropped_unpaired:,}**\n"
            f"- PERSON_NAME pairs merged: **{stats.pairs_merged:,}**\n"
        )
    if isinstance(stats, CombineAddressStats):
        survival = (100.0 * stats.rows_out / stats.rows_in) if stats.rows_in else 0.0
        return (
            f"## `combine_addresses` - attrition\n\n"
            f"- Rows in: **{stats.rows_in:,}**\n"
            f"- Rows out: **{stats.rows_out:,}** ({survival:.1f}% survival)\n"
            f"- Dropped, scattered (multiple address clusters): **{stats.rows_dropped_scattered:,}**\n"
            f"- Pass-through, no address components: **{stats.rows_no_address:,}**\n"
            f"- Addresses merged: **{stats.addresses_merged:,}**\n"
        )
    if isinstance(stats, KeepOnlyAllowedStats):
        survival = (100.0 * stats.rows_out / stats.rows_in) if stats.rows_in else 0.0
        triggers = ", ".join(
            f"`{label}`={count}"
            for label, count in stats.label_trigger_counts.most_common()
        ) or "(none)"
        allow = ", ".join(f"`{l}`" for l in sorted(spec["allowed_labels"]))
        return (
            f"## `keep_only_allowed` - attrition\n\n"
            f"- Allowlist for this dataset: {allow}\n"
            f"- Rows in: **{stats.rows_in:,}**\n"
            f"- Rows out: **{stats.rows_out:,}** ({survival:.1f}% survival)\n"
            f"- Rows dropped: **{stats.rows_dropped:,}**\n"
            f"- Dropped because of out-of-allowlist label: {triggers}\n"
        )
    if isinstance(stats, NemotronQualityGateStats):
        survival = (100.0 * stats.rows_out / stats.rows_in) if stats.rows_in else 0.0
        reasons = ", ".join(
            f"`{reason}`={count}"
            for reason, count in stats.reason_counts.most_common()
        ) or "(none)"
        rewrites = ", ".join(
            f"`{label}`={count}"
            for label, count in stats.labels_rewritten.most_common()
        ) or "(none)"
        return (
            f"## `quality_gate_nemotron` - attrition\n\n"
            f"- Rows in: **{stats.rows_in:,}**\n"
            f"- Rows out: **{stats.rows_out:,}** ({survival:.1f}% survival)\n"
            f"- Rows dropped: **{stats.rows_dropped:,}**\n"
            f"- Drop reasons: {reasons}\n"
            f"- Label rewrites: {rewrites}\n"
        )
    if isinstance(stats, ProjectAllowedStats):
        survival = (100.0 * stats.rows_out / stats.rows_in) if stats.rows_in else 0.0
        triggers = ", ".join(
            f"`{label}`={count}"
            for label, count in stats.label_dropped_counts.most_common()
        ) or "(none)"
        allow = ", ".join(f"`{l}`" for l in sorted(spec["allowed_labels"]))
        return (
            f"## `project_allowed` - span projection\n\n"
            f"- Allowlist for this dataset: {allow}\n"
            f"- Rows in: **{stats.rows_in:,}**\n"
            f"- Rows out: **{stats.rows_out:,}** ({survival:.1f}% survival)\n"
            f"- Rows dropped with no allowed spans: **{stats.rows_dropped_no_allowed_spans:,}**\n"
            f"- Spans dropped: **{stats.spans_dropped:,}**\n"
            f"- Dropped labels: {triggers}\n"
        )
    return f"## `{name}` - (no stats renderer)\n"


def _run_one(
    name: str,
    limit: int | None,
    out_dir: Path,
    transforms: list[str],
) -> Path:
    spec = DATASETS[name]
    print(f"[data_import] {name}: streaming...", file=sys.stderr)
    stream = spec["loader"](limit)

    suffix = ""
    applied: list[tuple[str, object]] = []
    if transforms:
        stream, applied = _apply_transforms(stream, transforms, spec)
        suffix = "_" + "_".join(transforms)

    analysis = analyze(stream, source=spec["source"], locale=spec["locale"])
    md = render(analysis)
    if applied:
        prelude = "\n".join(_stats_section(n, s, spec) for n, s in applied)
        md = prelude + "\n" + md

    out_path = out_dir / f"{name}{suffix}.md"
    out_path.write_text(md, encoding="utf-8")
    print(
        f"[data_import] {name}: {analysis.n_rows} rows analyzed, "
        f"{analysis.quality.total_spans} spans -> {out_path}",
        file=sys.stderr,
    )
    return out_path


def _parse_transforms(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if p not in KNOWN_TRANSFORMS:
            raise SystemExit(
                f"unknown transform {p!r}; known: {','.join(KNOWN_TRANSFORMS)}"
            )
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import + audit third-party PII datasets.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Which dataset to pull. 'all' runs every entry in DATASETS.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Cap rows analyzed per dataset (0 = no cap). Default 2000 for fast smoke runs.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        help=(
            "Comma-separated transform pipeline applied before analysis. "
            f"Known: {','.join(KNOWN_TRANSFORMS)}. "
            "Order matters; 'quality_gate_nemotron,combine_names,project_allowed' "
            "is canonical for nemotron_en."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "reports",
        help="Where to write per-dataset Markdown reports.",
    )
    args = parser.parse_args(argv)

    transforms = _parse_transforms(args.transform)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    limit = None if args.limit == 0 else args.limit

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        _run_one(name, limit, args.out_dir, transforms)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
