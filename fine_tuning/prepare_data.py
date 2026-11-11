"""Convert a ``traciora_pii-bench-<locale>`` release into OPF training JSONL.

Input: a release directory produced by
``benchmarks.data_generator.build_release``, e.g.

    releases/v1/traciora_pii-bench-ru/
        train.jsonl   dev.jsonl   test.jsonl

Each row carries character-offset spans tagged by *placeholder*:

    {"text": "...", "spans": [{"placeholder": "[PERSON_NAME]", "start": 0, "end": 12, ...}, ...]}

OPF's training runner expects spans as a *mapping* keyed by OPF label:

    {"text": "...", "spans": {"private_person": [[0, 12]]}}

This module performs that conversion using the per-locale OPF policy
(``label_space.LabelSpace.placeholder_to_label``) as the placeholder→label
map. Spans whose placeholder isn't in the locale's OPF label space are
silently dropped — they remain ``O`` in the trained model. That's
deliberate: the OPF fine-tune targets the contextual labels regex can't
catch (PERSON_NAME, STREET_ADDRESS, free-text PHONE in oblique cases);
structured IDs are the regex layer's job.

Hard-negative records (``kind == "negative"`` from the generator) are
preserved with empty span maps so the model sees PII-shaped non-PII at
training time.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .label_space import LabelSpace, load_label_space, write_label_space_json

logger = logging.getLogger(__name__)


SPLIT_FILE_MAP = {
    "train": "train.jsonl",
    "validation": "dev.jsonl",  # bench calls it dev; OPF expects validation
    "test": "test.jsonl",
}


@dataclass
class PrepareReport:
    locale: str
    slug: str
    bench_dir: Path
    out_dir: Path
    rows_in: Dict[str, int]
    rows_out: Dict[str, int]
    spans_kept: Dict[str, int]
    spans_dropped: Dict[str, int]
    placeholder_kept_counts: Dict[str, int]
    placeholder_dropped_counts: Dict[str, int]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def _convert_record(
    record: dict,
    *,
    placeholder_to_label: Mapping[str, str],
    kept_counts: Counter,
    dropped_counts: Counter,
) -> Tuple[dict, int, int]:
    """Convert one bench row to OPF format. Returns (record, kept, dropped)."""
    text = record.get("text", "")
    raw_spans = record.get("spans") or []

    spans_by_label: Dict[str, List[List[int]]] = {}
    kept = 0
    dropped = 0
    for span in raw_spans:
        placeholder = span.get("placeholder")
        start = span.get("start")
        end = span.get("end")
        if placeholder is None or start is None or end is None:
            dropped += 1
            continue
        opf_label = placeholder_to_label.get(placeholder)
        if opf_label is None:
            dropped += 1
            dropped_counts[placeholder] += 1
            continue
        spans_by_label.setdefault(opf_label, []).append([int(start), int(end)])
        kept += 1
        kept_counts[placeholder] += 1

    out = {
        "text": text,
        "spans": spans_by_label,
    }
    # Pass through identifying metadata for traceability — OPF's runner
    # ignores unknown keys.
    if "id" in record:
        out["source_id"] = record["id"]
    if "locale" in record:
        out["locale"] = record["locale"]
    if "kind" in record:
        out["kind"] = record["kind"]
    return out, kept, dropped


def _convert_split(
    src: Path,
    dst: Path,
    *,
    placeholder_to_label: Mapping[str, str],
    kept_counts: Counter,
    dropped_counts: Counter,
) -> Tuple[int, int, int, int]:
    """Convert one split file. Returns (rows_in, rows_out, spans_kept, spans_dropped)."""
    rows_in = 0
    spans_kept = 0
    spans_dropped = 0

    def _gen() -> Iterable[dict]:
        nonlocal rows_in, spans_kept, spans_dropped
        for record in _read_jsonl(src):
            rows_in += 1
            converted, kept, dropped = _convert_record(
                record,
                placeholder_to_label=placeholder_to_label,
                kept_counts=kept_counts,
                dropped_counts=dropped_counts,
            )
            spans_kept += kept
            spans_dropped += dropped
            yield converted

    rows_out = _write_jsonl(dst, _gen())
    return rows_in, rows_out, spans_kept, spans_dropped


def run_prepare(
    *,
    locale: str,
    bench_dir: Path,
    out_dir: Path,
    slug: str,
) -> PrepareReport:
    """Read a bench release for one locale; emit OPF-format JSONL + label_space.json."""
    label_space: LabelSpace = load_label_space(locale)
    placeholder_to_label = label_space.placeholder_to_label

    if not bench_dir.is_dir():
        raise FileNotFoundError(
            f"Bench directory missing: {bench_dir}. "
            f"Build it with: python -m benchmarks.data_generator.build_release "
            f"--out releases/v1 --locales {locale}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    rows_in: Dict[str, int] = {}
    rows_out: Dict[str, int] = {}
    spans_kept: Dict[str, int] = {}
    spans_dropped: Dict[str, int] = {}
    placeholder_kept_counts: Counter = Counter()
    placeholder_dropped_counts: Counter = Counter()

    for opf_split, bench_filename in SPLIT_FILE_MAP.items():
        src = bench_dir / bench_filename
        dst = out_dir / f"{opf_split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing split file: {src}")
        ri, ro, sk, sd = _convert_split(
            src,
            dst,
            placeholder_to_label=placeholder_to_label,
            kept_counts=placeholder_kept_counts,
            dropped_counts=placeholder_dropped_counts,
        )
        rows_in[opf_split] = ri
        rows_out[opf_split] = ro
        spans_kept[opf_split] = sk
        spans_dropped[opf_split] = sd
        logger.info(
            "[%s] %s: rows=%d kept_spans=%d dropped_spans=%d -> %s",
            locale, opf_split, ri, sk, sd, dst,
        )

    label_path = out_dir / "label_space.json"
    write_label_space_json(label_path, locale=locale)

    manifest = {
        "slug": slug,
        "locale": locale,
        "policy": label_space.policy_name,
        "label_space_version": f"traciora_pii_v1_{locale}",
        "span_class_names": list(label_space.span_class_names),
        "placeholder_to_label": dict(placeholder_to_label),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "spans_kept": spans_kept,
        "spans_dropped": spans_dropped,
        "placeholder_kept_counts": dict(placeholder_kept_counts),
        "placeholder_dropped_counts": dict(placeholder_dropped_counts),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return PrepareReport(
        locale=locale,
        slug=slug,
        bench_dir=bench_dir,
        out_dir=out_dir,
        rows_in=rows_in,
        rows_out=rows_out,
        spans_kept=spans_kept,
        spans_dropped=spans_dropped,
        placeholder_kept_counts=dict(placeholder_kept_counts),
        placeholder_dropped_counts=dict(placeholder_dropped_counts),
    )
