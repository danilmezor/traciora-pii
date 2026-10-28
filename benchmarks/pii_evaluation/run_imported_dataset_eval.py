"""Evaluate the framework's regex + second-pass detector against an
imported HF dataset.

Pulls a dataset through ``benchmarks.data_import``, applies the canonical
transform chain for that dataset, maps source labels to traciora_pii
placeholders, runs the full detection pipeline (regex first, then OPF
over the same raw text with regex-overlap suppression), and scores
per-placeholder precision / recall via span-overlap matching against
the dataset's gold.

This is the harness for the independent-evaluation answer to "evaluate
your detector on something you didn't generate." The ``--dataset`` choice
and label maps are declared at the top.

Example::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
        --dataset wolframko_ru --target-rows 200 --raw-limit 500

    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
        --dataset nemotron_en --target-rows 200 --raw-limit 800

    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
        --dataset hivetrace_ru --detector regex_encoder --target-rows 1810 --raw-limit 1810
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmarks.pii_evaluation.cli_common import (
    add_detector_llm_flags,
    detector_llm_overrides,
)
from benchmarks.pii_evaluation.metrics import LENGTH_BUCKETS, length_bucket_for_text


# --- label maps from imported-dataset label -> framework placeholder ----
# Keys are the labels carried by NormalizedExample after the transform
# chain (combine_names emits PERSON_NAME; combine_addresses emits
# STREET_ADDRESS). Values are the bracketed framework placeholder strings
# the regex / OPF detectors emit.
WOLFRAMKO_RU_LABEL_MAP: Dict[str, str] = {
    "PERSON_NAME": "[PERSON_NAME]",
    "DATEOFBIRTH": "[DATE]",
    "EMAIL": "[EMAIL]",
    "TELEPHONENUM": "[PHONE]",
    "CREDITCARDNUMBER": "[CARD]",
    "TAXNUM": "[RU_INN]",
    "SOCIALNUM": "[RU_SNILS]",
    # Passport + driver's license collapsed into [PERSONAL_ID] — both
    # share the XX YY ZZZZZZ shape and were structurally indistinguishable.
    "IDCARDNUM": "[PERSONAL_ID]",
    "DRIVERLICENSENUM": "[PERSONAL_ID]",
    "STREET_ADDRESS": "[STREET_ADDRESS]",
}


NEMOTRON_EN_LABEL_MAP: Dict[str, str] = {
    "PERSON_NAME": "[PERSON_NAME]",
    "email": "[EMAIL]",
    "phone_number": "[PHONE]",
    "credit_debit_card": "[CARD]",
    # SSN collapsed into [PERSONAL_ID] — 9-digit raw SSN and 9-digit
    # US passport are structurally indistinguishable.
    "ssn": "[PERSONAL_ID]",
    "date": "[DATE]",
    "date_of_birth": "[DATE]",
    "street_address": "[STREET_ADDRESS]",
    "url": "[URL]",
    "ipv4": "[IP]",
    "license_plate": "[LICENSE_PLATE]",
    "vehicle_identifier": "[VIN]",
    "password": "[SECRET]",
    "api_key": "[SECRET]",
}


# 5-label clean-subset projection of ai4privacy_de. The full ai4privacy
# DE label set has known format/context failures on the structured-ID
# classes (DE_STEUER_ID, DE_PERSONALAUSWEIS). Per the audit at
# ``benchmarks/data_import/reports/ai4privacy_de.md`` these labels pass:
# zero span-level quality failures, no value/text mismatches.
# Source labels post-transform: combine_names emits PERSON_NAME from
# GIVENNAME+SURNAME; combine_addresses emits STREET_ADDRESS from
# STREET+BUILDINGNUM+CITY+ZIPCODE; EMAIL/TELEPHONENUM/DATE are unchanged.
# DATE gold spans whose value doesn't match the DE regex layer's date
# pattern (German month-name forms like ``Mai/04`` or ``1. September
# 1976``) are dropped at load time — see _filter_unsupported_date_spans.
AI4PRIVACY_DE_LABEL_MAP: Dict[str, str] = {
    "PERSON_NAME": "[PERSON_NAME]",
    "EMAIL": "[EMAIL]",
    "TELEPHONENUM": "[PHONE]",
    "STREET_ADDRESS": "[STREET_ADDRESS]",
    "DATE": "[DATE]",
}


HIVETRACE_RU_LABEL_MAP: Dict[str, str] = {
    "NAME": "[PERSON_NAME]",
    "ADDRESS": "[STREET_ADDRESS]",
}


DATASETS = {
    "nemotron_en": {
        "locale": "en",
        "opf_policy": "en_comprehensive",
        "label_map": NEMOTRON_EN_LABEL_MAP,
        "allowed_labels": frozenset(NEMOTRON_EN_LABEL_MAP) | {
            "first_name", "last_name",  # pre-combine_names
        },
    },
    "wolframko_ru": {
        "locale": "ru",
        "opf_policy": "ru_comprehensive",
        "label_map": WOLFRAMKO_RU_LABEL_MAP,
        "allowed_labels": frozenset(WOLFRAMKO_RU_LABEL_MAP) | {
            "GIVENNAME", "SURNAME",  # pre-combine_names
            "STREET", "BUILDINGNUM", "CITY", "ZIPCODE",  # pre-combine_addresses
        },
    },
    "ai4privacy_de": {
        "locale": "de",
        "opf_policy": "de_comprehensive",
        "label_map": AI4PRIVACY_DE_LABEL_MAP,
        "allowed_labels": frozenset(AI4PRIVACY_DE_LABEL_MAP) | {
            "GIVENNAME", "SURNAME",  # pre-combine_names
            "STREET", "BUILDINGNUM", "CITY", "ZIPCODE",  # pre-combine_addresses
        },
    },
    "hivetrace_ru": {
        "locale": "ru",
        "opf_policy": "ru_comprehensive",
        "label_map": HIVETRACE_RU_LABEL_MAP,
        "allowed_labels": frozenset(HIVETRACE_RU_LABEL_MAP),
    },
    # Internal synthetic held-out test split. Sources from
    # ``fine_tuning/data/<slug>/test.jsonl`` written by
    # ``fine_tuning.prepare_data``. The slug is supplied at runtime via
    # ``--internal-slug``; locale, policy, and label_map are derived from
    # the slug's manifest.json. Stub values here are placeholders that
    # _load_examples overwrites before scoring runs.
    "internal_synthetic": {
        "locale": "_dynamic_",
        "opf_policy": "_dynamic_",
        "label_map": {},
        "allowed_labels": frozenset(),
    },
}


RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class GoldSpan:
    placeholder: str
    start: int
    end: int
    value: str


@dataclass
class FindingRecord:
    placeholder: str
    start: int
    end: int
    value: str
    source: str  # "regex" | "opf"


@dataclass
class PerPlaceholder:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class EvalRun:
    dataset: str
    rows_evaluated: int = 0
    rows_with_zero_gold: int = 0
    elapsed_seconds: float = 0.0
    counts: Dict[str, PerPlaceholder] = field(default_factory=lambda: defaultdict(PerPlaceholder))
    # First few mismatch examples per category, for debugging / paper appendix.
    fp_examples: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    fn_examples: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    # Per-length-bucket counters (sentence / paragraph / multi_paragraph /
    # structured). Buckets the synthetic eval already breaks down by; mirrored
    # here so external-eval reports carry the same surface.
    bucket_counts: Dict[str, Dict[str, PerPlaceholder]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(PerPlaceholder))
    )
    bucket_rows: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Per-row records for bootstrap CIs. Each entry is a dict
    # {placeholder: (tp, fp, fn)} for that one row, plus the row's length
    # bucket (so the bootstrap can filter per-bucket subsets without
    # mutating shared state).
    per_row_records: List[Dict[str, Tuple[int, int, int]]] = field(default_factory=list)
    per_row_buckets: List[str] = field(default_factory=list)
    # Projection-symmetric eval bookkeeping. ``gold_placeholders`` is the
    # set of placeholders any gold span can carry for this dataset; when
    # ``projection_symmetric`` is on, findings outside this set are
    # dropped before scoring so detectors that emit a broader label
    # taxonomy than the corpus annotates are not penalised for it.
    gold_placeholders: List[str] = field(default_factory=list)
    projection_symmetric: bool = True


# -----------------------------------------------------------------------
# Pipeline plumbing
# -----------------------------------------------------------------------


def _setup_opf_env(policy_name: str) -> None:
    # PII_OPF_ENABLED / PII_OPF_POLICY were removed in the routed-OPF cleanup;
    # second_pass and per-entry policy now live in detector.opf.models.<name>
    # in traciora.yaml. We only nudge the device default here.
    os.environ.setdefault("PII_OPF_DEVICE", "cpu")


def _load_examples(
    dataset_slug: str,
    raw_limit: int,
    target_rows: int,
    *,
    internal_slug: Optional[str] = None,
) -> List[Any]:
    """Pull raw rows, apply the canonical transform chain, return up to
    ``target_rows`` :class:`NormalizedExample`. ``raw_limit`` is the
    pre-pipeline cap; pick larger than ``target_rows / typical_survival``.

    For ``internal_synthetic`` the loader bypasses the data_import chain
    and reads the held-out test split that ``fine_tuning.prepare_data``
    emits, mutating ``DATASETS[dataset_slug]`` in place to set the locale
    / policy / label_map / allowed_labels derived from the slug's manifest.
    """
    from benchmarks.data_import.loaders import (
        NormalizedExample,
        Span,
        load_ai4privacy,
        load_hivetrace_ru,
        load_nemotron_pii,
        load_wolframko_ru,
    )
    from benchmarks.data_import.transforms import (
        combine_addresses,
        combine_names,
        keep_only_allowed_labels,
        project_allowed_labels,
        quality_gate_nemotron,
    )

    spec = DATASETS[dataset_slug]
    if dataset_slug == "wolframko_ru":
        raw = load_wolframko_ru(limit=raw_limit)
        s1, _ = combine_names(raw)
        s2, _ = combine_addresses(s1)
        s3, _ = keep_only_allowed_labels(s2, allowlist=spec["allowed_labels"])
    elif dataset_slug == "hivetrace_ru":
        # Already consolidated: one NAME / ADDRESS span per entity, no combine steps.
        raw = load_hivetrace_ru(limit=raw_limit)
        s3, _ = keep_only_allowed_labels(raw, allowlist=spec["allowed_labels"])
    elif dataset_slug == "nemotron_en":
        raw = load_nemotron_pii(limit=raw_limit)
        s0, _ = quality_gate_nemotron(raw)
        s1, _ = combine_names(s0)
        s3, _ = project_allowed_labels(s1, allowlist=spec["allowed_labels"])
    elif dataset_slug == "ai4privacy_de":
        # 5-label clean-subset projection: PERSON_NAME, EMAIL, PHONE,
        # STREET_ADDRESS, DATE. The structured-ID classes (DE_STEUER_ID,
        # DE_PERSONALAUSWEIS) are known broken in this dataset per the
        # audit and are deliberately not in AI4PRIVACY_DE_LABEL_MAP.
        # DATE gold spans are filtered to formats the regex layer
        # actually supports — see _filter_unsupported_date_spans.
        raw = load_ai4privacy(locale="de", limit=raw_limit)
        s1, _ = combine_names(raw)
        s2, _ = combine_addresses(s1)
        s3a, _ = project_allowed_labels(s2, allowlist=spec["allowed_labels"])
        s3 = _filter_unsupported_date_spans(s3a, locale="de")
    elif dataset_slug == "internal_synthetic":
        if not internal_slug:
            raise ValueError(
                "--internal-slug is required when --dataset internal_synthetic. "
                "Example: --internal-slug ru-v1"
            )
        s3 = _load_internal_synthetic(internal_slug, spec)
    else:
        raise NotImplementedError(f"loader for {dataset_slug!r} not wired yet")

    out: list = []
    for ex in s3:
        out.append(ex)
        if len(out) >= target_rows:
            break
    return out


def _load_internal_synthetic(slug: str, spec: Dict[str, Any]) -> Iterable[Any]:
    """Read OPF-format test.jsonl for ``slug`` and yield NormalizedExample.

    Mutates ``spec`` in place: locale, opf_policy, label_map,
    allowed_labels are derived from the slug's manifest.json. The
    label_map inverts the manifest's ``placeholder_to_label`` so each
    OPF label in the test split maps back to its runtime placeholder.

    Test rows whose ``kind`` is ``"negative"`` (hard-negatives) yield an
    example with an empty span list — scoring still counts FPs on them,
    which is exactly the PII-shaped-non-PII measurement we want.
    """
    import json as _json
    from benchmarks.data_import.loaders import NormalizedExample, Span

    data_dir = Path(__file__).resolve().parents[1] / ".." / "fine_tuning" / "data" / slug
    data_dir = data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    test_path = data_dir / "test.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest missing at {manifest_path}. Run "
            f"`python -m fine_tuning prepare --slug {slug} --locale <en|de|ru>` first."
        )
    if not test_path.exists():
        raise FileNotFoundError(f"test.jsonl missing at {test_path}")

    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    locale = manifest["locale"]
    placeholder_to_label = manifest["placeholder_to_label"]
    # Invert: OPF label seen in test.jsonl -> placeholder we score against.
    label_map = {opf_label: placeholder for placeholder, opf_label in placeholder_to_label.items()}

    spec["locale"] = locale
    spec["opf_policy"] = f"{locale}_comprehensive"
    spec["label_map"] = label_map
    spec["allowed_labels"] = frozenset(label_map)

    print(
        f"[eval] internal_synthetic slug={slug} locale={locale} "
        f"labels={len(label_map)}",
        file=sys.stderr,
    )

    with test_path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            text = row.get("text", "")
            row_spans = row.get("spans") or {}
            spans: list = []
            for opf_label, ranges in row_spans.items():
                for span_range in ranges:
                    if len(span_range) < 2:
                        continue
                    start, end = int(span_range[0]), int(span_range[1])
                    if start < 0 or end > len(text) or start >= end:
                        continue
                    spans.append(Span(
                        label=opf_label,
                        value=text[start:end],
                        start=start,
                        end=end,
                    ))
            yield NormalizedExample(
                id=str(row.get("source_id") or f"{slug}-row-{line_idx}"),
                locale=row.get("locale") or locale,
                text=text,
                spans=spans,
                source=f"internal_synthetic:{slug}",
                meta={"kind": row.get("kind", "")},
            )


def _filter_unsupported_date_spans(examples: Iterable[Any], *, locale: str) -> Iterable[Any]:
    """Drop DATE gold spans whose value doesn't match the locale's regex
    layer's date pattern.

    ai4privacy_de DATE includes formats our regex never claimed to
    handle — German month-name forms (``Mai/04``, ``März/90``,
    ``1. September 1976``, ``Februar/62``). Counting those as FNs would
    punish the detector for capability it doesn't advertise. Drop them
    at load time instead.

    Span-level (not row-level) filter on purpose: the rows still carry
    valuable PERSON_NAME / EMAIL / PHONE / STREET_ADDRESS gold; dropping
    only the unsupported DATE spans preserves that signal. If you want
    row-level filtering instead (skip the entire row when any DATE span
    is unsupported), invert: ``yield ex`` only when ``len(kept_dates) ==
    len(original_dates)``.
    """
    import re as _re

    from traciora_pii.policies.loader import (
        compile_regex_policy,
        load_regex_policy,
    )
    from benchmarks.data_import.loaders import NormalizedExample

    policy = load_regex_policy(locale)
    date_patterns = [
        _re.compile(p.pattern, _re.IGNORECASE)
        for p in policy.patterns
        if p.placeholder == "[DATE]"
    ]
    if not date_patterns:
        # Locale has no DATE regex — pass examples through unchanged.
        yield from examples
        return

    def _value_supported(value: str) -> bool:
        # fullmatch on at least one DATE pattern → supported. Any pattern
        # that requires the value to lie within a longer match (lookaround
        # wrappers) is still satisfied because we test the value in isolation.
        return any(p.fullmatch(value) for p in date_patterns)

    for ex in examples:
        kept_spans = [
            sp for sp in ex.spans
            if sp.label != "DATE" or _value_supported(sp.value)
        ]
        if len(kept_spans) == len(ex.spans):
            yield ex
        else:
            yield NormalizedExample(
                id=ex.id,
                locale=ex.locale,
                text=ex.text,
                spans=kept_spans,
                source=ex.source,
                meta=ex.meta,
            )


def _build_detectors(
    spec: dict,
    *,
    opf_model: Optional[str] = None,
    encoder_model: Optional[str] = None,
    second_pass: str = "opf",
):
    """Return (regex_detectors, second_pass_detector_or_None).

    ``second_pass`` selects the layer-2 implementation:
      * ``"opf"`` — our fine-tuned OPF; ``opf_model`` overrides the registry
        entry, default from ``detector.opf.default_model``.
      * ``"encoder"`` — the encoder second pass; ``encoder_model`` overrides
        the ``detector.encoder.models`` entry, default the entry registered
        under the dataset locale, else ``detector.encoder.default_model``.
      * ``"presidio"`` — Microsoft Presidio analyzer with the WIDE_LABEL_MAP
        (PERSON/LOCATION/PHONE/EMAIL/SSN/IP/CARD/IBAN/URL/DATE/passport/
        driver_license/medical_license → our placeholders). Industry NER
        baseline for §1 comparison.
      * ``"llm"`` — OpenAI-compatible LLM via OpenRouter (or any local
        endpoint pointed at by ``detector.llm.base_url``). Model name
        must be set via ``--detector-llm-model`` on the CLI.
      * ``"none"`` — regex-only ablation; returns ``(regex_detectors, None)``.
    """
    from traciora_pii.config import load_config
    from traciora_pii.policies.loader import compile_regex_policy, load_regex_policy

    cfg = load_config()
    regex_detectors = compile_regex_policy(load_regex_policy(spec["locale"]))

    if second_pass == "none":
        return regex_detectors, None

    if second_pass == "llm":
        from traciora_pii.llm_pii_detector import LLMPIIDetector

        print(
            f"[eval] loading LLM detector (model={cfg.detector.llm.model}, "
            f"base_url={cfg.detector.llm.base_url})...",
            file=sys.stderr,
        )
        llm = LLMPIIDetector.from_config(cfg.detector.llm)
        return regex_detectors, llm

    if second_pass == "encoder":
        from traciora_pii.pii_virtualizer import build_second_pass_detector

        chosen = encoder_model or (
            spec["locale"] if spec["locale"] in cfg.detector.encoder.models
            else cfg.detector.encoder.default_model
        )
        print(f"[eval] loading encoder detector (model={chosen})...", file=sys.stderr)
        t0 = time.perf_counter()
        encoder = build_second_pass_detector(
            cfg.with_overrides({"detector": {"second_pass": "encoder"}}),
            encoder_model_name=chosen,
        )
        print(f"[eval] encoder detector ready in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        return regex_detectors, encoder

    if second_pass == "presidio":
        from traciora_pii.presidio_pii_detector import (
            PresidioPIIDetector,
            WIDE_LABEL_MAP,
        )

        print(
            f"[eval] loading Presidio detector "
            f"(language={spec['locale']}, label_map=WIDE_LABEL_MAP)...",
            file=sys.stderr,
        )
        t0 = time.perf_counter()
        # Use WIDE_LABEL_MAP so Presidio's outputs map to our placeholders.
        # This is the §1-fair-comparison map (PERSON, LOCATION, US_SSN, etc.
        # → [PERSON_NAME], [STREET_ADDRESS], [PERSONAL_ID], ...).
        # Score threshold and prefilter default to the project config; the
        # language is forced to match the dataset's locale.
        presidio = PresidioPIIDetector(
            language=spec["locale"],
            score_threshold=cfg.detector.presidio.score_threshold,
            label_map=WIDE_LABEL_MAP,
            prefilter_enabled=False,  # don't skip rows in a benchmark eval
        )
        print(
            f"[eval] Presidio detector ready in {time.perf_counter() - t0:.1f}s",
            file=sys.stderr,
        )
        return regex_detectors, presidio

    # Default: OPF
    from traciora_pii.opf_pii_detector import OPFPIIDetector

    chosen = opf_model or cfg.detector.opf.default_model
    print(
        f"[eval] loading OPF detector "
        f"(model={chosen}, policy={spec['opf_policy']})...",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    opf = OPFPIIDetector.from_config(cfg.detector.opf, model_name=opf_model) \
        if opf_model is not None else OPFPIIDetector.from_config(cfg.detector.opf)
    print(f"[eval] OPF detector ready in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return regex_detectors, opf


def _detect_combined(text: str, regex_detectors, opf_detector) -> list[FindingRecord]:
    """Run regex first, then (optionally) OPF on the same raw text.

    When ``opf_detector`` is None this acts as a regex-only ablation —
    no second pass runs. Otherwise both passes are merged with regex
    winning on span overlap (mirrors ``redact_pii_text``).
    """
    from traciora_pii.pii import detect_pii

    regex_findings = detect_pii(text, detectors=regex_detectors)

    # Merge overlapping regex findings - mirrors redact_pii_text. The
    # first match in document order wins on overlap.
    merged_regex: list = []
    for f in regex_findings:
        if merged_regex and f.start <= merged_regex[-1].end:
            last = merged_regex[-1]
            merged_regex[-1] = type(last)(
                detector=last.detector,
                start=last.start,
                end=max(last.end, f.end),
                placeholder=last.placeholder,
            )
        else:
            merged_regex.append(f)

    out: list[FindingRecord] = []
    for f in merged_regex:
        out.append(
            FindingRecord(
                placeholder=f.placeholder,
                start=f.start,
                end=f.end,
                value=text[f.start:f.end],
                source="regex",
            )
        )

    if opf_detector is None:
        return out

    opf_findings = opf_detector.detect(text) or []
    for f in opf_findings:
        if any(f.start < r.end and r.start < f.end for r in out):
            continue
        out.append(
            FindingRecord(
                placeholder=f.placeholder,
                start=f.start,
                end=f.end,
                value=text[f.start:f.end],
                source="opf",
            )
        )
    return out


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------


def _score_one(
    gold: list[GoldSpan],
    findings: list[FindingRecord],
    counts: Dict[str, PerPlaceholder],
    fp_examples: Dict[str, list],
    fn_examples: Dict[str, list],
    *,
    text: str,
    row_id: str,
    bucket_counts_for_row: Optional[Dict[str, PerPlaceholder]] = None,
    max_examples_per_bucket: int = 5,
) -> Dict[str, Tuple[int, int, int]]:
    """Score one row. Mutates ``counts`` (and optionally bucket counts) and
    returns this row's contribution as ``{placeholder: (tp, fp, fn)}`` so
    the caller can store it for bootstrap resampling."""
    matched: set = set()
    row_record: Dict[str, Tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))

    def _bump(ph: str, delta: Tuple[int, int, int]) -> None:
        cur = row_record[ph]
        row_record[ph] = (cur[0] + delta[0], cur[1] + delta[1], cur[2] + delta[2])

    for f in findings:
        hit_idx = None
        for i, g in enumerate(gold):
            if i in matched:
                continue
            if f.placeholder != g.placeholder:
                continue
            if f.start < g.end and g.start < f.end:
                hit_idx = i
                break
        if hit_idx is not None:
            matched.add(hit_idx)
            counts[f.placeholder].tp += 1
            _bump(f.placeholder, (1, 0, 0))
            if bucket_counts_for_row is not None:
                bucket_counts_for_row[f.placeholder].tp += 1
        else:
            counts[f.placeholder].fp += 1
            _bump(f.placeholder, (0, 1, 0))
            if bucket_counts_for_row is not None:
                bucket_counts_for_row[f.placeholder].fp += 1
            if len(fp_examples[f.placeholder]) < max_examples_per_bucket:
                fp_examples[f.placeholder].append({
                    "row_id": row_id,
                    "value": f.value,
                    "source": f.source,
                    "context": text[max(0, f.start - 30):f.end + 30],
                })

    for i, g in enumerate(gold):
        if i in matched:
            continue
        counts[g.placeholder].fn += 1
        _bump(g.placeholder, (0, 0, 1))
        if bucket_counts_for_row is not None:
            bucket_counts_for_row[g.placeholder].fn += 1
        if len(fn_examples[g.placeholder]) < max_examples_per_bucket:
            fn_examples[g.placeholder].append({
                "row_id": row_id,
                "value": g.value,
                "context": text[max(0, g.start - 30):g.end + 30],
            })

    return dict(row_record)


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------


def _render_pr_block(
    counts: Dict[str, PerPlaceholder],
    *,
    per_row_records: Optional[Sequence[Dict[str, Tuple[int, int, int]]]] = None,
    n_resamples: int = 1000,
) -> List[str]:
    """Render the per-placeholder P/R/F1 table body for a counts dict.

    Thin adapter: convert the local ``Dict[str, PerPlaceholder]`` shape to
    the canonical ``_summarize_pr`` shape and delegate to the shared
    ``metrics.render_detection_pr_table``. Keeps callers in this module
    stable while routing through one renderer.
    """
    from benchmarks.pii_evaluation.metrics import (
        _summarize_pr,
        render_detection_pr_table,
    )

    tp_map = {ph: c.tp for ph, c in counts.items()}
    fp_map = {ph: c.fp for ph, c in counts.items()}
    fn_map = {ph: c.fn for ph, c in counts.items()}
    summary = _summarize_pr(tp_map, fp_map, fn_map)
    return render_detection_pr_table(
        summary,
        per_query_records=per_row_records,
        n_resamples=n_resamples,
    )


def _bucket_sums_match_global(run: EvalRun) -> bool:
    """Self-check: per-bucket TP/FP/FN sums must equal the global counts."""
    placeholders = set(run.counts) | {
        ph for b in run.bucket_counts.values() for ph in b
    }
    for ph in placeholders:
        g = run.counts.get(ph, PerPlaceholder())
        s_tp = sum(run.bucket_counts[b].get(ph, PerPlaceholder()).tp for b in run.bucket_counts)
        s_fp = sum(run.bucket_counts[b].get(ph, PerPlaceholder()).fp for b in run.bucket_counts)
        s_fn = sum(run.bucket_counts[b].get(ph, PerPlaceholder()).fn for b in run.bucket_counts)
        if (s_tp, s_fp, s_fn) != (g.tp, g.fp, g.fn):
            return False
    return True


def _render_markdown(run: EvalRun) -> str:
    proj_line = (
        f"- Projection-symmetric scoring: **{'on' if run.projection_symmetric else 'off'}** "
        f"(gold placeholder space: `{', '.join(run.gold_placeholders) or '(none)'}`)"
    )
    lines = [
        f"# Imported-dataset eval - {run.dataset}",
        "",
        f"- Rows evaluated: **{run.rows_evaluated:,}**",
        f"- Rows with zero gold spans (after label-map filter): **{run.rows_with_zero_gold:,}**",
        f"- Elapsed: **{run.elapsed_seconds:.1f}s** "
        f"({run.elapsed_seconds / max(run.rows_evaluated, 1) * 1000:.0f}ms/row)",
        proj_line,
        "",
        "## Per-placeholder precision / recall",
        "",
    ]
    lines.extend(_render_pr_block(run.counts, per_row_records=run.per_row_records))

    if run.bucket_rows:
        lines.extend(["", "## Per-length-bucket precision / recall", ""])
        bucket_dist = ", ".join(
            f"`{b}`: {run.bucket_rows[b]:,}"
            for b in LENGTH_BUCKETS
            if run.bucket_rows.get(b)
        )
        lines.append(f"Row distribution across buckets: {bucket_dist}")
        lines.append("")
        if not _bucket_sums_match_global(run):
            lines.append("> ⚠️ bucket sums do not match global counts — wiring bug")
            lines.append("")
        for b in LENGTH_BUCKETS:
            rows = run.bucket_rows.get(b, 0)
            if not rows:
                continue
            bucket_row_records = [
                rec for rec, bk in zip(run.per_row_records, run.per_row_buckets) if bk == b
            ]
            lines.append(f"### `{b}` ({rows:,} rows)")
            lines.append("")
            lines.extend(_render_pr_block(run.bucket_counts[b], per_row_records=bucket_row_records))
            lines.append("")

    if run.fp_examples:
        lines.extend(["", "## False-positive samples", ""])
        for ph in sorted(run.fp_examples):
            lines.append(f"### `{ph}`")
            for ex in run.fp_examples[ph]:
                lines.append(
                    f"- `{ex['row_id']}` [{ex['source']}] value=`{ex['value']}`  context: ...{ex['context']!r}..."
                )
            lines.append("")

    if run.fn_examples:
        lines.extend(["", "## False-negative samples", ""])
        for ph in sorted(run.fn_examples):
            lines.append(f"### `{ph}`")
            for ex in run.fn_examples[ph]:
                lines.append(
                    f"- `{ex['row_id']}` value=`{ex['value']}`  context: ...{ex['context']!r}..."
                )
            lines.append("")

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="wolframko_ru")
    parser.add_argument(
        "--target-rows",
        type=int,
        default=200,
        help="Stop after this many post-pipeline rows have been scored.",
    )
    parser.add_argument(
        "--raw-limit",
        type=int,
        default=600,
        help="Pre-pipeline row cap (must exceed target-rows / survival rate).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown report path. Defaults to results/imported_eval_<dataset>[__suffix].md",
    )
    parser.add_argument(
        "--detector",
        choices=("regex_only", "regex_opf", "regex_encoder", "regex_llm", "both", "regex_presidio"),
        default="both",
        help="Which detection layer(s) to run. 'regex_only' = first-pass only "
             "(layer 1 baseline). 'regex_opf' / 'both' = layer 1 + our "
             "fine-tuned OPF (default). 'regex_encoder' = layer 1 + the "
             "encoder second pass. 'regex_llm' = layer 1 + an OpenAI-"
             "compatible LLM via OpenRouter (requires --detector-llm-model). "
             "'regex_presidio' = layer 1 + Microsoft Presidio analyzer "
             "with WIDE_LABEL_MAP — industry NER baseline for §1 comparison.",
    )
    parser.add_argument(
        "--opf-model",
        default=None,
        help="Override the registry entry to load (e.g. 'base', 'ru', "
             "'ru-v6-published'). Default: detector.opf.default_model from config.",
    )
    parser.add_argument(
        "--encoder-model",
        default=None,
        help="Override the detector.encoder.models entry to load for "
             "--detector regex_encoder. Default: the entry registered under the "
             "dataset locale, else detector.encoder.default_model.",
    )
    add_detector_llm_flags(parser)
    parser.add_argument(
        "--internal-slug",
        default=None,
        help=(
            "Required when --dataset internal_synthetic. Picks which prepared "
            "fine_tuning/data/<slug>/test.jsonl to evaluate against. Example: "
            "'ru-v1', 'de-v1', 'en-v2'."
        ),
    )
    parser.add_argument(
        "--projection-symmetric",
        dest="projection_symmetric",
        action="store_true",
        default=True,
        help=(
            "Drop detector findings whose placeholder is not in the dataset's "
            "gold label space. The cross-source datasets project gold spans down "
            "to a 4-5 label subset; without this flag the unfiltered findings on "
            "out-of-subset placeholders (e.g. DE_STEUER_ID on ai4privacy_de, "
            "which projects to PERSON_NAME / EMAIL / PHONE / STREET_ADDRESS / DATE) "
            "all count as FPs and inflate the apparent error rate. Default: on. "
            "Pass --no-projection-symmetric for the legacy asymmetric eval that "
            "penalises every out-of-subset finding."
        ),
    )
    parser.add_argument(
        "--no-projection-symmetric",
        dest="projection_symmetric",
        action="store_false",
        help="See --projection-symmetric.",
    )
    args = parser.parse_args(argv)

    # Detector-LLM CLI flags push into env so downstream load_config()
    # inside _build_detectors picks them up without threading cfg through
    # every call.
    if detector_llm_overrides(args):
        if args.detector_llm_model:
            os.environ["PII_LLM_MODEL"] = args.detector_llm_model
        if args.detector_llm_base_url:
            os.environ["PII_LLM_BASE_URL"] = args.detector_llm_base_url

    spec = DATASETS[args.dataset]
    print(f"[eval] dataset={args.dataset} target_rows={args.target_rows} raw_limit={args.raw_limit} "
          f"detector={args.detector} opf_model={args.opf_model or '<default>'} "
          f"encoder_model={args.encoder_model or '<locale>'}"
          + (f" internal_slug={args.internal_slug}" if args.internal_slug else ""),
          file=sys.stderr)
    examples = _load_examples(
        args.dataset, args.raw_limit, args.target_rows,
        internal_slug=args.internal_slug,
    )
    # internal_synthetic mutates spec at load time; set env vars after the load.
    _setup_opf_env(spec["opf_policy"])
    print(f"[eval] {len(examples)} examples after transforms", file=sys.stderr)

    if args.detector == "regex_only":
        second_pass = "none"
    elif args.detector == "regex_presidio":
        second_pass = "presidio"
    elif args.detector == "regex_llm":
        second_pass = "llm"
    elif args.detector == "regex_encoder":
        second_pass = "encoder"
    else:  # "regex_opf" or "both"
        second_pass = "opf"
    regex_detectors, opf_detector = _build_detectors(
        spec, opf_model=args.opf_model, encoder_model=args.encoder_model,
        second_pass=second_pass,
    )

    run = EvalRun(dataset=args.dataset)
    label_map = spec["label_map"]
    # Projection-symmetric eval: only score detector findings whose
    # placeholder is in the dataset's gold-side label space. This is the
    # set of placeholders the gold annotation pipeline can produce — the
    # values of label_map after dataset-side filtering. Findings outside
    # this set are silently dropped before scoring (otherwise they all
    # become FPs that have no possible gold counterpart to match).
    gold_placeholders: frozenset = frozenset(label_map.values())
    run.gold_placeholders = sorted(gold_placeholders)
    run.projection_symmetric = bool(args.projection_symmetric)
    print(
        f"[eval] projection_symmetric={run.projection_symmetric}  "
        f"gold_placeholders={sorted(gold_placeholders)}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    for i, ex in enumerate(examples):
        gold = []
        for sp in ex.spans:
            mapped = label_map.get(sp.label)
            if mapped is None:
                continue  # label not mappable to a framework placeholder
            gold.append(GoldSpan(placeholder=mapped, start=sp.start, end=sp.end, value=sp.value))
        if not gold:
            run.rows_with_zero_gold += 1
        findings = _detect_combined(ex.text, regex_detectors, opf_detector)
        if run.projection_symmetric:
            findings = [f for f in findings if f.placeholder in gold_placeholders]
        bucket = length_bucket_for_text(ex.text)
        run.bucket_rows[bucket] += 1
        row_record = _score_one(
            gold,
            findings,
            run.counts,
            run.fp_examples,
            run.fn_examples,
            text=ex.text,
            row_id=ex.id,
            bucket_counts_for_row=run.bucket_counts[bucket],
        )
        run.per_row_records.append(row_record)
        run.per_row_buckets.append(bucket)
        run.rows_evaluated += 1
        if (i + 1) % 50 == 0:
            print(f"[eval] {i + 1}/{len(examples)} rows scored", file=sys.stderr)
    run.elapsed_seconds = time.perf_counter() - t0
    print(f"[eval] done: {run.elapsed_seconds:.1f}s total", file=sys.stderr)

    if args.out is not None:
        out_path = args.out
    else:
        if args.detector == "regex_only":
            suffix = "regex_only"
        elif args.detector == "regex_presidio":
            suffix = "regex_presidio"
        elif args.detector == "regex_llm":
            # Encode the LLM model in the filename so 4-LLM comparison
            # runs don't overwrite each other.
            llm_tag = (
                re.sub(r"[^a-zA-Z0-9._-]+", "-", args.detector_llm_model).strip("-").lower()
                if args.detector_llm_model
                else "default"
            )
            suffix = f"regex_llm__{llm_tag}"
        elif args.detector == "regex_encoder":
            suffix = f"regex_encoder__{args.encoder_model or 'locale'}"
        else:
            suffix = f"regex_opf__{args.opf_model or 'default'}"
        prefix = (
            f"internal_synthetic_{args.internal_slug}"
            if args.dataset == "internal_synthetic" and args.internal_slug
            else args.dataset
        )
        out_path = RESULTS_DIR / f"imported_eval_{prefix}__{suffix}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_markdown(run), encoding="utf-8")
    print(f"[eval] report -> {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    # _rc captures main()'s exit code so the bash caller can branch on it.
    # We use os._exit (not SystemExit) to bypass the HF `datasets`
    # library's atexit retry loop on streaming iterators: once the report
    # is written to disk, lingering retries against the upstream dataset
    # host can stall the process for tens of minutes on a flaky network.
    # Nothing in this module depends on Python's atexit hooks running.
    import os as _os
    import sys as _sys
    _rc = main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_rc)
