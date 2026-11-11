"""Build the canonical traciora_pii-bench release artifact set.

One command, deterministic, reproducible from seed alone:

    python -m benchmarks.data_generator.build_release \\
        --out releases/v1 --locales ru --n 3000

For each locale this produces:

    <out>/traciora_pii-bench-<locale>/
        train.jsonl
        dev.jsonl
        test.jsonl
        datasheet.md     — Gebru et al. outline + per-locale distribution

And at the top level:

    <out>/RELEASE_CARD.md  — cross-locale summary
    <out>/MANIFEST.json    — sha256 of every output file

The build also runs the regex policy over the produced JSONLs and fails
with non-zero exit if a placeholder the generator emits as regex-backed
is *not* detected by the policy (drift guard).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .noise import NoiseApplier, NoiseConfig
from .pipeline import GeneratedExample, Pipeline, split_examples, write_jsonl
from .report import (
    Distribution,
    compute_distribution,
    render_distribution_md,
    render_release_card,
)
from .seeds import SeedBundle, load_seed_bundle


# Default soft-tier minimum recall floor. The benchmark intentionally
# carries ~50% naturalistic templates for soft-tier slots (templates
# that evade the keyword anchor) — those are the surface OPF must
# learn. So expected soft-tier recall sits around 40-60% on a clean
# build. The floor is a structural-failure detector: below 30% means
# templates universally evade the regex (something is broken), not
# normal naturalistic mix.
_SOFT_RECALL_FLOOR = 0.30


# Frozen seeds per locale. Bumping the artifact `version` is the right way
# to invalidate previously published hashes; do not change these constants.
FROZEN_SEEDS = {
    "ru": 20240501,
    "de": 20240502,
    "en": 20240503,
}


# ======================================================================
# Helpers
# ======================================================================


def _detect_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return ""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_noise(args: argparse.Namespace, seeds: SeedBundle) -> Optional[NoiseApplier]:
    if (args.noise_typo_rate == 0 and args.noise_ocr_rate == 0
            and args.noise_whitespace_rate == 0 and args.noise_code_switch_rate == 0):
        return None
    cfg = NoiseConfig(
        typo_rate=args.noise_typo_rate,
        ocr_confusable_rate=args.noise_ocr_rate,
        whitespace_jitter_rate=args.noise_whitespace_rate,
        code_switch_rate=args.noise_code_switch_rate,
        target=args.noise_target,
        locale=seeds.locale,
    )
    return NoiseApplier(config=cfg, seeds=seeds)


# ======================================================================
# Datasheet rendering
# ======================================================================


def _render_datasheet_md(
    locale: str, distribution: Distribution, *,
    name: str, version: str, license_str: str, git_sha: str,
    seed: int, n_train: int, n_dev: int, n_test: int,
    noise_summary: Optional[str], timestamp: str,
) -> str:
    """Per-locale Markdown datasheet (Gebru et al. outline + distribution)."""
    parts: List[str] = []

    parts.append(
        f"# {name}-{locale} {version}\n\n"
        f"Datasheet for the {locale.upper()} synthetic-PII benchmark in the "
        f"`traciora_pii-bench` release. Format follows Gebru et al. (2018), "
        f"\"Datasheets for Datasets.\"\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Name | `{name}-{locale}` |\n"
        f"| Version | `{version}` |\n"
        f"| Locale | `{locale}` |\n"
        f"| License | `{license_str}` |\n"
        f"| Generation seed | `{seed}` |\n"
        f"| Generator commit | `{git_sha or 'unknown'}` |\n"
        f"| Generated on | {timestamp} |\n"
        f"| Train / dev / test | {n_train} / {n_dev} / {n_test} |\n"
        f"| Noise | {noise_summary or 'disabled'} |\n"
    )

    parts.append(
        "## Motivation\n\n"
        "Real personally identifiable information cannot be released. This "
        "benchmark addresses the resulting evaluation gap with a slot-template "
        "synthesis pipeline: every PII identifier is checksum-correct synthetic "
        "data; every host sentence is a hand-curated business-register template "
        "from the matching domain (customer service / KYC / billing / internal "
        "log / legal). Hard negatives — text with PII-shaped patterns that do "
        "NOT carry actual personal data — are mixed in at the configured rate.\n\n"
        "Intended uses: cross-source generalisation studies for locale-aware "
        "PII detectors, reproducibility of the numbers in the accompanying paper, "
        "and stress evaluation when paired with the noise pipeline."
    )

    parts.append(
        "## Composition\n\n"
        "Each instance is one labeled sentence (or short multi-sentence span "
        "when fillers are used) with character-offset gold spans. Format: "
        "JSONL with the schema in `benchmarks.data_generator.pipeline.GeneratedExample`. "
        "The invariant `text[span.start:span.end] == span.value` is enforced "
        "by the generator's smoke tests."
    )

    parts.append(
        "## Collection process\n\n"
        "Generation is purely synthetic. Pools (names, addresses, identifier "
        "tables) and slot-templates live in `benchmarks/data_generator/seeds/<locale>/`. "
        "Adding or modifying examples is a YAML edit; no Python change is required."
    )

    parts.append(
        "## Preprocessing / cleaning / labeling\n\n"
        "* Sentence templates are hand-curated; no real-text scaffolding.\n"
        "* Slot fills come from checksum-correct algorithms (ISO 7064, Luhn, "
        "ISO 3779, FNS / PFR formulas) wired through `faker_providers.py`.\n"
        "* Optional noise stage applies typos / OCR confusables / whitespace "
        "jitter / code-switching while preserving the span invariant.\n"
    )

    parts.append(
        "## Uses\n\n"
        "### Intended\n\n"
        "* Evaluation of PII detectors across the declared placeholder taxonomy.\n"
        "* Cross-generator generalisation studies (e.g. evaluate against "
        "`ai4privacy/pii-masking-openpii-1m`, `nvidia/Nemotron-PII`).\n"
        "* Stress evaluation under adversarial perturbation when paired with "
        "`NoiseConfig`-enabled runs.\n\n"
        "### Not intended\n\n"
        "* Training a detector that will then be evaluated on this same "
        "benchmark — the slot-fill distributions are shaped by the generator's "
        "own algorithms.\n"
        "* Any inference about real individuals — every PII value is drawn "
        "from synthetic generators.\n"
    )

    parts.append(
        f"## Distribution\n\n"
        f"Released under `{license_str}`. Reproducible from seed `{seed}` "
        f"and generator commit `{git_sha or 'unknown'}` via "
        f"`python -m benchmarks.data_generator.build_release`. The accompanying "
        f"`MANIFEST.json` lists sha256 of every file in the release."
    )

    parts.append(
        "## Maintenance\n\n"
        "Versioning: incrementing `version` produces a new artifact set; older "
        "versions are not retroactively patched. Generator-policy contract "
        "drift is guarded by `test_policy_catches_generator_output_ru` in CI."
    )

    # The actual stats — placeholder counts, category breakdown, etc.
    parts.append(render_distribution_md(distribution))

    return "\n\n".join(p.rstrip() for p in parts) + "\n"


# ======================================================================
# Policy contract assertion
# ======================================================================


# Patterns whose regex bakes the keyword anchor INTO the pattern string
# itself (e.g. ``\b(?:инн|inn)\s*:?\s*\d{10,12}\b``) rather than declaring
# it via the schema's ``keyword_anchors`` field. They are effectively
# keyword-anchored (the regex won't fire on a bare value lacking the
# keyword) and belong in the soft tier despite having an empty
# ``keyword_anchors`` list. Add a pattern's ``name`` here when adding such
# a pattern to a policy.
_INLINE_KEYWORD_PATTERN_NAMES = {"ru_inn"}


def _classify_placeholders(regex_policy) -> Tuple[Set[str], Set[str]]:
    """Split regex-backed placeholders into hard-fail and soft-report tiers.

    A placeholder is *hard-fail* if at least one of its regex patterns
    fires unconditionally on shape match — neither declared
    ``keyword_anchors`` nor a keyword baked into the pattern string.
    Misses for these indicate genuine drift between the generator and the
    policy.

    A placeholder is *soft-report* if every one of its patterns is
    effectively keyword-gated. Misses for those are expected at low rates
    whenever templates vary the surrounding prose; that's the surface the
    OPF second-pass covers at runtime, not the first-pass regex.
    """
    hard: Set[str] = set()
    soft: Set[str] = set()
    for pattern in regex_policy.patterns:
        is_anchored = bool(pattern.keyword_anchors) or pattern.name in _INLINE_KEYWORD_PATTERN_NAMES
        if is_anchored:
            soft.add(pattern.placeholder)
        else:
            hard.add(pattern.placeholder)
    # A placeholder with mixed-anchoring patterns lands in `hard` by virtue
    # of having ≥1 unconditional pattern. Strip from soft to avoid double-counting.
    soft -= hard
    return hard, soft


def _check_policy_contract(
    examples: Sequence[GeneratedExample], locale: str,
) -> Dict:
    """Compute per-placeholder regex recall over produced examples.

    Returns a dict with:
      ``hard_set``  — placeholders we hard-fail on misses.
      ``soft_set``  — placeholders we soft-report only.
      ``total``     — emitted-span counts per placeholder.
      ``caught``    — caught-span counts per placeholder.
    """
    from traciora_pii.pii import detect_pii
    from traciora_pii.policies.loader import compile_regex_policy, load_regex_policy

    regex_policy = load_regex_policy(locale)
    detectors = compile_regex_policy(regex_policy)
    hard_set, soft_set = _classify_placeholders(regex_policy)
    measured = hard_set | soft_set

    total: collections.Counter = collections.Counter()
    caught: collections.Counter = collections.Counter()

    for ex in examples:
        if ex.kind != "positive":
            continue
        findings = detect_pii(ex.text, detectors=detectors)
        found = {(f.placeholder, f.start, f.end) for f in findings}
        for span in ex.spans:
            if span.placeholder not in measured:
                continue
            total[span.placeholder] += 1
            hit = any(
                p == span.placeholder and not (e <= span.start or s >= span.end)
                for p, s, e in found
            )
            if hit:
                caught[span.placeholder] += 1

    return {
        "hard_set": hard_set,
        "soft_set": soft_set,
        "total": dict(total),
        "caught": dict(caught),
    }


def _format_recall_lines(locale: str, report: Dict) -> List[str]:
    """One-per-placeholder summary lines for stderr output."""
    lines: List[str] = []
    for ph in sorted(report["total"]):
        tier = "hard" if ph in report["hard_set"] else "soft"
        c = report["caught"].get(ph, 0)
        t = report["total"][ph]
        rate = (c / t * 100.0) if t else 100.0
        lines.append(
            f"  [contract {locale}] {tier:4s} {ph:24s} {c:>4}/{t:<4} = {rate:5.1f}%"
        )
    return lines


def _gate_failures(
    locale: str, report: Dict, soft_floor: float = _SOFT_RECALL_FLOOR,
) -> List[str]:
    """Return human-readable messages for placeholders that fail the gate.

    Hard tier fails when recall < 100% (any miss is drift).
    Soft tier fails when recall < ``soft_floor`` (anything below the floor
    means the templates universally evade the regex — likely a structural
    problem worth knowing about).
    """
    fails: List[str] = []
    for ph, t in report["total"].items():
        c = report["caught"].get(ph, 0)
        rate = (c / t) if t else 1.0
        if ph in report["hard_set"] and c < t:
            fails.append(
                f"{locale}: HARD {ph} drift: {c}/{t} = {rate * 100:.1f}% "
                f"(unconditional regex; missing hits indicate generator/policy drift)"
            )
        elif ph in report["soft_set"] and rate < soft_floor:
            fails.append(
                f"{locale}: SOFT {ph} below floor: {c}/{t} = {rate * 100:.1f}% "
                f"< {soft_floor * 100:.0f}% (keyword-anchored regex; templates "
                f"appear to universally evade the keyword anchor)"
            )
    return fails


# ======================================================================
# Main
# ======================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.data_generator.build_release",
        description="Build the canonical traciora_pii-bench release artifacts.",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--locales", default="ru",
        help="comma-separated subset of locales with seeds available "
             "(default: ru)",
    )
    parser.add_argument("--n", type=int, default=3000,
                        help="examples per locale (split into train/dev/test)")
    parser.add_argument("--split", default="0.8,0.1,0.1",
                        help="comma-separated train,dev,test ratios")
    parser.add_argument("--name", default="traciora_pii-bench")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--license", default="CC-BY-4.0")
    # Noise (default: off)
    parser.add_argument("--noise-typo-rate", type=float, default=0.0)
    parser.add_argument("--noise-ocr-rate", type=float, default=0.0)
    parser.add_argument("--noise-whitespace-rate", type=float, default=0.0)
    parser.add_argument("--noise-code-switch-rate", type=float, default=0.0)
    parser.add_argument("--noise-target", choices=("scaffold", "pii", "both"),
                        default="scaffold")
    parser.add_argument("--skip-policy-check", action="store_true",
                        help="skip the post-build policy-contract assertion")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute everything but write nothing")
    args = parser.parse_args(argv)

    locales = [loc.strip() for loc in args.locales.split(",") if loc.strip()]
    ratios_raw = tuple(float(r) for r in args.split.split(","))
    if len(ratios_raw) != 3:
        raise SystemExit("--split needs exactly 3 ratios")

    git_sha = _detect_git_sha()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    manifest_files: Dict[str, Dict[str, str]] = {}
    summaries: List[str] = []
    contract_failures: List[str] = []
    distributions: Dict[str, Distribution] = {}

    for locale in locales:
        if locale not in FROZEN_SEEDS:
            raise SystemExit(f"locale {locale!r} not in FROZEN_SEEDS")
        try:
            seeds = load_seed_bundle(locale)
        except FileNotFoundError as e:
            raise SystemExit(f"no seeds for locale {locale!r}: {e}") from e

        seed = FROZEN_SEEDS[locale]
        noise = _build_noise(args, seeds)
        pipeline = Pipeline(seeds=seeds, seed=seed, noise=noise)

        examples = list(pipeline.run(args.n))
        train, dev, test = split_examples(examples, ratios=ratios_raw, seed=seed)
        distribution = compute_distribution(examples, locale=locale)
        distributions[locale] = distribution

        summaries.append(
            f"{locale}: train={len(train)} dev={len(dev)} test={len(test)} "
            f"positives={distribution.n_positive} negatives={distribution.n_negative} "
            f"categories={len(distribution.by_category)} "
            f"placeholders={len(distribution.by_placeholder)}"
        )

        if not args.skip_policy_check:
            report = _check_policy_contract(examples, locale=locale)
            # Always print the per-placeholder recall table — useful even
            # when nothing fails the gate.
            for line in _format_recall_lines(locale, report):
                print(line, file=sys.stderr)
            for msg in _gate_failures(locale, report):
                contract_failures.append(msg)

        if args.dry_run:
            continue

        bench_dir = os.path.join(args.out, f"{args.name}-{locale}")
        os.makedirs(bench_dir, exist_ok=True)
        train_p = os.path.join(bench_dir, "train.jsonl")
        dev_p = os.path.join(bench_dir, "dev.jsonl")
        test_p = os.path.join(bench_dir, "test.jsonl")
        write_jsonl(iter(train), train_p)
        write_jsonl(iter(dev), dev_p)
        write_jsonl(iter(test), test_p)

        # Datasheet
        noise_summary = None
        if noise is not None:
            cfg = noise.config
            noise_summary = (
                f"typo={cfg.typo_rate} ocr={cfg.ocr_confusable_rate} "
                f"ws={cfg.whitespace_jitter_rate} cs={cfg.code_switch_rate} "
                f"target={cfg.target}"
            )
        datasheet_p = os.path.join(bench_dir, "datasheet.md")
        with open(datasheet_p, "w", encoding="utf-8") as f:
            f.write(_render_datasheet_md(
                locale=locale, distribution=distribution,
                name=args.name, version=args.version,
                license_str=args.license, git_sha=git_sha, seed=seed,
                n_train=len(train), n_dev=len(dev), n_test=len(test),
                noise_summary=noise_summary, timestamp=timestamp,
            ))

        for path in (train_p, dev_p, test_p, datasheet_p):
            rel = os.path.relpath(path, args.out)
            manifest_files[rel] = {
                "sha256": _sha256_file(path),
                "size_bytes": str(os.path.getsize(path)),
            }

    # Top-level RELEASE_CARD + MANIFEST
    if not args.dry_run and distributions:
        card_p = os.path.join(args.out, "RELEASE_CARD.md")
        with open(card_p, "w", encoding="utf-8") as f:
            f.write(render_release_card(
                distributions, version=args.version,
                license=args.license, git_sha=git_sha,
            ))
        manifest_files["RELEASE_CARD.md"] = {
            "sha256": _sha256_file(card_p),
            "size_bytes": str(os.path.getsize(card_p)),
        }

        manifest_p = os.path.join(args.out, "MANIFEST.json")
        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump({
                "name": args.name,
                "version": args.version,
                "git_sha": git_sha,
                "generated_at": timestamp,
                "files": {k: manifest_files[k] for k in sorted(manifest_files)},
            }, f, indent=2, sort_keys=True)
            f.write("\n")

    for line in summaries:
        print(line, file=sys.stderr)

    if contract_failures:
        for line in contract_failures:
            print(f"POLICY-CONTRACT FAILURE: {line}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
