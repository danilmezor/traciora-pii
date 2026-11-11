"""Per-query diagnostic for cross-turn lemma failures.

Walks the trilingual ``multi_turn`` bucket, runs both base OPF and the
routed fine-tune on each turn, prints the detected [PERSON_NAME] spans
and lemmas per turn, and flags pairs where turn-0's lemma does not
appear in turn-1's lemma set. Companion to ``measure_vault_ref_stability``
for localizing cross-turn failures in the fine-tune.
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

from traciora_pii.config import OPFConfig, OPFModelEntry, load_config
from traciora_pii.opf_pii_detector import OPFPIIDetector
from benchmarks.pii_evaluation.dataset import EvalQuery, load_dataset


def _person_spans(det, text: str) -> List[Tuple[str, str]]:
    """Return [(surface, lemma), ...] for [PERSON_NAME] findings."""
    out: List[Tuple[str, str]] = []
    for f in det.detect(text) or []:
        if f.placeholder == "[PERSON_NAME]":
            out.append((text[f.start:f.end], f.lemma))
    return out


def _diagnose(det, label: str, queries: List[EvalQuery]) -> None:
    print(f"\n{'=' * 72}")
    print(f"Detector: {label}")
    print(f"{'=' * 72}")
    collapsed = 0
    total = 0
    failures: List[EvalQuery] = []

    for q in queries:
        if len(q.turns) < 2:
            continue
        turn_spans = [_person_spans(det, t) for t in q.turns]
        turn0_lemmas = {l for _, l in turn_spans[0]}
        for later_idx in range(1, len(q.turns)):
            later_lemmas = {l for _, l in turn_spans[later_idx]}
            for lemma in turn0_lemmas:
                total += 1
                if lemma in later_lemmas:
                    collapsed += 1
                else:
                    if q not in failures:
                        failures.append(q)

    print(f"cross-turn collapse: {collapsed}/{total} = "
          f"{collapsed/total:.3f}" if total else "no pairs")

    print(f"\n--- failures ({len(failures)} queries) ---")
    for q in failures:
        print(f"\n[{q.id}] {q.language}")
        for i, turn in enumerate(q.turns):
            print(f"  turn {i}: {turn!r}")
            spans = _person_spans(det, turn)
            if spans:
                for surface, lemma in spans:
                    print(f"    [PERSON_NAME] surface={surface!r:<40} lemma={lemma!r}")
            else:
                print(f"    (no [PERSON_NAME] detections)")

    print(f"\n--- successes (first 3) ---")
    success_count = 0
    for q in queries:
        if q in failures or len(q.turns) < 2:
            continue
        success_count += 1
        if success_count > 3:
            break
        print(f"\n[{q.id}] {q.language}")
        for i, turn in enumerate(q.turns):
            print(f"  turn {i}: {turn!r}")
            spans = _person_spans(det, turn)
            for surface, lemma in spans:
                print(f"    [PERSON_NAME] surface={surface!r:<40} lemma={lemma!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help="Path or HF repo id for the fine-tuned model under test.",
    )
    parser.add_argument(
        "--policy",
        default="ru_names_addresses",
        help="OPF policy name (label_map). Default: ru_names_addresses.",
    )
    args = parser.parse_args(argv)

    queries = [q for q in load_dataset() if q.bucket == "multi_turn"]
    print(f"Multi-turn bucket: {len(queries)} queries")

    cfg = load_config()

    # Build a base-OPF view from the configured 'base' model entry, falling
    # back to a synthetic registry entry if the user hasn't registered one.
    base_cfg = cfg.detector.opf
    if "base" not in base_cfg.models:
        base_cfg = OPFConfig(
            default_model="base",
            models={"base": OPFModelEntry(checkpoint="openai/privacy-filter", policy=args.policy)},
            device=base_cfg.device,
            prefilter=base_cfg.prefilter,
        )
    base = OPFPIIDetector.from_config(base_cfg, model_name="base")
    _diagnose(base, "base OPF (no fine-tune)", queries)

    # Build a finetune view from the user-supplied path + policy. Synthesize
    # a single-entry OPFConfig so the diagnostic doesn't require the user to
    # pre-register the model in traciora.yaml.
    ft_cfg = OPFConfig(
        default_model="diag",
        models={"diag": OPFModelEntry(checkpoint=args.finetuned_model, policy=args.policy)},
        device=base_cfg.device,
        prefilter=base_cfg.prefilter,
    )
    finetuned = OPFPIIDetector.from_config(ft_cfg, model_name="diag")
    _diagnose(finetuned, "fine-tuned (single-pass)", queries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
