"""One-off measurement: vault ref-token stability across multi-turn pairs.

The paper's §4.3 claim — that the *vault* collapses referring expressions
to one token per entity across turns — is measured here end-to-end by
running the full ``PIIVirtualizer`` anonymization path on the trilingual
``multi_turn`` bucket and checking whether a person-name ref token from
turn 0 appears in later turns of the same session.

Usage:

  python -m benchmarks.pii_evaluation.measure_vault_ref_stability              # base OPF
  python -m benchmarks.pii_evaluation.measure_vault_ref_stability --routed ... # fine-tuned routed
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from cryptography.fernet import Fernet

from traciora_pii.config import OPFConfig, OPFModelEntry, load_config
from traciora_pii.opf_pii_detector import OPFPIIDetector
from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer
from benchmarks.pii_evaluation.dataset import EvalQuery, load_dataset


_REF_RE = re.compile(r"person_name_ref:pe_[0-9a-f]+")


def vault_ref_stability(
    virtualizer: PIIVirtualizer,
    dataset: Sequence[EvalQuery],
    *,
    bucket: str = "multi_turn",
) -> Dict[str, float]:
    """Ref-token cross-turn stability on the multi-turn bucket.

    For each query we open a fresh session and anonymize every turn in
    order within that session. Cross-turn is measured as: for each turn-0
    person-name ref, does it appear in the set of person-name refs in
    each later turn of the same session? If lemma-based vault keying is
    working, inflected referring expressions across turns collapse to a
    single ref token and every cross-turn pair counts as a match.
    """
    target = [q for q in dataset if q.bucket == bucket]
    if not target:
        return {"cross_turn": 1.0, "queries": 0, "cross_total": 0, "cross_match": 0}

    cross_total = cross_match = 0

    for query in target:
        session_key = f"ref-stability-{query.id}"
        virtualizer._vault.open_session(session_key)
        per_turn: List[set] = []
        for turn in query.turns:
            anon = virtualizer.anonymize_text(session_key, turn)
            per_turn.append(set(_REF_RE.findall(anon)))

        if len(query.turns) >= 2:
            turn0_refs = per_turn[0]
            for later_idx in range(1, len(query.turns)):
                later_refs = per_turn[later_idx]
                for ref in turn0_refs:
                    cross_total += 1
                    if ref in later_refs:
                        cross_match += 1

    return {
        "cross_turn": cross_match / cross_total if cross_total else 1.0,
        "queries": len(target),
        "cross_total": cross_total,
        "cross_match": cross_match,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opf-model",
        default=None,
        help=(
            "OPF registry key (e.g. base | en | de | ru). Defaults to "
            "detector.opf.default_model."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Override the registry entry's checkpoint with an explicit "
            "path or HF repo id (useful for ad-hoc experiments)."
        ),
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Override the entry's OPF policy (e.g. ru_names_addresses).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--label", default=None,
                        help="Free-text label printed alongside the numbers "
                             "so multiple invocations can be tagged in a log.")
    args = parser.parse_args(argv)

    cfg = load_config()
    opf_cfg = cfg.detector.opf
    if args.device:
        opf_cfg = opf_cfg.model_copy(update={"device": args.device})
    chosen = args.opf_model or opf_cfg.default_model
    if args.checkpoint or args.policy:
        # Synthesize a one-entry registry overlay for the run.
        original = opf_cfg.models.get(chosen)
        new_entry = OPFModelEntry(
            checkpoint=args.checkpoint or (original.checkpoint if original else "openai/privacy-filter"),
            policy=args.policy or (original.policy if original else "ru_names_addresses"),
            description=(original.description if original else ""),
        )
        new_models = dict(opf_cfg.models)
        new_models[chosen] = new_entry
        opf_cfg = opf_cfg.model_copy(update={"models": new_models, "default_model": chosen})
    detector = OPFPIIDetector.from_config(opf_cfg, model_name=chosen)

    vault = PIIVaultStore(db_path=":memory:", encryption_key=Fernet.generate_key())
    virt = PIIVirtualizer(vault, second_pass_detector=detector)

    dataset = load_dataset()
    result = vault_ref_stability(virt, dataset)

    label = args.label or chosen
    print(f"[{label}] vault ref-token cross-turn stability on multi-turn bucket "
          f"(queries={result['queries']}):")
    print(f"  cross-turn : {result['cross_turn']:.3f} "
          f"({result['cross_match']}/{result['cross_total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
