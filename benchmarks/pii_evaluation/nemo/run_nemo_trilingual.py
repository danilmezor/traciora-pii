"""Trilingual detector comparison: regex_only / regex_presidio / regex_opf_routed /
regex_nemo, per language and aggregated, using the harness's own scoring
(metrics.detection_precision_recall_with_records) and the ablation's merge rule."""
import csv
import os
import json
import sys
from collections import defaultdict

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Cached inputs remain in the ignored results tree, separate from source code.
HERE = str(Path(os.environ.get(
    "TRACIORA_NEMO_DATA_DIR", REPO / "benchmarks/pii_evaluation/results/nemo",
)))
sys.path.insert(0, str(REPO))

from benchmarks.pii_evaluation.dataset import load_dataset  # noqa: E402
from benchmarks.pii_evaluation import metrics  # noqa: E402
from traciora_pii.config import load_config  # noqa: E402
from traciora_pii.pii_virtualizer import build_second_pass_detector  # noqa: E402
from benchmarks.pii_evaluation.nemo.run_nemo_ablation import NemoCachedDetector  # noqa: E402

LANGS = ("en", "de", "ru")


def counts(pr):
    tp = sum(r.get("tp") or 0 for k, r in pr.items() if not k.startswith("__"))
    fp = sum(r.get("fp") or 0 for k, r in pr.items() if not k.startswith("__"))
    fn = sum(r.get("fn") or 0 for k, r in pr.items() if not k.startswith("__"))
    return tp, fp, fn


def show(tag, pr):
    ma, mi = pr.get("__macro__", {}), pr.get("__micro__", {})
    tp, fp, fn = counts(pr)
    print(f"  {tag:18s} macroP={ma.get('precision', 0):.3f} macroR={ma.get('recall', 0):.3f} "
          f"microP={mi.get('precision', 0):.3f} microR={mi.get('recall', 0):.3f} "
          f"TP/FP/FN={tp}/{fp}/{fn}")


def merge(prs):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for pr in prs:
        for ph, row in pr.items():
            if ph.startswith("__"):
                continue
            tp[ph] += row.get("tp") or 0
            fp[ph] += row.get("fp") or 0
            fn[ph] += row.get("fn") or 0
    return metrics._summarize_pr(tp, fp, fn)


def nemo_cache(lang):
    id_to_text = {}
    with open(f"{HERE}/turns_{lang}.csv") as f:
        for row in csv.DictReader(f):
            id_to_text[row["id"]] = row["text"]
    ents = json.load(open(f"{HERE}/nemo_entities_{lang}.json"))
    return {id_to_text[i]: e for i, e in ents.items() if i in id_to_text}


def main():
    full = list(load_dataset())
    base_cfg = load_config(None)
    agg = defaultdict(list)

    for lang in LANGS:
        ds = [q for q in full if q.language == lang]
        print(f"\n=== {lang.upper()} ({len(ds)} queries) ===")

        pr, _ = metrics.detection_precision_recall_with_records(ds)
        show("regex_only", pr)
        agg["regex_only"].append(pr)

        try:
            cfg = base_cfg.with_overrides(
                {"detector": {"second_pass": "presidio", "presidio": {"language": lang}}})
            det = build_second_pass_detector(cfg)
            pr, _ = metrics.detection_precision_recall_with_records(ds, second_pass_detector=det)
            show("regex_presidio", pr)
            agg["regex_presidio"].append(pr)
            close = getattr(det, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            print(f"  regex_presidio     skipped: {type(exc).__name__}: {exc}")

        try:
            cfg = base_cfg.with_overrides({"detector": {"second_pass": "opf"}})
            det = build_second_pass_detector(cfg, opf_model_name=lang)
            pr, _ = metrics.detection_precision_recall_with_records(ds, second_pass_detector=det)
            show("regex_opf_routed", pr)
            agg["regex_opf_routed"].append(pr)
            close = getattr(det, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            print(f"  regex_opf_routed   skipped: {type(exc).__name__}: {exc}")

        nemo = NemoCachedDetector(nemo_cache(lang))
        pr, _ = metrics.detection_precision_recall_with_records(ds, second_pass_detector=nemo)
        show("regex_nemo", pr)
        agg["regex_nemo"].append(pr)
        if nemo.misses:
            print(f"  !! cache misses: {len(nemo.misses)}")

    print(f"\n=== TRILINGUAL AGGREGATE (all {sum(1 for q in full if q.language in LANGS)} queries) ===")
    for tag in ("regex_only", "regex_presidio", "regex_opf_routed", "regex_nemo"):
        if agg[tag]:
            show(tag, merge(agg[tag]))


if __name__ == "__main__":
    main()
