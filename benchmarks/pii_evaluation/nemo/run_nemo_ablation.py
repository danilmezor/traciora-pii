"""Score NeMo Anonymizer (batch-cached) as a second-pass detector on the EN slice,
side by side with regex_only and regex_presidio, using the harness's own
metrics.detection_precision_recall_with_records — identical semantics to Table I.
"""
import csv
import os
import json
import sys

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Cached inputs remain in the ignored results tree, separate from source code.
HERE = str(Path(os.environ.get(
    "TRACIORA_NEMO_DATA_DIR", REPO / "benchmarks/pii_evaluation/results/nemo",
)))
sys.path.insert(0, str(REPO))

from benchmarks.pii_evaluation.dataset import load_dataset  # noqa: E402
from benchmarks.pii_evaluation import metrics  # noqa: E402
from traciora_pii.pii import PIIFinding  # noqa: E402

# Nearest-native-placeholder map, mirroring the conventions of
# presidio_pii_detector.WIDE_LABEL_MAP: unmappable labels are dropped.
NEMO_MAP = {
    "phone_number": "[PHONE]", "fax_number": "[PHONE]",
    "email": "[EMAIL]",
    "first_name": "[PERSON_NAME]", "last_name": "[PERSON_NAME]",
    "street_address": "[STREET_ADDRESS]",
    "credit_debit_card": "[CARD]",
    "account_number": "[IBAN]",
    "date": "[DATE]", "date_of_birth": "[DATE]", "date_time": "[DATE]",
    "ssn": "[PERSONAL_ID]", "national_id": "[PERSONAL_ID]",
    "certificate_license_number": "[PERSONAL_ID]",
    "tax_id": "[DE_STEUER_ID]",
    "url": "[URL]", "ipv4": "[IP]", "ipv6": "[IP]",
    "vehicle_identifier": "[VIN]", "license_plate": "[LICENSE_PLATE]",
    "password": "[SECRET]", "api_key": "[SECRET]", "http_cookie": "[SECRET]",
}


class NemoCachedDetector:
    """detect(text) served from the abrabox batch run's entity cache."""

    def __init__(self, cache):
        self.cache = cache
        self.misses = []

    def detect(self, text):
        ents = self.cache.get(text)
        if ents is None:
            self.misses.append(text[:60])
            return []
        finds = []
        for e in ents:
            ph = NEMO_MAP.get(e["label"])
            if ph:
                finds.append(PIIFinding(detector="nemo", start=e["start"],
                                        end=e["end"], placeholder=ph))
        # Merge adjacent same-placeholder findings (first_name+last_name -> one
        # PERSON_NAME span) so granularity differences don't count as FPs.
        finds.sort(key=lambda f: (f.start, f.end))
        merged = []
        for f in finds:
            if merged and f.placeholder == merged[-1].placeholder and f.start - merged[-1].end <= 2:
                prev = merged[-1]
                merged[-1] = PIIFinding(detector="nemo", start=prev.start,
                                        end=max(prev.end, f.end), placeholder=f.placeholder)
            else:
                merged.append(f)
        return merged


def summarize(name, pr):
    macro, micro = pr.get("__macro__", {}), pr.get("__micro__", {})
    tp = sum(r.get("tp") or 0 for k, r in pr.items() if not k.startswith("__"))
    fp = sum(r.get("fp") or 0 for k, r in pr.items() if not k.startswith("__"))
    fn = sum(r.get("fn") or 0 for k, r in pr.items() if not k.startswith("__"))
    print(f"{name:16s} macroP={macro.get('precision', 0):.3f} macroR={macro.get('recall', 0):.3f} "
          f"microP={micro.get('precision', 0):.3f} microR={micro.get('recall', 0):.3f} "
          f"TP/FP/FN={tp}/{fp}/{fn}")
    return pr


def per_label(pr):
    for ph in sorted(k for k in pr if not k.startswith("__")):
        r = pr[ph]
        print(f"    {ph:18s} P={r.get('precision')} R={r.get('recall')} "
              f"tp/fp/fn={r.get('tp')}/{r.get('fp')}/{r.get('fn')}")


def main():
    dataset = [q for q in load_dataset() if q.language == "en"]
    print(f"EN queries: {len(dataset)}")

    id_to_text = {}
    with open(f"{HERE}/turns_en.csv") as f:
        for row in csv.DictReader(f):
            id_to_text[row["id"]] = row["text"]
    ents_by_id = json.load(open(f"{HERE}/nemo_entities_en.json"))
    cache = {id_to_text[i]: e for i, e in ents_by_id.items() if i in id_to_text}
    print(f"cache: {len(cache)} texts")

    print("\n== regex_only (EN slice) ==")
    pr, _ = metrics.detection_precision_recall_with_records(dataset)
    summarize("regex_only", pr)

    print("\n== regex_presidio (EN slice) ==")
    try:
        from traciora_pii.config import load_config
        from traciora_pii.pii_virtualizer import build_second_pass_detector
        cfg = load_config(None).with_overrides(
            {"detector": {"second_pass": "presidio", "presidio": {"language": "en"}}})
        det = build_second_pass_detector(cfg)
        try:
            pr, _ = metrics.detection_precision_recall_with_records(
                dataset, second_pass_detector=det)
            summarize("regex_presidio", pr)
        finally:
            close = getattr(det, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # noqa: BLE001
        print(f"presidio row error: {type(exc).__name__}: {exc}")

    print("\n== regex_nemo (EN slice) ==")
    nemo = NemoCachedDetector(cache)
    pr, _ = metrics.detection_precision_recall_with_records(
        dataset, second_pass_detector=nemo)
    summarize("regex_nemo", pr)
    if nemo.misses:
        print(f"CACHE MISSES ({len(nemo.misses)}):", nemo.misses[:3])
    per_label(pr)


if __name__ == "__main__":
    main()
