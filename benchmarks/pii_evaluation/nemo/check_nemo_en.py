"""Forensics on the regex_nemo EN row: attribute every FP/FN to regex vs NeMo,
show the exact texts, and verify cache coverage — same composition and matching
logic as metrics._score_query_turn."""
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
from traciora_pii.pii import redact_pii_text  # noqa: E402
from benchmarks.pii_evaluation.nemo.run_nemo_ablation import NemoCachedDetector, NEMO_MAP  # noqa: E402

id_to_text = {}
with open(f"{HERE}/turns_en.csv") as f:
    for row in csv.DictReader(f):
        id_to_text[row["id"]] = row["text"]
ents_by_id = json.load(open(f"{HERE}/nemo_entities_en.json"))
cache = {id_to_text[i]: e for i, e in ents_by_id.items() if i in id_to_text}
nemo = NemoCachedDetector(cache)

dataset = [q for q in load_dataset() if q.language == "en"]

missing = [t for q in dataset for t in q.turns if t not in cache]
print(f"cache coverage: {len(cache)} texts; turns not in cache: {len(missing)}")

fp_by_src = defaultdict(int)
rows_fp, rows_fn = [], []

for q in dataset:
    gt_by_turn = metrics._gt_spans_for_query(q)
    for ti, text in enumerate(q.turns):
        gt = gt_by_turn.get(ti, [])
        _red, regex_f = redact_pii_text(text)
        findings = [(f, "regex") for f in regex_f]
        for sf in nemo.detect(text):
            if any(sf.start < rf.end and rf.start < sf.end for rf in regex_f):
                continue
            findings.append((sf, "nemo"))
        matched = set()
        for f, src in findings:
            hit = None
            for i, (ph, s, e) in enumerate(gt):
                if i in matched or f.placeholder != ph:
                    continue
                if f.start < e and s < f.end:
                    hit = i
                    break
            if hit is not None:
                matched.add(hit)
            else:
                fp_by_src[(f.placeholder, src)] += 1
                raw = [x for x in cache.get(text, [])
                       if x["start"] < f.end and f.start < x["end"]] if src == "nemo" else []
                rows_fp.append((q.id, q.bucket, f.placeholder, src,
                                text[f.start:f.end][:40],
                                (raw[0]["label"] if raw else "")))
        for i, (ph, s, e) in enumerate(gt):
            if i not in matched:
                overlapping = [x for x in cache.get(text, []) if x["start"] < e and s < x["end"]]
                rows_fn.append((q.id, q.bucket, ph, text[s:e][:40],
                                [(x["label"], round(x["score"], 2)) for x in overlapping]))

print(f"\n=== FALSE POSITIVES ({len(rows_fp)}) — placeholder, source, matched text, raw nemo label ===")
for r in rows_fp:
    print(f"  {r[0]:20s} {r[1]:15s} {r[2]:16s} src={r[3]:5s} '{r[4]}' {('nemo_label='+r[5]) if r[5] else ''}")
print("\nFP counts by (placeholder, source):")
for k, v in sorted(fp_by_src.items(), key=lambda kv: -kv[1]):
    print(f"  {k[0]:16s} {k[1]:5s} : {v}")

print(f"\n=== FALSE NEGATIVES ({len(rows_fn)}) — gold placeholder, gold value, nemo raw overlapping ===")
for r in rows_fn:
    print(f"  {r[0]:20s} {r[1]:15s} {r[2]:16s} gold='{r[3]}' nemo_overlap={r[4]}")
