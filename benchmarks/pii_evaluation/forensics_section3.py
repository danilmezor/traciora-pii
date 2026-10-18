"""Forensic dive into §3 virtualization-pipeline failures.

Two artifacts per (model, language) cell:

  1. Leaks — every raw-PII literal that crossed the boundary under
     virtualization, with the placeholder it *should* have been replaced
     by, the message role it appeared in, and the originating record id.
  2. Tool-call failures — every query whose virtualization run failed to
     match all expected tool calls, with the actual invocations the model
     made and the agent's final response. Lets us see whether the failure
     was rehydration-induced (bad arg fidelity) vs. agent-routing
     (wrong/missing tool name) vs. model-side semantic confusion.

The script loads the same dataset + RunResult shape the metrics module
consumes, so leak detection is byte-identical to ``model_pii_exposure_*``.

Usage::

    PYTHONPATH=src python -m benchmarks.pii_evaluation.forensics_section3 \
        --out benchmarks/pii_evaluation/results/section3_forensics.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarks.pii_evaluation.dataset import EvalQuery, load_dataset
from benchmarks.pii_evaluation.pipelines import RunResult
from benchmarks.pii_evaluation.tools import ToolInvocation

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------
# Cell discovery
# ----------------------------------------------------------------------

def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-").lower()


def discover_cells() -> List[Tuple[str, str, Path]]:
    """Return [(model_slug, language, json_path)] for every complete §3 cell.

    Smoke artifacts (slugs containing "smoke") are excluded so only the
    full 309-query cells contribute to the forensic report.
    """
    cells: List[Tuple[str, str, Path]] = []
    for p in sorted(RESULTS_DIR.glob("results_*__opf-*.json")):
        m = re.match(r"results_(.+)__opf-(en|de|ru)$", p.stem)
        if not m:
            continue
        slug = m.group(1)
        if "smoke" in slug:
            continue
        cells.append((slug, m.group(2), p))
    return cells


def _load_results(json_path: Path) -> Tuple[int, List[RunResult]]:
    raw = json.loads(json_path.read_text())
    results = []
    for r in raw["results"]:
        results.append(
            RunResult(
                query_id=r["query_id"],
                config=r["config"],
                final_response=r.get("final_response", ""),
                transcript=r.get("transcript", []),
                sent_to_llm=r.get("sent_to_llm", []),
                tool_invocations=[
                    ToolInvocation(
                        tool_name=ti.get("tool_name", ""),
                        raw_args=ti.get("raw_args", ti.get("raw_arguments", {})) or {},
                        result=ti.get("result", ""),
                        record_id=ti.get("record_id"),
                    )
                    for ti in r.get("tool_invocations", [])
                ],
                elapsed_seconds=r.get("elapsed_seconds", 0.0),
                leak_guard_triggers=r.get("leak_guard_triggers", 0),
                error=r.get("error"),
            )
        )
    return raw["dataset_size"], results


# ----------------------------------------------------------------------
# Leak forensics
# ----------------------------------------------------------------------

def _boundary_lookup(query: EvalQuery) -> Dict[str, object]:
    values = {}
    for item in getattr(query, "boundary_pii_values", []) or []:
        if item.value and item.value not in values:
            values[item.value] = item
    if values:
        return values
    for span in query.pii_spans:
        if span.value and span.value not in values:
            values[span.value] = span
    return values


def find_leaks(
    results: Iterable[RunResult],
    by_id: Dict[str, EvalQuery],
    *,
    config: str = "virtualization",
) -> List[Dict[str, object]]:
    leaks: List[Dict[str, object]] = []
    for r in results:
        if r.config != config:
            continue
        query = by_id.get(r.query_id)
        if query is None:
            continue
        boundary = _boundary_lookup(query)
        if not boundary:
            continue
        # Aggregate per (query, value, role) so a value appearing many
        # times in one message isn't double-counted as 'distinct leak'.
        seen: Dict[Tuple[str, str, str], int] = defaultdict(int)
        contexts: Dict[Tuple[str, str, str], str] = {}
        for entry in r.sent_to_llm:
            content = entry.get("content")
            role = entry.get("role", "?")
            if not isinstance(content, str) or not content:
                continue
            for value in boundary:
                hits = content.count(value)
                if hits == 0:
                    continue
                key = (r.query_id, value, role)
                seen[key] += hits
                if key not in contexts:
                    idx = content.find(value)
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(value) + 40)
                    ctx = content[start:end].replace("\n", " ")
                    contexts[key] = ctx
        for (qid, value, role), count in seen.items():
            item = boundary[value]
            leaks.append({
                "query_id": qid,
                "bucket": getattr(query, "bucket", ""),
                "config": r.config,
                "role": role,
                "value": value,
                "placeholder": getattr(item, "placeholder", ""),
                "source": getattr(item, "source", ""),
                "record_id": getattr(item, "record_id", None),
                "occurrences_in_sent": count,
                "context": contexts[(qid, value, role)],
            })
    return leaks


# ----------------------------------------------------------------------
# Tool-call failure forensics (virtualization-only)
# ----------------------------------------------------------------------

def find_tool_failures(
    results: Iterable[RunResult],
    by_id: Dict[str, EvalQuery],
    *,
    config: str = "virtualization",
) -> List[Dict[str, object]]:
    failures: List[Dict[str, object]] = []
    for r in results:
        if r.config != config:
            continue
        query = by_id.get(r.query_id)
        if query is None:
            continue
        expected = query.effective_tool_calls
        if not expected:
            # No expected tool calls → not counted toward TSR denominator.
            continue
        missing: List[Dict[str, str]] = []
        for exp in expected:
            hit = False
            for inv in r.tool_invocations:
                if inv.tool_name != exp.tool_name:
                    continue
                if exp.expected_record_id and inv.record_id != exp.expected_record_id:
                    continue
                hit = True
                break
            if not hit:
                missing.append({
                    "tool_name": exp.tool_name,
                    "expected_record_id": exp.expected_record_id or "",
                })
        if not missing:
            continue
        invoked = [
            {
                "tool_name": inv.tool_name,
                "record_id": inv.record_id or "",
                "raw_arguments": str(inv.raw_args)[:200] if inv.raw_args else "",
            }
            for inv in r.tool_invocations
        ]
        failures.append({
            "query_id": r.query_id,
            "bucket": getattr(query, "bucket", ""),
            "expected_count": len(expected),
            "missing": missing,
            "invoked": invoked,
            "final_response": (r.final_response or "")[:240].replace("\n", " "),
            "error": r.error,
        })
    return failures


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _bucket_summary(items: Sequence[Dict[str, object]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.get("bucket", "?")] += 1
    if not counts:
        return ""
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return parts


def _placeholder_summary(leaks: Sequence[Dict[str, object]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for leak in leaks:
        counts[leak.get("placeholder", "?") or "?"] += 1
    if not counts:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def render_cell_leaks(model_slug: str, language: str, leaks: Sequence[Dict[str, object]]) -> str:
    lines = [
        f"### Leaks — `{model_slug}` / {language}  (n={len(leaks)})",
        "",
    ]
    if not leaks:
        lines.append("_No raw-PII transmissions crossed the boundary._")
        return "\n".join(lines)
    lines.append(f"By placeholder: {_placeholder_summary(leaks)}")
    lines.append("")
    lines.append("| query_id | bucket | role | placeholder | source | record_id | n | value | context |")
    lines.append("|---|---|---|---|---|---|---:|---|---|")
    for leak in sorted(leaks, key=lambda x: (str(x["query_id"]), x["placeholder"])):
        ctx = (leak["context"] or "").replace("|", "\\|").replace("`", "")
        val = str(leak["value"]).replace("|", "\\|").replace("`", "")
        record_id = leak.get("record_id") or ""
        lines.append(
            f"| {leak['query_id']} | {leak.get('bucket','')} | "
            f"{leak['role']} | `{leak['placeholder']}` | "
            f"{leak.get('source','')} | {record_id} | "
            f"{leak['occurrences_in_sent']} | `{val}` | {ctx} |"
        )
    return "\n".join(lines)


def render_cell_failures(model_slug: str, language: str, fails: Sequence[Dict[str, object]]) -> str:
    lines = [
        f"### Tool-call failures — `{model_slug}` / {language}  (n={len(fails)})",
        "",
    ]
    if not fails:
        lines.append("_All expected tool calls matched._")
        return "\n".join(lines)
    lines.append(f"By bucket: {_bucket_summary(fails)}")
    lines.append("")
    for fail in sorted(fails, key=lambda x: str(x["query_id"])):
        lines.append(f"**`{fail['query_id']}`**  (bucket={fail.get('bucket','')}, expected={fail['expected_count']})")
        lines.append("")
        if fail["missing"]:
            lines.append("- Missing:")
            for m in fail["missing"]:
                rid = f" record_id=`{m['expected_record_id']}`" if m["expected_record_id"] else ""
                lines.append(f"  - `{m['tool_name']}`{rid}")
        if fail["invoked"]:
            lines.append("- Invoked:")
            for inv in fail["invoked"]:
                rid = f" record_id=`{inv['record_id']}`" if inv["record_id"] else ""
                args = inv["raw_arguments"]
                args_part = f"  args=`{args}`" if args else ""
                lines.append(f"  - `{inv['tool_name']}`{rid}{args_part}")
        else:
            lines.append("- Invoked: _(none)_")
        if fail["error"]:
            lines.append(f"- Error: `{fail['error']}`")
        final = fail.get("final_response") or ""
        if final:
            lines.append(f"- Final response (truncated 240ch): {final}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "section3_forensics.md")
    parser.add_argument(
        "--config", default="virtualization",
        help="Pipeline config to dive into (default: virtualization).",
    )
    args = parser.parse_args()

    dataset = list(load_dataset())
    by_id = {q.id: q for q in dataset}

    cells = discover_cells()
    if not cells:
        print("No §3 cell JSONs found under results/.")
        return 1

    sections: List[str] = ["# §3 Forensics — virtualization leaks & tool-call failures", ""]

    # Aggregate summary table first.
    summary_rows: List[Tuple[str, str, int, int]] = []  # (model, lang, leak_count, fail_count)
    leaks_by_cell: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    fails_by_cell: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for model_slug, language, path in cells:
        _, results = _load_results(path)
        leaks = find_leaks(results, by_id, config=args.config)
        fails = find_tool_failures(results, by_id, config=args.config)
        leaks_by_cell[(model_slug, language)] = leaks
        fails_by_cell[(model_slug, language)] = fails
        summary_rows.append((model_slug, language, len(leaks), len(fails)))

    sections.append("## Summary")
    sections.append("")
    sections.append("Leak count counts **distinct (query, value, role) triples** under the "
                    "virtualization pipeline — different from the cumulative `model_pii_exposure_count` metric "
                    "(which double-counts repeated occurrences in the same message). Failure count is the "
                    "number of queries whose expected tool calls were not all matched.")
    sections.append("")
    sections.append("| Model | Lang | Distinct leak triples | Failing queries |")
    sections.append("|---|---|---:|---:|")
    for model, lang, lcount, fcount in summary_rows:
        sections.append(f"| `{model}` | {lang} | {lcount} | {fcount} |")
    sections.append("")

    # Aggregate: which placeholders & buckets drive the leak / failure distribution?
    sections.append("## Leak distribution by placeholder")
    sections.append("")
    placeholder_counts: Dict[str, int] = defaultdict(int)
    for leaks in leaks_by_cell.values():
        for leak in leaks:
            placeholder_counts[leak.get("placeholder", "?") or "?"] += 1
    if placeholder_counts:
        sections.append("| Placeholder | Count |")
        sections.append("|---|---:|")
        for k, v in sorted(placeholder_counts.items(), key=lambda kv: -kv[1]):
            sections.append(f"| `{k}` | {v} |")
    sections.append("")

    sections.append("## Failure distribution by bucket")
    sections.append("")
    bucket_counts: Dict[str, int] = defaultdict(int)
    for fails in fails_by_cell.values():
        for fail in fails:
            bucket_counts[fail.get("bucket", "?") or "?"] += 1
    if bucket_counts:
        sections.append("| Bucket | Count |")
        sections.append("|---|---:|")
        for k, v in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
            sections.append(f"| `{k}` | {v} |")
    sections.append("")

    # Per-cell details
    sections.append("## Per-cell details")
    sections.append("")
    for model_slug, language, _ in cells:
        sections.append(render_cell_leaks(model_slug, language, leaks_by_cell[(model_slug, language)]))
        sections.append("")
        sections.append(render_cell_failures(model_slug, language, fails_by_cell[(model_slug, language)]))
        sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections))
    print(f"Wrote {args.out}")
    print(f"Cells analyzed: {len(cells)}")
    print(f"Total leak triples: {sum(s[2] for s in summary_rows)}")
    print(f"Total failing queries: {sum(s[3] for s in summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
