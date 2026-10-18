#!/usr/bin/env bash
#
# claims/1/run.sh — reproduce paper §1 (cross-source detection generalization).
#
# Two modes:
#   --smoke (default, ~10 min, no API cost):
#     9 cells (3 corpora × 3 detectors) at n=200 rows. Fast verification
#     that the regex+OPF and regex+Presidio pipelines work end-to-end on
#     each corpus and that OPF beats Presidio.
#     Diff target: claims/1/expected/smoke/
#
#   --full (~3-4 h, ~$0.05 OpenRouter for the four LLM cells):
#     21 cells (3 corpora × 7 detectors) at n=1000 rows. Matches the
#     paper headline tables exactly. Detectors: regex_only, regex_presidio,
#     regex_opf, regex_llm[{nvidia/nemotron-nano-9b-v2, qwen/qwen3.5-9b,
#     google/gemma-3-4b-it, mistralai/ministral-3b-2512}].
#     Diff target: claims/1/expected/full/
#
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="smoke"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)  MODE="full";  shift;;
    --smoke) MODE="smoke"; shift;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

if [ ! -f .env ]; then
  echo "✗ .env missing. Run bash install.sh first." >&2
  exit 1
fi
set -a; source .env; set +a

export PYTHONPATH=src
export OPF_MOE_TRITON=0

results_dir="benchmarks/pii_evaluation/results"
mkdir -p "$results_dir"

if [ "$MODE" = "full" ]; then
  echo "================================================================"
  echo "Claim 1 — §1 SMOKE+LLM matrix (21 cells x 1000 rows, ~3-4 hours)"
  echo "          Compare against claims/1/expected/full/"
  echo "================================================================"
  if [ -z "${PII_EVAL_API_KEY:-}" ]; then
    echo "✗ PII_EVAL_API_KEY not set in .env (required for the 4 LLM cells)." >&2
    exit 1
  fi
  bash scripts/run_section1_matrix_n1000.sh
  echo
  echo "✓ Claim 1 FULL reproduction complete."
  echo "  Outputs under ${results_dir}/n1000/."
  echo "  Diff against claims/1/expected/full/."
else
  echo "================================================================"
  echo "Claim 1 — §1 SMOKE (9 cells x 200 rows, ~10 min)"
  echo "          Compare against claims/1/expected/smoke/"
  echo "================================================================"
  for spec in "nemotron_en:en" "ai4privacy_de:de" "wolframko_ru:ru"; do
    ds="${spec%:*}"
    lang="${spec##*:}"
    for det in regex_only regex_opf regex_presidio; do
      suffix=""
      if [ "$det" = "regex_opf" ]; then suffix="__${lang}"; fi
      out="${results_dir}/imported_eval_${ds}__${det}${suffix}.md"
      echo "  -> ${ds} / ${det}"
      if [ "$det" = "regex_opf" ]; then
        python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
          --dataset "$ds" --detector "$det" --opf-model "$lang" \
          --target-rows 200 --raw-limit 600 \
          --out "$out" >/dev/null
      else
        python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
          --dataset "$ds" --detector "$det" \
          --target-rows 200 --raw-limit 600 \
          --out "$out" >/dev/null
      fi
    done
  done
  echo
  echo "✓ Claim 1 SMOKE reproduction complete."
  echo "  Outputs at ${results_dir}/imported_eval_*.md."
  echo "  Diff against claims/1/expected/smoke/."
  echo "  For paper-grade numbers, rerun: bash claims/1/run.sh --full"
fi
