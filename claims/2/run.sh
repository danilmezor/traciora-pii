#!/usr/bin/env bash
#
# claims/2/run.sh — reproduce paper §2 (detector ablation).
#
# Default mode (--smoke, ~15 min, no API cost):
#   4 local/non-LLM configs on the 309-query trilingual benchmark.
#
# Full mode (--full, detector LLM backend/OpenRouter required):
#   Smoke lineup plus the 3 LLM-detector rows used in the paper table.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="smoke"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) MODE="full"; shift;;
    --smoke) MODE="smoke"; shift;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
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

echo "================================================================"
if [ "$MODE" = "full" ]; then
  echo "Claim 2 — §2 FULL Detector Ablation (7 paper rows)"
else
  echo "Claim 2 — §2 SMOKE Detector Ablation (4 local rows)"
fi
echo "================================================================"

PII_EVAL_MODEL_NAME='ablation' \
python -m benchmarks.pii_evaluation.run_detector_ablation \
  --language all \
  --only regex_only,regex_opf_base,regex_opf_routed,regex_presidio

if [ "$MODE" = "full" ]; then
  if [ -z "${PII_EVAL_API_KEY:-}" ]; then
    echo "✗ PII_EVAL_API_KEY not set in .env (required for detector LLM rows)." >&2
    exit 1
  fi
  for llm in \
    "nvidia/nemotron-nano-9b-v2:free" \
    "qwen/qwen3.5-9b" \
    "mistralai/ministral-3b-2512"
  do
    echo "  -> regex_llm[$llm]"
    PII_EVAL_MODEL_NAME='ablation' \
    python -m benchmarks.pii_evaluation.run_detector_ablation \
      --language all \
      --only regex_llm \
      --detector-llm-model "$llm"
  done
  python -m benchmarks.pii_evaluation.aggregate_detector_ablation \
    --out benchmarks/pii_evaluation/results/section2_ablation_matrix.md
fi

echo
echo "✓ Claim 2 reproduction complete."
if [ "$MODE" = "full" ]; then
  echo "  Compare section2_ablation_matrix.md against claims/2/expected/full/."
else
  echo "  Compare detector_ablation_ablation.{md,json} against claims/2/expected/smoke/."
  echo "  For the full paper table, rerun: bash claims/2/run.sh --full"
fi
