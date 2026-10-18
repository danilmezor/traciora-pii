#!/usr/bin/env bash
#
# claims/3/run.sh — reproduce paper §3 (full virtualization framework).
#
# Default mode (--smoke, ~5 min, ~$1):
#   Single English cell with claude-haiku-4.5, --limit 10 queries.
#   Verifies end-to-end virtualization on a small query slice.
#
# Full mode (--full, ~16-20h, ~$20-50):
#   Full 8 LLMs x 3 languages x 3 pipelines x 309 queries matrix.
#   Reproduces the paper's headline TSR and bounded-residual leak numbers.
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

if [ -z "${PII_EVAL_API_KEY:-}" ]; then
  echo "✗ PII_EVAL_API_KEY not set in .env." >&2
  echo "  Get an OpenRouter key at https://openrouter.ai (format sk-or-v1-...)" >&2
  echo "  and set PII_EVAL_API_KEY=sk-or-v1-... in .env." >&2
  exit 1
fi

export PYTHONPATH=src
export OPF_MOE_TRITON=0

results_dir="benchmarks/pii_evaluation/results"
mkdir -p "$results_dir"

if [ "$MODE" = "full" ]; then
  echo "================================================================"
  echo "Claim 3 — §3 FULL matrix (8 LLMs x 3 langs x 3 pipelines x 309q)"
  echo "          ~16-20 hours wall clock, \$20-50 OpenRouter spend"
  echo "================================================================"
  bash scripts/run_section3_matrix.sh
  # Aggregator autodetects every results_*__opf-{en,de,ru}.json triple
  # under results/, so it stays correct as the model set evolves.
  python -m benchmarks.pii_evaluation.aggregate_section3 \
    --out "${results_dir}/section3_matrix.md"
  python -m benchmarks.pii_evaluation.forensics_section3 \
    --out "${results_dir}/section3_forensics.md"
  echo
  echo "✓ Claim 3 FULL reproduction complete."
  echo "  See section3_matrix.md (TSR per cell) and section3_forensics.md"
  echo "  (raw-PII leak audit). Acceptance bounds in claims/3/claim.txt."
else
  echo "================================================================"
  echo "Claim 3 — §3 SMOKE (en x claude-haiku-4.5 x 10 queries)"
  echo "          ~5 minutes, ~\$1 OpenRouter spend"
  echo "================================================================"
  PII_EVAL_LLM_ENABLED=1 \
  PII_EVAL_MODEL_NAME=anthropic-claude-haiku-4.5 \
  python -m benchmarks.pii_evaluation.run_experiment \
    --language en --limit 10 \
    --second-pass opf --opf-model en \
    --llm-model anthropic/claude-haiku-4.5 \
    --llm-base-url https://openrouter.ai/api/v1
  echo
  echo "✓ Claim 3 SMOKE reproduction complete."
  echo "  See results_anthropic-claude-haiku-4.5__opf-en.{json,md}"
  echo "  Diff against claims/3/expected/smoke/."
  echo "  For the full 24-cell matrix that produced the paper's headline"
  echo "  numbers, rerun: bash claims/3/run.sh --full"
fi
