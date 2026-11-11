#!/usr/bin/env bash
#
# scripts/reproduce/tool_calling.sh — tool-calling experiment (full
# virtualization framework against an LLM agent).
#
# Default mode (--smoke, ~5 min, ~$1):
#   Single English cell with claude-haiku-4.5, --limit 10 queries.
#   Verifies end-to-end virtualization on a small query slice.
#
# Full mode (--full, ~16-20h, ~$20-50):
#   8 LLMs x 3 languages x 3 pipelines x 309 queries matrix: task
#   success rate and bounded-residual leak numbers per cell.
#
# Reference: benchmarks/pii_evaluation/results/reference/tool_calling/{smoke,full}/
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
  echo "✗ .env missing. Run bash scripts/setup.sh first." >&2
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
reference_dir="${results_dir}/reference/tool_calling"
mkdir -p "$results_dir"

if [ "$MODE" = "full" ]; then
  echo "================================================================"
  echo "Tool calling — FULL matrix (8 LLMs x 3 langs x 3 pipelines x 309q)"
  echo "               ~16-20 hours wall clock, \$20-50 OpenRouter spend"
  echo "================================================================"
  bash scripts/run_section3_matrix.sh
  # Aggregator autodetects every results_*__opf-{en,de,ru}.json triple
  # under results/, so it covers whatever model set was run.
  python -m benchmarks.pii_evaluation.aggregate_section3 \
    --out "${results_dir}/section3_matrix.md"
  python -m benchmarks.pii_evaluation.forensics_section3 \
    --out "${results_dir}/section3_forensics.md"
  echo
  echo "✓ Tool calling FULL run complete."
  echo "  See section3_matrix.md (task success per cell) and section3_forensics.md"
  echo "  (raw-PII leak audit). Compare against ${reference_dir}/full/."
else
  echo "================================================================"
  echo "Tool calling — SMOKE (en x claude-haiku-4.5 x 10 queries)"
  echo "               ~5 minutes, ~\$1 OpenRouter spend"
  echo "================================================================"
  PII_EVAL_LLM_ENABLED=1 \
  PII_EVAL_MODEL_NAME=anthropic-claude-haiku-4.5 \
  python -m benchmarks.pii_evaluation.run_experiment \
    --language en --limit 10 \
    --second-pass opf --opf-model en \
    --llm-model anthropic/claude-haiku-4.5 \
    --llm-base-url https://openrouter.ai/api/v1
  echo
  echo "✓ Tool calling SMOKE run complete."
  echo "  See results_anthropic-claude-haiku-4.5__opf-en.{json,md}"
  echo "  Compare against ${reference_dir}/smoke/."
  echo "  For the 24-cell matrix, rerun: bash scripts/reproduce/tool_calling.sh --full"
fi
