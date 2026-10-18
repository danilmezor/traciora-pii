#!/usr/bin/env bash
#
# scripts/reproduce.sh — reviewer SMOKE verification of the pipeline.
#
# This script is the fast end-to-end sanity check, NOT a paper-grade
# reproduction. It confirms every pipeline works and that the headline
# trends hold qualitatively on reduced N. For paper-grade numbers
# matching the PDF tables exactly, use one of:
#   - `claims/1/run.sh --full`         (§1 at n=1000)
#   - `claims/2/run.sh --full`         (§2 with the 3 paper LLM rows)
#   - `claims/3/run.sh --full`         (§3 full 8×3×3 matrix)
#   - `scripts/run_paper_eval.sh`      (full paper sweep §1+§2+§3+§4)
#
# Default mode (~30 min, ~$1 OpenRouter) — SMOKE across §1/§2/§3:
#   - §1 smoke: 3 datasets × 3 non-LLM configs at n=200/corpus
#     (paper §1 table uses n=1000 + 4 LLM detector configs;
#      that's `claims/1/run.sh --full`).
#   - §2 smoke: 4 non-LLM detector configs on the 309-query bench
#     (paper §2 table also includes 3 LLM rows;
#      that's `claims/2/run.sh --full`).
#   - Unit tests (283/283 should pass: library + benchmarks + fine-tuning).
#   - §3 smoke: 1 LLM (claude-haiku-4.5) × English × --limit 10 queries
#     (paper §3 uses 8 LLMs × 3 languages × 309 queries;
#      that's `--full-section3` below or `claims/3/run.sh --full`).
#
# Opt-in full §3 mode (`--full-section3`, ~16-20h, ~$20-50 OpenRouter):
#   - §3 only at paper N: 8 LLMs × 3 languages × 3 pipelines × 309 queries.
#     Reproduces the paper's headline TSR / leak numbers exactly.
#     §1 and §2 stay in smoke mode under this flag — for the full
#     §1+§2+§3+§4 sweep use `scripts/run_paper_eval.sh`.
#
# Requires: scripts/setup.sh must have completed first. PII_EVAL_API_KEY
# must be set to a valid OpenRouter key in .env.
#
set -euo pipefail
cd "$(dirname "$0")/.."

FULL_S3=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-section3|--full-s3) FULL_S3=1; shift;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

# Sanity: have we been set up?
if [ ! -f .env ]; then
  echo "✗ .env missing. Run scripts/setup.sh first." >&2
  exit 1
fi
# Load .env into the environment. (`set -a` exports every var.)
set -a; source .env; set +a

export PYTHONPATH=src
export OPF_MOE_TRITON=0   # macOS-safe; harmless on Linux

results_dir="benchmarks/pii_evaluation/results"
mkdir -p "$results_dir"

# ----------------------------------------------------------------------
# §1 Detection Generalization — 9 cells (3 datasets × 3 configs)
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "[1/4] §1 Detection Generalization — SMOKE  (~10 min, no API cost)"
echo "      9 cells (3 datasets × 3 non-LLM configs) at n=200/corpus."
echo "      For paper-grade n=1000 + LLM configs: claims/1/run.sh --full"
echo "================================================================"
for spec in "nemotron_en:en" "ai4privacy_de:de" "wolframko_ru:ru"; do
  ds="${spec%:*}"; lang="${spec##*:}"
  for det in regex_only regex_opf regex_presidio; do
    suffix=""
    if [ "$det" = "regex_opf" ]; then suffix="__${lang}"; fi
    out="${results_dir}/imported_eval_${ds}__${det}${suffix}.md"
    echo "  -> $ds / $det / opf=$lang"
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
echo "  ✓ §1 done — see imported_eval_*.md under $results_dir"

# ----------------------------------------------------------------------
# §2 Detector Ablation — 4 configs across all 3 languages
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "[2/4] §2 Detector Ablation — SMOKE  (~15 min, no API cost)"
echo "      4 non-LLM detector configs on the 309-query bench."
echo "      For the 3 paper-reported LLM rows: claims/2/run.sh --full"
echo "================================================================"
python -m benchmarks.pii_evaluation.run_detector_ablation --language all
echo "  ✓ §2 done — see detector_ablation_*.{json,md}"

# ----------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "[3/4] Unit tests  (~30 s)"
echo "================================================================"
python -m pytest -q 2>&1 | tail -5

# ----------------------------------------------------------------------
# §3 Full Framework — smoke (default) or full matrix (opt-in)
# ----------------------------------------------------------------------
echo
echo "================================================================"
if [ "$FULL_S3" -eq 1 ]; then
  echo "[4/4] §3 FULL matrix  (~16-20h sequential, \$20-50 OpenRouter)"
  echo "================================================================"
  if [ -z "${PII_EVAL_API_KEY:-}" ]; then
    echo "✗ PII_EVAL_API_KEY not set. Cannot run §3." >&2
    exit 1
  fi
  bash scripts/run_section3_matrix.sh
  # Aggregator autodetects every results_*__opf-{en,de,ru}.json triple
  # under results/, so the script stays correct as the model set evolves.
  python -m benchmarks.pii_evaluation.aggregate_section3 \
    --out "${results_dir}/section3_matrix.md"
  python -m benchmarks.pii_evaluation.forensics_section3 \
    --out "${results_dir}/section3_forensics.md"
  echo "  ✓ §3 done — see section3_matrix.md + section3_forensics.md"
else
  echo "[4/4] §3 Smoke  (~5 min, ~\$1 OpenRouter)"
  echo "================================================================"
  if [ -z "${PII_EVAL_API_KEY:-}" ]; then
    echo "✗ PII_EVAL_API_KEY not set. Skipping §3 smoke."
    echo "   To run §3, fill PII_EVAL_API_KEY in .env (OpenRouter key)."
    echo "================================================================"
  else
    PII_EVAL_LLM_ENABLED=1 \
    PII_EVAL_MODEL_NAME=anthropic-claude-haiku-4.5 \
    python -m benchmarks.pii_evaluation.run_experiment \
      --language en --limit 10 \
      --second-pass opf --opf-model en \
      --llm-model anthropic/claude-haiku-4.5 \
      --llm-base-url https://openrouter.ai/api/v1
    echo "  ✓ §3 smoke done — see results_anthropic-claude-haiku-4.5__opf-en.{json,md}"
  fi
fi

echo
echo "================================================================"
echo "Smoke verification complete. Files to diff against claims/N/expected/:"
echo
echo "  §1 — cross-source detection: imported_eval_*.md"
echo "  §2 — detector ablation:       detector_ablation_*.{json,md}"
echo "  §3 — full framework:          results_*__opf-*.{json,md} (+ matrix/forensics if --full-section3)"
echo
echo "All output files live under: $results_dir"
echo "================================================================"
