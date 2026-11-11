#!/usr/bin/env bash
#
# scripts/run_paper_eval.sh — every experiment at full N in one sweep.
#
# Phases:
#   0  pre-flight  — env + model availability probe + cost gate
#   1  detection   — cross-source detection (3 datasets × 7 detector configs)
#   2  ablation    — detector ablation (1 non-LLM run + 3 regex_llm runs)
#   3  tool calling — full framework (8 LLMs × 3 languages × 3 pipelines)
#   4  security    — security robustness (per-cell stress reports + pivot)
#   5  aggregate   — final summary tables (markdown)
#
# Resume-safe: every cell skips when its output file already exists.
# Sequential by design (predictable cost, avoids OpenRouter rate-limit fan-out).
# Paid tool-calling models run before free-tier ones so quota loss on free models doesn't
# block the headline-cost cells.
#
# Usage:
#   bash scripts/run_paper_eval.sh                # all phases, ask before tool-calling cost
#   bash scripts/run_paper_eval.sh --phase 3      # just tool calling
#   bash scripts/run_paper_eval.sh --phase 1,2,4  # multiple
#   bash scripts/run_paper_eval.sh --dry-run      # print plan + cost estimate
#   bash scripts/run_paper_eval.sh --yes          # auto-confirm cost gate
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ----------------------------------------------------------------------
# Model lineups
# ----------------------------------------------------------------------

# Layer-2 LLM detectors in the detection experiment (n=1000 matrix).
DETECTION_LLMS=(
  "nvidia/nemotron-nano-9b-v2"
  "qwen/qwen3.5-9b"
  "google/gemma-3-4b-it"
  "mistralai/ministral-3b-2512"
)

# Layer-2 LLM detector rows in the ablation (309-query table).
# gemma is left out here because the detection experiment already covers it.
ABLATION_LLMS=(
  "nvidia/nemotron-nano-9b-v2:free"
  "qwen/qwen3.5-9b"
  "mistralai/ministral-3b-2512"
)

# Tool-calling eval LLMs (the agent's LLM). Paid first, free last so rate-limit failures
# on free-tier models don't waste paid quota on retries.
EVAL_LLMS=(
  "anthropic/claude-sonnet-4.6"
  "anthropic/claude-haiku-4.5"
  "openai/gpt-5.4-mini"
  "openai/gpt-5.4-nano"
  "z-ai/glm-5.1"
  "z-ai/glm-4.7"
  "openai/gpt-oss-120b:free"
  "z-ai/glm-4.5-air:free"
)

# Detection dataset → OPF policy language.
DATASETS=("nemotron_en:en" "ai4privacy_de:de" "wolframko_ru:ru")

RESULTS_DIR="benchmarks/pii_evaluation/results"
OPENROUTER_URL="https://openrouter.ai/api/v1"

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------

PHASES="0,1,2,3,4,5"
DRY_RUN=0
AUTO_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASES="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --yes)
      AUTO_YES=1; shift;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0;;
    *)
      echo "✗ unknown arg: $1" >&2; exit 1;;
  esac
done

run_phase() { [[ ",${PHASES}," == *",$1,"* ]]; }

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_msafe() {
  # Filesystem-safe slug matching benchmarks/pii_evaluation/cli_common.py.
  printf '%s' "$1" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9._-]\{1,\}/-/g; s/^-//; s/-$//'
}

_cell_complete() {
  # Args: <json_path> <md_path>. Returns 0 (success) iff both exist and non-empty.
  [[ -s "$1" && -s "$2" ]]
}

_tool_calling_cell_complete() {
  # Full tool-calling language cells contain 103 queries. Smoke cells use the same
  # filename pattern with dataset_size=10, so size-check before skipping.
  [[ -s "$1" && -s "$2" ]] || return 1
  python -c 'import json, sys; print(0 if json.load(open(sys.argv[1])).get("dataset_size") == 103 else 1)' "$1" | grep -q '^0$'
}

# ----------------------------------------------------------------------
# Phase 0 — pre-flight
# ----------------------------------------------------------------------

if run_phase 0; then
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] Phase 0 — pre-flight"
  echo "================================================================"

  if [[ ! -f .env ]]; then
    echo "✗ .env not found. Run scripts/setup.sh first." >&2; exit 1
  fi
  set -a; source .env; set +a
  if [[ -z "${PII_EVAL_API_KEY:-}" ]]; then
    echo "✗ PII_EVAL_API_KEY missing from .env." >&2; exit 1
  fi
  if [[ ! -f traciora.local.yaml ]]; then
    echo "! traciora.local.yaml not present — OPF models will be auto-fetched"
    echo "  from HuggingFace on first use. Continuing."
  fi
  export PYTHONPATH=src
  export OPF_MOE_TRITON=0
  # Only point TRACIORA_PII_CONFIG at the local override if the file exists; otherwise
  # let load_config() read the shipped traciora.yaml (which registers the public
  # pii-anon/opf-*-v2 HF checkpoints).
  if [[ -f "$PWD/traciora.local.yaml" ]]; then
    export TRACIORA_PII_CONFIG="${TRACIORA_PII_CONFIG:-$PWD/traciora.local.yaml}"
  fi

  echo
  echo "Detection layer-2 LLM detectors (${#DETECTION_LLMS[@]}):"
  printf '  - %s\n' "${DETECTION_LLMS[@]}"
  echo
  echo "Ablation layer-2 LLM detector rows (${#ABLATION_LLMS[@]}):"
  printf '  - %s\n' "${ABLATION_LLMS[@]}"
  echo
  echo "Tool-calling eval LLMs (${#EVAL_LLMS[@]}):"
  printf '  - %s\n' "${EVAL_LLMS[@]}"
  echo
  echo "Cost / time envelope (sequential):"
  echo "  detection    — 21 cells, ~3-4h,       ~\$0.05 (n=1000 matrix)"
  echo "  ablation     — 4 invocations, ~25 min, ~\$0"
  echo "  tool calling — 24 cells, ~16-20h,      ~\$20-50"
  echo "  security+agg — derived, ~7 min,        ~\$0"
  echo "  -----"
  echo "  TOTAL: ~20-24h,  ~\$20-50 OpenRouter"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "(dry-run) stopping before any work; rerun without --dry-run to proceed."
    exit 0
  fi

  if [[ "$AUTO_YES" -ne 1 ]]; then
    echo
    read -r -p "Proceed with full eval? [y/N] " yn
    case "$yn" in [yY]*) echo "Confirmed.";; *) echo "Aborted."; exit 0;; esac
  fi
fi

# Re-source .env in case Phase 0 was skipped (rerun + phase-restart).
[[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH=src
export OPF_MOE_TRITON=0
if [[ -f "$PWD/traciora.local.yaml" ]]; then
  export TRACIORA_PII_CONFIG="${TRACIORA_PII_CONFIG:-$PWD/traciora.local.yaml}"
fi

# ----------------------------------------------------------------------
# Phase 1 — detection
# ----------------------------------------------------------------------

if run_phase 1; then
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] Phase 1 — detection"
  echo "================================================================"

  bash scripts/run_section1_matrix_n1000.sh
fi

# ----------------------------------------------------------------------
# Phase 2 — detector ablation
# ----------------------------------------------------------------------

if run_phase 2; then
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] Phase 2 — detector ablation"
  echo "================================================================"

  # Non-LLM lineup (4 configs in one invocation).
  non_llm_slug="$(PII_EVAL_MODEL_NAME='ablation' \
    python -c 'from benchmarks.pii_evaluation.cli_common import model_slug; print(model_slug(fallback="ablation"))')"
  non_llm_out="${RESULTS_DIR}/detector_ablation_${non_llm_slug}.md"
  if [[ -s "$non_llm_out" ]]; then
    echo "  ✓ ablation non-LLM lineup (cached)"
  else
    echo "[$(date +%H:%M:%S)] ablation non-LLM lineup (regex_only + opf_base + opf_routed + presidio)"
    PII_EVAL_MODEL_NAME='ablation' \
    python -m benchmarks.pii_evaluation.run_detector_ablation \
      --language all \
      --only regex_only,regex_opf_base,regex_opf_routed,regex_presidio \
      >/dev/null 2>&1 || echo "    ✗ failed; continuing"
  fi

  # 3 LLM ablation rows.
  for llm in "${ABLATION_LLMS[@]}"; do
    llm_tag="$(_msafe "$llm")"
    llm_slug="${non_llm_slug}__llm-${llm_tag}"
    out="${RESULTS_DIR}/detector_ablation_${llm_slug}.md"
    if [[ -s "$out" ]]; then
      echo "  ✓ ablation regex_llm[$llm] (cached)"; continue
    fi
    echo "[$(date +%H:%M:%S)] ablation regex_llm[$llm]"
    PII_EVAL_MODEL_NAME='ablation' \
    python -m benchmarks.pii_evaluation.run_detector_ablation \
      --language all --only regex_llm \
      --detector-llm-model "$llm" \
      >/dev/null 2>&1 || echo "    ✗ failed; continuing"
  done
fi

# ----------------------------------------------------------------------
# Phase 3 — tool-calling matrix (8 models × 3 langs)
# ----------------------------------------------------------------------

if run_phase 3; then
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] Phase 3 — tool-calling matrix"
  echo "================================================================"
  export PII_EVAL_LLM_ENABLED=1

  total=$(( ${#EVAL_LLMS[@]} * 3 ))
  i=0
  for model in "${EVAL_LLMS[@]}"; do
    msafe="$(_msafe "$model")"
    for lang in en de ru; do
      i=$((i + 1))
      slug="${msafe}__opf-${lang}"
      json="${RESULTS_DIR}/results_${slug}.json"
      md="${RESULTS_DIR}/results_${slug}.md"
      if _tool_calling_cell_complete "$json" "$md"; then
        echo "  ✓ [$i/$total] tool calling $model / $lang (cached)"; continue
      fi
      log="${RESULTS_DIR}/run_${slug}.log"
      echo "[$(date +%H:%M:%S)] [$i/$total] tool calling $model / $lang"
      PII_EVAL_MODEL_NAME="$msafe" \
      python -m benchmarks.pii_evaluation.run_experiment \
        --language "$lang" \
        --second-pass opf --opf-model "$lang" \
        --llm-model "$model" \
        --llm-base-url "$OPENROUTER_URL" \
        > "$log" 2>&1 || echo "    ✗ failed; log=$log; continuing"
    done
  done
fi

# ----------------------------------------------------------------------
# Phase 4 — security robustness (per-cell + pivot aggregator)
# ----------------------------------------------------------------------

if run_phase 4; then
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] Phase 4 — security robustness"
  echo "================================================================"

  shopt -s nullglob
  for json in "${RESULTS_DIR}"/results_*__opf-*.json; do
    slug="$(basename "$json" .json | sed 's/^results_//')"
    out="${RESULTS_DIR}/stress_report_${slug}.md"
    if [[ -s "$out" ]]; then
      echo "  ✓ security stress_report $slug (cached)"; continue
    fi
    echo "[$(date +%H:%M:%S)] security stress_report $slug"
    python -m benchmarks.pii_evaluation.run_stress_report \
      --results "$json" --out "$out" >/dev/null 2>&1 || echo "    ✗ failed; continuing"
  done
  shopt -u nullglob

  echo "[$(date +%H:%M:%S)] security archetype × model matrix"
  python -m benchmarks.pii_evaluation.aggregate_stress_reports \
    --models "${EVAL_LLMS[@]}" \
    --out "${RESULTS_DIR}/section4_stress_matrix.md" || \
    echo "    ✗ matrix aggregator failed"
fi

# ----------------------------------------------------------------------
# Phase 5 — final aggregation
# ----------------------------------------------------------------------

if run_phase 5; then
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] Phase 5 — final aggregation"
  echo "================================================================"

  echo "[$(date +%H:%M:%S)] tool-calling matrix"
  python -m benchmarks.pii_evaluation.aggregate_section3 \
    --models "${EVAL_LLMS[@]}" \
    --out "${RESULTS_DIR}/section3_matrix.md" || echo "    ✗ failed"

  echo "[$(date +%H:%M:%S)] tool-calling forensics"
  python -m benchmarks.pii_evaluation.forensics_section3 \
    --out "${RESULTS_DIR}/section3_forensics.md" || echo "    ✗ failed"

  echo "[$(date +%H:%M:%S)] ablation matrix"
  python -m benchmarks.pii_evaluation.aggregate_detector_ablation \
    --out "${RESULTS_DIR}/section2_ablation_matrix.md" || echo "    ✗ failed"
fi

echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] DONE"
echo "================================================================"
echo "Outputs:"
echo "  detection:    ${RESULTS_DIR}/imported_eval_*.md"
echo "  ablation:     ${RESULTS_DIR}/section2_ablation_matrix.md"
echo "  tool calling: ${RESULTS_DIR}/section3_matrix.md + section3_forensics.md"
echo "  security:     ${RESULTS_DIR}/section4_stress_matrix.md"
