#!/usr/bin/env bash
# §3 full framework matrix: 8 models × 3 languages × 3 pipelines.
# Each (model, lang) cell runs the matching OPF fine-tune so the
# second pass sees its in-distribution language.
#
# The model set matches the paper-grade run that produced the
# reference outputs under results/results_*__opf-{en,de,ru}.{json,md}
# and the aggregated section3_matrix.md / section3_forensics.md.
#
# Resume-friendly: each cell writes results_<model>__opf-<lang>.{json,md}
# and is skipped if both files already exist.
set -euo pipefail
cd "$(dirname "$0")/.."

models=(
  "anthropic/claude-sonnet-4.6"
  "anthropic/claude-haiku-4.5"
  "openai/gpt-5.4-mini"
  "openai/gpt-5.4-nano"
  "z-ai/glm-5.1"
  "z-ai/glm-4.7"
  "openai/gpt-oss-120b:free"
  "z-ai/glm-4.5-air:free"
)
langs=(en de ru)

results_dir="benchmarks/pii_evaluation/results"
mkdir -p "$results_dir"

export PII_EVAL_LLM_ENABLED=1
export OPF_MOE_TRITON=0

cell_complete() {
  local json=$1
  local md=$2
  [[ -s "$json" && -s "$md" ]] || return 1
  PYTHONPATH=src python -c 'import json, sys; print(0 if json.load(open(sys.argv[1])).get("dataset_size") == 103 else 1)' "$json" | grep -q '^0$'
}

total=$((${#models[@]} * ${#langs[@]}))
i=0
for model in "${models[@]}"; do
  # Slug-safe model name for filenames (matches _model_slug() rules).
  msafe=$(printf '%s' "$model" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9._-]\{1,\}/-/g; s/^-//; s/-$//')
  for lang in "${langs[@]}"; do
    i=$((i + 1))
    slug="${msafe}__opf-${lang}"
    json="$results_dir/results_${slug}.json"
    md="$results_dir/results_${slug}.md"
    log="$results_dir/run_${slug}.log"
    echo "[$i/$total] $model  lang=$lang  slug=$slug"
    if cell_complete "$json" "$md"; then
      echo "  -> skipping (already complete)"
      continue
    fi
    PII_EVAL_MODEL_NAME="$msafe" \
    PYTHONPATH=src python -m benchmarks.pii_evaluation.run_experiment \
      --language "$lang" \
      --second-pass opf --opf-model "$lang" \
      --llm-model "$model" \
      --llm-base-url https://openrouter.ai/api/v1 \
      2>&1 | tee "$log"
    echo "  -> done -> $md"
  done
done

echo "Matrix complete. $total result pairs under $results_dir."
