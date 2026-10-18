#!/usr/bin/env bash
# §1 detection-generalization matrix at n=1000 rows per corpus.
# 3 corpora × 7 detector configs = 21 cells.
#
# Cells per corpus:
#   regex_only
#   regex_presidio
#   regex_opf__<lang>            (v2 fine-tune for the corpus's locale)
#   regex_llm__nvidia/nemotron-nano-9b-v2:free
#   regex_llm__qwen/qwen3.5-9b
#   regex_llm__google/gemma-3-4b-it
#   regex_llm__mistralai/ministral-3b-2512
#
# Resume-friendly: each cell writes results/n1000/imported_eval_<corpus>__<suffix>.md
# and is skipped if that file already exists. Re-run after a partial sweep to
# fill in only the missing cells.
#
# Output lives under results/n1000/ so legacy smaller imported-eval runs in
# results/ are preserved alongside the current paper matrix.
set -euo pipefail
cd "$(dirname "$0")/.."

# Route OPF model lookups at the local v2 fine-tunes under fine_tuning/runs/
# when the local override exists. In a fresh clone (no traciora.local.yaml) fall
# through to traciora.yaml's public pii-anon/opf-*-v2 HF refs.
if [[ -f "$PWD/traciora.local.yaml" ]]; then
  export TRACIORA_PII_CONFIG="${TRACIORA_PII_CONFIG:-$PWD/traciora.local.yaml}"
fi

results_root="benchmarks/pii_evaluation/results"
out_dir="$results_root/n1000"
log_dir="$out_dir/_logs"
mkdir -p "$out_dir" "$log_dir"

# Dataset → (opf model id, raw_limit). raw_limit is target/survival × ~1.3
# safety margin based on n=200 runs (ai4privacy ~40%, wolframko ~33%,
# nemotron ~25% survival through the transform chain).
declare -A OPF_MODEL=(
  [ai4privacy_de]=de
  [wolframko_ru]=ru
  [nemotron_en]=en
)
declare -A RAW_LIMIT=(
  [ai4privacy_de]=3000
  [wolframko_ru]=3600
  [nemotron_en]=5000
)

datasets=(ai4privacy_de wolframko_ru nemotron_en)

llm_models=(
  "nvidia/nemotron-nano-9b-v2"
  "qwen/qwen3.5-9b"
  "google/gemma-3-4b-it"
  "mistralai/ministral-3b-2512"
)

# Mirror the filename-slug rule from run_imported_dataset_eval.py:
#   non-[a-zA-Z0-9._-] collapsed to '-', stripped from ends, lowercased.
slugify() {
  printf '%s' "$1" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9._-]\{1,\}/-/g; s/^-//; s/-$//'
}

run_cell() {
  local dataset=$1
  local suffix=$2
  shift 2
  local out_path="$out_dir/imported_eval_${dataset}__${suffix}.md"
  local log_path="$log_dir/${dataset}__${suffix}.log"
  if [[ -s "$out_path" ]]; then
    echo "  [skip] $out_path already exists"
    return 0
  fi
  echo "  [run]  $suffix  -> $out_path"
  PYTHONPATH=src python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
    --dataset "$dataset" \
    --target-rows 1000 \
    --raw-limit "${RAW_LIMIT[$dataset]}" \
    --out "$out_path" \
    "$@" \
    >"$log_path" 2>&1
  echo "         done -> $(grep -E '^- Elapsed' "$out_path" | head -1)"
}

total_cells=$(( ${#datasets[@]} * (3 + ${#llm_models[@]}) ))
echo "==> §1 matrix sweep @ n=1000: $total_cells cells total"
echo "==> output: $out_dir"
echo

cell_idx=0
for dataset in "${datasets[@]}"; do
  echo "=== dataset: $dataset (raw_limit=${RAW_LIMIT[$dataset]}, opf_model=${OPF_MODEL[$dataset]}) ==="

  cell_idx=$((cell_idx + 1))
  echo "[$cell_idx/$total_cells] regex_only"
  run_cell "$dataset" "regex_only" --detector regex_only

  cell_idx=$((cell_idx + 1))
  echo "[$cell_idx/$total_cells] regex_presidio"
  run_cell "$dataset" "regex_presidio" --detector regex_presidio

  cell_idx=$((cell_idx + 1))
  opf_model=${OPF_MODEL[$dataset]}
  echo "[$cell_idx/$total_cells] regex_opf__${opf_model}"
  run_cell "$dataset" "regex_opf__${opf_model}" \
    --detector regex_opf --opf-model "$opf_model"

  for llm in "${llm_models[@]}"; do
    cell_idx=$((cell_idx + 1))
    llm_slug=$(slugify "$llm")
    echo "[$cell_idx/$total_cells] regex_llm__${llm_slug}"
    run_cell "$dataset" "regex_llm__${llm_slug}" \
      --detector regex_llm --detector-llm-model "$llm"
  done

  echo
done

echo "==> sweep complete. Results in $out_dir"
