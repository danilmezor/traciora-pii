#!/usr/bin/env bash
#
# scripts/setup.sh — one-shot setup for project users.
#
# Idempotent: re-running re-checks each phase and skips finished work.
# Total time on a fresh box: ~10-15 minutes.
#
# What it does:
#   1. Installs the framework + the four optional dependency groups
#      (opf, presidio, benchmarks, dev) into the current Python env.
#   2. Downloads the three spaCy NER models Presidio needs.
#   3. Bootstraps `.env` from `.env.example` if missing, and prompts
#      the reviewer to fill in the OpenRouter API key.
#   4. Surfaces the macOS-specific OPF_MOE_TRITON=0 workaround.
#   5. Runs the unit-test suite (no network, no API key needed) to
#      confirm the install is healthy.
#
# What it does NOT do:
#   - Eagerly fetch the fine-tuned OPF models from HuggingFace. The
#     framework's OPF loader (`traciora_pii.opf_pii_detector.
#     _resolve_opf_checkpoint`) fetches them lazily on first use, so
#     the first §1 / §2 / §3 invocation will block briefly while it
#     pulls ~600 MB per language into `~/.cache/huggingface/`.
#
set -euo pipefail
cd "$(dirname "$0")/.."

OK="✓"; WARN="!"; FAIL="✗"

# ----------------------------------------------------------------------
# Pre-flight: Python version + venv guard
# ----------------------------------------------------------------------
# requires-python = ">=3.11" in pyproject.toml. The lockfile is compiled
# against 3.11 so we don't want reviewers running into version-specific
# resolver surprises on 3.10 or earlier.
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
  pyver=$(python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || echo unknown)
  echo "${FAIL} Python 3.11+ required (current: ${pyver})." >&2
  echo "       Create a fresh env:  conda create -y -n traciora_pii python=3.11 && conda activate traciora_pii" >&2
  exit 1
fi
# Refuse to pollute system Python — installs must go into a venv/conda env.
# Three accepted signals: a venv-style prefix mismatch, an active conda env
# that isn't `base`, or the legacy `VIRTUAL_ENV` env var.
in_venv=0
if python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null; then
  in_venv=1
elif [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
  in_venv=1
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  in_venv=1
fi
if [[ $in_venv -eq 0 ]]; then
  echo "${FAIL} Not running inside a virtualenv / conda env." >&2
  echo "       Create a fresh env:  conda create -y -n traciora_pii python=3.11 && conda activate traciora_pii" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Phase 1 — Python deps
# ----------------------------------------------------------------------
echo
echo "==> [1/5] Installing traciora_pii + optional dependency groups"
python -m pip install --upgrade pip --quiet
# Quote the extras so zsh doesn't try to expand the brackets as a glob.
pip install --quiet -e '.[opf,presidio,benchmarks,dev]'
echo "  ${OK} traciora_pii installed (with extras: opf, presidio, benchmarks, dev)"

# ----------------------------------------------------------------------
# Phase 2 — spaCy NER models (Presidio dependency)
# ----------------------------------------------------------------------
echo
echo "==> [2/5] spaCy NER models (used by the Presidio baseline)"
for model in en_core_web_lg de_core_news_lg ru_core_news_lg; do
  if python -c "import spacy; spacy.load('${model}')" >/dev/null 2>&1; then
    echo "  ${OK} ${model} already installed"
  else
    echo "  ${WARN} downloading ${model} (~500-700 MB)..."
    # Don't suppress stderr — surface download failures to the operator
    # instead of silently falling through to the next iteration.
    python -m spacy download "${model}" >/dev/null
    echo "  ${OK} ${model} installed"
  fi
done

# ----------------------------------------------------------------------
# Phase 3 — .env wizard
# ----------------------------------------------------------------------
echo
echo "==> [3/5] Environment file"
if [ -f .env ]; then
  echo "  ${OK} .env already exists"
else
  cp .env.example .env
  echo "  ${OK} created .env from .env.example"
fi

# Prompt only if the eval API key is missing or unset.
if ! grep -Eq '^PII_EVAL_API_KEY=sk-' .env 2>/dev/null; then
  echo
  echo "  ${WARN} .env needs PII_EVAL_API_KEY set to your OpenRouter key"
  echo "       (https://openrouter.ai/keys — required only for §3 eval)"
  echo "       Open .env and fill in:"
  echo "         PII_EVAL_API_KEY=sk-or-v1-..."
fi

# ----------------------------------------------------------------------
# Phase 4 — macOS workaround
# ----------------------------------------------------------------------
echo
echo "==> [4/5] Platform-specific notes"
if [[ "$(uname)" == "Darwin" ]]; then
  echo "  ${WARN} macOS detected"
  echo "       Triton-backed MoE kernels are not available on macOS. Before"
  echo "       running any OPF-backed eval (§1/§2/§3), export:"
  echo "         export OPF_MOE_TRITON=0"
  echo "       scripts/reproduce.sh sets this automatically."
else
  echo "  ${OK} Linux / non-Darwin platform — no platform workarounds needed"
fi

# ----------------------------------------------------------------------
# Phase 5 — Smoke pytest (no network, no API key)
# ----------------------------------------------------------------------
echo
echo "==> [5/5] Running unit tests (no network, no API key required)"
python -m pytest -q 2>&1 | tail -5

echo
echo "================================================================"
echo "Setup complete."
echo
echo "Next steps:"
echo "  1. (If skipped) fill PII_EVAL_API_KEY in .env"
echo "  2. Run a smoke verification:    bash scripts/reproduce.sh"
echo "     (~30 min, ~\$1 OpenRouter)"
echo "  3. Full §3 matrix (optional):   bash scripts/reproduce.sh --full-section3"
echo "     (~16-20h, ~\$20-50 OpenRouter)"
echo "================================================================"
