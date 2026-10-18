#!/usr/bin/env bash
#
# install.sh — project installation entry point.
#
# Thin wrapper that runs scripts/setup.sh. Kept at the repo root so
# installation follows the standard install.sh convention.
#
# scripts/setup.sh performs:
#   - pip install of the four extras groups in pyproject.toml
#     ([dev,opf,presidio,benchmarks]).
#   - spaCy NER model download (en_core_web_lg, de_core_news_lg,
#     ru_core_news_lg) for Presidio.
#   - .env wizard (copies .env.example to .env if missing).
#
# Time: ~10-15 min on a fresh machine.
#
set -euo pipefail
cd "$(dirname "$0")"
exec bash scripts/setup.sh "$@"
