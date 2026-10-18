Experiment runners and acceptance criteria
=========================================

The numbered directories retain executable experiment runners and the
acceptance criteria used to assess cross-source detection, detector ablation,
and tool-calling utility.

Run from the repository root:
  bash claims/1/run.sh          Cross-source detection
  bash claims/2/run.sh          Detector comparison
  bash claims/3/run.sh          Tool-calling comparison

Each runner accepts --full for the broader experiment matrix. Consult its
source and --help behavior before running; optional models, corpora, and
configured LLM credentials may be needed.

Generated results go to benchmarks/pii_evaluation/results/ and are ignored.
Frozen reference snapshots under claims/N/expected/ are also ignored and are
not included in fresh clones. To compare against a previous run, supply its
snapshot locally. A fresh run cannot verify historical numbers without the
corresponding reference files and experimental setup.

Synthetic source fixtures remain tracked; the evaluation loader regenerates
its synthetic dataset when no local frozen dataset exists. See README.md and
CONFIGURATION.md for dependency setup and local data paths.
