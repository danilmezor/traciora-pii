# Reproducing the paper

The repository carries the runners behind the paper on PII virtualization for
LLM tool-calling agents. Three experiments back its headline results:
cross-source detection, a detector comparison and tool-calling utility under
virtualization. Each has a fast smoke mode and a full mode at paper scale.

## Setup

```bash
bash scripts/setup.sh
```

This installs the `dev`, `opf`, `presidio` and `benchmarks` extras, downloads
the spaCy models Presidio needs (`en_core_web_lg`, `de_core_news_lg`,
`ru_core_news_lg`), creates `.env` from `.env.example` and runs the unit tests.
Then set an OpenRouter key in `.env`:

```bash
PII_EVAL_API_KEY=sk-or-v1-...
```

The fine-tuned OPF checkpoints (`pii-anon/opf-{en,de,ru}-v2`) download from
Hugging Face on first use. The runners set `OPF_MOE_TRITON=0`, which macOS
needs. Outputs go to `benchmarks/pii_evaluation/results/`.

## Quick check of everything

```bash
bash scripts/reproduce.sh                       # ~30 min, ~$1
bash scripts/reproduce.sh --full-tool-calling   # tool calling at full scale
```

The default runs the smoke mode of all three experiments plus the unit tests.
It confirms every pipeline works and that the headline trends hold on reduced
data; it does not reproduce the paper's numbers.

## Cross-source detection

**Statement.** Per-language fine-tuned OPF checkpoints beat Microsoft Presidio
on held-out PII corpora from other sources, in all three languages, under
projection-symmetric scoring.

| Corpus | Language | Micro F1, regex+OPF minus regex+Presidio |
|---|---|---:|
| `nemotron_en` | English | +0.110 |
| `ai4privacy_de` | German | +0.059 |
| `wolframko_ru` | Russian | +0.051 |

n = 1000 rows per corpus. The OPF lead holds in every language.

```bash
bash scripts/reproduce/detection.sh          # smoke: ~10 min, no API cost
bash scripts/reproduce/detection.sh --full   # ~3-4 h, ~$0.05
```

- **Smoke:** 3 corpora × 3 detectors (`regex_only`, `regex_opf` with the
  language's fine-tune, `regex_presidio`) at 200 rows. Expected: every cell
  completes, OPF beats Presidio by at least 0.025 micro F1 in each language,
  and the ranking OPF > Presidio > regex only holds.
- **Full:** 3 corpora × 7 detectors at 1000 rows, adding four small LLM
  detectors: `nvidia/nemotron-nano-9b-v2`, `qwen/qwen3.5-9b`,
  `google/gemma-3-4b-it`, `mistralai/ministral-3b-2512`.
- **Needs:** `opf` and `presidio` extras, spaCy models, Hugging Face access to
  the three corpora; the full mode also needs `PII_EVAL_API_KEY`.
- **Variance:** Presidio scores can move by about ±0.02 micro F1 across spaCy
  model releases. The regex-only and OPF cells are deterministic.
- **Output:** `results/imported_eval_*.md` (smoke), `results/n1000/` (full).
  The micro F1 row of each per-placeholder table is the headline metric.

## Detector comparison

**Statement.** On the 309-query trilingual benchmark, layered regex plus routed
per-language OPF has the best recall and micro F1. The Qwen 9B LLM detector has
the best precision, which is the deployment trade-off.

| Detector | Micro P | Micro R | Micro F1 |
|---|---:|---:|---:|
| regex only | 0.908 | 0.692 | 0.785 |
| regex + Presidio | 0.883 | 0.724 | 0.796 |
| regex + base OPF | 0.880 | 0.891 | 0.885 |
| regex + routed OPF fine-tunes | 0.899 | 0.965 | 0.930 |
| regex + LLM (ministral-3b) | 0.912 | 0.731 | 0.811 |
| regex + LLM (nemotron-9b) | 0.917 | 0.779 | 0.842 |
| regex + LLM (qwen-9b) | 0.918 | 0.792 | 0.850 |

```bash
bash scripts/reproduce/detectors.sh          # smoke: ~15 min, no API cost
bash scripts/reproduce/detectors.sh --full   # adds the three LLM rows
```

- **Smoke:** the four local detectors. Expected: routed OPF micro recall in
  [0.93, 0.97], micro F1 in [0.91, 0.95], and both above every other smoke row.
- **Full:** adds `nvidia/nemotron-nano-9b-v2:free`, `qwen/qwen3.5-9b` and
  `mistralai/ministral-3b-2512` as LLM detectors, then builds the seven-row
  table. Expected: routed OPF has the highest recall and F1; qwen the highest
  precision.
- **Needs:** `opf` and `presidio` extras; the full mode also needs
  `PII_EVAL_API_KEY`.
- **Output:** `results/detector_ablation_ablation.{md,json}` (smoke),
  `results/section2_ablation_matrix.md` (full).

## Tool-calling utility under virtualization

**Statement.** With the full pipeline on (regex plus per-language OPF
fine-tune, vault, broker, leak guard), an LLM tool-calling agent keeps most of
its baseline tool-call success while raw personal data sent to the model falls
by more than 99.5%. The small residual comes from typed identifiers in
code-switched text.

| Measure | Baseline | Virtualization |
|---|---:|---:|
| Raw PII transmissions, all models | 17,631 | 65 |
| Raw PII transmissions, per model | 2,066-2,311 | 4-12 |
| Average task success rate (TSR) | ~92.3% | ~90.5% |

That is a 99.63% reduction for a ~1.8 point TSR cost. Seven of eight models stay
within 5.5 points of baseline; `anthropic/claude-haiku-4.5` drops 20.3 points,
mostly on Russian.

```bash
bash scripts/reproduce/tool_calling.sh          # smoke: ~5 min, ~$1
bash scripts/reproduce/tool_calling.sh --full   # ~16-20 h, ~$20-50
```

- **Smoke:** `anthropic/claude-haiku-4.5`, English, 10 queries. Expected: the run
  completes and virtualization TSR is in [0.7, 1.0].
- **Full:** 8 models × 3 languages × 3 pipelines (baseline, destructive,
  virtualization) × 309 queries (103 per language). Models:
  `anthropic/claude-sonnet-4.6`, `anthropic/claude-haiku-4.5`,
  `openai/gpt-5.4-mini`, `openai/gpt-5.4-nano`, `z-ai/glm-5.1`, `z-ai/glm-4.7`,
  `openai/gpt-oss-120b:free`, `z-ai/glm-4.5-air:free`. Expected: mean TSR change
  within ±5 points, fewer than 100 raw transmissions under virtualization, a
  reduction above 99.5%, and a 0.0% leak-guard trigger rate in every cell.
- **Needs:** `opf` and `benchmarks` extras, `PII_EVAL_API_KEY`.
- **Output:** `results/results_<model>__opf-<lang>.{json,md}` per cell,
  `results/section3_matrix.md` (TSR) and `results/section3_forensics.md`
  (raw-PII leak audit). Cells that already have output are skipped, so a
  stopped run resumes; delete a cell's files to rerun it.

## Every experiment in one sweep

```bash
bash scripts/run_paper_eval.sh --dry-run   # plan and cost estimate
bash scripts/run_paper_eval.sh             # asks before spending
bash scripts/run_paper_eval.sh --phase 3   # one phase; also --phase 1,2,4
bash scripts/run_paper_eval.sh --yes       # skip the cost prompt
```

Phases: 0 pre-flight, 1 cross-source detection, 2 detector comparison,
3 tool calling, 4 security robustness (stress reports per result and a
scenario × model matrix), 5 final tables. About 20-24 hours and $20-50 of
OpenRouter spend in total, run sequentially and resume-safe.

## Comparing with a previous run

Reference outputs are not shipped. To compare, put a previous run's outputs
under `benchmarks/pii_evaluation/results/reference/<detection|detectors|tool_calling>/{smoke,full}/`
and diff.

Exact numbers can drift: OpenRouter providers update served models, and
third-party responses vary slightly even at temperature 0. Bootstrap
confidence intervals use seed 42 and 1000 resamples and are deterministic for
identical inputs. Hugging Face cache state affects first-run timings only.
