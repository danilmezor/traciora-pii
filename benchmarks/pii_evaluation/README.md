# PII Virtualization Evaluation Experiment

> Data policy: source tools and synthetic fixture definitions are tracked.
> Generated datasets, imported caches, reports, result snapshots, logs,
> training runs, and model checkpoints stay local and are ignored by Git.
> Historical result files referenced below must be supplied locally.

This folder contains the end-to-end experiment that produces the metrics reported in **§1–§4** of the paper. It is **not** a unit test — it makes real LLM calls and writes a results table that gets pasted directly into the paper.

## What it measures

| Metric | Source |
|---|---|
| Tool-call success rate per configuration | `metrics.tool_call_success_rate` |
| Raw PII reaching the LLM per configuration | `metrics.model_pii_exposure_count` |
| End-to-end latency (median, p95, mean) | `metrics.latency_stats` |
| Per-detector latency contribution (regex / second-pass) | `metrics.latency_contribution_report` |
| Leak-guard trigger rate (virtualization only) | `metrics.leak_guard_trigger_rate` |
| Per-detector precision and recall | `metrics.detection_precision_recall` |
| Per-(language × placeholder) precision and recall | `metrics.detection_precision_recall_per_language` |
| Per-(length-bucket × language × placeholder) precision and recall | `metrics.detection_precision_recall_by_length_bucket` |

Each query is run through three configurations:
1. **baseline** — no PII protection at all (utility upper bound)
2. **destructive** — packaged `PIIRedactor` applied to every message and tool result
3. **virtualization** — packaged vault + tool broker + leak guard

## Files

```
__init__.py                       package marker
dataset/                          trilingual EN/DE/RU scenario-driven dataset
                                  (YAML scenarios + frozen JSONL + renderer)
tools.py                          real LangChain @tool functions over an in-memory CRM
pipelines.py                      shared agent loop + the three pipeline configurations
metrics.py                        M1–M5 computation + per-language / length-bucket breakdowns
metrics_argfidelity.py            argument fidelity + cross-turn token stability
run_experiment.py                 main §3 driver (full framework eval)
run_detector_ablation.py          §2 consolidated 5-row detector ablation table driver
run_imported_dataset_eval.py      §1 cross-source detection generalization
                                  (ai4privacy_de / nemotron_en / wolframko_ru)
measure_vault_ref_stability.py    vault ref-token cross-turn stability probe
                                  (maintainer-only forensic tool; not in any claim runner)
diagnose_cross_turn_failures.py   companion diagnostic for cross-turn lemma failures
                                  (maintainer-only)
tests/                            pytest regression suite (parity, metrics, leakage, ablation)
results/                          gitignored output (JSON + markdown tables)
                                  — overwritten by every run; the
                                    frozen reference snapshots live under
                                    claims/N/expected/.
```

## Running

### Dataset only (no LLM, free, fast)

Computes only the detection precision/recall table — useful for iterating on the dataset and the regex detectors without burning API budget.

```bash
python -m benchmarks.pii_evaluation.run_experiment --dataset-only
```

### Full experiment (real LLM)

Requires both an env-var enable flag and a working LLM provider key. The standalone harness builds `langchain_openai.ChatOpenAI` directly from `PII_EVAL_API_KEY`, `PII_EVAL_BASE_URL`, and `PII_EVAL_MODEL_NAME`.

```bash
PII_EVAL_LLM_ENABLED=1 python -m benchmarks.pii_evaluation.run_experiment
```

To override the model just for this experiment without touching package defaults:

```bash
PII_EVAL_LLM_ENABLED=1 \
PII_EVAL_MODEL_NAME=deepseek-chat \
PII_EVAL_API_KEY=sk-... \
PII_EVAL_BASE_URL=https://api.deepseek.com/v1 \
python -m benchmarks.pii_evaluation.run_experiment
```

Cost estimate: 300 queries × 3 configurations × ~2 LLM calls per query ≈ 1,800 calls per model. On `deepseek-chat` this is a few dollars per full run.

## Output

After a full run:

- `results/results.json` — complete per-query transcripts and the metric breakdowns
- `results/results.md` — two markdown tables (end-to-end metrics + per-detector P/R) shaped for direct paste into Section 6

The script also prints a one-line summary to stdout for at-a-glance verification.

## Sanity bounds for the numbers

If the experiment produces numbers outside these bounds, **investigate before publishing**:

- baseline tool-call success: should be at or near 100% (the LLM has nothing in its way)
- destructive tool-call success: should be substantially below 100% — the tool receives placeholder strings on every PII-dependent query and returns "not found"
- virtualization tool-call success: should be at or near 100% — any failure is a finding worth understanding
- raw PII reaching the model under virtualization: must be **0** — anything else is a wiring bug
- leak-guard trigger rate under virtualization: probably non-zero on the multi-turn bucket; this is the most interesting empirical result

## Reproducibility

Temperature is fixed at 0.0. The experiment is otherwise free of randomness. Two runs against the same model should produce identical transcripts; if they do not, the provider is not honoring `temperature=0` and that fact should be noted in Section 6.

## Detector ablation via `run_experiment.py`

Second-pass detector selection lives on the main driver (`--second-pass {none,opf,llm,presidio}`) plus the orthogonal `--opf-model` flag for OPF model registry selection. Detection P/R is reported per language (en / de / ru).

```bash
# Regex-only baseline
python -m benchmarks.pii_evaluation.run_experiment --dataset-only --second-pass none

# OPF, default registered model
python -m benchmarks.pii_evaluation.run_experiment --dataset-only --second-pass opf

# LLM second-pass via OpenRouter (default) — model is required at runtime
python -m benchmarks.pii_evaluation.run_experiment \
    --dataset-only --second-pass llm \
    --detector-llm-model 'nvidia/nemotron-nano-9b-v2:free'

# LLM second-pass via a local OpenAI-compatible endpoint (LM Studio / Ollama / vLLM)
python -m benchmarks.pii_evaluation.run_experiment \
    --dataset-only --second-pass llm \
    --detector-llm-model 'nvidia/nemotron-nano-9b-v2' \
    --detector-llm-base-url 'http://localhost:1234/v1'

# Presidio (third-party detector baseline). Requires the optional extra
# and the matching spaCy model:
#   pip install -e '.[presidio]'
#   python -m spacy download en_core_web_sm  # or de_core_news_sm / ru_core_news_sm
PII_PRESIDIO_LANGUAGE=en \
python -m benchmarks.pii_evaluation.run_experiment --dataset-only --second-pass presidio
```

## Consolidated detector ablation

`run_detector_ablation.py` runs all five detector configurations against the same dataset slice and emits one comparison table:

```bash
# Full lineup (skips regex_llm if --detector-llm-model is not supplied)
PYTHONPATH=src python -m benchmarks.pii_evaluation.run_detector_ablation

# All five configs including the LLM second-pass
PYTHONPATH=src python -m benchmarks.pii_evaluation.run_detector_ablation \
    --detector-llm-model 'nvidia/nemotron-nano-9b-v2:free'

# Slice + opt-in subset
PYTHONPATH=src python -m benchmarks.pii_evaluation.run_detector_ablation \
    --language en --bucket single_turn \
    --only regex_only,regex_opf_base,regex_presidio
```

Configurations:

1. `regex_only` — first-pass regex detectors only.
2. `regex_opf_base` — regex + base OPF token classifier.
3. `regex_opf_routed` — regex + per-language fine-tuned OPF; languages without a registered fine-tune fall back to the registry default. Matches production routing.
4. `regex_llm` — regex + OpenAI-compatible LLM via OpenRouter (or any local endpoint). Requires `--detector-llm-model`; otherwise the row is reported with `status="skipped: …"`.
5. `regex_presidio` — regex + Microsoft Presidio (third-party baseline).

Configurations whose dependencies are missing are reported with `status="skipped: <reason>"` rather than dropped silently. Output lands in `results/detector_ablation_<slug>.{md,json}`.

### Anchor, 2026-10-27, M1 Pro CPU, commit 9b48661

`dataset_v1.jsonl`, 309 queries, 312 gold spans; OPF on `device: cpu` with the local
`fine_tuning/runs/{ru,en,de}-v2/checkpoint` fine-tunes for the routed row and
`openai/privacy-filter` for the base row. `results/` is gitignored, so this table is
the only kept copy; a run overwrites `results/detector_ablation_<slug>.*`.

| Configuration | TP | FP | FN | micro P | micro R | macro P | macro R | s/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `regex_only` | 272 | 0 | 40 | 1.000 | 0.872 | 0.857 | 0.802 | 0.000 |
| `regex_opf_base` | 300 | 13 | 12 | 0.958 | 0.962 | 0.872 | 0.831 | 2.773 |
| `regex_opf_routed` | 311 | 11 | 1 | 0.966 | 0.997 | 0.784 | 0.817 | 1.373 |

Regex-only misses are by design (30 names, 3 addresses) plus 6 German Steuer-IDs and one
Russian passport without a keyword. Routed false positives: 2 each of `[DE_STEUER_ID]`,
`[PERSONAL_ID]`, `[PERSON_NAME]`, `[SECRET]`, one each of `[RU_BANK_ACCOUNT]`, `[RU_OGRN]`,
`[STREET_ADDRESS]`.

The routed row on `device: mlx` (2026-10-27, same Mac, base 628aca5 plus the MLX backend;
other sessions kept the load average between 7 and 80, so s/query is indicative). The
same three local fine-tunes, converted by `prepare_opf`, run in 512-token windows with
128 tokens of overlap:

| Configuration | TP | FP | FN | micro P | micro R | macro P | macro R | s/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `regex_opf_routed`, mlx bf16 | 311 | 11 | 1 | 0.966 | 0.997 | 0.784 | 0.817 | 0.097 |
| `regex_opf_routed`, mlx 8-bit | 311 | 10 | 1 | 0.969 | 0.997 | 0.785 | 0.817 | 0.088 |

bf16 has the anchor's counts and the same false positives by placeholder. 8-bit differs
on one query: `scn-sec-710-ru` ("Найди клиента по референс-токену
phone_ref:ph_deadbeef00.") loses its `[PERSON_NAME]` false positive.

`anonymize_text` end to end on the four Russian profile workloads (regex, vault, MLX
second pass with the ru fine-tune, lemma), median of 3 after one warm-up, one process per
row, load average 8-10:

| workload | chars | mlx bf16 (ms) | mlx 8-bit (ms) |
|---|---:|---:|---:|
| small | 126 | 15.2 | 21.8 |
| medium | 1,629 | 62.2 | 78.9 |
| document | 39,466 | 1,265 | 1,381 |
| db_result | 66,974 | 3,847 | 4,053 |
| MLX active after load / peak | | 2.61 / 3.33 GiB | 1.39 / 2.11 GiB |
| phys_footprint peak | | 4.34 GB | 3.64 GB |
| process start to ready (incl. `prepare_opf` and self-check) | | 12.5 s | 10.3 s |

## Security stress — archetype 3 (tool-exception leakage)

The `tool_exception_leakage` workflow drives a tool that raises with the raw card number embedded in its `ValueError`. The agent loop in `pipelines.py` catches the exception and stringifies it verbatim into the next ToolMessage — that's the leak surface this archetype tests. Pass conditions are inverted from the other archetypes:

- **baseline / destructive**: `exposure > 0` is the *expected* leak. Zero would mean the harness silently scrubbed an exception body, which would be a wiring regression.
- **virtualization**: `exposure == 0`. The leak guard inspects ToolMessage content and triggers re-anonymization before the message reaches the model.

## Imported-dataset detection eval

Runs the full regex+OPF detector against an external HF dataset and scores per-placeholder P/R/F1.

```bash
PYTHONPATH=src python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
    --dataset wolframko_ru --target-rows 200 --raw-limit 600

PYTHONPATH=src python -m benchmarks.pii_evaluation.run_imported_dataset_eval \
    --dataset nemotron_en --target-rows 200 --raw-limit 800
```

## Vault ref-token stability

```bash
python -m benchmarks.pii_evaluation.measure_vault_ref_stability                    # default OPF model
python -m benchmarks.pii_evaluation.measure_vault_ref_stability --opf-model en     # registry override
```
