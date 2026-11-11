# Benchmarks and tools

The repository ships the tools used to measure the library: an evaluation
harness, a synthetic data generator, an importer for third-party corpora and an
OPF fine-tuning pipeline. Install them with:

```bash
pip install -e '.[benchmarks]'
```

Generated datasets, caches, reports, results, logs and checkpoints stay local
and are ignored by Git. The tracked fixtures (templates, seed vocabulary,
scenarios, policies) are synthetic.

## Evaluation harness

[`benchmarks/pii_evaluation/`](../benchmarks/pii_evaluation/README.md) runs an
LLM tool-calling agent over a trilingual (EN/DE/RU) scenario dataset of 309
queries, against real LangChain tools over an in-memory CRM. Each query runs
through three pipelines:

1. **baseline**: no protection (the utility upper bound);
2. **destructive**: `PIIRedactor` on every message and tool result;
3. **virtualization**: vault, tool broker and leak guard.

It reports tool-call success rate, raw PII reaching the model, end-to-end and
per-detector latency, leak-guard trigger rate, and detection precision and
recall per detector, language, placeholder and text length.

| Driver | What it does |
|---|---|
| `run_experiment` | The agent experiment above; `--dataset-only` scores detection without LLM calls |
| `run_detector_ablation` | One table comparing `regex_only`, `regex_opf_base`, `regex_opf_routed`, `regex_llm`, `regex_presidio` and `regex_encoder_routed` |
| `run_imported_dataset_eval` | Detection P/R/F1 on an imported third-party corpus |
| `run_stress_report`, `aggregate_stress_reports` | Security stress reports per result file and across models |
| `aggregate_section3`, `forensics_section3`, `aggregate_detector_ablation` | Aggregate tables and the raw-PII leak audit |
| `measure_vault_ref_stability` | Checks that tokens stay stable across turns |

```bash
# detection only, no LLM spend
python -m benchmarks.pii_evaluation.run_experiment --dataset-only --second-pass opf

# full agent experiment through OpenRouter
PII_EVAL_LLM_ENABLED=1 python -m benchmarks.pii_evaluation.run_experiment \
  --language en --second-pass opf --opf-model en \
  --llm-model anthropic/claude-haiku-4.5

# detector comparison
python -m benchmarks.pii_evaluation.run_detector_ablation --language all
```

LLM runs need `PII_EVAL_LLM_ENABLED=1` and `PII_EVAL_API_KEY`. There is no
default model; pass `--llm-model` (and `--detector-llm-model` for the LLM
detector) so every result names its model. Temperature is 0. Results go to
`benchmarks/pii_evaluation/results/`.

The stress suite includes a tool that raises with a raw card number in its
error. Under baseline and destructive the leak is expected; under
virtualization raw exposure must be zero.

The dataset renders from tracked scenario fixtures when no frozen copy exists.
To freeze a local copy:

```bash
python -m benchmarks.pii_evaluation.dataset.render --freeze
```

NeMo comparison tools live in `benchmarks/pii_evaluation/nemo/`; their inputs
are read from `results/nemo/` or `TRACIORA_NEMO_DATA_DIR`.

## Synthetic data generator

[`benchmarks/data_generator/`](../benchmarks/data_generator/README.md) renders
per-locale slot templates with synthetic personal data and emits examples with
character-aligned gold spans. Structured identifiers satisfy their real
checksums (Luhn, IBAN mod 97, VIN, Steuer-ID, INN, SNILS), using reserved
ranges so outputs are useless as forgery material. Optional typo, OCR and
whitespace noise tests robustness. A new locale is a new `seeds/<locale>/`
directory.

```bash
python -m benchmarks.data_generator.build_release \
  --out releases/v1 --locales ru,de,en --n 3000
```

This writes train/dev/test splits, a datasheet per locale and a sha256
manifest. The release feeds both evaluation and fine-tuning.

## Data import

[`benchmarks/data_import/`](../benchmarks/data_import/README.md) loads public
PII corpora from Hugging Face into one schema and writes an audit report per
corpus (length and label distributions, quality checks, flagged examples).

| Slug | Source | Language |
|---|---|---|
| `nemotron_en` | `nvidia/Nemotron-PII` | en |
| `ai4privacy_de` | `ai4privacy/pii-masking-openpii-1m`, German subset, five reliable labels | de |
| `wolframko_ru` | `wolframko/russian-pii-66k` | ru |

```bash
python -m benchmarks.data_import.cli --dataset all --limit 2000
```

These corpora are made by parties unrelated to this project, so they test
detectors on data independent of the generator.

## Fine-tuning

[`fine_tuning/`](../fine_tuning/README.md) fine-tunes the base
`openai/privacy-filter` checkpoint per language on the generator's release. The
label space comes straight from `policies/opf/<locale>_comprehensive.yaml`.

```bash
pip install -e '.[opf-finetune,benchmarks]'
python -m fine_tuning full --slug ru-v1 --locale ru --low-vram --epochs 3
```

`full` runs prepare, train and evaluate. `--low-vram` uses 8-bit AdamW to fit a
16 GB GPU (Linux x86_64 with CUDA only). Run locales one after another. Then
register each checkpoint under `detector.opf.models` (see
[Detectors](detectors.md#opf-token-classifier)). The data pipeline is tested on
CPU with `python -m pytest fine_tuning/tests/`.
