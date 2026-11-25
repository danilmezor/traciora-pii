# Detectors

Detection runs in two passes. The regex pass always runs. One optional second
pass, set by `detector.second_pass`, then runs over the regex-tokenized text,
so it never sees a value the regex pass already replaced. Before both, values
the session's vault already holds are replaced by their tokens, so a value
either pass found once stays a token in every later text of the session.

## Regex pass

Built-in policies for English, German and Russian live in
`src/traciora_pii/policies/regex/`. The default `regex_policies: [en, de, ru]`
unions all three, so a German tax ID or a Russian SNILS is caught whatever
language the surrounding text is in.

| Policy | Categories |
|---|---|
| `en` | phone, email, card, iban, ip, url, date, plate, vin, personal_id, secret |
| `de` | the `en` set plus steuer_id |
| `ru` | the `en` set plus inn, snils, ogrn, kpp, bank_acct, sts, osago |

Each policy also declares `person_name` and `street_address`, but only a second
pass emits them. Checksum-bearing identifiers are validated where the policy
defines a validator, and some patterns (passport numbers, for example) need a
nearby keyword. Point `regex_policy` at your own YAML file to use a custom policy.

The regex pass is precise but finds no names or addresses: on the 309-query
trilingual benchmark it reaches micro precision 1.000 and recall 0.872, and its
misses are mostly names. Use a second pass when names or addresses matter.

## Second-pass backends

| `second_pass` | Extra | Finds | Runs |
|---|---|---|---|
| `opf` | `opf` (`mlx` on Apple silicon) | names, addresses, emails, phones, secrets, personal IDs; more structured types in the `de` and `ru` label maps | locally |
| `encoder` | `encoder` | names and street addresses | locally |
| `llm` | `llm` | names and street addresses | any OpenAI-compatible endpoint |
| `presidio` | `presidio` + spaCy model | names (`PERSON`) by default | locally |

### OPF token classifier

OPF is the OpenAI Privacy Filter token classifier. The registry in
`detector.opf.models` ships:

| Key | Checkpoint | Policy |
|---|---|---|
| `base` | `openai/privacy-filter` | `en_comprehensive` |
| `en` | `pii-anon/opf-en-v2` | `en_comprehensive` |
| `de` | `pii-anon/opf-de-v2` | `de_comprehensive` |
| `ru` | `pii-anon/opf-ru-v2` | `ru_comprehensive` |

The `en`, `de` and `ru` checkpoints are fine-tuned on synthetic data from the
benchmark generator; train your own with the [fine-tuning tools](benchmarks.md#fine-tuning).
Each policy under `policies/opf/` maps OPF labels onto vault categories and
drops labels the regex pass handles better (dates, URLs). Checkpoints download
from Hugging Face into `~/.cache/traciora_pii/models/opf`: ahead of time with
`prepare_second_pass` (required by the shared runtime), or on first use when
built with `build_second_pass_detector`. Devices: `cpu`, `cuda`, `mps`,
`mlx`. On macOS, set `OPF_MOE_TRITON=0` before running OPF on CPU or MPS.

### Encoder

Token-classification encoders pinned to one Hugging Face revision each:

| Key | Checkpoint | Family |
|---|---|---|
| `ru` | `redmadrobot-rnd/rubert-base-pii-ner` | ruBERT |
| `en`, `de` | `Horizon-Labs/pii-redactor-small` | Horizon ModernBERT |

Only name and street-address labels are kept. `threshold` (default `0.5`) sets
the minimum mean token probability per fragment. Devices: `cpu`, `cuda`.

### LLM

Sends the regex-tokenized text to an OpenAI-compatible chat endpoint and asks
for names and street addresses. There is no default model: set
`detector.llm.model` or `PII_LLM_MODEL`. Point `base_url` at OpenRouter or at a
local server (LM Studio, Ollama, vLLM). A cheap prefilter (`prefilter: true`)
skips the call when no name or address is plausible. Reasoning is suppressed by
an assistant prefill, so reasoning and non-reasoning models behave alike. A
hosted endpoint receives the text it inspects; run it locally when that matters.

### Presidio

Microsoft Presidio with a spaCy NER model, one language per detector. By default
only `PERSON` is mapped (to `[PERSON_NAME]`); extend `label_map` to map more.
Install the matching spaCy model, for example
`python -m spacy download de_core_news_lg`.

## Choosing a backend

| Need | Use |
|---|---|
| Structured identifiers only, no models | `none` |
| Best recall on names and addresses, local | `opf` with the language's fine-tune |
| Small CPU footprint, local, names and addresses | `encoder` |
| No local model, or an existing LLM endpoint | `llm` |
| A familiar third-party baseline | `presidio` |

Only `none`, `opf` and `encoder` are supported by the shared runtime
(`prepare_second_pass` / `get_shared_second_pass`). Build `llm` and `presidio`
detectors with `build_second_pass_detector(config)`. Any configured second pass
either runs or raises; there is no silent regex-only fallback.

## Measured quality and latency

On the 309-query trilingual benchmark (312 gold spans), Apple M1 Pro:

| Configuration | micro P | micro R | s/query |
|---|---:|---:|---:|
| regex only | 1.000 | 0.872 | 0.000 |
| regex + base OPF, CPU | 0.958 | 0.962 | 2.773 |
| regex + routed OPF fine-tunes, CPU | 0.966 | 0.997 | 1.373 |
| regex + routed OPF fine-tunes, MLX bf16 | 0.966 | 0.997 | 0.097 |
| regex + routed OPF fine-tunes, MLX 8-bit | 0.969 | 0.997 | 0.088 |
| regex + routed encoders, CPU | 0.987 | 0.974 | 0.095 |

`anonymize_text` end to end on Russian text (regex, vault and second pass):

| Text | Chars | OPF MLX bf16 | Encoder CPU | Encoder CUDA |
|---|---:|---:|---:|---:|
| short message | 126 | 15 ms | 53 ms | 14 ms |
| medium | 1,629 | 62 ms | 101 ms | 15 ms |
| document | 39,466 | 1.3 s | 2.0 s | 84 ms |
| database result | 66,974 | 3.8 s | 8.1 s | 351 ms |
| peak memory | | 4.3 GB | 486 MB | 449 MB GPU |

On a Russian names-and-addresses benchmark (1,810 rows), the ruBERT encoder
reached precision 0.970 / recall 0.996 on names and 0.933 / 0.955 on addresses.
Known limitation: both OPF and encoder models can tag room numbers as addresses.

For the comparison against Presidio and small LLM detectors across corpora, see
[Reproducing the paper](reproduce.md). Run the measurements yourself with the
[evaluation harness](benchmarks.md).
