# Configuration

The distribution is `traciora-pii`; the import namespace is `traciora_pii`.

```python
from traciora_pii import PIIVaultStore, PIIVirtualizer
from traciora_pii.config import PIIConfig, load_config
```

## Where settings come from

Runtime settings live in `traciora.yaml`. Secrets never do: a setting that
needs a key names an environment variable (`api_key_env: PII_EVAL_API_KEY`) and
the value is read from the environment or `.env` when it is used.

`load_config()` picks the file in this order:

1. a path passed to `load_config(path)`;
2. `TRACIORA_PII_CONFIG=/absolute/path/config.yaml`;
3. `traciora.yaml` in the working directory;
4. `traciora.yaml` in an ancestor directory of the package.

`<stem>.local.yaml` beside the selected file (for example
`traciora.local.yaml`) is deep-merged over it. Keep private overrides there;
it is ignored by Git. Precedence, highest first:

```
CLI flags  >  PII_* environment variables  >  <stem>.local.yaml  >  YAML  >  built-in defaults
```

`PIIConfig.with_overrides({...})` returns a copy with a nested dict merged in.
The configuration is frozen; rebuild virtualizers and redactors after a change.

## Category exclusions

The default keeps every detection rule on:

```yaml
detector:
  regex_policies: [en, de, ru]
  second_pass: none
  excluded_categories: []
```

To leave licence plates and URLs visible:

```yaml
detector:
  excluded_categories: [plate, url]
```

Exclusions apply to the regex pass, every second-pass backend, destructive
masking and the leak guard's detection policy. Excluded findings are dropped
before replacement and never stored in the vault. Tokens already in the vault
still resolve, but a value of an excluded category that is already in the
vault is left as text. Input Unicode normalization still runs.

An exclusion does not shield its whole span: a phone number or email inside a
URL is still masked when `url` is excluded. An excluded value stays in the text
that a hosted second-pass detector receives.

Valid categories (`VaultCategory` in `src/traciora_pii/policies/schemas.py`):
`phone`, `email`, `card`, `ip`, `url`, `personal_id`, `inn`, `snils`, `ogrn`,
`bank_acct`, `kpp`, `date`, `plate`, `iban`, `secret`, `person_name`,
`street_address`, `vin`, `sts`, `osago`, `steuer_id`.

In code, pass `excluded_categories=["plate", "url"]` to `PIIVirtualizer` or
`PIIRedactor`; an explicit empty list overrides the configured exclusions.
`PIIRedactor.from_config(config)` uses the configuration's exclusions;
`PIIVirtualizer(vault, config=config)` does too when you pass no explicit list.

## `detector` settings

| Key | Default in `traciora.yaml` | Meaning |
|---|---|---|
| `second_pass` | `none` | `none`, `opf`, `encoder`, `llm` or `presidio` |
| `excluded_categories` | `[]` | Categories left visible |
| `regex_policies` | `[en, de, ru]` | Built-in regex policies, unioned in order; first listed wins on overlap |
| `regex_policy` | unset | One built-in name or a path to a custom policy YAML; use instead of `regex_policies`, not with it (with neither set, `ru` applies) |
| `masking_enabled` | `true` | Turns `PIIRedactor` (destructive masking) on or off |
| `leak_guard_enabled` | `true` | Turns `assert_model_safe` on or off |

### `detector.opf`

| Key | Default | Meaning |
|---|---|---|
| `default_model` | `base` | Registry key used when `second_pass: opf` |
| `models.<key>.checkpoint` | see below | Hugging Face repo id or local path |
| `models.<key>.policy` | | OPF label policy under `policies/opf/` (`en_comprehensive`, ...) |
| `device` | `cpu` | `cpu`, `cuda`, `mps` or `mlx` |
| `mlx_bits` | `null` | MLX only: `null` for bf16, `8` for 8-bit |
| `prefilter` | `false` | Cheap regex check that skips clearly clean text |
| `cache_dir` | `null` | Defaults to `~/.cache/traciora_pii/models/opf` |

Shipped registry: `base` (`openai/privacy-filter`), `en`, `de`, `ru`
(`pii-anon/opf-<lang>-v2`). Add an entry to register your own fine-tune.

### `detector.encoder`

| Key | Default | Meaning |
|---|---|---|
| `default_model` | `ru` | Registry key used when `second_pass: encoder` |
| `models.<key>` | `checkpoint`, `revision`, `family` | A repo id must be pinned by a 40-hex `revision`; a local path needs none |
| `device` | `cpu` | `cpu` or `cuda` |
| `threshold` | `0.5` | Minimum mean probability over a fragment's tokens |
| `threads` | `null` | torch intra-op threads on CPU |
| `cache_dir` | `null` | Defaults to `~/.cache/traciora_pii/models/encoder` |

### `detector.llm`

| Key | Default | Meaning |
|---|---|---|
| `base_url` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM work unchanged) |
| `model` | `null` | Required; there is no default model |
| `api_key_env` | `PII_EVAL_API_KEY` | Name of the variable holding the key |
| `timeout_seconds` | `30.0` | Request timeout |
| `prefilter` | `true` | Skip the call when no name or address is plausible |
| `max_tokens` | `1024` | Headroom for reasoning models |

### `detector.presidio`

| Key | Default | Meaning |
|---|---|---|
| `language` | `en` | `en`, `de` or `ru`; one language per detector |
| `score_threshold` | `0.5` | Minimum Presidio confidence |
| `prefilter` | `false` | Same cheap gate as OPF/LLM |
| `nlp_model` | `""` | spaCy model; empty means `en_core_web_lg`, `de_core_news_lg` or `ru_core_news_lg` |
| `label_map` | `{}` | Empty means `PERSON` → `[PERSON_NAME]` only |
| `entities` | `[]` | Empty means derived from `label_map` |

### `vault` and `eval`

`vault.encryption_key_env` (`PII_VAULT_KEY`) names the key variable the
evaluation harness reads; without it the harness uses a fresh key per run.
In your own code, pass the key to `PIIVaultStore` directly.

`eval` configures the [benchmark agent](benchmarks.md): `llm.enabled` (a spend
gate, `false`), `llm.model` (no default), `llm.api_key_env`, `llm.base_url`,
`llm.temperature` (`0.0`), `llm.max_tokens` (`512`), `results_dir`,
`max_agent_iterations` (`4`).

## Environment overrides

| Variable | Sets |
|---|---|
| `TRACIORA_PII_CONFIG` | Path of the YAML file |
| `PII_SECOND_PASS` | `detector.second_pass` |
| `PII_OPF_MODEL`, `PII_OPF_DEVICE`, `PII_OPF_PREFILTER_ENABLED`, `PII_OPF_CACHE_DIR` | `detector.opf.*` |
| `PII_ENCODER_MODEL`, `PII_ENCODER_DEVICE`, `PII_ENCODER_CACHE_DIR` | `detector.encoder.*` |
| `PII_LLM_BASE_URL`, `PII_LLM_MODEL`, `PII_LLM_TIMEOUT`, `PII_LLM_PREFILTER` | `detector.llm.*` |
| `PII_PRESIDIO_LANGUAGE`, `PII_PRESIDIO_SCORE_THRESHOLD`, `PII_PRESIDIO_PREFILTER` | `detector.presidio.*` |
| `PII_REGEX_POLICY` | `detector.regex_policy` |
| `PII_MASKING_ENABLED`, `PII_LEAK_GUARD_ENABLED` | the two switches |
| `PII_EVAL_LLM_ENABLED`, `PII_EVAL_MODEL_NAME`, `PII_EVAL_BASE_URL` | `eval.llm.*` |

Secrets: `PII_VAULT_KEY` (Fernet key) and `PII_EVAL_API_KEY` (OpenRouter or
other provider key, shared by the evaluation agent and the LLM detector). For a
separate detector key, set `detector.llm.api_key_env` to another variable name.
Copy `.env.example` to `.env` to start.

## Fixed per-language profile

`traciora_pii.runtime.traciora_config(language)` with `en`, `de` or `ru` builds
a configuration without reading any YAML or environment, so it works from an
installed wheel. It selects that language's regex policy, its fine-tuned OPF
entry and its encoder entry, and excludes URLs. Pass the result explicitly to
`PIIVirtualizer`, `PIIRedactor.from_config`, `detect_pii`, `redact_pii_text`
and `assert_model_safe`. Per-call exclusions override the profile's.

For structured message content, pass your application's text-mapping function
as `map_text` to `PIIVirtualizer`; `anonymize_content` and `rehydrate_content`
then keep your block schema. Pass a binary-block predicate as `skip_block` to
`assert_model_safe`.

## Shared second pass

`prepare_second_pass(config)` downloads and validates only the selected
checkpoint; call it at startup. With `second_pass: none` it does nothing.
`get_shared_second_pass(config)` opens only a prepared local copy, never
downloads, serializes inference and reuses one detector across sessions. Both
support `none`, `opf` and `encoder`.

- On open, the detector must find the name in a self-check sentence in its
  language, or it raises `SecondPassUnavailable` until restart, without retrying.
- An inference error propagates from `anonymize_text`; no text is returned.
  There is no regex-only fallback. Changing settings requires a restart.
- For `encoder`, the registry key must be `en`, `de` or `ru` (it picks the
  self-check sentence). Process shutdown closes the detector.

### OPF on Apple silicon

`detector.opf.device: mlx` runs the OPF forward pass through MLX. It needs the
`mlx` extra (macOS arm64) and fails at construction otherwise; it never falls
back to CPU. `prepare_second_pass` converts the checkpoint once into a sibling
directory (`<checkpoint>-mlx-bf16`, or `-mlx-8bit` with `mlx_bits: 8`) and
converts again when the source weights change. Text is labelled in 512-token
windows with 128 tokens of overlap. 8-bit halves memory and load time; it is
not faster. On macOS without MLX, set `OPF_MOE_TRITON=0` for OPF on CPU or MPS.

### Encoder models

`second_pass: encoder` runs ruBERT for `ru` and Horizon ModernBERT for `en` and
`de`, each pinned to one Hugging Face revision. Startup is refused when the
`encoder` extra is missing, when the checkpoint is absent, incomplete, of
another revision or lacks the family's labels, when the self-check finds no
name, or when `device: cuda` is set without a CUDA build of torch.

## Logging and local data

`scrub(text)` is regex-only with its own logging policy, including long-number
masking; conversational exclusions never weaken it. Generated data stays out of
Git and packages: `benchmarks/pii_evaluation/results/`,
`benchmarks/pii_evaluation/dataset/data/`, `benchmarks/data_import/reports/`,
`fine_tuning/data/`, `fine_tuning/runs/`. Tracked fixtures are synthetic; never
replace them with real data.
