# Repository and runtime settings

The distribution name is `traciora-pii`; the Python import
namespace is `traciora_pii` (Python module names cannot contain hyphens).

```python
from traciora_pii import PIIVaultStore, PIIVirtualizer
from traciora_pii.config import PIIConfig, load_config
```

## Configuration file

The repository configuration is `traciora.yaml`. To use another file, set
`TRACIORA_PII_CONFIG=/absolute/path/to/config.yaml`, or pass its path to
`load_config()`. Explicit file selection takes precedence over the environment,
then configuration discovery checks the working directory and package ancestors.
Individual supported `PII_*` environment overrides take precedence over YAML.

Local overrides can live in `traciora.local.yaml` (generally `<stem>.local.yaml`
beside the selected file), which is ignored by Git and deep-merged over that file
before environment overrides apply: environment > local YAML > YAML > defaults.
The previous names were `piiv.yaml`, `PIIV_CONFIG`, `PIIVConfig`, and the
`piiv` import namespace; consumers should migrate to the names above.

## Category exclusions

The checked-in default preserves all enabled detection rules:

```yaml
detector:
  regex_policies: [en, de, ru]
  second_pass: none
  excluded_categories: []
```

To leave license plates and URLs visible, change only:

```yaml
detector:
  excluded_categories: [plate, url]
```

Exclusions apply to the regex pass, every second-pass backend (OPF, LLM,
Presidio), destructive masking, and the global leak-guard detection policy.
Excluded findings are dropped before replacement and are not stored as new
vault entries. Existing tokens still resolve.

Excluding a category does not protect its whole span from other categories.
For example, a phone number or email inside a URL can still be masked when
`url` is excluded. Input Unicode normalization still runs.

Category names are validated against `VaultCategory` in
`src/traciora_pii/policies/schemas.py`. Supported values are:
`phone`, `email`, `card`, `ip`, `url`, `personal_id`, `inn`, `snils`, `ogrn`,
`bank_acct`, `kpp`, `date`, `plate`, `iban`, `secret`, `person_name`,
`street_address`, `vin`, `sts`, `osago`, and `steuer_id`.

Programmatic callers can pass `excluded_categories=["plate", "url"]` to
`PIIVirtualizer` or `PIIRedactor`. An explicit empty list overrides global
exclusions. `PIIRedactor.from_config(config)` uses the supplied configuration;
for a virtualizer, pass `config.detector.excluded_categories` explicitly.
Rebuild these objects after changing configuration. Excluding a category does
not prevent a hosted second-pass detector from receiving the retained text.

## Validation

Run `pytest` with the `dev` dependencies installed. Tests use simulated
second-pass findings and do not require downloading OPF or spaCy models.

## What belongs in Git

Track runtime code, benchmark and training tools, tests, documentation, and
synthetic source fixtures required by those tools (seed vocabulary, templates,
scenario definitions, and detector policies). These fixtures are source inputs,
not customer records or personal exports. Do not replace them with real data.

Keep personal records, credentials, local settings, generated datasets,
transcripts, benchmark outputs, frozen result snapshots, logs, caches, vaults,
model checkpoints, and training artifacts outside Git. `.gitignore` covers
these paths and common formats. Ignored directories remain on disk. Personal paper notes and review material
under `docs/` or `traciora-pii-paper/` also remain local.

Benchmark outputs go under `benchmarks/pii_evaluation/results/`, imported-data
reports under `benchmarks/data_import/reports/`, frozen benchmark data under
`benchmarks/pii_evaluation/dataset/data/`, and training data/runs under
`fine_tuning/data/` and `fine_tuning/runs/`. `claims/*/expected/` contains only
local comparison snapshots. These locations are also excluded from packages.

NeMo evaluation scripts live in `benchmarks/pii_evaluation/nemo/`; their cached
CSV/JSON inputs remain in the ignored `results/nemo/` directory. Override that
input directory with `TRACIORA_NEMO_DATA_DIR`.

Ignore rules prevent future additions; removing a tracked file from the index
keeps its local copy but does not erase earlier commits containing that file.

## Repository history

The Git history starts from the cleaned source tree. Personal data and prior
result snapshots are excluded from that initial commit. Keep generated outputs
local, and inspect staged files before committing. History replacement cannot
erase copies in other clones, forks, or hosting-provider caches.

## Embedding Traciora core

Use `traciora_pii.runtime.traciora_config(language)` with `en`, `de`, or `ru`.
It selects that language's packaged regex policy, its fine-tuned OPF registry
entry and its encoder registry entry, and excludes URLs. It reads no environment or checkout YAML, so it works
from an installed wheel and does not interpret the core application's switches.
Pass the resulting configuration explicitly to `PIIVirtualizer`,
`PIIRedactor.from_config`, `detect_pii`, `redact_pii_text`, and `assert_model_safe`.
Explicit per-call category exclusions override those on the configuration.

For attachments, pass the application's text-mapping function as `map_text`
to `PIIVirtualizer`; `anonymize_content` and `rehydrate_content` then preserve
the application's block schema. Pass its binary-block predicate as `skip_block`
to `assert_model_safe`. The package never inspects image contents.

`prepare_second_pass(config)` downloads and validates only the selected checkpoint at
startup; with `second_pass: none` it does nothing. `get_shared_second_pass(config)` opens only a prepared local checkpoint,
serializes inference, and reuses one detector across session virtualizers. A
backend/model/configuration change requires a process restart. A configured second pass
either runs or refuses; there is no regex-only fallback. Once opened, the detector
runs one self-check sentence in the selected language. If the checkpoint cannot
be opened, or the self-check finds no name, `get_shared_second_pass` raises
`SecondPassUnavailable`; if inference fails, it raises the inference error, and
`anonymize_text` propagates it without returning text, so a failed second pass
fails the call. It
raises again on every call until restart, without retrying the load. Session owners close
their own vaults; only process shutdown closes the shared detector.

On Apple silicon, `detector.opf.device: mlx` runs the OPF forward pass through
MLX instead of PyTorch; it needs the `mlx` extra (`pip install -e '.[mlx]'`,
macOS on arm64 only) and fails at construction otherwise, never falling back to
CPU. `prepare_second_pass` converts the selected checkpoint once into a sibling
directory in the cache (`<checkpoint>-mlx-bf16`, or `-mlx-8bit` with
`mlx_bits: 8`) and converts again when the source weights change. The text is
labelled in 512-token windows with 128 tokens of overlap, and opf's own
aggregation, Viterbi decoder and span conversion produce the spans. 8-bit
halves the memory and load time; it is not faster.

`detector.second_pass: encoder` runs a token-classification encoder instead of
OPF: ruBERT for `ru`, Horizon ModernBERT for `en` and `de`, each pinned to one
Hugging Face revision in `detector.encoder.models`. It needs the `encoder` extra
(`pip install -e '.[encoder]'`). `prepare_second_pass` downloads the selected
checkpoint at its pinned revision into its own cache directory and validates its
files and labels; `get_shared_second_pass` opens only that local copy, never
downloads, and runs the same self-check sentence, which must yield a
`[PERSON_NAME]` finding. For the shared runtime the registry key names the
language, `en`, `de` or `ru`, because the self-check sentence is chosen by that
key; a key such as `ru-local` is refused. It also refuses to start when the
extra is missing, when the checkpoint is absent, incomplete, of another revision
or without the family's labels, when the self-check finds no name, or when `device: cuda` is set without
a CUDA build of torch. An inference error propagates from `anonymize_text` like
OPF's; there is no regex-only fallback.

The package's `scrub` is regex-only and uses a separate logging policy, including
long-number masking. Conversational category exclusions do not weaken logging.
Historical core token formats are not supported by this release; consumers must
start fresh contexts and vault storage when migrating.

Version 0.2.1 buffers every possible token prefix, including one-character
chunks, so streaming rehydration does not depend on provider chunk sizes.
