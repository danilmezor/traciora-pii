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

Local overrides can live in `traciora.local.yaml`, which is ignored by Git.
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
