# Third-party PII dataset import + audit

> Data policy: source tools and synthetic fixture definitions are tracked.
> Generated datasets, imported caches, reports, result snapshots, logs,
> training runs, and model checkpoints stay local and are ignored by Git.
> Historical result files referenced below must be supplied locally.

Pulls public PII benchmarks from Hugging Face, normalizes them to the same
in-memory schema used by `benchmarks/data_generator/`, and emits per-dataset
audit reports under `reports/`.

## Why this exists

`data_generator/` produces our **training-side** synthetic corpus. Independent
evaluation checks whether `traciora_pii`'s detector is being
evaluated on something independent of its own generator. This module
provides that independent evaluation surface.

Real-world labeled PII corpora cannot be lawfully published under
GDPR / CCPA / 152-FZ, so every dataset here is itself synthetic - but
produced by parties unrelated to this project.

## Datasets

| Slug | HF repo | Locale | Notes |
|---|---|---|---|
| `nemotron_en` | `nvidia/Nemotron-PII` | en | 200k English / US-locale synthetic rows, stringified spans; quality-gated before use |
| `ai4privacy_de` | `ai4privacy/pii-masking-openpii-1m` (DE subset) | de | German rows; projected at load time to a 5-label clean subset (PERSON_NAME, EMAIL, PHONE, STREET_ADDRESS, DATE) per the audit at `reports/ai4privacy_de.md`. The full label set is broken on structured-ID classes (DE_STEUER_ID, DE_PERSONALAUSWEIS) and those labels are deliberately not mapped. |
| `wolframko_ru` | `wolframko/russian-pii-66k` | ru | ~65k synthetic Russian rows, span-only |

The English subset of `ai4privacy` remains in `loaders.py` for reuse,
but is not part of the §1 primary eval surface — its sampled rows
showed format/context failures on critical EN classes.

## Install

The `datasets` dependency is bundled with the `[benchmarks]` extra:

```bash
pip install -e '.[benchmarks]'
```

(`pyarrow` is pulled transitively by `datasets`.) HF caches downloads under
`~/.cache/huggingface/`; nothing is written to this directory by default.

## Usage

Smoke run (caps each dataset at 2k rows; useful for iterating on the report):

```bash
python -m benchmarks.data_import.cli --dataset all --limit 2000
```

Full pull for a single dataset after the canonical eval transform:

```bash
python -m benchmarks.data_import.cli \
  --dataset nemotron_en \
  --limit 0 \
  --transform quality_gate_nemotron,combine_names,project_allowed
```

Reports land in `benchmarks/data_import/reports/<slug>.md`.

## Report contents

For each dataset:

- **Length distributions** - chars, whitespace tokens, spans/row, span chars.
- **Label distribution** - count + share + one example value per label.
- **Quality checks** - rows with zero spans, value/text mismatch, out-of-bounds
  offsets, overlapping spans, dropped spans at load, non-NFC text, control
  characters.
- **Flagged examples** - one example per failure mode (deterministic).
- **Random samples** - five deterministic samples across the stream.

## Schema

```python
@dataclass(frozen=True)
class Span:
    label: str
    value: str
    start: int
    end: int

@dataclass
class NormalizedExample:
    id: str
    locale: str
    text: str
    spans: list[Span]
    source: str
    meta: dict
```

Source-specific metadata (Nemotron domain/document fields, wolframko locale
fields) is preserved on `meta` rather than promoted into top-level
fields - the analyzer doesn't need it but downstream consumers might.

## What's *not* here yet

- **Mapping to traciora_pii's placeholder vocabulary** (`[RU_INN]`, `[PERSON_NAME]`,
  ...) - imported labels stay faithful to source. The mapping layer belongs
  next to the evaluation harness.
- **Train/dev/test split materialization** - we audit splits as the source
  publishes them. Re-splitting for our own protocol is a downstream concern.
- **Annotation-quality re-labeling** - if the audit surfaces systematic
  issues, partial re-annotation lives outside this module.
