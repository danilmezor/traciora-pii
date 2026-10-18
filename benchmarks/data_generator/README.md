# Slot-Template PII Benchmark Generator

> Data policy: source tools and synthetic fixture definitions are tracked.
> Generated datasets, imported caches, reports, result snapshots, logs,
> training runs, and model checkpoints stay local and are ignored by Git.
> Historical result files referenced below must be supplied locally.

Generates evaluation benchmarks for the `traciora_pii` detection layer from
per-locale **slot templates** seeded with **synthetic locale-specific
PII** (RU INN/SNILS/passport/OGRN, DE Steuer-ID/IBAN/Personalausweis,
EN SSN/IBAN/VIN, …). Synthetic-only is a hard constraint of the domain
(real values cannot be released under GDPR / CCPA / 152-FZ).

## Why it's credible

| Reviewer concern | How this addresses it |
|---|---|
| "Trained on the test set" | Generator is a separate program from the detector; structured-ID checksums use FNS/PFR/BZSt/ISO algorithms unrelated to the detector regex. |
| "Templated/memorizable" | Multiple insertion templates per profile + slot vocabularies that vary the surface form (names, addresses, dates) across rows. Adversarial noise (typo / OCR / whitespace) is opt-in. |
| "Where does the data come from?" | Every example is reproducible from a single seed + the frozen seed YAMLs under `seeds/<locale>/`. |
| "Why synthetic only?" | Real INN/SNILS/Steuer-ID values cannot be released — this is a hard constraint of the domain. We disclose it explicitly and mitigate per the items above. |

## Pipeline stages

```
Seeds          → per-locale templates + slot vocabularies (seeds/<locale>/*.yaml)
SlotFillEngine → renders templates with synthetic PII values (checksumed
                 INN/SNILS/Steuer-ID/IBAN/VIN/...; locale-aware names)
NoiseApplier   → optional adversarial perturbation (default off; see below)
Pipeline.run   → yields GeneratedExample with character-aligned gold spans
to_eval_query  → adapter into the §3 EvalQuery shape, or write_jsonl()
```

Adding a locale: drop a new `seeds/<locale>/` directory; no Python edit
required as long as the slots used appear in `_SLOT_METHODS` in
`injectors.py`. See `seeds/ru/` for the canonical layout (one YAML per
profile, plus a `shared.yaml` for cross-locale templates).

## Quick start

```python
from benchmarks.data_generator import Pipeline, load_seed_bundle, write_jsonl

pipeline = Pipeline(seeds=load_seed_bundle("ru"), seed=42)
examples = list(pipeline.run(100))
write_jsonl(examples, "/tmp/ru_smoke.jsonl")
```

To build the canonical release artifact (train/dev/test split, datasheet,
sha256 manifest), see "Building the v1 release artifact" below.

## Output schema

JSONL where each row is:

```json
{
  "id": "ru-gen-00042",
  "locale": "ru",
  "profile": "ru_inn_phone",
  "text": "Заявка зарегистрирована и передана в работу профильному специалисту, ИНН 7707083893, телефон +7 925 123-45-67.",
  "spans": [
    {"placeholder": "[RU_INN]", "value": "7707083893", "start": 81, "end": 91},
    {"placeholder": "[RU_PHONE]", "value": "+7 925 123-45-67", "start": 102, "end": 118}
  ]
}
```

To consume directly in `benchmarks.pii_evaluation`:

```python
from benchmarks.data_generator import Pipeline, load_seed_bundle, to_eval_query

pipeline = Pipeline(seeds=load_seed_bundle("ru"), seed=42)
queries = [to_eval_query(ex) for ex in pipeline.run(100)]
```

## Coverage (RU + DE + EN)

| Locale | Structured IDs | Names / addresses | Generic |
|---|---|---|---|
| RU | INN, SNILS, passport, OGRN, KPP, BIK, bank account, license plate, VIN, drv. license, СТС, ОСАГО | `[PERSON_NAME]`, `[STREET_ADDRESS]` | phone, email, card, IP, DOB, URL, secret token |
| DE | Steuer-ID, Personalausweis, IBAN-DE, VIN, license plate | `[PERSON_NAME]`, `[STREET_ADDRESS]`, PLZ + city | phone, email, card, IP, DOB, URL, secret token |
| EN | US SSN, US phone, IBAN (intl.), card, VIN, license plate | `[PERSON_NAME]`, `[STREET_ADDRESS]` | email, IP, DOB, URL, secret token |

Every checksum-bearing identifier is generated to satisfy the originating
standard (ISO 3779 for VIN, ISO 7064 MOD 11,10 for Steuer-ID and
Personalausweis, mod 97 for IBAN, FNS algorithms for INN, PFR algorithm
for SNILS, Luhn for cards). See module docstrings in
`faker_providers.py` for the citations.

## Adversarial noise

Default off. Opt in via flags on the release CLI:

```bash
python -m benchmarks.data_generator.build_release \
  --out /tmp/ru_stress --locales ru --n 500 \
  --noise-typo-rate 0.05 --noise-ocr-rate 0.02 \
  --noise-whitespace-rate 0.05 --noise-target scaffold
```

Span invariance is preserved by construction:

* `target=scaffold` (default) — only non-PII characters are perturbed; PII
  spans are re-anchored against the perturbed text.
* `target=pii` — PII values themselves are perturbed; `span.value` and
  `span.end` are updated; `text[start:end] == value` still holds. The
  `placeholder` continues to label the *intended* entity type, so this
  mode tests detector robustness ("does our SSN regex still match a typo'd
  SSN?").

## Building the v1 release artifact

The `build_release` CLI runs the canonical pipeline per locale with frozen
seeds, writes split JSONLs + datasheets + a sha256 manifest, and asserts
the policy contract.

```bash
# Offline smoke (no network, no spaCy)
python -m benchmarks.data_generator.build_release \
  --out /tmp/bench-v1 --locales ru,de,en --n 50

# Full release (3000 examples per locale, 80/10/10 split)
python -m benchmarks.data_generator.build_release \
  --out releases/v1 --locales ru,de,en --n 3000
```

Output structure:

```
<out>/
  traciora_pii-bench-ru/
    train.jsonl   dev.jsonl   test.jsonl   datasheet.md
  traciora_pii-bench-de/   ...
  traciora_pii-bench-en/   ...
  MANIFEST.json   # sha256 + size for every file above
```

Frozen seeds (`FROZEN_SEEDS` in `build_release.py`) — bumping artifact
`version` is the right way to invalidate previously published hashes.

## Datasheet

Each release directory ships a `datasheet.md` rendered from
`datasheet.py` (Gebru et al. 2018 outline). The Markdown is deterministic
given identical metadata so two runs produce byte-identical bytes.
