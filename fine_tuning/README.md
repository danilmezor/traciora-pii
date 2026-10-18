# OPF fine-tuning pipeline (EN / DE / RU)

> Data policy: source tools and synthetic fixture definitions are tracked.
> Generated datasets, imported caches, reports, result snapshots, logs,
> training runs, and model checkpoints stay local and are ignored by Git.
> Training data can be regenerated from the tracked synthetic fixtures; no
> historical result files are required.

Fine-tunes the base `openai/privacy-filter` checkpoint on synthetic
PII data produced by `benchmarks.data_generator.build_release`.

The runtime label space is sourced **directly** from
`src/traciora_pii/policies/opf/<locale>_comprehensive.yaml`. Adding or
removing a label there flows automatically into the next prepare run.

## Layout

```
fine_tuning/
├── __main__.py        # argparse dispatcher
├── label_space.py     # reads policies/opf/*.yaml → OPF runner JSON
├── prepare_data.py    # bench release → OPF train/validation/test JSONL
├── train.py           # opf._train.runner wrapper + bnb.AdamW8bit patch
├── evaluate.py        # detector ablation harness wrapper
├── data/<slug>/       # gitignored
└── runs/<slug>/       # gitignored
    └── checkpoint/    # the fine-tune output
```

## Install (on the GPU host)

```bash
pip install -e '.[opf-finetune,benchmarks]'
```

This pulls `opf` from upstream, plus `torch` and
(Linux x86_64 only) `bitsandbytes` for the 8-bit AdamW path.

## Step 1 — build the bench release

It's deterministic given a seed (slot-template generator; no HF scaffolds,
no spaCy required).

```bash
pip install -e .

python -m benchmarks.data_generator.build_release \
  --out releases/v1 \
  --locales ru,de,en \
  --n 3000
```

Output:

```
releases/v1/
  traciora_pii-bench-ru/  train.jsonl  dev.jsonl  test.jsonl  datasheet.md
  traciora_pii-bench-de/  ...
  traciora_pii-bench-en/  ...
  RELEASE_CARD.md  MANIFEST.json
```

## Step 2 — prepare per-locale training data

```bash
python -m fine_tuning prepare --slug ru-v1 --locale ru
python -m fine_tuning prepare --slug de-v1 --locale de
python -m fine_tuning prepare --slug en-v1 --locale en
```

Each writes:

```
fine_tuning/data/<slug>/
  train.jsonl         # spans rewritten as {opf_label: [[start, end]]}
  validation.jsonl    # bench's dev.jsonl, renamed
  test.jsonl
  label_space.json    # ("O", *sorted(policy.label_map.keys()))
  manifest.json       # placeholder→label map + per-split kept/dropped counts
```

Placeholders not in the locale's OPF `label_map` (e.g. `[CARD]` in EN)
are dropped — the regex layer covers structured IDs at runtime; the
fine-tune targets contextual labels only.

## Step 3 — fine-tune (16 GB → use `--low-vram`)

All three start from the same base checkpoint (OPF resolves it via
`OPF_CHECKPOINT` env var or `~/.opf/privacy_filter`, downloading it if
the default directory is absent). Use `--base-checkpoint /path/to/model`
to supply an existing local checkpoint. Model weights are not stored in Git.
Training requires suitable hardware; GPU training is not verified by the
CPU-only smoke tests below.

```bash
# Russian
python -m fine_tuning train --slug ru-v1 --low-vram --epochs 3

# German
python -m fine_tuning train --slug de-v1 --low-vram --epochs 3

# English
python -m fine_tuning train --slug en-v1 --low-vram --epochs 3
```

`--low-vram` swaps `torch.optim.AdamW` → `bitsandbytes.optim.AdamW8bit`
process-wide, dropping optimizer state from ~12 GB to ~3 GB. Defaults
under that flag: `batch_size=1`, `grad_accum_steps=8` (effective
batch 8). Override with explicit `--batch-size` / `--grad-accum-steps`
if you have headroom.

Each run writes:

```
fine_tuning/runs/<slug>/checkpoint/
  config.json
  pytorch_model.safetensors  (or .bin)
  ... (OPF runner artifacts)
```

## Step 4 — end-to-end (one command per locale)

```bash
python -m fine_tuning full --slug ru-v1 --locale ru --low-vram --epochs 3
python -m fine_tuning full --slug de-v1 --locale de --low-vram --epochs 3
python -m fine_tuning full --slug en-v1 --locale en --low-vram --epochs 3
```

`full` = prepare + train + evaluate.

## Step 5 — register the checkpoints in `traciora.yaml`

After training succeeds, register each checkpoint under
`detector.opf.models` so the runtime can route by language:

```yaml
detector:
  opf:
    models:
      ru:
        checkpoint: fine_tuning/runs/ru-v1/checkpoint
        policy: ru_comprehensive
      de:
        checkpoint: fine_tuning/runs/de-v1/checkpoint
        policy: de_comprehensive
      en:
        checkpoint: fine_tuning/runs/en-v1/checkpoint
        policy: en_comprehensive
```

## Verifying without GPU access

The smoke tests don't need torch / opf / bitsandbytes — they exercise
the data pipeline end-to-end against the policies in this repo:

```bash
python -m pytest fine_tuning/tests/ -v
```

## Notes

- `--low-vram` is gated to Linux x86_64 + CUDA (bitsandbytes constraint).
  On macOS or non-x86_64 the flag raises a clear error before any
  training begins.
- Three runs in parallel will conflict on the global optimizer
  monkey-patch — run them sequentially.
- The harness writes the OPF spans schema (`{label: [[s, e], ...]}`),
  not the legacy list-of-objects form. If a future OPF release changes
  this, the rewrite lives in `prepare_data._convert_record`.
