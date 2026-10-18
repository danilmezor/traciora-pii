# traciora-pii

Reversible PII replacement for LLM tool-calling agents. The library detects
personal information, replaces it with session-scoped reference tokens, stores
original values in an encrypted vault, restores tool arguments before execution,
and tokenizes tool results before returning them to the model. Final answers
can restore the original values for the user.

## Install

Python 3.11 or newer is required.

```bash
pip install -e .
pip install -e '.[dev]'          # tests and development tools
pip install -e '.[opf]'          # OPF token-classification detector
pip install -e '.[llm]'          # OpenAI-compatible second-pass detector
pip install -e '.[presidio]'     # Presidio baseline; also needs spaCy models
pip install -e '.[benchmarks]'   # benchmark tools
```

Use `requirements-lock.txt` when the pinned dependency set is needed.
`scripts/setup.sh` and `install.sh` provide the broader benchmark setup,
including optional model dependencies and a local `.env` template.

## Library usage

```python
from cryptography.fernet import Fernet
from traciora_pii import PIIVaultStore, PIIVirtualizer

vault = PIIVaultStore(":memory:", Fernet.generate_key())
vault.open_session("example-session")
virtualizer = PIIVirtualizer(vault)
safe = virtualizer.anonymize_text("example-session", "Email test@example.com")
restored = virtualizer.rehydrate_text("example-session", safe)
vault.close()
```

Use `make_pii_brokered_tool` to restore arguments and anonymize results around
tool execution. Use `assert_model_safe` at the model boundary; it checks for
regex-detectable raw values already tracked in the current session vault.
`StreamingRefRehydrator` supports restoring tokens in streamed text.

## Configuration

The runtime configuration is `traciora.yaml`. Select another file with
`TRACIORA_PII_CONFIG=/absolute/path/config.yaml`. Keep private overrides in
ignored `*.local.yaml` files and credentials in `.env` or environment variables.

To leave license plates and URLs visible:

```yaml
detector:
  excluded_categories: [plate, url]
```

The default is `[]`. Exclusions apply to regex detection, every second-pass
backend, and destructive masking. Other categories inside excluded values can
still be masked, such as an email inside a URL. Existing vault tokens still
resolve. See [CONFIGURATION.md](CONFIGURATION.md) for the full category list,
precedence, programmatic settings, and migration names.

The regex pass supports English, German, and Russian policies. The optional
second pass is selected with `detector.second_pass`:

| Mode | Backend |
|---|---|
| `none` | Regex only; checked-in default |
| `opf` | Token classifier selected from `detector.opf.models` |
| `llm` | Configured OpenAI-compatible endpoint |
| `presidio` | Presidio with the configured spaCy model |

Custom regex and OPF policy YAML paths are supported. Excluding a category
leaves its text available to downstream detectors and the model. Image pixels
are outside this text-only mechanism.

## Tools and tests

- [Benchmark generator](benchmarks/data_generator/README.md): synthetic source
  templates and seed vocabulary; generated datasets stay local.
- [Data import](benchmarks/data_import/README.md): corpus loading and quality
  analysis; caches and reports stay local.
- [Evaluation harness](benchmarks/pii_evaluation/README.md): detection and
  tool-calling comparisons; results and logs stay local.
- [Training tools](fine_tuning/README.md): data preparation, training, and
  evaluation; training data, checkpoints, and run artifacts stay local.
- [Experiment runners](claims/README.txt): retained scripts and acceptance
  criteria; reference snapshots must be supplied locally when needed.

The evaluation loader can render synthetic data from tracked source fixtures
when the ignored frozen dataset is absent. To freeze a local copy:

```bash
python -m benchmarks.pii_evaluation.dataset.render --freeze
```

Run the library and tool tests with `pytest` after installing the development
dependencies. The tests use synthetic examples; optional model integration
checks may be skipped when their dependencies are unavailable.

NeMo comparison tools live under `benchmarks/pii_evaluation/nemo/`. Their cached
inputs are local to `benchmarks/pii_evaluation/results/nemo/`, or the directory
selected by `TRACIORA_NEMO_DATA_DIR`.

## Repository contents

Git tracks source code, tests, tooling, relevant documentation, detector
policies, and the synthetic source fixtures needed by the tools. Personal
records, credentials, local settings, generated datasets, result snapshots,
reports, logs, vault databases, and model artifacts are ignored. Package
manifests also exclude local data and output directories.

Untracking a file keeps its local copy and removes it from the current tree;
it does not remove the file from previous commits.

## License

Apache 2.0. See [LICENSE](LICENSE) for terms and third-party notices.
