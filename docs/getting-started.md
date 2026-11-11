# Getting started

## Install

Python 3.11 or newer. From a checkout of the repository:

```bash
pip install -e .
```

The core install (cryptography, pydantic, PyYAML) gives you the regex pass,
the vault, the broker, the leak guard and streaming. Add extras for more:

| Extra | Adds |
|---|---|
| `opf` | OPF token-classifier second pass (`detector.second_pass: opf`) |
| `encoder` | ruBERT / Horizon ModernBERT encoder second pass (torch, transformers) |
| `mlx` | OPF on Apple silicon through MLX (`detector.opf.device: mlx`); macOS arm64 only |
| `llm` | Second pass through any OpenAI-compatible endpoint (httpx) |
| `presidio` | Microsoft Presidio second pass; also needs a spaCy model |
| `benchmarks` | Evaluation harness, data import and agent experiments (LangChain, datasets) |
| `opf-finetune` | OPF fine-tuning (torch; bitsandbytes on Linux x86_64) |
| `dev` | pytest and the libraries the test suite needs |

```bash
pip install -e '.[opf]'
pip install -e '.[presidio]' && python -m spacy download en_core_web_lg
```

`requirements-lock.txt` pins a known-good dependency set.
`bash scripts/setup.sh` installs the `dev`, `opf`, `presidio` and `benchmarks`
extras, downloads the spaCy models and creates `.env` from `.env.example`.

## First example

```python
from cryptography.fernet import Fernet
from traciora_pii import PIIVaultStore, PIIVirtualizer

vault = PIIVaultStore(":memory:", Fernet.generate_key())
vault.open_session("conv-1")
virtualizer = PIIVirtualizer(vault)

safe = virtualizer.anonymize_text(
    "conv-1", "Email anna.berg@example.com or call +49 30 1234 5678.")
# 'Email email_ref:em_18a53099 or call phone_ref:ph_43538438.'

virtualizer.rehydrate_text("conv-1", safe)
# 'Email anna.berg@example.com or call +49 30 1234 5678.'

vault.close()
```

Send `safe` to the model. Rehydrate what comes back before showing it to the
user. For a persistent vault pass a file path and a stable key, for example
one read from `PII_VAULT_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Wrap a tool

`make_pii_brokered_tool(tool, session_scope_key, virtualizer)` patches the
tool in place and returns it. Arguments arrive with real values; the result
goes back to the model tokenized. LangChain tools (anything with `_run`) are
patched at `_run`; other objects are patched at `invoke`.

```python
from langchain_core.tools import tool
from traciora_pii import make_pii_brokered_tool

@tool
def lookup_customer(email: str) -> str:
    """Look up a customer by email."""
    return f"Customer {email}, phone +1 415 555 0134"

make_pii_brokered_tool(lookup_customer, "conv-1", virtualizer)

lookup_customer.invoke({"email": "email_ref:em_18a53099"})
# the tool receives 'anna.berg@example.com'
# returns 'Customer email_ref:em_18a53099, phone phone_ref:ph_352829ee'
```

Wrap every tool the agent can call, once per session. The broker does not
touch exceptions: if a tool can raise with personal data in the message,
pass the error text through `virtualizer.anonymize_text` yourself.

## Guard the model boundary

Call `assert_model_safe` on every payload before it goes to the provider:

```python
from traciora_pii import PIILeakError, assert_model_safe

try:
    assert_model_safe(messages, "conv-1", vault)
except PIILeakError as exc:
    # exc.pii_type, e.g. '[EMAIL]'; exc.preview, e.g. 'ann***'
    messages = virtualizer.anonymize_structured("conv-1", messages)
```

The payload can be any nesting of dicts, lists and strings. The guard raises
only for a raw value that the regex pass detects **and** that this session's
vault already holds, so unrelated digit strings do not trip it. Re-anonymizing
and continuing, as above, is the pattern the evaluation harness uses.
Pass `skip_block=` to skip binary blocks such as images. The guard is on by
default; `detector.leak_guard_enabled: false` turns it off.

## Stream the answer

```python
from traciora_pii import StreamingRefRehydrator

rehydrator = StreamingRefRehydrator(virtualizer, "conv-1")
for chunk in model_stream:
    send_to_user(rehydrator.feed(chunk))
send_to_user(rehydrator.flush())
```

`feed` holds back any tail that could be the start of a token, down to a
single character, so results do not depend on provider chunk sizes. Always
call `flush` at the end of the stream.

## Add a second pass

The regex pass does not find most names or street addresses. Add a second pass
for those. For OPF or encoder models, prepare the model once at startup and
share one detector across sessions:

```python
from traciora_pii import (PIIVirtualizer, get_shared_second_pass,
                          prepare_second_pass, traciora_config)

config = traciora_config("en").with_overrides({"detector": {"second_pass": "opf"}})
prepare_second_pass(config)            # downloads and validates; call at startup
detector = get_shared_second_pass(config)   # opens the local copy, runs a self-check

virtualizer = PIIVirtualizer(vault, second_pass_detector=detector, config=config)
```

A configured second pass either runs or raises; it never falls back to regex
only. For the LLM and Presidio backends, build the detector from a loaded
configuration:

```python
from traciora_pii.config import load_config
from traciora_pii.pii_virtualizer import build_second_pass_detector

config = load_config()                 # reads traciora.yaml
detector = build_second_pass_detector(config)
virtualizer = PIIVirtualizer(vault, second_pass_detector=detector, config=config)
```

See [Detectors](detectors.md) to choose a backend and
[Configuration](configuration.md) for every setting.

## One-way redaction

When you do not need the original values back, use `redact_pii_text` (returns
the text with `[EMAIL]`-style placeholders and the findings) or `scrub` (regex
only, also masks long numbers; meant for logs). `detect_pii` returns the raw
findings.

## Run the tests

```bash
pip install -e '.[dev]'
pytest
```

Tests use synthetic examples and simulated second-pass findings; they do not
download models.
