<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
  <img alt="traciora-pii: your agents work with real customer data, your model never sees it" src="docs/assets/hero-light.svg">
</picture>

**traciora-pii is reversible PII protection for LLM tool-calling agents.**

An agent that books, invoices or replies to customers needs their names, emails and phone
numbers. The language model behind it does not. traciora-pii swaps personal data for
tokens before the model sees it and swaps the real values back in only where they are
needed: inside your tools and in the final answer to your user.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flow-dark.svg">
  <img alt="Swap, reason, restore for tools, answer" src="docs/assets/flow-light.svg">
</picture>

Tokens are stable within a session, so the model can still tell two customers apart
and refer back to the same one. Originals live in an encrypted vault scoped to that session.

## Results

Eight LLMs ran the same 309 tool-calling tasks in English, German and Russian, with and without traciora-pii.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/stats-dark.svg">
  <img alt="65 raw personal values reached the model, down from 17,631; 90.5% of tasks completed against 92.3% unprotected; 0.93 detection F1" src="docs/assets/stats-light.svg">
</picture>

Deleting personal data instead (masking) leaves agents unable to finish two tasks in three. Swapping it for tokens keeps them working.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/dots-dark.svg">
  <img alt="Tasks completed by model, with traciora-pii, with no protection and with masking" src="docs/assets/dots-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bars-dark.svg">
  <img alt="Detector micro F1: the fine-tuned per-language OPF detector scores 0.93, ahead of OPF, LLM, Presidio and regex-only detection" src="docs/assets/bars-light.svg">
</picture>

Numbers, methods and how to rerun them: [Reproduce the results](docs/reproduce.md).

## What you get

- **A vault, not a mask.** Values come back exactly, per session, encrypted at rest.
- **A tool broker.** Wrap any tool once; it receives real arguments and returns tokenized results.
- **A leak guard.** A check at the model boundary raises if a tracked raw value slipped through.
- **Streaming.** Tokens are restored in streamed answers, even when a token is split across chunks.
- **Your choice of detector.** Fast regex for English, German and Russian, plus an optional
  second pass: a fine-tuned token classifier, an encoder, any OpenAI-compatible LLM, or Presidio.
- **Your rules.** Leave chosen categories visible, such as plates or URLs, with one setting.

## Quick start

```bash
pip install "traciora-pii @ git+https://github.com/danilmezor/traciora-pii.git"
```

```python
from cryptography.fernet import Fernet
from traciora_pii import PIIVaultStore, PIIVirtualizer

vault = PIIVaultStore(":memory:", Fernet.generate_key())
vault.open_session("chat-1")
pii = PIIVirtualizer(vault)

safe = pii.anonymize_text("chat-1", "Email anna@acme.com about the invoice")
# "Email email_ref:em_… about the invoice"
pii.rehydrate_text("chat-1", safe)
# "Email anna@acme.com about the invoice"
```

Next: [Getting started](docs/getting-started.md) wraps a tool and guards the model boundary.

## Documentation

[How it works](docs/how-it-works.md) · [Getting started](docs/getting-started.md) ·
[Configuration](docs/configuration.md) · [Detectors](docs/detectors.md) ·
[Benchmarks](docs/benchmarks.md) · [Reproduce the results](docs/reproduce.md)

## License

[Apache-2.0](LICENSE). Made by [Traciora](https://traciora.com).
