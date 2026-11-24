# How it works

traciora-pii sits between your agent runtime and the model provider. The model
works with reference tokens; your tools and your user see real values.

## The flow

```
user text ──► detect ──► tokenize ──► model sees tokens
                │            │
                └──► vault (encrypted originals, per session)
                             │
model calls a tool ──► broker restores real values in the arguments ──► tool runs
                                                                         │
model sees tokens ◄── broker tokenizes the result ◄──────────────────────┘
                             │
streamed answer ──► StreamingRefRehydrator ──► user sees real values
```

1. **Detect.** The regex pass finds structured values (phones, emails, cards,
   IBANs, national IDs and more). An optional second pass finds contextual
   values such as names and street addresses. See [Detectors](detectors.md).
2. **Tokenize.** `PIIVirtualizer.anonymize_text` replaces each finding with a
   reference token. The second pass runs on the already-tokenized text, so it
   never sees a value the regex pass caught. Before either pass, every value
   the session's vault already holds is replaced by its token wherever it
   appears, so a value a detector misses in one text (a plate written without
   the word "plate", a name the second pass catches only once) still reaches
   the model as its token.
3. **Vault.** `PIIVaultStore` keeps each original value, encrypted with Fernet,
   in SQLite under the session it belongs to.
4. **Restore tool arguments.** A tool wrapped with `make_pii_brokered_tool`
   receives real values: tokens in its arguments (plain strings, dicts, nested
   structures and JSON strings) are resolved just before it runs.
5. **Tokenize tool results.** The broker passes the tool's return value through
   `anonymize_text` / `anonymize_structured` before the model sees it.
6. **Guard the boundary.** `assert_model_safe` scans a model-bound payload and
   raises `PIILeakError` if it contains a raw value this session has already
   tokenized.
7. **Restore the answer.** `rehydrate_text` restores a whole answer;
   `StreamingRefRehydrator` restores tokens in a stream, holding back any
   chunk tail that could be the start of a token.

## Sessions

Every call takes a `session_scope_key`, usually one per conversation.

- `vault.open_session(key)` opens or resumes a session.
- Within a session, the same value always gets the same token, so the model
  can refer to "the same customer" across turns. Before hashing, whitespace,
  dashes and parentheses are removed and the value is lowercased, so
  `+49 30 1234 5678` and `+49-30-1234-5678` share one token.
- Russian names are keyed by their dictionary form, so inflected forms of one
  name share a token. The first surface form seen is what rehydration returns.
- The same value gets a different token in a different session.
- A value the session already holds is found without a detector when it is
  written with the same characters, in any case for values with a digit or an
  `@` and in the stored case for names, with any whitespace, dashes or
  parentheses between them. It is not replaced inside a longer word or
  number. Values shorter than four characters, or than six digits for a bare
  number, are left to the detectors.
- `vault.close_session(key)` stops resolution. Rows stay for audit but no
  longer resolve.
- `vault.cleanup_expired(ttl_hours=24)` expires sessions idle longer than the TTL.

The vault stores, per token: category, encrypted value, a short audit preview
(first three and last two characters), first/last seen times and the source.
Use `":memory:"` for a throwaway vault or a file path for a persistent one.
Keep the Fernet key stable to read a persistent vault across restarts.

## Token format

```
<category>_ref:<two letters>_<8 hex digits>
```

Examples: `email_ref:em_18a53099`, `phone_ref:ph_43538438`,
`iban_ref:ib_1831c318`. The category is one of the vault categories listed in
[Configuration](configuration.md#category-exclusions). The exported
`REF_TOKEN_PATTERN` matches every token the library can emit.

Before detection, input text is NFKC-normalized and zero-width characters are
removed, so a value split by invisible characters is still found.

## What is protected

- Text passed through `anonymize_text`, `anonymize_structured` or
  `anonymize_content`, including tool results returned through the broker.
- Categories covered by the regex policies and the selected second pass.
- Model-bound payloads checked with `assert_model_safe`, for values the session
  vault already holds.

## What is not protected

- **Image pixels and binary content.** The library is text-only. Pass a
  `skip_block` predicate to `assert_model_safe` to skip binary blocks; the
  package never inspects image contents.
- **Types outside the taxonomy.** Values outside the regex policies and the
  second-pass label space (for example medical record numbers or biometric
  templates) are not detected.
- **Other languages.** Regex policies and fine-tuned models cover English,
  German and Russian. Other languages get degraded coverage.
- **Detector misses.** The leak guard only knows values already in the vault.
  A value that was never detected can still reach the model provider.
- **Excluded categories.** A category you exclude stays visible to the model
  and to any downstream detector.
- **Hosted second passes.** An LLM or other hosted detector receives the text
  that remains after the regex pass, including the names and addresses it is
  asked to find.
- **Tool exceptions.** The broker tokenizes return values only. If a tool can
  raise with personal data in the message, anonymize the error text yourself
  before it goes back to the model.

## Intended use

The library is for defensive use: reducing the personal data an agent runtime
sends to LLM inference providers. Do not use it to build surveillance or
user-identification systems, or to strip identifiers from records that must
stay attributable for audit, consent or accountability. Apply legal review for
your jurisdiction before processing real personal data.
