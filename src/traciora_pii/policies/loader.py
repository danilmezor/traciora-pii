"""Policy loader: YAML name/path -> validated, compiled policy objects.

Both ``load_regex_policy`` and ``load_opf_policy`` take an explicit
``name_or_path`` argument and never read the environment. Callers
resolve their preferred policy via the config layer
(``cfg.detector.regex_policy`` or ``cfg.detector.opf.models[X].policy``)
and pass the value in. Tests can swap policies by passing a different
argument.

A built-in name resolves to a YAML file shipped inside the package
(``traciora_pii/policies/regex/<name>.yaml`` or
``traciora_pii/policies/opf/<name>.yaml``). A path is taken verbatim.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import yaml

from ..pii import PIIDetector
from . import validators
from .schemas import OPFPolicy, RegexPolicy, validate_opf_against_regex

_REGEX_FLAGS = re.IGNORECASE | re.UNICODE
_BUILTIN_REGEX_PKG = "traciora_pii.policies.regex"
_BUILTIN_OPF_PKG = "traciora_pii.policies.opf"


def _looks_like_path(spec: str) -> bool:
    import os as _os
    return _os.sep in spec or spec.endswith(".yaml") or spec.startswith("/")


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_builtin(pkg: str, name: str) -> dict:
    resource = files(pkg).joinpath(f"{name}.yaml")
    with resource.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_spec(spec: str, builtin_pkg: str) -> dict:
    """Return parsed YAML for either a builtin name or a filesystem path."""
    if _looks_like_path(spec):
        return _read_yaml(Path(spec).expanduser())
    return _load_builtin(builtin_pkg, spec)


def load_regex_policy(name_or_path: str = "ru") -> RegexPolicy:
    """Load and validate a regex policy.

    ``name_or_path`` is either a built-in name (``ru`` / ``en`` / ``de``)
    that resolves to ``traciora_pii/policies/regex/<name>.yaml`` or an
    absolute / relative path to a custom YAML file.
    """
    spec = (name_or_path or "ru").strip()
    if not spec:
        raise ValueError("regex policy name/path required")
    return RegexPolicy.model_validate(_resolve_spec(spec, _BUILTIN_REGEX_PKG))


def load_opf_policy(name_or_path: str = "ru_comprehensive") -> OPFPolicy:
    """Load and validate an OPF policy.

    ``name_or_path`` is either a built-in name
    (e.g. ``ru_comprehensive`` / ``en_names_addresses``) that resolves
    to ``traciora_pii/policies/opf/<name>.yaml`` or an absolute / relative
    path to a custom YAML file.
    """
    spec = (name_or_path or "ru_comprehensive").strip()
    if not spec:
        raise ValueError("opf policy name/path required")
    return OPFPolicy.model_validate(_resolve_spec(spec, _BUILTIN_OPF_PKG))


def load_multi_regex_policy(names_or_paths: Sequence[str]) -> RegexPolicy:
    """Load and merge multiple regex policies into a single synthetic policy.

    Each input is resolved by :func:`load_regex_policy`; the merged result
    concatenates the input policies' patterns in declaration order
    (preserving the "first-listed wins on overlap" semantics across the
    union) and unions their placeholder declarations, deduping by
    ``(name, pattern)`` for patterns and by ``placeholder`` for declarations.

    Use this when the runtime may receive content in any of several
    languages: the regex layer then catches every locale's typed
    identifiers regardless of which language the surrounding prose is in.
    """
    if not names_or_paths:
        raise ValueError("at least one policy name/path required")
    loaded = [(name, load_regex_policy(name)) for name in names_or_paths]
    if len(loaded) == 1:
        return loaded[0][1]

    # Dedupe patterns. Same (name, pattern) → drop. Same name with a
    # different pattern (e.g. `passportnum` in both en and ru) → suffix
    # the later occurrence with the locale tag so the merged policy
    # validates and both patterns still run.
    seen_pattern_keys: set[tuple[str, str]] = set()
    name_to_pattern_text: dict[str, str] = {}
    merged_patterns: list = []
    for locale_tag, policy in loaded:
        for pat in policy.patterns:
            key = (pat.name, pat.pattern)
            if key in seen_pattern_keys:
                continue
            seen_pattern_keys.add(key)
            existing_pattern_text = name_to_pattern_text.get(pat.name)
            if existing_pattern_text is not None and existing_pattern_text != pat.pattern:
                pat = pat.model_copy(update={"name": f"{pat.name}__{locale_tag}"})
            else:
                name_to_pattern_text[pat.name] = pat.pattern
            merged_patterns.append(pat)

    # Dedupe placeholders by placeholder string. First definition wins
    # (so `en` carries the canonical `[EMAIL]` description even when
    # `de`/`ru` declare it).
    seen_placeholders: set[str] = set()
    merged_placeholders: list = []
    for _, policy in loaded:
        for ph in policy.placeholders:
            if ph.placeholder in seen_placeholders:
                continue
            seen_placeholders.add(ph.placeholder)
            merged_placeholders.append(ph)

    _base_name, base_policy = loaded[0]
    merged_dict = base_policy.model_dump()
    merged_dict["name"] = "+".join(name for name, _ in loaded)
    merged_dict["description"] = "Merged: " + " + ".join(name for name, _ in loaded)
    merged_dict["patterns"] = [p.model_dump() for p in merged_patterns]
    merged_dict["placeholders"] = [p.model_dump() for p in merged_placeholders]
    return RegexPolicy.model_validate(merged_dict)


def compile_multi_regex_policy(names_or_paths: Sequence[str]) -> list[PIIDetector]:
    """Compile a list of regex policies into a single deduped detector list.

    Equivalent to :func:`compile_regex_policy` applied to
    :func:`load_multi_regex_policy`. Convenience helper used by
    :func:`pii._get_default_detectors` when ``detector.regex_policies`` is
    set in the runtime config.
    """
    return compile_regex_policy(load_multi_regex_policy(names_or_paths))


def compile_regex_policy(policy: RegexPolicy) -> list[PIIDetector]:
    """Compile a :class:`RegexPolicy` into ready-to-execute detectors.

    Resolves named validators against the registry, compiles regex
    patterns once, and returns detectors in policy-declaration order.
    """
    detectors: list[PIIDetector] = []
    for pat in policy.patterns:
        validator_fn = validators.get(pat.validator) if pat.validator else None
        detectors.append(
            PIIDetector(
                name=pat.name,
                pattern=re.compile(pat.pattern, _REGEX_FLAGS),
                placeholder=pat.placeholder,
                keywords=tuple(k.lower() for k in pat.keyword_anchors),
                lookaround_chars=max(1, pat.lookaround_chars),
                validator=validator_fn,
            )
        )
    return detectors


def build_placeholder_to_prefix(policy: RegexPolicy) -> dict[str, str]:
    """Map ``[PLACEHOLDER]`` -> vault-category string (the ref-token prefix)."""
    return {p.placeholder: p.vault_category.value for p in policy.placeholders}


def build_ref_token_pattern(policy: RegexPolicy) -> re.Pattern:
    """Compile the ref-token regex used to find vault refs in text.

    Matches ``<category>_ref:<scope>_<hex8>`` for every category declared
    in the policy. Categories are emitted as a regex alternation so the
    pattern reflects the policy's actual taxonomy.
    """
    categories = sorted({p.vault_category.value for p in policy.placeholders})
    if not categories:
        raise ValueError("policy has no placeholders; cannot build ref-token pattern")
    alternation = "|".join(re.escape(c) for c in categories)
    return re.compile(rf"(?:{alternation})_ref:[a-z]{{2}}_[a-f0-9]{{8}}")


def build_partial_ref_token_pattern(policy: RegexPolicy) -> re.Pattern:
    """Compile a regex matching any *prefix* of a full ref token at end-of-buffer.

    Used by streaming rehydrators to decide whether to hold a chunk back
    in case the next chunk completes a `<category>_ref:<scope>_<hex8>`
    sequence. Anchored to ``$`` so it only matches the trailing characters
    of the buffer. Derives the category alternation from the same policy
    that drives :func:`build_ref_token_pattern`, so the streaming and
    completion matchers can't drift.
    """
    categories = sorted(
        {p.vault_category.value for p in policy.placeholders}, key=len, reverse=True,
    )
    if not categories:
        raise ValueError("policy has no placeholders; cannot build partial-ref pattern")
    cat_alt = "|".join(re.escape(c) for c in categories)
    # Optional partial suffix: 0-1 of `_`, then 0-1 of `r`, `e`, `f`, `:`,
    # then 0-2 scope letters, then 0-1 `_`, then 0-8 hex chars. Each layer
    # nests as `(?:x(?:y)?)?` so any prefix of the suffix matches.
    suffix = (
        r"(?:_(?:r(?:e(?:f(?::[a-z]{0,2}(?:_[a-f0-9]{0,8})?)?)?)?)?)?"
    )
    # Proper prefixes of category names — for chunks that split inside a
    # category name itself (e.g. buffer ends with `pho` and the next chunk
    # delivers `ne_ref:...`). Sorted longest-first so the regex tries the
    # most specific prefix first. Include even single-letter prefixes: SSE
    # and provider token streams may split at any character. Holding a short
    # ordinary suffix until the next chunk is preferable to leaking a ref.
    proper_prefixes = sorted(
        {cat[:i] for cat in categories for i in range(1, len(cat))},
        key=len,
        reverse=True,
    )
    prefix_alt = "|".join(re.escape(p) for p in proper_prefixes) or "(?!)"
    return re.compile(rf"(?:(?:{cat_alt}){suffix}|{prefix_alt})$")


def cross_validate(opf: OPFPolicy, regex: RegexPolicy) -> None:
    """Re-export of :func:`schemas.validate_opf_against_regex` for convenience."""
    validate_opf_against_regex(opf, regex)


__all__ = [
    "build_partial_ref_token_pattern",
    "build_placeholder_to_prefix",
    "build_ref_token_pattern",
    "compile_multi_regex_policy",
    "compile_regex_policy",
    "cross_validate",
    "load_multi_regex_policy",
    "load_opf_policy",
    "load_regex_policy",
]
