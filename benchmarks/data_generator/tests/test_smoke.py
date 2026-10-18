"""Smoke tests for the slot-template PII benchmark generator.

These run offline (no network, no spaCy, no HuggingFace) using only the
RU seed bundle shipped under ``seeds/ru/``. They cover:

  * SeedBundle loading + pydantic validation.
  * Checksum correctness for every locale-specific identifier.
  * Span invariant ``text[start:end] == value`` across positives, fillers,
    and noise.
  * Determinism: same seed → byte-identical output.
  * Negative-mix behaviour: the negative-rate config matches realised mix.
  * Filler placement: matches source category, never crosses categories.
  * Policy-contract: every regex-backed placeholder the generator emits is
    detected by ``policies/regex/ru.yaml``.
  * JSONL serialisation + EvalQuery roundtrip.
"""
from __future__ import annotations

import json
import random
import re

import pytest

from benchmarks.data_generator import (
    GeneratedExample,
    NoiseApplier,
    NoiseConfig,
    Pipeline,
    SlotFillEngine,
    load_seed_bundle,
    load_shared_seeds,
    split_examples,
    to_eval_query,
    write_jsonl,
)
from benchmarks.data_generator.faker_providers import RuPII, _format_person_name_variant


# ======================================================================
# SeedBundle loading
# ======================================================================


def test_shared_seeds_load():
    s = load_shared_seeds()
    assert s.secrets.shapes
    assert s.domains.reserved
    assert s.ip_blocks.test_nets
    assert s.ocr_confusables.single_char


def test_ru_bundle_loads_and_validates():
    ru = load_seed_bundle("ru")
    assert ru.locale == "ru"
    assert ru.names.last_names
    assert ru.names.first_names_male
    assert ru.names.first_names_female
    assert ru.addresses.city_streets
    assert ru.addresses.formats
    assert ru.identifiers.model_extra["plate_letters"]
    assert ru.templates.categories
    # Negatives + fillers are present (their YAMLs ship)
    assert ru.negatives.probability > 0
    assert ru.fillers.prepend_probability > 0


def test_en_bundle_loads_and_renders():
    en = load_seed_bundle("en")
    assert en.locale == "en"
    assert en.names.last_names
    assert en.names.first_names_male
    assert en.names.first_names_female
    assert en.addresses.city_streets
    assert en.addresses.formats
    assert en.identifiers.model_extra["phone_area_codes"]
    assert en.identifiers.model_extra["ssn_area_numbers"]
    assert en.templates.categories
    assert en.negatives.probability > 0
    assert en.fillers.prepend_probability > 0

    examples = list(Pipeline(seeds=en, seed=42).run(25))
    assert examples
    assert any(ex.kind == "positive" and ex.spans for ex in examples)
    for ex in examples:
        for span in ex.spans:
            assert ex.text[span.start:span.end] == span.value


def test_de_bundle_loads_and_renders():
    de = load_seed_bundle("de")
    assert de.locale == "de"
    assert de.names.last_names
    assert de.names.first_names_male
    assert de.names.first_names_female
    assert de.addresses.city_streets
    assert de.addresses.plz_city_streets
    assert de.addresses.formats
    assert de.identifiers.model_extra["phone_area_codes"]
    assert de.identifiers.model_extra["steuer_id_formats"]
    assert de.templates.categories
    assert de.negatives.probability > 0
    assert de.fillers.prepend_probability > 0

    examples = list(Pipeline(seeds=de, seed=42).run(25))
    assert examples
    assert any(ex.kind == "positive" and ex.spans for ex in examples)
    for ex in examples:
        for span in ex.spans:
            assert ex.text[span.start:span.end] == span.value


def test_unknown_locale_raises():
    with pytest.raises(FileNotFoundError):
        load_seed_bundle("xx")


def test_person_name_formatter_covers_order_and_initial_variants():
    """Formatter currently emits 3 variants: given+surname, surname+given,
    given+surname-initial. The reverse ``S. Name`` form was deliberately
    dropped (see _format_person_name_variant docstring) because it reads
    as inverted in EN/DE/RU CRM contexts. Update this set in lockstep if
    the formatter taxonomy changes."""
    rng = random.Random(123)
    variants = {
        _format_person_name_variant("Ivan", "Petrov", rng)
        for _ in range(100)
    }
    assert variants == {
        "Ivan Petrov",
        "Petrov Ivan",
        "Ivan P.",
    }


def test_template_unknown_slot_rejected(tmp_path):
    """Templates referencing slots not in _KNOWN_SLOTS must fail at load."""
    from benchmarks.data_generator.seeds import TemplatesConfig
    bad = {
        "categories": {
            "x": {
                "templates": [{"text": "Hello {NOT_A_REAL_SLOT}."}],
            },
        },
    }
    with pytest.raises(Exception):
        TemplatesConfig.model_validate(bad)


# ======================================================================
# Checksum correctness
# ======================================================================


def _inn10_check(s: str) -> bool:
    weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
    digits = [int(c) for c in s]
    return digits[-1] == sum(d * w for d, w in zip(digits[:-1], weights)) % 11 % 10


def _inn12_check(s: str) -> bool:
    a = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    b = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    digits = [int(c) for c in s]
    c1 = sum(d * w for d, w in zip(digits[:10], a)) % 11 % 10
    c2 = sum(d * w for d, w in zip(digits[:11], b)) % 11 % 10
    return digits[10] == c1 and digits[11] == c2


def _snils_check(s: str) -> bool:
    body = s.replace(" ", "").replace("-", "")
    if len(body) != 11:
        return False
    digits = [int(c) for c in body[:9]]
    expected = int(body[9:])
    weights = (9, 8, 7, 6, 5, 4, 3, 2, 1)
    sm = sum(d * w for d, w in zip(digits, weights))
    if sm < 100:
        check = sm
    elif sm in (100, 101):
        check = 0
    else:
        r = sm % 101
        check = 0 if r in (100, 101) else r
    return check == expected


def _ogrn_check(s: str) -> bool:
    return int(s[-1]) == int(s[:-1]) % 11 % 10


def _luhn_ok(s: str) -> bool:
    digs = [int(c) for c in s if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digs)):
        if i % 2 == 0:
            total += d
        else:
            total += d * 2 if d * 2 < 10 else d * 2 - 9
    return total % 10 == 0


def _vin_check_ok(vin: str) -> bool:
    if len(vin) != 17 or any(c in "IOQ" for c in vin):
        return False
    translit = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
        "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
        "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
    }
    for d in range(10):
        translit[str(d)] = d
    weights = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
    s = sum(translit[c] * w for c, w in zip(vin, weights))
    expected = vin[8]
    actual = "X" if s % 11 == 10 else str(s % 11)
    return expected == actual


def test_ru_pii_checksums():
    seeds = load_seed_bundle("ru")
    p = RuPII(seeds=seeds, rng=random.Random(0))
    for _ in range(50):
        assert _inn10_check(p.inn_legal())
        assert _inn12_check(p.inn_individual())
        assert _snils_check(p.snils())
        assert _ogrn_check(p.ogrn())
        assert _vin_check_ok(p.vin())
        assert _luhn_ok(p.card())


# ======================================================================
# Pipeline span / kind / category invariants
# ======================================================================


def test_pipeline_span_invariant_ru():
    pipe = Pipeline(seeds=load_seed_bundle("ru"), seed=42)
    examples = list(pipe.run(120))
    assert len(examples) == 120
    for ex in examples:
        for span in ex.spans:
            assert ex.text[span.start:span.end] == span.value, (
                f"{ex.id} span: {ex.text[span.start:span.end]!r} != {span.value!r}"
            )
        if ex.kind == "negative":
            assert ex.spans == ()
            assert ex.category == "hard_negative"
        else:
            assert ex.spans  # positives have at least one slot
            assert ex.category in load_seed_bundle("ru").templates.categories


def test_pipeline_negative_mix_in_range():
    pipe = Pipeline(seeds=load_seed_bundle("ru"), seed=7)
    examples = list(pipe.run(500))
    n_neg = sum(1 for ex in examples if ex.kind == "negative")
    target = load_seed_bundle("ru").negatives.probability
    # Allow ±5% tolerance on a 500-sample run.
    assert abs(n_neg / len(examples) - target) < 0.05, (
        f"negative mix off: {n_neg}/{len(examples)} vs target {target}"
    )


def test_pipeline_filler_categories_match_source():
    """A prepended/appended filler must come from the source template's category.

    We verify by checking that for every example with a filler, the filler
    text appears in the source category's filler list.
    """
    seeds = load_seed_bundle("ru")
    pipe = Pipeline(seeds=seeds, seed=11)
    n_with_filler = 0
    for ex in pipe.run(300):
        if ex.kind == "negative":
            continue
        if ex.fillers_prepended == 0 and ex.fillers_appended == 0:
            continue
        n_with_filler += 1
        # Reconstruct: text outside of (prefix-filler + sep + template + sep + suffix-filler)
        # Easier check: every filler chosen for this example must be a string from
        # the matching category's filler list.
        category_fillers = set(seeds.fillers.for_category(ex.category))
        # Each filler shows up as a substring in ex.text
        # (We don't know which one was picked, but at least one filler from this
        # category must appear as a substring of ex.text somewhere outside the
        # template_text region.)
        outside_template = ex.text.replace(ex.template_text, "", 1)
        # Some category filler should match a portion of `outside_template`.
        assert any(filler in outside_template for filler in category_fillers), (
            f"{ex.id} carries a filler not from category {ex.category}"
        )
    assert n_with_filler > 0, "no fillers exercised in 300-example run"


def test_pipeline_determinism():
    def run_once(seed: int):
        return [
            ex.to_dict()
            for ex in Pipeline(seeds=load_seed_bundle("ru"), seed=seed).run(50)
        ]
    assert run_once(42) == run_once(42)
    assert run_once(42) != run_once(43)


# ======================================================================
# Slot coverage
# ======================================================================


def test_every_template_slot_has_generator():
    """All template slot names must have a registered generator method."""
    from benchmarks.data_generator.injectors import _SLOT_METHODS
    seeds = load_seed_bundle("ru")
    used = seeds.templates.all_slots_used()
    for slot in used:
        assert slot in _SLOT_METHODS, f"slot {{{slot}}} has no _SLOT_METHODS entry"


def test_engine_can_fill_every_slot_used_in_seeds():
    """Smoke: invoke the generator for every slot a template references."""
    seeds = load_seed_bundle("ru")
    engine = SlotFillEngine(seeds=seeds, rng=random.Random(0))
    for slot in seeds.templates.all_slots_used():
        value = engine._generate_for_slot(slot)
        assert isinstance(value, str) and len(value) > 0, slot


# ======================================================================
# Noise — span invariance preserved under default-zero AND under perturbation
# ======================================================================


def test_noise_zero_rates_is_identity():
    seeds = load_seed_bundle("ru")
    pipe = Pipeline(seeds=seeds, seed=42)
    applier = NoiseApplier(config=NoiseConfig(locale="ru"), seeds=seeds)
    for ex in list(pipe.run(20)):
        text2, spans2 = applier.apply(ex.text, list(ex.spans))
        assert text2 == ex.text
        assert spans2 == list(ex.spans)


@pytest.mark.parametrize("typo_rate", [0.05, 0.10])
def test_noise_scaffold_preserves_span_invariant(typo_rate):
    seeds = load_seed_bundle("ru")
    noise = NoiseApplier(
        config=NoiseConfig(
            typo_rate=typo_rate,
            ocr_confusable_rate=0.05,
            whitespace_jitter_rate=0.05,
            target="scaffold",
            locale="ru",
        ),
        seeds=seeds,
        rng=random.Random(int(typo_rate * 1000)),
    )
    pipe = Pipeline(seeds=seeds, seed=42, noise=noise)
    for ex in pipe.run(60):
        for s in ex.spans:
            assert ex.text[s.start:s.end] == s.value


def test_noise_pii_target_updates_span():
    seeds = load_seed_bundle("ru")
    noise = NoiseApplier(
        config=NoiseConfig(typo_rate=0.30, target="pii", locale="ru"),
        seeds=seeds,
        rng=random.Random(7),
    )
    pipe = Pipeline(seeds=seeds, seed=11, noise=noise)
    for ex in pipe.run(40):
        for s in ex.spans:
            assert ex.text[s.start:s.end] == s.value


# ======================================================================
# Policy-contract — generator output detected by regex policy
# ======================================================================


@pytest.mark.parametrize("locale,catchable", [
    (
        "ru",
        # Hard-tier only: regex catches these unconditionally on shape.
        # Soft-tier slots (RU_INN/SNILS/PASSPORT/OGRN/KPP/BIK/BANK_ACCOUNT)
        # are intentionally exercised by ~50% naturalistic templates that
        # evade the keyword anchor — those are OPF training
        # surface, not regex's responsibility. The build_release two-tier
        # gate enforces the 30% floor on soft-tier recall instead.
        {
            "[PHONE]", "[EMAIL]", "[LICENSE_PLATE]", "[URL]", "[DATE]",
        },
    ),
    (
        "de",
        # Hard-tier only: regex catches these unconditionally on shape.
        # Soft-tier slots ([DE_STEUER_ID] / [DE_PERSONALAUSWEIS]) are
        # exercised by ~50% keyword-evading templates by design — those
        # are OPF training surface, not regex's responsibility. The
        # build_release two-tier gate enforces the recall floor on them.
        {
            "[PHONE]", "[EMAIL]", "[IBAN]", "[URL]", "[DATE]",
        },
    ),
    (
        "en",
        # EN/US identifiers + universals. PERSONAL_ID is now emitted for
        # SSN/passport (formerly [US_SSN] / [PASSPORTNUM]).
        {
            "[PERSONAL_ID]", "[PHONE]", "[EMAIL]", "[IBAN]", "[URL]",
        },
    ),
])
def test_policy_catches_generator_output(locale, catchable):
    """Every regex-backed placeholder the locale generator emits must be
    detected by the matching `policies/regex/<locale>.yaml`.

    OPF-only placeholders are excluded from `catchable` (the regex layer
    isn't expected to catch them), as are placeholders whose regex requires
    a keyword anchor that some templates don't carry.
    """
    from traciora_pii.pii import detect_pii
    from traciora_pii.policies.loader import compile_regex_policy, load_regex_policy

    detectors = compile_regex_policy(load_regex_policy(locale))
    pipe = Pipeline(seeds=load_seed_bundle(locale), seed=99)

    misses = []
    for ex in pipe.run(150):
        if ex.kind != "positive":
            continue
        findings = detect_pii(ex.text, detectors=detectors)
        found = {(f.placeholder, f.start, f.end) for f in findings}
        for span in ex.spans:
            if span.placeholder not in catchable:
                continue
            hit = any(
                p == span.placeholder and not (e <= span.start or s >= span.end)
                for p, s, e in found
            )
            if not hit:
                misses.append((ex.id, span.placeholder, span.value))
    assert not misses, (
        f"{len(misses)} regex-backed spans not detected by {locale} policy. "
        f"Examples: {misses[:5]}"
    )


def test_policy_catches_every_en_template_regex_backed_slot():
    """Every HARD-tier regex-backed slot in each EN template must be detected.

    Hard-tier slots are those whose regex fires unconditionally on shape
    (no keyword anchor). Soft-tier slots are exercised by keyword-evasion
    templates that the regex misses by design — those are OPF training
    surface and are excluded from this strict per-template check.
    """
    from traciora_pii.pii import detect_pii
    from traciora_pii.policies.loader import compile_regex_policy, load_regex_policy

    detectors = compile_regex_policy(load_regex_policy("en"))
    catchable = {
        "[EMAIL]", "[CARD]", "[IP]", "[PERSONAL_ID]", "[PHONE]",
        "[IBAN]", "[URL]", "[DATE]",
    }
    seeds = load_seed_bundle("en")
    engine = SlotFillEngine(seeds=seeds, rng=random.Random(20260430))

    misses = []
    for cat in seeds.templates.categories.values():
        for template in cat.templates:
            text, spans = engine.render(template)
            findings = detect_pii(text, detectors=detectors)
            found = {(f.placeholder, f.start, f.end) for f in findings}
            for span in spans:
                if span.placeholder not in catchable:
                    continue
                hit = any(
                    p == span.placeholder and not (e <= span.start or s >= span.end)
                    for p, s, e in found
                )
                if not hit:
                    misses.append((template.text, span.placeholder, span.value))

    assert not misses, (
        f"{len(misses)} EN regex-backed spans not detected. "
        f"Examples: {misses[:5]}"
    )


def test_policy_catches_every_de_template_regex_backed_slot():
    """Every HARD-tier regex-backed slot in each DE template must be detected.

    Soft-tier slots ``[DE_STEUER_ID]`` / ``[DE_PERSONALAUSWEIS]`` are
    excluded — they have keyword-evasion templates that intentionally
    evade the regex anchor. The build_release two-tier gate enforces the
    30% recall floor on those.
    """
    from traciora_pii.pii import detect_pii
    from traciora_pii.policies.loader import compile_regex_policy, load_regex_policy

    detectors = compile_regex_policy(load_regex_policy("de"))
    catchable = {
        "[PHONE]", "[EMAIL]", "[CARD]", "[IP]", "[IBAN]", "[URL]", "[DATE]",
    }
    seeds = load_seed_bundle("de")
    engine = SlotFillEngine(seeds=seeds, rng=random.Random(20260430))

    misses = []
    for cat in seeds.templates.categories.values():
        for template in cat.templates:
            text, spans = engine.render(template)
            findings = detect_pii(text, detectors=detectors)
            found = {(f.placeholder, f.start, f.end) for f in findings}
            for span in spans:
                if span.placeholder not in catchable:
                    continue
                hit = any(
                    p == span.placeholder and not (e <= span.start or s >= span.end)
                    for p, s, e in found
                )
                if not hit:
                    misses.append((template.text, span.placeholder, span.value))

    assert not misses, (
        f"{len(misses)} DE regex-backed spans not detected. "
        f"Examples: {misses[:5]}"
    )


# ======================================================================
# JSONL + EvalQuery roundtrip
# ======================================================================


def test_jsonl_roundtrip(tmp_path):
    pipe = Pipeline(seeds=load_seed_bundle("ru"), seed=11)
    out = tmp_path / "ru.jsonl"
    n = write_jsonl(pipe.run(15), str(out))
    assert n == 15
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 15
    for line in lines:
        row = json.loads(line)
        for sp in row["spans"]:
            assert row["text"][sp["start"]:sp["end"]] == sp["value"]


def test_to_eval_query_roundtrip():
    pipe = Pipeline(seeds=load_seed_bundle("ru"), seed=7)
    examples = list(pipe.run(10))
    queries = [to_eval_query(ex) for ex in examples]
    for ex, q in zip(examples, queries):
        assert q.id == ex.id
        assert q.language == ex.locale
        assert q.turns == (ex.text,)
        assert len(q.pii_spans) == len(ex.spans)


# ======================================================================
# Train/dev/test split
# ======================================================================


def test_split_ratios():
    pipe = Pipeline(seeds=load_seed_bundle("ru"), seed=42)
    all_examples = list(pipe.run(100))
    train, dev, test = split_examples(all_examples, ratios=(0.8, 0.1, 0.1), seed=0)
    assert len(train) == 80 and len(dev) == 10 and len(test) == 10
    # No overlap.
    ids = {e.id for e in train} | {e.id for e in dev} | {e.id for e in test}
    assert len(ids) == 100
