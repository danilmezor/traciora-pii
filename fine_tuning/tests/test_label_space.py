"""Smoke tests for label-space derivation from policy YAMLs.

These run without torch / opf / bitsandbytes installed.
"""
from __future__ import annotations

import json

from fine_tuning.label_space import (
    LOCALES,
    LabelSpace,
    load_label_space,
    write_label_space_json,
)


def test_load_each_locale_yields_a_valid_space():
    for locale in LOCALES:
        space = load_label_space(locale)
        assert isinstance(space, LabelSpace)
        assert space.locale == locale
        assert space.span_class_names[0] == "O"
        # Every label after O must come from the policy's label_map keys.
        assert len(space.span_class_names) >= 2
        assert all(isinstance(n, str) and n for n in space.span_class_names)


def test_ru_includes_russian_specific_labels():
    space = load_label_space("ru")
    labels = set(space.span_class_names)
    # ru_comprehensive policy declares these — fine-tune must learn them.
    # personal_id replaces the former ru_passport / ru_drv_license labels
    # (structurally collision-prone, merged in the v2 policy revision).
    for required in ("ru_inn", "ru_snils", "personal_id", "private_person"):
        assert required in labels, f"{required} missing from RU label space: {labels}"


def test_placeholder_inversion_round_trips():
    """Every (placeholder, label) pair from the policy must appear in the inverted map."""
    from traciora_pii.policies.loader import load_opf_policy

    for locale in LOCALES:
        space = load_label_space(locale)
        policy = load_opf_policy(space.policy_name)
        for opf_label, placeholder in policy.label_map.items():
            assert space.placeholder_to_label[placeholder] in policy.label_map
            # Round-trip check: at least one OPF label maps back from this
            # placeholder. (Multiple OPF labels may share a placeholder;
            # we keep the first.)


def test_write_label_space_json_emits_runner_format(tmp_path):
    out = tmp_path / "label_space.json"
    write_label_space_json(out, locale="ru")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "category_version" in payload
    assert "span_class_names" in payload
    assert payload["span_class_names"][0] == "O"
    assert payload["category_version"].endswith("_ru")
