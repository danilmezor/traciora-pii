"""End-to-end deterministic dataset pipeline.

The pipeline is a thin orchestration around three stages:

    SeedBundle  → SlotFillEngine.render()  → optional Noise.apply()  → GeneratedExample

There is no scaffold source, scrubber, or filter step in the slot-template
architecture: every template is hand-curated to host its PII naturally and
there is no real-text scaffold that could carry pre-existing entities.

Reproducibility: every randomness flows from ``seed`` (the int) through
``random.Random`` instances re-seeded via fixed XOR offsets so two runs
with the same seed produce byte-identical output.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

from .injectors import SlotFillEngine, Span
from .seeds import SeedBundle


# ======================================================================
# Output record
# ======================================================================


@dataclass(frozen=True)
class GeneratedExample:
    """One generated benchmark example.

    ``id`` is stable across runs given the same ``(seed, locale, sequence_index)``
    so reruns produce diff-friendly JSONL. ``kind`` is ``"positive"`` for
    slot-filled examples and ``"negative"`` for hard-negative samples
    (zero-span text drawn from ``seeds/<locale>/negatives.yaml``).
    """

    id: str
    locale: str
    kind: str                # "positive" | "negative"
    category: str            # template category for positives; "hard_negative" for negatives
    template_text: str       # source template (positives); the raw negative line (negatives)
    text: str                # final rendered text after fillers + noise
    spans: Tuple[Span, ...]
    fillers_prepended: int = 0
    fillers_appended: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "locale": self.locale,
            "kind": self.kind,
            "category": self.category,
            "template": self.template_text,
            "text": self.text,
            "spans": [
                {"placeholder": s.placeholder, "value": s.value,
                 "start": s.start, "end": s.end}
                for s in self.spans
            ],
            "fillers_prepended": self.fillers_prepended,
            "fillers_appended": self.fillers_appended,
        }


# ======================================================================
# Pipeline
# ======================================================================


@dataclass
class Pipeline:
    """Slot-template dataset pipeline. Deterministic from ``seed`` alone.

    Per example, the pipeline:
      1. Decides positive vs negative by ``seeds.negatives.probability``.
      2. For positives: samples a category and template, fills slots,
         randomly prepends and/or appends a same-category filler, applies
         noise (if configured), and emits.
      3. For negatives: samples a hard-negative line from ``seeds.negatives``,
         emits with empty span list. Fillers and noise are not applied to
         negatives (the negative line is a complete, hand-curated example).
    """

    seeds: SeedBundle
    seed: int = 0
    noise: Optional[object] = None      # NoiseApplier; typed Object to avoid hard import

    rng: random.Random = field(init=False)
    engine: SlotFillEngine = field(init=False)
    _filler_rng: random.Random = field(init=False)
    _negative_rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        # Distinct RNG streams per stage so they don't phase-align.
        self.rng = random.Random(self.seed)
        engine_rng = random.Random(self.seed ^ 0xA1_B2_C3)
        self.engine = SlotFillEngine(seeds=self.seeds, rng=engine_rng)
        self._filler_rng = random.Random(self.seed ^ 0x77_BB_99)
        self._negative_rng = random.Random(self.seed ^ 0x12_34_56)
        if self.noise is not None:
            self.noise.rng = random.Random(self.seed ^ 0xF6_07_18)  # type: ignore[attr-defined]

    def run(self, n: int) -> Iterator[GeneratedExample]:
        """Yield ``n`` deterministic examples."""
        neg_p = self.seeds.negatives.probability
        all_negatives = self.seeds.negatives.all_examples()

        for i in range(n):
            example_id = f"{self.seeds.locale}-gen-{i:05d}"

            # Decide kind. If negatives are configured AND the roll lands,
            # emit a negative; otherwise emit a positive.
            if all_negatives and self._negative_rng.random() < neg_p:
                line = self._negative_rng.choice(all_negatives)
                yield GeneratedExample(
                    id=example_id,
                    locale=self.seeds.locale,
                    kind="negative",
                    category="hard_negative",
                    template_text=line,
                    text=line,
                    spans=(),
                )
                continue

            cat_name, cat = self.engine.sample_category()
            template = self.engine.sample_template(cat)
            text, spans = self.engine.render(template)

            text, spans, n_pre, n_app = self._apply_fillers(text, spans, cat_name)

            if self.noise is not None:
                text, spans = self.noise.apply(text, spans)  # type: ignore[attr-defined]

            yield GeneratedExample(
                id=example_id,
                locale=self.seeds.locale,
                kind="positive",
                category=cat_name,
                template_text=template.text,
                text=text,
                spans=tuple(spans),
                fillers_prepended=n_pre,
                fillers_appended=n_app,
            )

    # ------------------------------------------------------------------

    def _apply_fillers(
        self, text: str, spans: List[Span], category: str,
    ) -> Tuple[str, List[Span], int, int]:
        """Maybe prepend / append a filler from the same category.

        Independent rolls per side per the seed-config probabilities. Spans
        are shifted forward by the prepended-text length so the invariant
        ``text[start:end] == value`` holds throughout.
        """
        fc = self.seeds.fillers
        category_fillers = fc.for_category(category)
        if not category_fillers:
            return text, spans, 0, 0

        n_pre = 0
        n_app = 0

        # Prepend.
        if self._filler_rng.random() < fc.prepend_probability:
            filler = self._filler_rng.choice(category_fillers)
            sep = " "
            shift = len(filler) + len(sep)
            text = filler + sep + text
            spans = [
                Span(placeholder=s.placeholder, value=s.value,
                     start=s.start + shift, end=s.end + shift)
                for s in spans
            ]
            n_pre = 1

        # Append.
        if self._filler_rng.random() < fc.append_probability:
            filler = self._filler_rng.choice(category_fillers)
            text = text + " " + filler
            n_app = 1

        return text, spans, n_pre, n_app


# ======================================================================
# EvalQuery adapter and JSONL writer
# ======================================================================


def to_eval_query(example: GeneratedExample, *, bucket: Optional[str] = None):
    """Adapt to ``benchmarks.pii_evaluation.dataset.EvalQuery`` for harness reuse."""
    from benchmarks.pii_evaluation.dataset import EvalQuery, PIIGroundTruth
    spans = tuple(
        PIIGroundTruth(placeholder=s.placeholder, value=s.value, turn=0)
        for s in example.spans
    )
    return EvalQuery(
        id=example.id,
        language=example.locale,
        bucket=bucket or f"{example.locale}_{example.category}",
        turns=(example.text,),
        pii_spans=spans,
    )


def write_jsonl(examples: Iterator[GeneratedExample], path: str) -> int:
    """Serialize examples to JSONL. Returns count written."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def split_examples(
    examples: List[GeneratedExample],
    *,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Tuple[List[GeneratedExample], List[GeneratedExample], List[GeneratedExample]]:
    """Deterministic train/dev/test split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n = len(examples)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    train_idx = sorted(indices[:n_train])
    dev_idx = sorted(indices[n_train:n_train + n_dev])
    test_idx = sorted(indices[n_train + n_dev:])
    return (
        [examples[i] for i in train_idx],
        [examples[i] for i in dev_idx],
        [examples[i] for i in test_idx],
    )
