"""Encoder token classifiers as the second pass: ruBERT for Russian, Horizon ModernBERT for en/de.

The text is tokenised once, cut into overlapping 512-token windows, and each token takes its
probabilities from the window where it sits farthest from an edge. One BIO decode over the
stitched sequence gives fragments; fragments below the threshold are dropped, the rest snap
outward to whole words and merge into ``[PERSON_NAME]`` and ``[STREET_ADDRESS]`` findings by
the family's label scheme.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._detector_common import LLMFinding, normalize_person_lemma
from .encoder_checkpoint import cached_encoder_path, validate_encoder_checkpoint

if TYPE_CHECKING:
    import torch

    from .config import EncoderConfig

RUBERT_NAME_LABELS = frozenset({"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"})
RUBERT_ADDRESS_CORE = frozenset({"STREET", "HOUSE"})
RUBERT_ADDRESS_CONTEXT = frozenset({"CITY", "REGION", "DISTRICT", "COUNTRY"})
HORIZON_PLACEHOLDERS = {"PERSON": "[PERSON_NAME]", "STREET_ADDRESS": "[STREET_ADDRESS]"}

# Between two parts of one Russian address: whitespace, punctuation, address keywords
# ("г.", "ул.", "д.", "кв." ...) and the prepositions that introduce a street ("по улице").
_RU_ADDRESS_KEYWORDS = (
    r"(?:г|гор|город|ул|улица|улице|улицу|пр|пр-т|просп|проспект|пер|переулок|ш|шоссе|"
    r"б-р|бульвар|пл|площадь|наб|набережная|д|дом|корп|к|стр|строение|кв|квартира|"
    r"оф|офис|обл|область|р-н|район|пос|п|с|дер|мкр|респ|край|ао|на|по|в|во)"
)
_RU_ADDRESS_GAP = re.compile(
    rf"^[\s,./\-–№]*(?:{_RU_ADDRESS_KEYWORDS}[\s,./\-–№]+)*$", re.IGNORECASE)


class EncoderInferenceError(RuntimeError):
    """The encoder could not run on a text; the second pass has no answer."""


@dataclass(frozen=True)
class Fragment:
    """One decoded entity: the model label without ``B-``/``I-``, character offsets, confidence."""
    label: str
    start: int
    end: int
    score: float


def _short_gap(gap: str) -> bool:
    return len(gap) <= 3 and not any(c.isalpha() for c in gap)


def _ru_address_gap(gap: str) -> bool:
    return len(gap) <= 20 and bool(_RU_ADDRESS_GAP.match(gap))


def group_fragments(text: str, fragments: list[Fragment], family: str) -> list[tuple[str, int, int]]:
    """Merge fragments of one placeholder into ``(placeholder, start, end)`` spans.

    Names (both families) and Horizon addresses merge across at most 3 characters with no
    letter; ruBERT addresses merge across whitespace, punctuation and Russian address keywords
    and are kept only when a street or house is inside.
    """
    if family == "rubert":
        typed = [("[PERSON_NAME]" if f.label in RUBERT_NAME_LABELS else "[STREET_ADDRESS]", f)
                 for f in fragments
                 if f.label in RUBERT_NAME_LABELS | RUBERT_ADDRESS_CORE | RUBERT_ADDRESS_CONTEXT]
        gap_ok = {"[PERSON_NAME]": _short_gap, "[STREET_ADDRESS]": _ru_address_gap}
    elif family == "horizon":
        typed = [(HORIZON_PLACEHOLDERS[f.label], f) for f in fragments if f.label in HORIZON_PLACEHOLDERS]
        gap_ok = {"[PERSON_NAME]": _short_gap, "[STREET_ADDRESS]": _short_gap}
    else:
        raise ValueError(f"Unknown encoder family {family!r}; expected rubert or horizon")

    merged: list[list] = []
    for placeholder, f in sorted(typed, key=lambda item: (item[1].start, item[1].end)):
        last = merged[-1] if merged else None
        if last and last[0] == placeholder and (f.start < last[2] or gap_ok[placeholder](text[last[2]:f.start])):
            last[2] = max(last[2], f.end)
            last[3].add(f.label)
        else:
            merged.append([placeholder, f.start, f.end, {f.label}])

    spans = []
    for placeholder, start, end, labels in merged:
        if family == "rubert" and placeholder == "[STREET_ADDRESS]" and not labels & RUBERT_ADDRESS_CORE:
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((placeholder, start, end))
    return spans


def window_bounds(n_tokens: int, window: int = 512, overlap: int = 128) -> list[tuple[int, int]]:
    """Content-token ``(start, end)`` ranges; each window adds two special tokens to ``window - 2``."""
    if n_tokens == 0:
        return []
    body = window - 2
    starts = [0]
    while starts[-1] + body < n_tokens:
        starts.append(starts[-1] + body - overlap)
    return [(start, min(start + body, n_tokens)) for start in starts]


def _alnum_joint(text: str, left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Two adjacent pieces of one word whose joint is a letter or digit on both sides."""
    left_text = text[left[0]:left[1]].strip()
    right_text = text[right[0]:right[1]].strip()
    return (left[1] == right[0] and bool(left_text) and bool(right_text)
            and left_text[-1].isalnum() and right_text[0].isalnum())


class EncoderPIIDetector:
    """A token-classification encoder behind the ``OPFPIIDetector`` contract."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        family: str,
        device: str = "cpu",
        threshold: float = 0.5,
        threads: int | None = None,
        batch_windows: int = 8,
    ):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "EncoderPIIDetector requires torch and transformers. "
                "Install with: pip install -e '.[encoder]'"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda but this torch build sees no CUDA device")
        if threads is not None:
            torch.set_num_threads(threads)

        self.family = family
        self.device = device
        self.threshold = threshold
        self.batch_windows = batch_windows
        self.prefilter_enabled = False
        self.lm_calls = 0
        self.prefilter_skips = 0
        self.hallucinated_drops = 0
        self.parse_failures = 0

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint, dtype=torch.float32)
        self._model = (model.to("cuda").half() if device == "cuda" else model).eval()
        self._labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        tok = self._tokenizer
        self._open_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        self._close_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id

    @classmethod
    def from_config(cls, encoder_cfg: "EncoderConfig", *, model_name: str | None = None) -> "EncoderPIIDetector":
        """Open the cached, validated checkpoint of ``model_name`` (default ``default_model``)."""
        chosen = model_name or encoder_cfg.default_model
        if chosen not in encoder_cfg.models:
            available = ", ".join(sorted(encoder_cfg.models)) or "(none registered)"
            raise KeyError(
                f"Encoder model {chosen!r} is not registered in detector.encoder.models. "
                f"Available: {available}. Add an entry to traciora.yaml or pick one of the "
                f"existing keys.",
            )
        entry = encoder_cfg.models[chosen]
        path = cached_encoder_path(entry, encoder_cfg.cache_dir)
        validate_encoder_checkpoint(path, entry)
        return cls(checkpoint=path, family=entry.family, device=encoder_cfg.device,
                   threshold=encoder_cfg.threshold, threads=encoder_cfg.threads)

    def detect(self, text: str) -> list[LLMFinding]:
        """Return ``[PERSON_NAME]`` / ``[STREET_ADDRESS]`` spans; raise when the model cannot run."""
        if not text:
            return []
        self.lm_calls += 1
        try:
            fragments = self._fragments(text)
        except Exception as exc:
            raise EncoderInferenceError(
                f"Encoder inference failed on device={self.device}: {type(exc).__name__}: {exc}",
            ) from exc
        findings = []
        kept = [f for f in fragments if f.score >= self.threshold]
        for placeholder, start, end in group_fragments(text, kept, self.family):
            span_text = text[start:end]
            if placeholder == "[PERSON_NAME]":
                lemma = normalize_person_lemma(span_text, span_text.lower())
            else:
                lemma = " ".join(span_text.split()).lower()
            findings.append(LLMFinding(detector="encoder", start=start, end=end,
                                       placeholder=placeholder, lemma=lemma))
        return findings

    def close(self) -> None:
        self._model = None

    def _window_probs(self, ids: list[int], bounds: list[tuple[int, int]]) -> list["torch.Tensor"]:
        """Label probabilities per window over its content tokens, in batches padded to the longest."""
        torch = self._torch
        probs = []
        for first in range(0, len(bounds), self.batch_windows):
            batch = bounds[first:first + self.batch_windows]
            width = max(end - start for start, end in batch) + 2
            input_ids = torch.full((len(batch), width), self._tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            for row, (start, end) in enumerate(batch):
                input_ids[row, :end - start + 2] = torch.tensor([self._open_id, *ids[start:end], self._close_id])
                attention[row, :end - start + 2] = 1
            inputs = {"input_ids": input_ids.to(self.device), "attention_mask": attention.to(self.device)}
            if self.family == "rubert":
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
            with torch.inference_mode():
                logits = self._model(**inputs).logits.float().cpu()
            for row, (start, end) in enumerate(batch):
                probs.append(logits[row, 1:end - start + 1].softmax(-1))
        return probs

    def _fragments(self, text: str) -> list[Fragment]:
        torch = self._torch
        encoding = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        word_ids = encoding.word_ids()
        n = len(ids)
        bounds = window_bounds(n)
        if not bounds:
            return []

        stitched = torch.zeros((n, len(self._labels)))
        best = torch.full((n,), -1.0)
        for (start, end), window in zip(bounds, self._window_probs(ids, bounds)):
            positions = torch.arange(start, end, dtype=torch.float32)
            from_left = positions - start if start > 0 else torch.full_like(positions, float("inf"))
            from_right = (end - 1) - positions if end < n else torch.full_like(positions, float("inf"))
            distance = torch.minimum(from_left, from_right)
            take = distance > best[start:end]
            stitched[start:end][take] = window[take]
            best[start:end][take] = distance[take]
        confidence, label_ids = stitched.max(-1)
        tags = [self._labels[i] for i in label_ids.tolist()]
        confidence = confidence.tolist()

        spans: list[list] = []
        for i, tag in enumerate(tags):
            if tag == "O":
                continue
            bio, label = tag.split("-", 1)
            if bio == "I" and spans and spans[-1][0] == label and spans[-1][2] == i - 1:
                spans[-1][2] = i
            else:
                spans.append([label, i, i])

        fragments = []
        for label, first, last in spans:
            score = sum(confidence[first:last + 1]) / (last - first + 1)
            while first > 0 and word_ids[first - 1] == word_ids[first] and _alnum_joint(
                    text, offsets[first - 1], offsets[first]):
                first -= 1
            while last + 1 < n and word_ids[last + 1] == word_ids[last] and _alnum_joint(
                    text, offsets[last], offsets[last + 1]):
                last += 1
            fragments.append(Fragment(label, offsets[first][0], offsets[last][1], score))
        return fragments
