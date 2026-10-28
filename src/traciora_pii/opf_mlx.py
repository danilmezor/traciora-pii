"""OPF second pass with the forward pass on MLX and everything else from opf.

The text is tokenised with the checkpoint's tiktoken encoding and cut into
overlapping windows. The mlx-embeddings privacy-filter port labels the windows,
several at a time. Per-token log-probabilities are averaged over the windows that
saw the token, the way ``opf._core.runtime.predict_text`` aggregates them, and
the result goes through opf's own constrained Viterbi decoder, span conversion,
whitespace trimming and output rendering. ``redact`` returns opf's
``RedactionResult``, so ``OPFPIIDetector.detect`` reads it unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from opf._api import RedactionResult, _redact_text, _warning_for_prediction
from opf._common.constants import SCHEMA_VERSION
from opf._common.label_space import resolve_label_space_from_config
from opf._core.decoding import build_sequence_decoder
from opf._core.runtime import (
    DetectedSpan,
    PredictionResult,
    _apply_output_mode_to_detected_spans,
    _label_placeholder,
    _select_non_overlapping_spans,
    build_detection_summary,
)
from opf._core.sequence_labeling import build_label_info
from opf._core.spans import (
    decode_text_with_offsets,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)

_OUTPUT_MODE = "typed"


def window_starts(n_tokens: int, window: int, overlap: int) -> list[int]:
    """Start offsets of windows of ``window`` tokens advancing by ``window - overlap``.

    The last window ends at ``n_tokens``; a text no longer than one window is one window.
    """
    if not 0 <= overlap < window:
        raise ValueError(f"overlap must be in [0, window), got window={window}, overlap={overlap}")
    if n_tokens <= 0:
        return []
    starts = [0]
    while starts[-1] + window < n_tokens:
        starts.append(starts[-1] + window - overlap)
    return starts


class MLXRuntime:
    """Drop-in for ``opf.OPF`` in typed/viterbi mode with the forward pass on MLX."""

    def __init__(self, mlx_path: Path, opf_checkpoint: Path, *, window: int = 512,
                 overlap: int = 128, batch: int = 8):
        import mlx.core as mx

        from .opf_mlx_convert import load_converted

        window_starts(1, window, overlap)  # validates the window shape
        self._mx = mx
        self.window, self.overlap, self.batch = window, overlap, batch
        checkpoint_config = json.loads((Path(opf_checkpoint) / "config.json").read_text())
        self.encoding = tiktoken.get_encoding(checkpoint_config["encoding"])
        self.pad_token_id = int(self.encoding.eot_token)
        _category, _spans, ner_class_names = resolve_label_space_from_config(
            checkpoint_config, context=str(Path(opf_checkpoint) / "config.json"))
        self.label_info = build_label_info(ner_class_names)
        self.decoder, _biases = build_sequence_decoder(
            decode_mode="viterbi", label_info=self.label_info, viterbi_calibration_path=None,
            checkpoint_dir=str(opf_checkpoint))
        self.model = load_converted(Path(mlx_path))
        if self.model.num_labels != len(ner_class_names):
            raise ValueError(f"Converted checkpoint {mlx_path} has {self.model.num_labels} labels; "
                             f"{opf_checkpoint} has {len(ner_class_names)}. Run prepare_opf again.")

    # ------------------------------------------------------------------

    def _window_log_probs(self, windows: list[list[int]]) -> list[torch.Tensor]:
        """Per-window log-probabilities, ``batch`` windows per forward pass."""
        mx = self._mx
        out: list[torch.Tensor] = []
        for first in range(0, len(windows), self.batch):
            chunk = windows[first:first + self.batch]
            width = max(len(w) for w in chunk)
            ids = np.full((len(chunk), width), self.pad_token_id, dtype=np.int32)
            mask = np.zeros((len(chunk), width), dtype=np.int32)
            for row, tokens in enumerate(chunk):
                ids[row, :len(tokens)] = tokens
                mask[row, :len(tokens)] = 1
            padded = any(len(w) < width for w in chunk)
            logits = self.model(mx.array(ids),
                                attention_mask=mx.array(mask) if padded else None).logits
            logits = logits.astype(mx.float32)
            mx.eval(logits)
            host = torch.from_numpy(np.array(logits))
            for row, tokens in enumerate(chunk):
                out.append(F.log_softmax(host[row, :len(tokens)], dim=-1))
        return out

    def token_log_probs(self, token_ids: list[int]) -> torch.Tensor:
        """Mean probability over the windows that saw each token, as opf aggregates it."""
        starts = window_starts(len(token_ids), self.window, self.overlap)
        windows = [token_ids[s:s + self.window] for s in starts]
        per_window = self._window_log_probs(windows)
        n_labels = per_window[0].shape[-1]
        summed = torch.empty((len(token_ids), n_labels), dtype=torch.float32)
        counts = torch.zeros(len(token_ids), dtype=torch.int64)
        covered = 0
        for start, log_probs in zip(starts, per_window):
            end = start + log_probs.shape[0]
            seen = max(0, min(end, covered) - start)
            if seen:
                summed[start:start + seen] = torch.logaddexp(summed[start:start + seen],
                                                             log_probs[:seen])
            summed[start + seen:end] = log_probs[seen:]
            counts[start:end] += 1
            covered = max(covered, end)
        for count in sorted(set(counts.tolist()) - {1}):
            rows = counts == count
            summed[rows] = summed[rows] - math.log(float(count))
        return summed

    def token_labels(self, token_ids: list[int]) -> list[int]:
        """Viterbi-decoded label index per token.

        opf's predict_text silently swaps in argmax when the decoder's length is
        off; here that decoder fault raises instead of degrading the labels.
        """
        scores = self.token_log_probs(token_ids)
        labels = self.decoder.decode(scores)
        if len(labels) != len(token_ids):
            raise RuntimeError(f"Viterbi decoder returned {len(labels)} labels "
                               f"for {len(token_ids)} tokens")
        return [int(label) for label in labels]

    @torch.inference_mode()
    def _predict(self, text: str) -> PredictionResult:
        token_ids = [int(t) for t in self.encoding.encode(text, allowed_special="all")]
        if not token_ids:
            return PredictionResult(text=text, spans=(), decoded_mismatch=False)
        labels = self.token_labels(token_ids)
        token_spans = labels_to_spans(dict(enumerate(labels)), self.label_info)
        decoded_text, char_starts, char_ends = decode_text_with_offsets(token_ids, self.encoding)
        decoded_mismatch = decoded_text != text
        source_text = decoded_text if decoded_mismatch else text
        char_spans = trim_char_spans_whitespace(
            token_spans_to_char_spans(token_spans, char_starts, char_ends), source_text)
        names = self.label_info.span_class_names
        detected = []
        for label_idx, start, end in char_spans:
            if not 0 <= start < end <= len(source_text):
                continue
            label = (str(names[label_idx]) if 0 <= int(label_idx) < len(names)
                     else f"label_{label_idx}")
            detected.append(DetectedSpan(label=label, start=int(start), end=int(end),
                                         text=source_text[start:end],
                                         placeholder=_label_placeholder(label)))
        spans = _apply_output_mode_to_detected_spans(
            _select_non_overlapping_spans(detected), output_mode=_OUTPUT_MODE)
        return PredictionResult(text=source_text, spans=tuple(spans),
                                decoded_mismatch=decoded_mismatch)

    def redact(self, text: str) -> RedactionResult:
        """Same result as ``opf.OPF(output_mode="typed", decode_mode="viterbi").redact``."""
        prediction = self._predict(text)
        return RedactionResult(
            schema_version=SCHEMA_VERSION,
            summary=build_detection_summary(
                output_mode=_OUTPUT_MODE, labels=[s.label for s in prediction.spans],
                decoded_mismatch=prediction.decoded_mismatch),
            text=prediction.text,
            detected_spans=tuple(prediction.spans),
            redacted_text=_redact_text(prediction.text, prediction.spans),
            warning=_warning_for_prediction(prediction),
        )
