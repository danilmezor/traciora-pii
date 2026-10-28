"""OpenAI Privacy Filter second-pass PII detector.

Sibling of ``LLMPIIDetector`` that swaps the generative-LM backend for
OpenAI's bidirectional token-classification model (``openai/privacy-filter``,
Apache 2.0). The classifier emits BIOES-tagged spans over the input in a
single forward pass plus a constrained Viterbi decode, rather than
decoding a JSON string from a chat completion. That changes three things
compared with the original detector:

  * No JSON parse failures — ``DetectedSpan`` tuples are structured output.
  * No hallucinated spans — every span is a verbatim contiguous slice of
    the input tokens by construction; the ``hallucinated_drops`` counter
    is kept for interface parity but should always read zero.
  * Label set is OPF's native taxonomy. We consume only ``private_person``
    and ``private_address`` here — the other six labels overlap with the
    regex layer, which has already tokenized those values before this
    detector runs.

Everything else — the pymorphy3 lemma pipeline, the cheap regex
prefilter, the ``LLMFinding`` output contract, the detector counter
interface — is shared with ``LLMPIIDetector`` via internal helpers
``normalize_person_lemma`` and ``prefilter``. The harness
treats the two detectors polymorphically.

Model selection: one fine-tune per language plus the base classifier,
registered in ``traciora.yaml`` under ``detector.opf.models`` and
selected by name (``base`` / ``en`` / ``de`` / ``ru`` / ...). See
:meth:`OPFPIIDetector.from_config`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ._detector_common import (
    LLMFinding,
    normalize_person_lemma,
    prefilter,
)
import logging

if TYPE_CHECKING:
    from .config import OPFConfig

logger = logging.getLogger(__name__)


class OPFInferenceError(RuntimeError):
    """The OPF model could not run on a text; the second pass has no answer."""


def _resolve_opf_checkpoint(checkpoint: str | None, *, cache_dir: str | None = None) -> str | None:
    """Resolve a checkpoint string to a local path.

    The upstream ``opf`` library treats ``model=`` as a local checkpoint
    directory. To allow registry entries to use HF repo ids (the natural
    way to ship fine-tunes alongside the base classifier), accept
    ``org/name`` strings here and pull them on first use into a slim
    cache. ``cache_dir`` (typically ``OPFConfig.cache_dir``) overrides the
    default ``~/.cache/traciora_pii/models/opf/`` root. Already-resolved local
    paths and ``None`` (use OPF default) pass through unchanged.

    "Slim" matters: ``openai/privacy-filter`` ships five ONNX variants
    (~9 GB) on top of its safetensors — and the upstream library's own
    default-checkpoint download uses ``allow_patterns=["original/*"]``
    to skip them (~3 GB final on disk). We mirror that for the base
    layout, and detect fine-tunes (no ``original/`` subfolder) so they
    download in their native layout without ONNX bloat.
    """
    if checkpoint is None:
        return None
    if os.path.exists(checkpoint):
        return checkpoint
    if "/" in checkpoint and not checkpoint.startswith((".", "/", "~")):
        return _ensure_opf_slim_checkpoint(checkpoint, cache_dir=cache_dir)
    return checkpoint


def _ensure_opf_slim_checkpoint(repo_id: str, *, cache_dir: str | None = None) -> str:
    """Download an OPF HF repo's slim canonical variant.

    Two layouts are recognized:

      * **Base OPF layout** (``original/`` subfolder present): pulls
        ``original/*`` only and flattens that subtree to the top level
        of the local dir. Mirrors
        ``opf._common.checkpoint_download.ensure_default_checkpoint``.
      * **Fine-tune layout** (no ``original/`` subfolder): pulls every
        file *except* the ``onnx/`` subtree. Fine-tunes don't ship ONNX
        variants, so this is usually a no-op safety filter.

    The resolved local dir is reused on subsequent calls via a sentinel
    file. The HF hub's blob-level cache still applies underneath, so a
    second registry entry pointing at the same repo doesn't re-fetch.
    """

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve OPF checkpoint {repo_id!r}: "
            "huggingface_hub is required to fetch HF-hosted fine-tunes. "
            "Install with: pip install huggingface_hub",
        ) from exc

    # Precedence: explicit cache_dir arg (OPFConfig.cache_dir) > legacy env
    # var > built-in default. The env-var path is kept for backwards-compat
    # with shells that already export it; new code should use the config.
    cache_root_env = os.getenv("PII_OPF_CACHE_DIR", "").strip()
    if cache_dir:
        cache_root = os.path.expanduser(cache_dir)
    elif cache_root_env:
        cache_root = os.path.expanduser(cache_root_env)
    else:
        cache_root = os.path.expanduser("~/.cache/traciora_pii/models/opf")
    os.makedirs(cache_root, exist_ok=True)
    target = os.path.join(cache_root, repo_id.replace("/", "--"))
    sentinel = os.path.join(target, ".traciora_pii-opf-ready")
    if os.path.exists(sentinel):
        return target

    # Decide the layout up front by listing the repo. One round-trip,
    # cheap; avoids downloading the wrong slice and re-downloading.
    try:
        repo_files = HfApi().list_repo_files(repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to list files for OPF repo {repo_id!r}: {exc}",
        ) from exc

    has_original = any(f.startswith("original/") for f in repo_files)

    if has_original:
        # Base layout — slim pull + promote original/* to top level.
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            allow_patterns=["original/*"],
        )
        _promote_original_subtree(target)
    else:
        # Fine-tune layout — already slim, defensively skip ONNX.
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data*"],
        )

    # Atomic-ish ready signal: only set after a complete pull so an
    # interrupted download forces a re-pull on the next run.
    Path(sentinel).touch()
    return target


def _promote_original_subtree(target: str) -> None:
    """Flatten ``<target>/original/<files>`` → ``<target>/<files>``.

    Mirrors ``opf._common.checkpoint_download._promote_original_subtree``
    so the upstream ``OPF(model=target)`` constructor finds files where
    it expects them (top level of the checkpoint dir).
    """
    import shutil

    original_dir = os.path.join(target, "original")
    if not os.path.isdir(original_dir):
        return
    for entry in sorted(os.listdir(original_dir)):
        src = os.path.join(original_dir, entry)
        dst = os.path.join(target, entry)
        if os.path.exists(dst):
            # Don't clobber a pre-existing top-level file; leave the
            # one in original/ in place for human inspection.
            continue
        shutil.move(src, dst)
    try:
        os.rmdir(original_dir)
    except OSError:
        pass  # Non-empty (e.g. file we left behind) — fine to leave.


class OPFPIIDetector:
    """OPF model wrapped as a drop-in second-pass detector.

    The ``label_map`` is supplied directly — a dict of ``opf_model_label
    -> [PLACEHOLDER]``. Use :meth:`from_config` to build one driven by the
    ``OPFPolicy`` named in the active ``OPFConfig`` model entry.
    """

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        device: str = "cpu",
        mlx_bits: int | None = None,
        output_mode: str = "typed",
        decode_mode: str = "viterbi",
        prefilter_enabled: bool = False,
        cache_dir: str | None = None,
        label_map: dict[str, str],
    ):
        # Deferred import: the `opf` package is installed only under the
        # `opf` optional extra. The production runtime must not
        # require it, so failure to import here is a clear message to
        # install the extra rather than a silent fallback.
        try:
            from opf import OPF
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "OPFPIIDetector requires the `opf` package. "
                "Install with: pip install -e '.[opf]'"
            ) from exc

        self.checkpoint = _resolve_opf_checkpoint(checkpoint, cache_dir=cache_dir)
        self.device = device
        self.output_mode = output_mode
        self.decode_mode = decode_mode
        self.prefilter_enabled = prefilter_enabled
        self._label_to_placeholder = dict(label_map)

        # Parity with LLMPIIDetector for the metrics harness — every
        # field is populated even when the underlying backend has no
        # structural way to fail that category (parse/hallucinate).
        self.lm_calls = 0
        self.prefilter_skips = 0
        self.hallucinated_drops = 0
        self.parse_failures = 0

        if self.device == "mlx":
            self._opf = _open_mlx_runtime(self.checkpoint, mlx_bits)
            return

        # opf picks its Triton MoE kernels on any non-CPU device, and macOS has
        # no Triton: on MPS every forward pass would fail. The torch-ops path
        # runs there, so it is the default unless the caller chose explicitly.
        if self.device == "mps":
            os.environ.setdefault("OPF_MOE_TRITON", "0")
        moe_triton = os.environ.get("OPF_MOE_TRITON")
        use_triton = (moe_triton.strip().lower() in {"1", "true", "yes", "on"}
                      if moe_triton is not None else self.device != "cpu")
        logger.info("OPF MoE path on device=%s: %s", self.device,
                    "triton" if use_triton else "torch-ops")
        self._opf = OPF(
            model=self.checkpoint,
            device=self.device,
            output_mode=self.output_mode,
            decode_mode=self.decode_mode,
        )

    @classmethod
    def from_config(
        cls,
        opf_cfg: "OPFConfig",
        *,
        model_name: str | None = None,
    ) -> "OPFPIIDetector":
        """Build the detector from a ``PIIConfig.detector.opf`` section.

        ``model_name`` selects which entry in ``opf_cfg.models`` to load.
        When omitted, ``opf_cfg.default_model`` is used. Selecting a name
        that is not in the registry raises ``KeyError`` with the list of
        available keys so the failure mode is loud, not silent.

        The label map for the OPF inference is pulled from the policy
        named by the registry entry (e.g. ``ru_comprehensive``); the
        policy YAML lives under ``traciora_pii/policies/opf/``.
        """
        from .policies.loader import load_opf_policy

        chosen = model_name or opf_cfg.default_model
        if chosen not in opf_cfg.models:
            available = ", ".join(sorted(opf_cfg.models)) or "(none registered)"
            raise KeyError(
                f"OPF model {chosen!r} is not registered in detector.opf.models. "
                f"Available: {available}. Add an entry to traciora.yaml or pick "
                f"one of the existing keys.",
            )
        entry = opf_cfg.models[chosen]
        opf_policy = load_opf_policy(entry.policy)
        return cls(
            checkpoint=entry.checkpoint,
            device=opf_cfg.device,
            mlx_bits=opf_cfg.mlx_bits,
            prefilter_enabled=opf_cfg.prefilter,
            cache_dir=opf_cfg.cache_dir,
            label_map=opf_policy.label_map,
        )

    # ------------------------------------------------------------------
    # Public API — matches LLMPIIDetector.detect()
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[LLMFinding]:
        """Return ``[PERSON_NAME]`` / ``[STREET_ADDRESS]`` spans in *text*.

        Empty list is a valid and frequent answer. All other OPF labels
        are dropped: the regex layer has already tokenized those types.
        Raises :class:`OPFInferenceError` when the model cannot run, so a
        failed pass is never mistaken for a text without names.
        """
        if not text:
            return []

        if self.prefilter_enabled and not prefilter(text):
            self.prefilter_skips += 1
            return []

        self.lm_calls += 1
        try:
            result = self._opf.redact(text)
        except Exception as exc:
            raise OPFInferenceError(
                f"OPF inference failed on device={self.device}: {type(exc).__name__}: {exc}",
            ) from exc

        # ``OPF.redact`` returns a ``RedactionResult`` when
        # ``output_text_only`` is False (our default). In the text-only
        # mode it returns a bare str — we never enable that.
        detected_spans = getattr(result, "detected_spans", None)
        if detected_spans is None:
            self.parse_failures += 1
            return []

        out: list[LLMFinding] = []
        for span in detected_spans:
            placeholder = self._label_to_placeholder.get(span.label)
            if placeholder is None:
                # Drop labels outside the selected policy. The
                # comprehensive policy intentionally excludes native
                # private_date and private_url.
                continue

            span_text = text[span.start:span.end]
            if not span_text:
                continue

            # OPF guarantees spans are contiguous slices of the input
            # tokens, but a whitespace-trim edge case could still put the
            # span offset outside the string. Defense in depth.
            if text.find(span_text, span.start) != span.start:
                self.hallucinated_drops += 1
                continue

            # OPF does not produce a lemma field — feed the span text
            # itself as the LM-emitted lemma, letting the shared
            # normalizer derive the canonical form.
            if placeholder == "[PERSON_NAME]":
                lemma = normalize_person_lemma(span_text, span_text.lower())
            elif placeholder == "[STREET_ADDRESS]":
                lemma = _address_lemma(span_text)
            else:
                lemma = " ".join(span_text.split()).lower()
            out.append(
                LLMFinding(
                    detector="opf",
                    start=span.start,
                    end=span.end,
                    placeholder=placeholder,
                    lemma=lemma,
                )
            )
        return out

    def close(self) -> None:
        """Release the cached runtime + decoder."""
        # OPF caches its runtime internally; dropping our reference is
        # enough for GC to reclaim it on the next allocation.
        self._opf = None


def _open_mlx_runtime(checkpoint: str | None, bits: int | None):
    """The MLX forward pass over the converted copy of ``checkpoint``; never a CPU fallback."""
    import platform
    import sys

    hint = ("device=mlx needs macOS on Apple silicon and the `mlx` extra. "
            "Install with: pip install -e '.[mlx]'")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ImportError(f"{hint} (this host is {sys.platform}/{platform.machine()})")
    try:
        import mlx.core  # noqa: F401
        import mlx_embeddings.models.openai_privacy_filter  # noqa: F401
    except ImportError as exc:
        raise ImportError(hint) from exc
    from .opf_mlx_convert import converted_path, is_current

    if checkpoint is None:
        raise ValueError("device=mlx needs an explicit OPF checkpoint; run prepare_second_pass first")
    source = Path(checkpoint)
    converted = converted_path(source, bits)
    if not is_current(source, converted, bits):
        raise ValueError(f"No current MLX conversion of {source} at {converted}; "
                         "run prepare_second_pass(config) before startup")
    from .opf_mlx import MLXRuntime

    return MLXRuntime(converted, source)


def _address_lemma(span_text: str) -> str:
    """Canonical form for a street-address span without LM-emitted lemma.

    The generative detector returns a structured ``city|street|building|unit``
    lemma that the vault uses directly. OPF emits only a span, so we fall
    back to a whitespace-normalized lowercase form. This is weaker
    cross-turn — an address worded two different ways will not collapse
    — but names are the dominant multi-turn case anyway; addresses
    rarely recur across turns with rephrasing.
    """
    return " ".join(span_text.split()).lower()
