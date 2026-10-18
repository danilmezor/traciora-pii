"""Training wrapper — delegates to ``opf._train.runner.main``.

Default mode is a straight pass-through to OPF's upstream training loop
(AdamW, bf16, whatever OPF ships). ``--low-vram`` monkey-patches the
optimizer to ``bitsandbytes.optim.AdamW8bit`` before invoking the
runner, which drops the Adam state from ~12 GB (fp32 m+v) to ~3 GB —
fits comfortably in a 16 GB card (5060 Ti / 4060 Ti / 4080) after
accounting for weights, gradients, and activations.

We do NOT vendor the training code. Every fine-tune uses upstream
``opf train`` semantics; the contribution is the data pipeline, the
optimizer swap, and the post-training evaluation harness.
"""
from __future__ import annotations

import contextlib
import logging
import platform
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _is_linux_x86_64() -> bool:
    return sys.platform == "linux" and platform.machine() in {"x86_64", "amd64"}


@contextlib.contextmanager
def _adamw8bit_monkey_patch():
    """Swap ``torch.optim.AdamW`` for ``bitsandbytes.optim.AdamW8bit``.

    Process-wide patch — don't run two trainers concurrently with
    conflicting optimizer configurations. Restored on exit even on raise.
    """
    if not _is_linux_x86_64():
        raise RuntimeError(
            "--low-vram requires Linux x86_64 with CUDA. "
            "bitsandbytes does not support macOS; "
            "run the fine-tune on a Linux host or rent a GPU instance."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "--low-vram requires torch. Install the opf-finetune extra: "
            "pip install -e '.[opf-finetune]'"
        ) from exc
    try:
        import bitsandbytes as bnb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "--low-vram requires bitsandbytes. Install the opf-finetune extra: "
            "pip install -e '.[opf-finetune]'"
        ) from exc

    original = torch.optim.AdamW
    torch.optim.AdamW = bnb.optim.AdamW8bit  # type: ignore[assignment]
    logger.warning(
        "[--low-vram] monkey-patched torch.optim.AdamW -> bitsandbytes.optim.AdamW8bit"
    )
    try:
        yield
    finally:
        torch.optim.AdamW = original  # type: ignore[assignment]


def _build_runner_argv(
    *,
    train_data: Path,
    validation_data: Optional[Path],
    output_dir: Path,
    base_checkpoint: Optional[Path],
    label_space_json: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_accum_steps: int,
    overwrite_output: bool,
) -> List[str]:
    argv: List[str] = [str(train_data)]
    if validation_data is not None:
        argv += ["--validation-dataset", str(validation_data)]
    argv += ["--output-dir", str(output_dir)]
    argv += ["--label-space-json", str(label_space_json)]
    argv += ["--epochs", str(epochs)]
    argv += ["--batch-size", str(batch_size)]
    argv += ["--learning-rate", str(learning_rate)]
    if grad_accum_steps > 1:
        argv += ["--grad-accum-steps", str(grad_accum_steps)]
    if base_checkpoint is not None:
        argv += ["--checkpoint", str(base_checkpoint)]
    if overwrite_output:
        argv += ["--overwrite-output"]
    return argv


def run_train(
    *,
    data_dir: Path,
    output_dir: Path,
    base_checkpoint: Optional[Path] = None,
    epochs: int = 3,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-5,
    grad_accum_steps: Optional[int] = None,
    low_vram: bool = False,
    overwrite_output: bool = False,
) -> Path:
    """Run the OPF fine-tune. Returns the output checkpoint directory.

    ``data_dir`` must contain ``train.jsonl``, ``validation.jsonl``, and
    ``label_space.json`` — the artifacts produced by ``prepare_data.run_prepare``.
    """
    train_data = data_dir / "train.jsonl"
    validation_data = data_dir / "validation.jsonl"
    label_space_json = data_dir / "label_space.json"
    for p in (train_data, validation_data, label_space_json):
        if not p.exists():
            raise FileNotFoundError(f"Missing prepared artifact: {p}")

    effective_batch = batch_size
    if effective_batch is None:
        effective_batch = 1 if low_vram else 4
    effective_grad_accum = grad_accum_steps
    if effective_grad_accum is None:
        effective_grad_accum = 8 if low_vram else 1

    argv = _build_runner_argv(
        train_data=train_data,
        validation_data=validation_data,
        output_dir=output_dir,
        base_checkpoint=base_checkpoint,
        label_space_json=label_space_json,
        epochs=epochs,
        batch_size=effective_batch,
        learning_rate=learning_rate,
        grad_accum_steps=effective_grad_accum,
        overwrite_output=overwrite_output,
    )

    try:
        from opf._train.runner import main as opf_train_main  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OPF package not available. Install with: pip install -e '.[opf-finetune]'"
        ) from exc

    logger.info("Invoking opf._train.runner with argv=%s", argv)

    if low_vram:
        with _adamw8bit_monkey_patch():
            rc = opf_train_main(argv)
    else:
        rc = opf_train_main(argv)

    if rc != 0:
        raise RuntimeError(f"opf._train.runner.main returned non-zero exit code {rc}")

    return output_dir
