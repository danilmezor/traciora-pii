"""CLI dispatcher for the traciora_pii OPF fine-tuning pipeline.

Subcommands:
  prepare   — convert a bench release into OPF training JSONL
  train     — fine-tune an OPF base checkpoint against the prepared data
  evaluate  — run the detector ablation harness against a checkpoint
  full      — prepare + train + evaluate end-to-end

``--slug`` threads the data directory, the checkpoint directory, and the
results-file suffix together so concurrent runs don't overwrite each other.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


PKG_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PKG_ROOT / "data"
RUNS_ROOT = PKG_ROOT / "runs"
RELEASES_ROOT = PKG_ROOT.parent / "releases"

DEFAULT_RELEASE_VERSION = "v1"
DEFAULT_BENCH_NAME = "traciora_pii-bench"


def _data_dir_for(slug: str) -> Path:
    return DATA_ROOT / slug


def _run_dir_for(slug: str) -> Path:
    return RUNS_ROOT / slug


def _checkpoint_dir_for(slug: str) -> Path:
    return _run_dir_for(slug) / "checkpoint"


def _bench_dir_for(locale: str, *, release_version: str, release_root: Path) -> Path:
    return release_root / release_version / f"{DEFAULT_BENCH_NAME}-{locale}"


def _cmd_prepare(args: argparse.Namespace) -> int:
    from fine_tuning.prepare_data import run_prepare

    bench_dir = (
        Path(args.bench_dir).expanduser()
        if args.bench_dir
        else _bench_dir_for(
            args.locale,
            release_version=args.release_version,
            release_root=Path(args.release_root).expanduser(),
        )
    )
    report = run_prepare(
        locale=args.locale,
        bench_dir=bench_dir,
        out_dir=_data_dir_for(args.slug),
        slug=args.slug,
    )
    print(
        f"prepared {args.locale}/{args.slug}: "
        f"train={report.rows_out['train']} "
        f"validation={report.rows_out['validation']} "
        f"test={report.rows_out['test']} "
        f"-> {report.out_dir}"
    )
    if report.placeholder_dropped_counts:
        top = sorted(report.placeholder_dropped_counts.items(), key=lambda x: -x[1])[:5]
        print(
            "  dropped (not in OPF label_map): "
            + ", ".join(f"{ph}={n}" for ph, n in top)
        )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from fine_tuning.train import run_train

    data_dir = _data_dir_for(args.slug)
    if not data_dir.exists():
        print(
            f"Data directory {data_dir} does not exist. "
            f"Run `python -m fine_tuning prepare --slug {args.slug} --locale <en|de|ru>` first.",
            file=sys.stderr,
        )
        return 1

    run_train(
        data_dir=data_dir,
        output_dir=_checkpoint_dir_for(args.slug),
        base_checkpoint=Path(args.base_checkpoint).expanduser() if args.base_checkpoint else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        low_vram=args.low_vram,
        overwrite_output=args.overwrite_output,
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from fine_tuning.evaluate import run_evaluate

    ckpt = _checkpoint_dir_for(args.slug)
    if not ckpt.exists():
        print(f"Checkpoint directory {ckpt} does not exist.", file=sys.stderr)
        return 1
    run_evaluate(
        checkpoint_dir=ckpt,
        locale=args.locale,
        slug=args.slug,
        runs_dir=_run_dir_for(args.slug),
    )
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    rc = _cmd_prepare(args)
    if rc != 0:
        return rc
    rc = _cmd_train(args)
    if rc != 0:
        return rc
    return _cmd_evaluate(args)


def _add_slug(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slug", required=True,
        help="Identifier shared across data/, runs/, and result files.",
    )


def _add_locale(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--locale", required=True, choices=("en", "de", "ru"),
        help="Locale of the bench release and the OPF policy to use.",
    )


def _add_prepare_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bench-dir", default=None,
        help=f"Override bench directory. Defaults to "
             f"<release-root>/<release-version>/{DEFAULT_BENCH_NAME}-<locale>.",
    )
    parser.add_argument(
        "--release-root", default=str(RELEASES_ROOT),
        help="Root directory containing release versions.",
    )
    parser.add_argument(
        "--release-version", default=DEFAULT_RELEASE_VERSION,
        help="Release version directory name.",
    )


def _add_train_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-checkpoint", default=None,
        help="Path to the base OPF checkpoint. Defaults to OPF's resolver "
             "(OPF_CHECKPOINT env var or ~/.opf/privacy_filter).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Defaults to 4 in normal mode, 1 under --low-vram.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Defaults to 8 under --low-vram, otherwise 1.",
    )
    parser.add_argument(
        "--low-vram", action="store_true",
        help="Route through bitsandbytes.AdamW8bit to fit a 16 GB GPU. "
             "Requires Linux x86_64 + CUDA.",
    )
    parser.add_argument("--overwrite-output", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fine_tuning",
        description="OPF fine-tuning pipeline for traciora_pii (EN / DE / RU).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser("prepare", help="Convert a bench release to OPF training JSONL.")
    _add_slug(p_prepare)
    _add_locale(p_prepare)
    _add_prepare_flags(p_prepare)
    p_prepare.set_defaults(func=_cmd_prepare)

    p_train = subparsers.add_parser("train", help="Fine-tune the base checkpoint.")
    _add_slug(p_train)
    _add_train_flags(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Run the ablation harness against a checkpoint.")
    _add_slug(p_eval)
    _add_locale(p_eval)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_full = subparsers.add_parser("full", help="prepare + train + evaluate end-to-end.")
    _add_slug(p_full)
    _add_locale(p_full)
    _add_prepare_flags(p_full)
    _add_train_flags(p_full)
    p_full.set_defaults(func=_cmd_full)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
