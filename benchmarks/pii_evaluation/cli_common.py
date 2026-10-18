"""Shared argparse helpers for the eval-harness entry points.

Three CLIs (``run_experiment``, ``run_detector_ablation``,
``run_imported_dataset_eval``) duplicate the same ``--language``,
``--bucket``, ``--detector-llm-model``, ``--detector-llm-base-url``
flags and the same filesystem-safe model-slug derivation. Sharing here
keeps a flag rename to one place.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict


def add_language_bucket_flags(parser: argparse.ArgumentParser) -> None:
    """Add ``--language`` and ``--bucket`` to *parser*."""
    parser.add_argument(
        "--language",
        choices=["en", "de", "ru", "all"],
        default="all",
        help="Restrict the run to one language (default: all three).",
    )
    parser.add_argument(
        "--bucket",
        default="",
        help="Restrict the run to a single bucket (single_turn, multi_turn, ...).",
    )


def add_detector_llm_flags(parser: argparse.ArgumentParser) -> None:
    """Add ``--detector-llm-model`` / ``--detector-llm-base-url`` to *parser*."""
    parser.add_argument(
        "--detector-llm-model",
        default=None,
        help=(
            "Override detector.llm.model. Required when the regex_llm second "
            "pass is active because the YAML has no default. Use OpenRouter "
            "prefix-routed names (e.g. nvidia/nemotron-nano-9b-v2:free, "
            "meta-llama/llama-3.2-3b-instruct:free)."
        ),
    )
    parser.add_argument(
        "--detector-llm-base-url",
        default=None,
        help=(
            "Override detector.llm.base_url. Default: "
            "https://openrouter.ai/api/v1. Point at LM Studio / Ollama / "
            "vLLM for a fully local detector."
        ),
    )


def detector_llm_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect ``{model, base_url}`` overrides from a parsed argparse namespace."""
    overrides: Dict[str, Any] = {}
    if args.detector_llm_model:
        overrides["model"] = args.detector_llm_model
    if args.detector_llm_base_url:
        overrides["base_url"] = args.detector_llm_base_url
    return overrides


def model_slug(env_var: str = "PII_EVAL_MODEL_NAME", fallback: str = "unknown-model") -> str:
    """Filesystem-safe slug derived from the model name the run targets.

    Used to suffix ``results_<slug>.{json,md}`` and ``vault_<slug>.db`` so
    that re-running against a different model does not overwrite prior
    artifacts.
    """
    raw = os.getenv(env_var) or fallback
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-").lower() or fallback
