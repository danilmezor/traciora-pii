"""Run a fine-tuned checkpoint through the detector ablation harness.

The evaluator points the OPF detector registry at the freshly trained
checkpoint and invokes ``benchmarks.pii_evaluation.run_detector_ablation``
with the ``regex_opf_routed`` configuration. The harness writes its
per-config P/R/F1 markdown under ``benchmarks/pii_evaluation/results/``;
this module copies the relevant slice into ``runs/<slug>/eval.md`` so
training runs don't clobber each other's results.

No new metric code lives here — the harness is the source of truth.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from traciora_pii.config import load_config

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_RESULTS_DIR = REPO_ROOT / "benchmarks" / "pii_evaluation" / "results"


def run_evaluate(
    *,
    checkpoint_dir: Path,
    locale: str,
    slug: str,
    runs_dir: Path,
    extra_argv: Optional[list[str]] = None,
) -> Path:
    """Evaluate ``checkpoint_dir`` against the ablation harness.

    Returns the path to the eval markdown copied into the run directory.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")

    runs_dir.mkdir(parents=True, exist_ok=True)

    # Register the new checkpoint using the current model registry schema.
    config = load_config().model_dump(mode="json")
    config["detector"]["opf"]["models"][locale] = {
        "checkpoint": str(checkpoint_dir.resolve()),
        "policy": f"{locale}_comprehensive",
    }
    env = os.environ.copy()
    env["PII_EVAL_MODEL_NAME"] = f"{slug}-eval"
    cmd = [
        sys.executable, "-m", "benchmarks.pii_evaluation.run_detector_ablation",
        "--only", "regex_only,regex_opf_base,regex_opf_routed",
        "--language", locale,
    ]
    if extra_argv:
        cmd.extend(extra_argv)

    with tempfile.TemporaryDirectory(prefix="traciora-eval-") as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        cmd.extend(["--config", str(config_path)])
        logger.info("Running ablation harness: %s", " ".join(cmd))
        subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=True)

    # Copy only the report from this invocation.
    source = HARNESS_RESULTS_DIR / f"detector_ablation_{slug}-eval.md"
    target = runs_dir / "eval.md"
    if not source.exists():
        raise FileNotFoundError(f"Ablation report missing: {source}")
    shutil.copy2(source, target)
    logger.info("Copied %s -> %s", source, target)
    return target
