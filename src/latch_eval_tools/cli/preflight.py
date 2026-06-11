"""Non-fatal preflight checks (Docker, provider keys, Latch token) for the CLI."""

import os
import shutil
import sys
from pathlib import Path

from latch_eval_tools.cli.harnesses import HARNESS_ENV_HINTS


def preflight(harness, docker_image, needs_latch_token=True):
    """Emit non-fatal warnings for missing Docker / API keys before a run."""
    warnings = []
    if shutil.which("docker") is None:
        warnings.append(
            "Docker not found on PATH. All harnesses run the agent inside a "
            "Docker container; the run will fail without it."
        )
    hints = HARNESS_ENV_HINTS.get(harness, [])
    if hints and not any(os.environ.get(k) for k in hints):
        warnings.append(
            f"None of {hints} is set; the {harness} harness needs one to reach "
            "its model provider."
        )
    if needs_latch_token and not (Path.home() / ".latch" / "token").exists():
        warnings.append(
            "No Latch token at ~/.latch/token; data_node downloads will fail. "
            "Run `latch login` (or pass --data to use local files)."
        )
    for w in warnings:
        print(f"[preflight] WARNING: {w}", file=sys.stderr)
