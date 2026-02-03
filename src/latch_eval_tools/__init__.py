"""latch-eval-tools: Shared evaluation harness tools for biology AI benchmarks."""

from latch_eval_tools.types import Eval, EvalResult, TestCase, TestResult, GraderResult
from latch_eval_tools.linter import lint_eval, lint_directory, LintResult
from latch_eval_tools.harness import (
    EvalRunner,
    run_minisweagent_task,
    run_claudecode_task,
    run_plotsagent_task,
    download_single_dataset,
    download_data,
    batch_download_datasets,
    setup_workspace,
    cleanup_workspace,
)

__all__ = [
    # Core types
    "Eval",
    "EvalResult",
    "TestCase",  # Backward compatibility alias
    "TestResult",  # Backward compatibility alias
    "GraderResult",
    # Linter
    "lint_eval",
    "lint_directory",
    "LintResult",
    # Harness
    "EvalRunner",
    "run_minisweagent_task",
    "run_claudecode_task",
    "run_plotsagent_task",
    "download_single_dataset",
    "download_data",
    "batch_download_datasets",
    "setup_workspace",
    "cleanup_workspace",
]

__version__ = "0.1.0"
