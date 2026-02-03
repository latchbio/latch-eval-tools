"""latch-eval-tools: Shared evaluation harness tools for biology AI benchmarks."""

from latch_eval_tools.types import Eval, EvalResult, TestCase, TestResult, GraderResult
from latch_eval_tools.linter import lint_eval, lint_directory, LintResult

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
]

__version__ = "0.1.0"
