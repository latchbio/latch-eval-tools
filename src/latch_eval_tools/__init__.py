from latch_eval_tools.types import Eval, EvalResult, TestCase, TestResult
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
from latch_eval_tools.graders import (
    BinaryGrader,
    GraderResult,
    get_nested_value,
    NumericToleranceGrader,
    MarkerGenePrecisionRecallGrader,
    MarkerGeneSeparationGrader,
    LabelSetJaccardGrader,
    DistributionComparisonGrader,
    SpatialAdjacencyGrader,
    MultipleChoiceGrader,
    GRADER_REGISTRY,
    get_grader,
)

__all__ = [
    # Types
    "Eval",
    "EvalResult",
    "TestCase",  # Backward compatibility alias
    "TestResult",  # Backward compatibility alias
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
    # Graders
    "BinaryGrader",
    "GraderResult",
    "get_nested_value",
    "NumericToleranceGrader",
    "MarkerGenePrecisionRecallGrader",
    "MarkerGeneSeparationGrader",
    "LabelSetJaccardGrader",
    "DistributionComparisonGrader",
    "SpatialAdjacencyGrader",
    "MultipleChoiceGrader",
    "GRADER_REGISTRY",
    "get_grader",
]

__version__ = "0.1.0"
