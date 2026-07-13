from latch_eval_tools.types import Eval, EvalResult, TestCase, TestResult
from latch_eval_tools.harness import (
    EvalRunner,
    run_minisweagent_task,
    run_claudecode_task,
    run_openaicodex_task,
    run_pi_task,
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
    NumericRangeGrader,
    NumericToleranceGrader,
    MarkerGenePrecisionRecallGrader,
    MarkerGeneSeparationGrader,
    LabelSetJaccardGrader,
    DistributionComparisonGrader,
    SpatialAdjacencyGrader,
    MultipleChoiceGrader,
    RefusalVocabGrader,
    PredicateLeafGrader,
    AllOfGrader,
    ListMatchGrader,
    DictMatchGrader,
    GRADER_REGISTRY,
    get_grader,
)
from latch_eval_tools.llm_refusal import (
    LLMRefusalDiagnostic,
    detect_llm_refusal,
)

__all__ = [
    # Types
    "Eval",
    "EvalResult",
    "TestCase",  # Backward compatibility alias
    "TestResult",  # Backward compatibility alias
    # Harness
    "EvalRunner",
    "run_minisweagent_task",
    "run_claudecode_task",
    "run_openaicodex_task",
    "run_pi_task",
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
    "NumericRangeGrader",
    "NumericToleranceGrader",
    "MarkerGenePrecisionRecallGrader",
    "MarkerGeneSeparationGrader",
    "LabelSetJaccardGrader",
    "DistributionComparisonGrader",
    "SpatialAdjacencyGrader",
    "MultipleChoiceGrader",
    "RefusalVocabGrader",
    "PredicateLeafGrader",
    "AllOfGrader",
    "ListMatchGrader",
    "DictMatchGrader",
    "GRADER_REGISTRY",
    "get_grader",
    # LLM refusal detection
    "LLMRefusalDiagnostic",
    "detect_llm_refusal",
]

__version__ = "0.1.0"
