from .base import BinaryGrader, GraderResult, get_nested_value
from .numeric import NumericRangeGrader, NumericToleranceGrader
from .marker_gene import MarkerGenePrecisionRecallGrader, MarkerGeneSeparationGrader
from .label_set import LabelSetJaccardGrader
from .distribution import DistributionComparisonGrader
from .spatial import SpatialAdjacencyGrader
from .multiple_choice import MultipleChoiceGrader
from .refusal import RefusalVocabGrader
from .predicate import PredicateLeafGrader
from .composite import AllOfGrader, ListMatchGrader, DictMatchGrader
from .longest_subsequence import LongestSubsequenceGrader
from .completion import FinishedFileGrader
from .rubric import (
    GraderError,
    GraderTransientError,
    RubricCriterion,
    RubricCriterionGraderOutput,
    RubricCriterionJudgment,
    RubricGrader,
    RubricGraderConfig,
    RubricGraderOutput,
    RubricGraderOutputParseError,
    RubricScoreResult,
    compute_rubric_reward,
    rubric_criterion_output_config,
)
from .helpers import grade_multiple_graders_single_answer  # noqa: E402 -- depends on GRADER_REGISTRY

GRADER_REGISTRY = {
    "numeric_tolerance": NumericToleranceGrader,
    "numeric_range": NumericRangeGrader,
    "label_set_jaccard": LabelSetJaccardGrader,
    "jaccard_label_set": LabelSetJaccardGrader,
    "distribution_comparison": DistributionComparisonGrader,
    "marker_gene_precision_recall": MarkerGenePrecisionRecallGrader,
    "marker_gene_separation": MarkerGeneSeparationGrader,
    "spatial_adjacency": SpatialAdjacencyGrader,
    "multiple_choice": MultipleChoiceGrader,
    "refusal_vocab": RefusalVocabGrader,
    "predicate_leaf": PredicateLeafGrader,
    "all_of": AllOfGrader,
    "list_match": ListMatchGrader,
    "dict_match": DictMatchGrader,
    "longest_subsequence": LongestSubsequenceGrader,
    "finished_file": FinishedFileGrader,
}

LLM_GRADER_REGISTRY = {
    "rubric": RubricGrader,
}


def get_grader(grader_type: str) -> BinaryGrader:
    if grader_type not in GRADER_REGISTRY:
        raise ValueError(
            f"Unknown grader type: {grader_type}. Available: {list(GRADER_REGISTRY.keys())}"
        )
    return GRADER_REGISTRY[grader_type]()


__all__ = [
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
    "LongestSubsequenceGrader",
    "FinishedFileGrader",
    "GraderError",
    "GraderTransientError",
    "RubricCriterion",
    "RubricCriterionGraderOutput",
    "RubricCriterionJudgment",
    "RubricGrader",
    "RubricGraderConfig",
    "RubricGraderOutput",
    "RubricGraderOutputParseError",
    "RubricScoreResult",
    "GRADER_REGISTRY",
    "LLM_GRADER_REGISTRY",
    "compute_rubric_reward",
    "rubric_criterion_output_config",
    "get_grader",
    "grade_multiple_graders_single_answer",
]
