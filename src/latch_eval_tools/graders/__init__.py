from .artifact import ArtifactGrader
from .base import BinaryGrader, GraderResult, get_nested_value, normalize_score
from .completion import FinishedFileGrader
from .composite import (
    AllOfGrader,
    AverageOfGrader,
    DictMatchGrader,
    ListMatchGrader,
    evaluate_composite_predicate_leaf,
)
from .distribution import DistributionComparisonGrader
from .helpers import (
    grade_multiple_graders_single_answer,  # noqa: E402 -- depends on GRADER_REGISTRY
)
from .label_set import LabelSetJaccardGrader
from .longest_subsequence import LongestSubsequenceGrader
from .marker_gene import MarkerGenePrecisionRecallGrader, MarkerGeneSeparationGrader
from .multiple_choice import MultipleChoiceGrader
from .numeric import NumericRangeGrader, NumericToleranceGrader
from .predicate import PredicateLeafGrader
from .refusal import RefusalVocabGrader
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
from .spatial import SpatialAdjacencyGrader

GRADER_REGISTRY = {
    "artifact": ArtifactGrader,
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
    "composite": AllOfGrader,
    "average_of": AverageOfGrader,
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
    "normalize_score",
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
    "AverageOfGrader",
    "ListMatchGrader",
    "DictMatchGrader",
    "evaluate_composite_predicate_leaf",
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
