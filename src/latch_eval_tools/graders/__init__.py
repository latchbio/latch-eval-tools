from .base import BinaryGrader, GraderResult, get_nested_value
from .numeric import NumericRangeGrader, NumericToleranceGrader
from .marker_gene import MarkerGenePrecisionRecallGrader, MarkerGeneSeparationGrader
from .label_set import LabelSetJaccardGrader
from .distribution import DistributionComparisonGrader
from .spatial import SpatialAdjacencyGrader
from .multiple_choice import MultipleChoiceGrader
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
    "GRADER_REGISTRY",
    "get_grader",
    "grade_multiple_graders_single_answer",
]
