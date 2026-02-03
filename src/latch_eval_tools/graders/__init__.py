from .base import BinaryGrader, GraderResult, get_nested_value
from .numeric import NumericToleranceGrader
from .marker_gene import MarkerGenePrecisionRecallGrader, MarkerGeneSeparationGrader
from .label_set import LabelSetJaccardGrader
from .distribution import DistributionComparisonGrader
from .spatial import SpatialAdjacencyGrader
from .multiple_choice import MultipleChoiceGrader

GRADER_REGISTRY = {
    "numeric_tolerance": NumericToleranceGrader,
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
        raise ValueError(f"Unknown grader type: {grader_type}. Available: {list(GRADER_REGISTRY.keys())}")
    return GRADER_REGISTRY[grader_type]()


__all__ = [
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
