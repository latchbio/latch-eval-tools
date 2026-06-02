from typing import Literal, NotRequired, TypeAlias, TypedDict

AgentAnswer: TypeAlias = dict[str, dict[str, list[str]] | list[str]]


class PassThresholds(TypedDict):
    min_recall_per_celltype: NotRequired[float]
    min_celltypes_passing: NotRequired[int]

    precision_at_k: NotRequired[float]
    recall_at_k: NotRequired[float]


class Scoring(TypedDict):
    pass_thresholds: NotRequired[PassThresholds]


class ConfigBase(TypedDict):
    scoring: NotRequired[Scoring]
    answer_field: NotRequired[str]


class ConfigCanonicalMarkers(ConfigBase):
    canonical_markers: list[str] | dict[str, list[str]]


class ConfigGroundTruthLabels(ConfigBase):
    ground_truth_labels: list[str] | dict[str, list[str]]


Config: TypeAlias = ConfigCanonicalMarkers | ConfigGroundTruthLabels


class Spec(TypedDict):
    type: Literal["marker_gene_precision_recall"]
    config: Config


class CellTypeMetricsError(TypedDict("Metrics", {"pass": Literal[False]})):
    recall: float  # always 0.0
    error: str


class CellTypeMetricsOk(TypedDict("Metrics", {"pass": bool})):
    recall: float
    num_predicted: int
    num_canonical: int
    true_positives: list[str]
    false_negatives: list[str]


CellTypeMetrics: TypeAlias = CellTypeMetricsError | CellTypeMetricsOk


class MetricsCellTypeMode(TypedDict):
    celltypes_passing: int
    total_celltypes: int
    min_celltypes_passing: int
    min_recall_per_celltype: float
    per_celltype: dict[str, CellTypeMetrics]
    answer_field_used: str


class MetricsFlatMode(TypedDict):
    k: int
    precision_at_k: float
    recall_at_k: float
    precision_threshold: float
    recall_threshold: float
    true_positives: list[str]
    false_positives: list[str]
    false_negatives: list[str]
    num_true_positives: int
    num_false_positives: int
    num_false_negatives: int
    num_canonical_markers: int
    precision_pass: bool
    recall_pass: bool
    answer_field_used: str
