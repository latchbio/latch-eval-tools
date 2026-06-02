from typing import Literal, NotRequired, TypedDict


class AgentAnswer(TypedDict):
    total_cells: NotRequired[int | float]
    cell_type_distribution: NotRequired[dict[str, int | float]]


class GroundTruth(TypedDict):
    total_cells: NotRequired[int | float]
    cell_type_distribution: NotRequired[dict[str, int | float]]


class ToleranceValue(TypedDict):
    value: NotRequired[int | float]


class Tolerances(TypedDict):
    total_cells: NotRequired[ToleranceValue]
    cell_type_percentages: NotRequired[ToleranceValue]


class Config(TypedDict):
    ground_truth: NotRequired[GroundTruth]
    tolerances: NotRequired[Tolerances]


class Spec(TypedDict):
    type: Literal["distribution_comparison"]
    config: Config
