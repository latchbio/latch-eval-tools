from typing import Literal, NotRequired, TypeAlias, TypedDict

AgentAnswer: TypeAlias = dict[str, int | float]


class Range(TypedDict):
    min: NotRequired[int | float]
    max: NotRequired[int | float]


class Config(TypedDict):
    ground_truth: dict[str, int | float]
    ranges: dict[str, Range]


class Spec(TypedDict):
    type: Literal["numeric_range"]
    config: Config
