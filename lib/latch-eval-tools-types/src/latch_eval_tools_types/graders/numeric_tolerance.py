from typing import Literal, NotRequired, TypeAlias, TypedDict

AgentAnswer: TypeAlias = dict[str, int | float]


class ToleranceAbsolute(TypedDict):
    type: Literal["absolute"]
    value: int | float
    upper: NotRequired[int | float]
    lower: NotRequired[int | float]


class ToleranceOther(TypedDict):
    type: Literal["relative", "min", "max"]
    value: int | float


Tolerance: TypeAlias = ToleranceAbsolute | ToleranceOther


class ConfigBase(TypedDict):
    ground_truth: dict[str, int | float]


class Config1Tolerance(ConfigBase):
    tolerance: dict[str, Tolerance] | Tolerance


class ConfigNTolerances(ConfigBase):
    tolerances: dict[str, Tolerance] | Tolerance


Config: TypeAlias = Config1Tolerance | ConfigNTolerances


class Spec(TypedDict):
    type: Literal["numeric_tolerance"]
    config: Config
