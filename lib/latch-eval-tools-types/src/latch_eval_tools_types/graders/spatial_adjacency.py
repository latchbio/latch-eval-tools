from typing import Literal, NotRequired, TypedDict


class AgentAnswer(TypedDict):
    median_ic_to_pc_um: int | float
    p90_ic_to_pc_um: int | float
    pct_ic_within_15um: int | float
    pct_ic_mixed_within_55um: int | float
    adjacency_pass: int | float


class PassThresholds(TypedDict):
    max_median_ic_to_pc_um: NotRequired[int | float]
    max_p90_ic_to_pc_um: NotRequired[int | float]
    min_pct_ic_within_15um: NotRequired[int | float]
    min_pct_ic_mixed_within_55um: NotRequired[int | float]


class Scoring(TypedDict):
    pass_thresholds: NotRequired[PassThresholds]


class Config(TypedDict):
    scoring: NotRequired[Scoring]


class Spec(TypedDict):
    type: Literal["spatial_adjacency"]
    config: Config


class Metrics(TypedDict):
    median_ic_to_pc_um: int | float
    p90_ic_to_pc_um: int | float
    pct_ic_within_15um: int | float
    pct_ic_mixed_within_55um: int | float
    adjacency_pass: bool
    max_median_threshold: int | float
    max_p90_threshold: int | float
    min_pct_15um_threshold: int | float
    min_pct_55um_threshold: int | float
    median_pass: bool
    p90_pass: bool
    within_15um_pass: bool
    mixed_55um_pass: bool
