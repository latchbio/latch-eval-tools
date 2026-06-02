from typing import Literal, NotRequired, TypedDict


class GeneStat(TypedDict):
    gene: str
    auroc: int | float


class AgentAnswer(TypedDict):
    per_gene_stats: list[GeneStat]
    mean_auroc: int | float


class PassThresholds(TypedDict):
    mean_auroc: NotRequired[float]
    fraction_high: NotRequired[float]
    per_gene_cutoff: NotRequired[float]


class Scoring(TypedDict):
    pass_thresholds: NotRequired[PassThresholds]


class Config(TypedDict):
    scoring: NotRequired[Scoring]


class Spec(TypedDict):
    type: Literal["marker_gene_separation"]
    config: Config
