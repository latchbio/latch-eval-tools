from typing import Literal, NotRequired, TypeAlias, TypedDict


class AgentAnswer(TypedDict): ...


class ConfigAll(TypedDict):
    children: NotRequired[...]
    pass_rule: NotRequired[Literal["all"]]


class ConfigMinPassing(TypedDict):
    children: NotRequired[...]
    pass_rule: Literal["min_passing"]
    min_passing_children: NotRequired[int]


class ConfigScoreThreshold(TypedDict):
    children: NotRequired[...]
    pass_rule: Literal["score_threshold"]
    score_threshold: NotRequired[float]


Config: TypeAlias = ConfigAll | ConfigMinPassing | ConfigScoreThreshold


class Spec(TypedDict):
    type: Literal["all_of"]
    config: Config
