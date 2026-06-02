from typing import Literal, NotRequired, TypeAlias, TypedDict

AgentAnswer: TypeAlias = dict[str, list[str]]


class Scoring(TypedDict):
    pass_threshold: NotRequired[int | float]


class Config(TypedDict):
    ground_truth_labels: list[str]
    scoring: NotRequired[Scoring]
    answer_field: NotRequired[str]


class Spec(TypedDict):
    type: Literal["label_set_jaccard", "jaccard_label_set"]
    config: Config
