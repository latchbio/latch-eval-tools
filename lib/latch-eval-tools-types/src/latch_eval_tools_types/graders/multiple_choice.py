from typing import Literal, TypeAlias, TypedDict


class AgentAnswer(TypedDict):
    answer: str


class Config1Answer(TypedDict):
    correct_answer: str


class ConfigNAnswers(TypedDict):
    correct_answers: list[str]


Config: TypeAlias = Config1Answer | ConfigNAnswers


class Spec(TypedDict):
    type: Literal["multiple_choice"]
    config: Config1Answer | ConfigNAnswers
