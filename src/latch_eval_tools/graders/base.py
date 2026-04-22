from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraderResult:
    passed: bool | None
    metrics: dict
    reasoning: str
    agent_answer: dict | None
    score: float | None = 1.0
    field_scores: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed is not None and self.score is None:
            raise ValueError(
                "GraderResult.score must not be None when passed is True or False; "
                "None is reserved for tooling misconfiguration (passed=None)"
            )


def get_nested_value(obj: dict, key: str) -> tuple[Any, bool]:
    if "." not in key:
        return obj.get(key), key in obj
    parts = key.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


class BinaryGrader:
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        raise NotImplementedError

    def evaluate(self, agent_answer: dict, config: dict) -> GraderResult:
        return self.evaluate_answer(agent_answer, config)
