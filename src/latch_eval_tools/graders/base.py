from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from latch_eval_tools_types.graders import all as models


@dataclass
class GraderResult:
    passed: bool
    metrics: dict[str, Any]
    reasoning: str
    agent_answer: dict[str, Any] | None
    score: float = 1.0
    field_scores: dict[str, Any] = field(default_factory=dict)


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


AA = TypeVar("AA", bound=models.AgentAnswer, default=models.AgentAnswer)
Cfg = TypeVar("Cfg", bound=models.Config, default=models.Config)


class BinaryGrader(ABC, Generic[AA, Cfg]):
    @abstractmethod
    def evaluate_answer(self, agent_answer: AA, config: Cfg) -> GraderResult:
        raise NotImplementedError

    def evaluate(self, agent_answer: AA, config: Cfg) -> GraderResult:
        return self.evaluate_answer(agent_answer, config)
