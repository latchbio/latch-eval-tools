from dataclasses import dataclass, field
from typing import Any


class _Missing:
    """Sentinel for a graded field the agent never supplied.

    Kept distinct from a supplied ``null``: an absent field is graded as a
    failure without consulting the predicate, while an explicit ``null`` is a
    real (wrong) answer and is still evaluated. Without the distinction a
    missing field binds to ``None`` and silently satisfies predicates that are
    vacuously true on it (``not``, ``every``, ``none``, a ``weighted_label``
    with a positive default), paying full credit for a skipped answer.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


@dataclass
class GraderResult:
    passed: bool
    metrics: dict
    reasoning: str
    agent_answer: dict | None
    # Defaults to 0.0 so any construction site that forgets to pass a score
    # fails closed. Reward is computed from `score`, not `passed`, so a
    # fail-open default here silently awards full credit for non-answers.
    score: float = 0.0
    field_scores: dict = field(default_factory=dict)


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
