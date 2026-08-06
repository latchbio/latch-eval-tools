import math
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


def normalize_score(score: float, score_max: float) -> float:
    """Normalize a raw grader score onto the required ``[0, 1]`` reward surface."""

    if not math.isfinite(score) or not math.isfinite(score_max) or score_max <= 0.0:
        return 0.0
    normalized = score / score_max
    return min(1.0, max(0.0, normalized))


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
    # Raw predicate scores may use a larger scale (for example weighted_label
    # values of 0, 1, and 2). Consumers that expose a bounded reward normalize
    # score by this maximum while retaining the raw score for thresholds and
    # diagnostics.
    score_max: float = 1.0

    def normalized_score(self) -> float:
        return normalize_score(float(self.score), float(self.score_max))


def configuration_error_result(
    agent_answer: object,
    grader_name: str,
    reason: str,
) -> GraderResult:
    """Return a fail-closed result for an invalid static grader specification.

    Configuration failures are distinct from malformed or missing agent output:
    callers use the ``configuration_error`` metric to avoid treating a broken
    grader as evidence that an agent answered incorrectly.
    """

    return GraderResult(
        passed=False,
        metrics={"configuration_error": reason},
        reasoning=f"{grader_name}: CONFIGURATION ERROR\n\n  x {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
        field_scores={},
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
