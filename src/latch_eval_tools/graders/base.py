from dataclasses import dataclass


@dataclass
class GraderResult:
    passed: bool
    metrics: dict
    reasoning: str
    agent_answer: dict | None
    score: float = 0.0


def get_nested_value(obj: dict, key: str) -> tuple[any, bool]:
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

    @staticmethod
    def clamp_score(score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def score_with_tolerance(error: float, tolerance: float) -> float:
        if error <= 0:
            return 1.0
        if tolerance > 0:
            if error <= tolerance:
                return 1.0
            return tolerance / error
        return 1.0 / (1.0 + error)
