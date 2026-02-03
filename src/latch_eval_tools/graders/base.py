from dataclasses import dataclass


@dataclass
class GraderResult:
    passed: bool
    metrics: dict
    reasoning: str
    agent_answer: dict | None


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
