import math

from .base import BinaryGrader, GraderResult


def _normalize_choice(value: object) -> str | None:
    """Return the comparison token for a JSON scalar choice.

    Multiple-choice answers are identifiers rather than quantities. Normalize
    strings case-insensitively and normalize the other JSON scalar types by
    their textual representation. This preserves the existing behavior where
    an agent answer of ``6`` and ``"6"`` are equivalent, while rejecting
    containers and null instead of accidentally stringifying them.
    """
    if isinstance(value, str):
        normalized = value.strip().upper()
        return normalized or None
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value).upper()
    return None


def _normalize_correct_answers(config: dict) -> tuple[list[str] | None, str | None]:
    has_single = "correct_answer" in config
    has_multiple = "correct_answers" in config
    if has_single == has_multiple:
        return None, "Configure exactly one of correct_answer or correct_answers"

    if has_multiple:
        raw_answers = config["correct_answers"]
        if not isinstance(raw_answers, list) or not raw_answers:
            return None, "correct_answers must be a non-empty list of JSON scalars"
    else:
        raw_answers = [config["correct_answer"]]

    correct_answers: list[str] = []
    for raw_answer in raw_answers:
        normalized = _normalize_choice(raw_answer)
        if normalized is None:
            return None, "Correct answers must be non-empty, finite JSON scalars"
        correct_answers.append(normalized)

    return correct_answers, None


class MultipleChoiceGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        correct_answers, config_error = _normalize_correct_answers(config)
        if correct_answers is None:
            return GraderResult(
                passed=False,
                metrics={"configuration_error": config_error},
                reasoning=f"Multiple Choice: CONFIGURATION ERROR\n\n  x {config_error}",
                agent_answer=agent_answer,
                score=0.0,
            )

        if "answer" not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="Agent answer missing required field: answer",
                agent_answer=agent_answer,
                score=0.0,
            )

        agent_choice = _normalize_choice(agent_answer["answer"])
        if agent_choice is None:
            return GraderResult(
                passed=False,
                metrics={
                    "correct_answers": correct_answers,
                    "agent_answer": agent_answer["answer"],
                },
                reasoning=(
                    "Multiple Choice: FAIL\n\n"
                    "  x Agent answer must be a non-empty JSON scalar"
                ),
                agent_answer=agent_answer,
                score=0.0,
            )

        passed = agent_choice in correct_answers

        display_correct = (
            correct_answers[0] if len(correct_answers) == 1 else correct_answers
        )
        metrics = {
            "correct_answers": correct_answers,
            "agent_answer": agent_choice,
        }

        if passed:
            reasoning = (
                f"Multiple Choice: PASS\n\n  + Agent answered: {agent_choice} (correct)"
            )
        else:
            reasoning = f"Multiple Choice: FAIL\n\n  x Agent answered: {agent_choice}\n    Correct answer(s): {display_correct}"

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )
