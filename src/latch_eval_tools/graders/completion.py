from .base import BinaryGrader, GraderResult


class FinishedFileGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        expected = config.get("expected", "finished")
        contents = agent_answer.get("finished_file_contents")
        if contents is None:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="agent_answer missing 'finished_file_contents'",
                agent_answer=agent_answer,
            )
        passed = contents.strip() == expected
        return GraderResult(
            passed=passed,
            metrics={"finished_file_contents": contents},
            reasoning=(
                f"finished.txt stripped to {contents.strip()!r}; "
                f"expected {expected!r}"
            ),
            agent_answer=agent_answer,
        )
