from .base import BinaryGrader, GraderResult, configuration_error_result


class FinishedFileGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return configuration_error_result(
                agent_answer, "Finished File", "config must be an object"
            )
        expected = config.get("expected", "finished")
        if not isinstance(expected, str) or expected.strip() == "":
            return configuration_error_result(
                agent_answer,
                "Finished File",
                "expected must be a non-empty string",
            )
        if not isinstance(agent_answer, dict):
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="agent answer must be an object",
                agent_answer=None,
                score=0.0,
            )
        contents = agent_answer.get("finished_file_contents")
        if contents is None:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="agent_answer missing 'finished_file_contents'",
                agent_answer=agent_answer,
                score=0.0,
            )
        if not isinstance(contents, str):
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="finished_file_contents must be a string",
                agent_answer=agent_answer,
                score=0.0,
            )
        passed = contents.strip() == expected
        return GraderResult(
            passed=passed,
            metrics={"finished_file_contents": contents},
            reasoning=(
                f"finished.txt stripped to {contents.strip()!r}; expected {expected!r}"
            ),
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )
