from .base import BinaryGrader, GraderResult


class MultipleChoiceGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if "correct_answers" in config:
            correct_answers = [a.strip().upper() for a in config["correct_answers"]]
        else:
            correct_answers = [config.get("correct_answer", "").strip().upper()]

        if "answer" not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={"score": 0.0},
                reasoning="Agent answer missing required field: answer",
                agent_answer=agent_answer,
                score=0.0,
            )

        agent_choice = str(agent_answer["answer"]).strip().upper()
        passed = agent_choice in correct_answers

        display_correct = correct_answers[0] if len(correct_answers) == 1 else correct_answers
        metrics = {
            "correct_answers": correct_answers,
            "agent_answer": agent_choice,
            "score": 1.0 if passed else 0.0,
        }

        if passed:
            reasoning = f"Multiple Choice: PASS\n\n  + Agent answered: {agent_choice} (correct)"
        else:
            reasoning = f"Multiple Choice: FAIL\n\n  x Agent answered: {agent_choice}\n    Correct answer(s): {display_correct}"

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )
