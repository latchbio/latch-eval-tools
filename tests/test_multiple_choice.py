import math

import pytest

from latch_eval_tools.graders.multiple_choice import MultipleChoiceGrader


@pytest.mark.parametrize(
    ("config", "agent_choice"),
    [
        ({"correct_answer": 6}, 6),
        ({"correct_answer": 6}, " 6 "),
        ({"correct_answer": 6}, 6.0),
        ({"correct_answer": 6.0}, 6),
        ({"correct_answer": -0.0}, 0),
        ({"correct_answers": [6, 8]}, 8),
        ({"correct_answer": True}, "true"),
        ({"correct_answer": 2.5}, "2.5"),
        ({"correct_answer": "option a"}, " OPTION A "),
    ],
)
def test_json_scalar_correct_answers_are_normalized_consistently(
    config: dict, agent_choice: object
) -> None:
    result = MultipleChoiceGrader().evaluate_answer({"answer": agent_choice}, config)

    assert result.passed
    assert result.score == 1.0


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"correct_answer": "A", "correct_answers": ["A"]},
        {"correct_answers": "A"},
        {"correct_answers": []},
        {"correct_answer": None},
        {"correct_answer": ""},
        {"correct_answer": ["A"]},
        {"correct_answer": math.nan},
    ],
)
def test_invalid_config_fails_closed(config: dict) -> None:
    result = MultipleChoiceGrader().evaluate_answer({"answer": "A"}, config)

    assert not result.passed
    assert result.score == 0.0
    assert result.metrics["configuration_error"]


@pytest.mark.parametrize("agent_choice", [None, "", ["A"], {"choice": "A"}, math.inf])
def test_non_scalar_agent_answer_fails_closed(agent_choice: object) -> None:
    result = MultipleChoiceGrader().evaluate_answer(
        {"answer": agent_choice}, {"correct_answer": "A"}
    )

    assert not result.passed
    assert result.score == 0.0


def test_wrong_numeric_choice_scores_zero() -> None:
    result = MultipleChoiceGrader().evaluate_answer(
        {"answer": 7}, {"correct_answer": 6}
    )

    assert not result.passed
    assert result.score == 0.0
