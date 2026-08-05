"""Regression test: numbered MC options use an int ``correct_answer``.

Some evals present numbered options ("1. ... 10. ...") rather than lettered
ones and store ``correct_answer`` as an int (e.g. ``7``) with the agent asked
to return ``"answer": <int>``. The grader used to call ``.strip()`` directly
on ``config["correct_answer"]``, which raised ``AttributeError`` for any
non-string value and crashed grading for every such eval.
"""

from __future__ import annotations

import pytest

from latch_eval_tools.graders import get_grader


def test_int_correct_answer_does_not_raise() -> None:
    grader = get_grader("multiple_choice")
    result = grader.evaluate_answer({"answer": 7}, {"correct_answer": 7})
    assert result.passed
    assert result.score == 1.0


def test_int_correct_answer_wrong_choice() -> None:
    grader = get_grader("multiple_choice")
    result = grader.evaluate_answer({"answer": 3}, {"correct_answer": 7})
    assert not result.passed
    assert result.score == 0.0


def test_int_correct_answer_agent_answers_as_string() -> None:
    grader = get_grader("multiple_choice")
    result = grader.evaluate_answer({"answer": "7"}, {"correct_answer": 7})
    assert result.passed
    assert result.score == 1.0


def test_int_correct_answers_list_does_not_raise() -> None:
    grader = get_grader("multiple_choice")
    result = grader.evaluate_answer({"answer": 2}, {"correct_answers": [2, 4]})
    assert result.passed
    assert result.score == 1.0
