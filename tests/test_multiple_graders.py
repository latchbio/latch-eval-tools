import pytest
from pydantic import ValidationError

from latch_eval_tools.graders import grade_answer_with_specs
from latch_eval_tools.types import Eval


def _numeric_spec(field: str, expected: float) -> dict:
    return {
        "type": "numeric_tolerance",
        "config": {
            "ground_truth": {field: expected},
            "tolerances": {field: {"type": "absolute", "value": 0}},
        },
    }


def test_eval_accepts_plural_graders() -> None:
    graders = [_numeric_spec("a", 1), _numeric_spec("b", 2)]
    eval_case = Eval(id="example", task="return numbers", graders=graders)

    assert eval_case.grader_specs == graders


def test_eval_rejects_conflicting_grader_fields() -> None:
    with pytest.raises(ValidationError):
        Eval(
            id="example",
            task="return numbers",
            grader=_numeric_spec("a", 1),
            graders=[_numeric_spec("b", 2)],
        )


def test_eval_rejects_empty_graders() -> None:
    with pytest.raises(ValidationError):
        Eval(id="example", task="return numbers", graders=[])


def test_multiple_graders_pass_and_aggregate_scores() -> None:
    per_grader, aggregate = grade_answer_with_specs(
        {"a": 1, "b": 2},
        [_numeric_spec("a", 1), _numeric_spec("b", 2)],
    )

    assert [result.passed for result in per_grader if result is not None] == [
        True,
        True,
    ]
    assert aggregate is not None
    assert aggregate.passed is True
    assert aggregate.score == 1.0
    assert aggregate.metrics["n_graders"] == 2
    assert aggregate.field_scores["graders[0].a"] == 1.0
    assert aggregate.field_scores["graders[1].b"] == 1.0


def test_multiple_graders_fail_if_any_child_fails() -> None:
    per_grader, aggregate = grade_answer_with_specs(
        {"a": 1, "b": 2},
        [_numeric_spec("a", 1), _numeric_spec("b", 3)],
    )

    assert [result.passed for result in per_grader if result is not None] == [
        True,
        False,
    ]
    assert aggregate is not None
    assert aggregate.passed is False
    assert aggregate.score == 0.5
    assert aggregate.field_scores["graders[0].a"] == 1.0
    assert aggregate.field_scores["graders[1].b"] == 0.0


def test_unknown_grader_type_is_aggregate_failure() -> None:
    per_grader, aggregate = grade_answer_with_specs(
        {"a": 1},
        [{"type": "missing_grader", "config": {}}],
    )

    assert per_grader == [None]
    assert aggregate is not None
    assert aggregate.passed is False
    assert aggregate.score == 0.0
    assert aggregate.metrics["grader_misconfigured"] is True
    assert aggregate.metrics["graders[0].misconfigured"] is True


def test_misconfigured_grader_forces_zero_aggregate_score() -> None:
    per_grader, aggregate = grade_answer_with_specs(
        {"a": 1},
        [_numeric_spec("a", 1), {"type": "missing_grader", "config": {}}],
    )

    assert per_grader[0] is not None
    assert per_grader[1] is None
    assert aggregate is not None
    assert aggregate.passed is False
    assert aggregate.score == 0.0
