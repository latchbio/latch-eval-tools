"""``points`` restates an ``average_of`` child's weight in the bundle."""

import pytest

from latch_eval_tools.graders import get_grader


def leaf(name: str, arg: int, *, role: str = "gate", **extra) -> dict:
    return {
        "name": name,
        "answer_field": name,
        "predicate": {"op": "equals", "arg": arg},
        "role": role,
        **extra,
    }


def grade(config: dict, answer: dict):
    return get_grader("average_of").evaluate_answer(answer, config)


def test_unweighted_children_keep_the_plain_mean() -> None:
    config = {"children": [leaf("a", 1), leaf("b", 1)]}

    result = grade(config, {"a": 1, "b": 2})

    assert result.score == pytest.approx(0.5)
    assert result.metrics["score_denominator"] == pytest.approx(2.0)


def test_points_weight_a_child_against_its_siblings() -> None:
    # 3-point decision vs 1-point check: getting only the decision right is 75%.
    config = {"children": [leaf("a", 1, points=3), leaf("b", 1)]}

    result = grade(config, {"a": 1, "b": 2})

    assert result.score == pytest.approx(0.75)
    assert result.metrics["scoring_total_score"] == pytest.approx(3.0)
    assert result.metrics["score_denominator"] == pytest.approx(4.0)


def test_points_apply_to_typed_children_too() -> None:
    config = {
        "children": [
            {
                "name": "counts",
                "type": "numeric_tolerance",
                "points": 4,
                "config": {
                    "ground_truth": {"n": 100},
                    "tolerances": {"n": {"type": "absolute", "value": 5}},
                },
            },
            leaf("b", 1),
        ]
    }

    result = grade(config, {"n": 100, "b": 2})

    assert result.score == pytest.approx(0.8)
    assert result.metrics["score_denominator"] == pytest.approx(5.0)


def test_fractional_points_are_allowed() -> None:
    config = {"children": [leaf("a", 1, points=0.5), leaf("b", 1, points=1.5)]}

    result = grade(config, {"a": 1, "b": 2})

    assert result.score == pytest.approx(0.25)


def test_score_threshold_is_measured_in_points() -> None:
    # 9-point bundle, pass at 75% => 6.75 points.
    config = {
        "pass_rule": "score_threshold",
        "score_threshold": 6.75,
        "children": [leaf("a", 1, points=5), leaf("b", 1, points=2), leaf("c", 1)],
    }

    assert grade(config, {"a": 1, "b": 1, "c": 2}).passed is True
    assert grade(config, {"a": 1, "b": 2, "c": 1}).passed is False


def test_points_do_not_change_which_children_passed() -> None:
    config = {
        "pass_rule": "min_passing",
        "min_passing_children": 2,
        "children": [leaf("a", 1, points=10), leaf("b", 1), leaf("c", 1)],
    }

    result = grade(config, {"a": 2, "b": 1, "c": 1})

    assert result.passed is True
    assert result.metrics["failed_children"] == ["a"]


def test_partial_credit_inside_a_child_scales_to_its_points() -> None:
    config = {
        "children": [
            {
                "name": "rows",
                "type": "list_match",
                "points": 4,
                "config": {
                    "answer_field": "rows",
                    "match_key": "id",
                    "ground_truth": [
                        {
                            "id": "a",
                            "fields": {
                                "v": {
                                    "predicate": {"op": "equals", "arg": 1},
                                    "role": "additive",
                                }
                            },
                        },
                        {
                            "id": "b",
                            "fields": {
                                "v": {
                                    "predicate": {"op": "equals", "arg": 2},
                                    "role": "additive",
                                }
                            },
                        },
                    ],
                },
            }
        ]
    }

    result = grade(config, {"rows": [{"id": "a", "v": 1}, {"id": "b", "v": 99}]})

    # Half the rows right on a 4-point child => 2 of 4 points.
    assert result.metrics["scoring_total_score"] == pytest.approx(2.0)
    assert result.metrics["score_denominator"] == pytest.approx(4.0)


@pytest.mark.parametrize("points", [0, -1, "3", True, float("inf"), None])
def test_invalid_points_fail_closed_as_misconfiguration(points: object) -> None:
    config = {"children": [leaf("a", 1, points=points), leaf("b", 1)]}

    result = grade(config, {"a": 1, "b": 1})

    assert result.passed is False
    assert result.score == 0.0
    assert "a" in result.metrics["misconfigured_children"]


def test_points_on_a_hard_fail_child_is_a_configuration_error() -> None:
    config = {
        "children": [leaf("a", 1), leaf("veto", 1, role="hard_fail", points=2)],
    }

    result = grade(config, {"a": 1, "veto": 1})

    assert result.passed is False
    assert "veto" in result.metrics["misconfigured_children"]


def test_hard_fail_child_without_points_still_vetoes() -> None:
    config = {
        "children": [leaf("a", 1, points=5), leaf("veto", 1, role="hard_fail")],
    }

    assert grade(config, {"a": 1, "veto": 99}).passed is True
    # The veto predicate matching is the disqualifying condition.
    blocked = grade(config, {"a": 1, "veto": 1})
    assert blocked.passed is False
    assert blocked.score == 0.0
