import pytest

from latch_eval_tools.graders.predicate import (
    PredicateLeafGrader,
    evaluate_predicate,
)


@pytest.mark.parametrize(
    ("value", "predicate"),
    [
        ("0.5", {"op": "equals", "arg": 0.5}),
        (" -0.5 ", {"op": "in", "args": [0, -0.5, 1]}),
        ("1e3", {"op": "in", "args": [10, 100, 1000]}),
    ],
)
def test_numeric_config_values_match_numeric_strings(
    value: str, predicate: dict
) -> None:
    assert evaluate_predicate(predicate, value) is True


@pytest.mark.parametrize("value", ["not-a-number", "nan", "inf", "-inf"])
def test_invalid_or_nonfinite_numeric_strings_do_not_match(value: str) -> None:
    assert evaluate_predicate({"op": "equals", "arg": 0.5}, value) is False


def test_string_config_values_keep_exact_string_semantics() -> None:
    assert evaluate_predicate({"op": "equals", "arg": "001"}, 1) is False
    assert evaluate_predicate({"op": "in", "args": ["1"]}, "01") is False


def test_numeric_string_triggers_hard_fail_veto() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"rho": "-0.5"},
        {
            "name": "canonical constant veto",
            "role": "hard_fail",
            "answer_field": "rho",
            "predicate": {"op": "in", "args": [0, 0.5, -0.5, 1, -1]},
        },
    )

    assert not result.passed
    assert result.score == 0.0
    assert result.metrics["raw_result"] is True


def test_weighted_label_reports_its_raw_score_scale() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"label": "best"},
        {
            "role": "gate",
            "answer_field": "label",
            "threshold": 1.0,
            "predicate": {
                "op": "weighted_label",
                "table": {"partial": 1.0, "best": 2.0},
                "default": 0.0,
            },
        },
    )

    assert result.passed is True
    assert result.score == 2.0
    assert result.score_max == 2.0
    assert result.normalized_score() == 1.0
