from latch_eval_tools.graders.predicate import (
    PredicateLeafGrader,
    evaluate_predicate,
)


def test_numeric_config_values_match_numeric_strings() -> None:
    cases = [
        ("0.5", {"op": "equals", "arg": 0.5}),
        (" -0.5 ", {"op": "in", "args": [0, -0.5, 1]}),
        ("1e3", {"op": "in", "args": [10, 100, 1000]}),
    ]
    for value, predicate in cases:
        assert evaluate_predicate(predicate, value) is True


def test_invalid_or_nonfinite_numeric_strings_do_not_match() -> None:
    for value in ["not-a-number", "nan", "inf", "-inf"]:
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


def test_in_does_not_coerce_against_boolean_candidates() -> None:
    assert evaluate_predicate({"op": "in", "args": [True]}, "1") is False
    assert evaluate_predicate({"op": "in", "args": [True]}, 1) is True


def test_in_never_matches_nan() -> None:
    nan = float("nan")
    assert evaluate_predicate({"op": "in", "args": [nan]}, nan) is False


def test_location_within_radius_predicate() -> None:
    predicate = {
        "op": "location_within_radius",
        "reference_location": [0, 0],
        "tolerance_radius": 5,
    }

    assert evaluate_predicate(predicate, [3, 4]) is True
    assert evaluate_predicate(predicate, [6, 0]) is False


def test_location_within_radius_predicate_leaf() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"location": [3, 4]},
        {
            "role": "gate",
            "answer_field": "location",
            "predicate": {
                "op": "location_within_radius",
                "reference_location": [0, 0],
                "tolerance_radius": 5,
            },
        },
    )

    assert result.passed is True
    assert result.score == 1.0


def test_location_within_radius_rejects_invalid_config() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"location": [0, 0]},
        {
            "role": "gate",
            "answer_field": "location",
            "predicate": {
                "op": "location_within_radius",
                "reference_location": [0, 0],
                "tolerance_radius": -1,
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")
