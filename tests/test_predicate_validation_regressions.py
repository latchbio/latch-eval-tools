from latch_eval_tools.graders import get_grader
from latch_eval_tools.graders.predicate import PredicateLeafGrader


def test_predicate_leaf_rejects_non_string_name() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"x": 1},
        {
            "name": ["invalid"],
            "role": "gate",
            "answer_field": "x",
            "predicate": {"op": "equals", "arg": 1},
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_composite_leaf_rejects_non_string_name_without_raising() -> None:
    result = get_grader("all_of").evaluate_answer(
        {"x": 1},
        {
            "children": [
                {
                    "name": ["invalid"],
                    "role": "gate",
                    "answer_field": "x",
                    "predicate": {"op": "equals", "arg": 1},
                }
            ]
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_bounded_scalar_predicate_rejects_impossible_threshold() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"labels": ["A"]},
        {
            "role": "gate",
            "answer_field": "labels",
            "threshold": 1.1,
            "predicate": {"op": "f1", "expected": ["A"]},
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_scalar_predicate_cannot_be_nested_as_implicit_boolean() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"labels": ["A"]},
        {
            "role": "gate",
            "answer_field": "labels",
            "predicate": {
                "op": "and",
                "args": [
                    {"op": "f1", "expected": ["A"]},
                    {"op": "equals", "arg": ["A"]},
                ],
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_weighted_label_rejects_threshold_above_score_capacity() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"label": "A"},
        {
            "role": "gate",
            "answer_field": "label",
            "threshold": 1.0,
            "predicate": {
                "op": "weighted_label",
                "table": {"A": 0.5},
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_composite_rejects_unknown_list_and_dict_modes() -> None:
    list_base = {
        "answer_field": "rows",
        "match_key": "id",
        "ground_truth": [
            {
                "id": "a",
                "fields": {
                    "x": {
                        "role": "gate",
                        "predicate": {"op": "equals", "arg": 1},
                    }
                },
            }
        ],
    }
    dict_base = {
        "answer_field": "values",
        "ground_truth": {
            "a": {
                "role": "gate",
                "predicate": {"op": "equals", "arg": 1},
            }
        },
    }

    for field, value in (
        ("match_key_normalize", "typo"),
        ("match_key_normalize", []),
        ("per_tuple_rule", {}),
    ):
        result = get_grader("list_match").evaluate_answer(
            {"rows": []},
            {**list_base, field: value},
        )
        assert result.metrics.get("configuration_error")

    for value in ("typo", []):
        result = get_grader("dict_match").evaluate_answer(
            {"values": {}},
            {**dict_base, "per_entry_rule": value},
        )
        assert result.metrics.get("configuration_error")
