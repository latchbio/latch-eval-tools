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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (155, True),
        (105, True),
        (205, True),
        (104, False),
        (206, False),
        (0, False),
        ("160", True),
        ("not-a-number", False),
    ],
)
def test_abs_diff_lte_matches_absolute_tolerance(value, expected) -> None:
    predicate = {"op": "abs_diff_lte", "arg": 155, "tolerance": 50}
    assert evaluate_predicate(predicate, value) is expected


def test_abs_diff_lte_hard_fail_gates_offset_field() -> None:
    # hard_fail triggers when the predicate is true, so the veto wraps the
    # "passing" check in `not`: the veto fires when the field is *outside*
    # tolerance.
    config = {
        "name": "offset consistency veto",
        "role": "hard_fail",
        "answer_field": "offset_kb_of_largest_contact_gain",
        "predicate": {
            "op": "not",
            "arg": {"op": "abs_diff_lte", "arg": 155, "tolerance": 50},
        },
    }

    triggered = PredicateLeafGrader().evaluate_answer(
        {
            "offset_kb_of_largest_contact_gain": 0,
            "observed_contacts_promoter_to_distal_candidate_activated": 69,
            "answer": 6,
            "gain_profile_topology": 2,
        },
        config,
    )
    assert not triggered.passed

    clean = PredicateLeafGrader().evaluate_answer(
        {
            "offset_kb_of_largest_contact_gain": 155,
            "observed_contacts_promoter_to_distal_candidate_activated": 69,
            "answer": 6,
            "gain_profile_topology": 2,
        },
        config,
    )
    assert clean.passed


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
