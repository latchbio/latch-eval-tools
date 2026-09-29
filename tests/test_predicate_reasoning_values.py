"""A predicate failure has to say what it saw and what it wanted.

Without both values a failing eval cannot be triaged: "predicate result:
False" is equally consistent with a wrong answer and a wrong grader, and
telling those apart is the first question anyone asks. numeric_tolerance
has always reported actual-vs-expected; predicate_leaf reported neither.
"""

from latch_eval_tools.graders import get_grader

GRADER = get_grader("predicate_leaf")


def reasoning(config: dict, answer: dict) -> str:
    return GRADER.evaluate_answer(answer, config).reasoning


def equals_config(expected: str) -> dict:
    return {
        "name": "conclusion",
        "role": "gate",
        "answer_field": "conclusion",
        "predicate": {"op": "equals", "arg": expected},
    }


def test_a_failure_reports_both_values():
    text = reasoning(
        equals_config("elevated_not_definitive"), {"conclusion": "Distinguishable"}
    )

    assert "observed: Distinguishable" in text
    assert "expected: equals elevated_not_definitive" in text


def test_a_pass_reports_them_too():
    """Useful when the grader is the thing under suspicion."""

    text = reasoning(
        equals_config("elevated_not_definitive"),
        {"conclusion": "elevated_not_definitive"},
    )

    assert "observed: elevated_not_definitive" in text
    assert "PASS" in text


def test_a_membership_predicate_lists_what_it_would_accept():
    text = reasoning(
        {
            "name": "pair",
            "role": "gate",
            "answer_field": "pair",
            "predicate": {"op": "in", "args": ["a|b", "b|a"]},
        },
        {"pair": "c|d"},
    )

    assert "observed: c|d" in text
    assert '"a|b"' in text and '"b|a"' in text


def test_a_long_answer_is_truncated_not_dumped():
    """Agents write paragraphs into fields that expect a token; the whole
    essay in a grader message buries the mismatch it is meant to show."""

    essay = "x" * 5000
    text = reasoning(equals_config("short"), {"conclusion": essay})

    assert len(text) < 600
    assert "5000 chars" in text


def test_a_compound_predicate_claims_no_single_expectation():
    """and/or/not have no one value to compare against, so nothing is
    asserted rather than something misleading."""

    text = reasoning(
        {
            "name": "c",
            "role": "gate",
            "answer_field": "c",
            "predicate": {
                "op": "or",
                "args": [{"op": "equals", "arg": 1}, {"op": "equals", "arg": 2}],
            },
        },
        {"c": 3},
    )

    assert "observed: 3" in text


def test_a_non_string_value_is_rendered_as_json():
    text = reasoning(
        {
            "name": "n",
            "role": "gate",
            "answer_field": "n",
            "predicate": {"op": "equals", "arg": 3},
        },
        {"n": [1, 2]},
    )

    assert "observed: [1, 2]" in text
