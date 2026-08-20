"""Regression tests for grader fail-closed behaviour.

``GraderResult.score`` is the reward-bearing value. Any code path that returns
``passed=False`` without an explicit ``score`` therefore silently awards full
credit. These tests pin that behaviour for every registered grader so the bug
cannot reappear in a new grader or a new early-return branch.

The second half of the file pins the related invariant that *skipping* a graded
field is never cheaper than getting it wrong. An asymmetry there is a direct
incentive to answer less, which is worse than a merely lenient grader: the
highest-scoring strategy becomes submitting as little as possible.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from latch_eval_tools.graders import (
    GRADER_REGISTRY,
    BinaryGrader,
    GraderResult,
    get_grader,
)

OVERFLOWING_INTEGER = int("9" * 400)

# Minimal-but-valid config per grader so each one reaches its own logic rather
# than bailing out on a config error.
CONFIGS: dict[str, dict] = {
    "numeric_tolerance": {
        "ground_truth": {"x": 1.0},
        "tolerances": {"x": {"type": "absolute", "value": 0.1}},
    },
    "numeric_range": {
        "ground_truth": {"x": 1.0},
        "ranges": {"x": {"min": 0.0, "max": 2.0}},
    },
    "label_set_jaccard": {"ground_truth_labels": ["A", "B"], "answer_field": "labels"},
    "jaccard_label_set": {"ground_truth_labels": ["A", "B"], "answer_field": "labels"},
    "distribution_comparison": {
        "ground_truth": {"total_cells": 10, "cell_type_distribution": {"T": 50.0}}
    },
    "marker_gene_precision_recall": {
        "canonical_markers": ["CD3D"],
        "answer_field": "genes",
    },
    "marker_gene_separation": {"scoring": {"pass_thresholds": {}}},
    "spatial_adjacency": {"scoring": {"pass_thresholds": {}}},
    "multiple_choice": {"correct_answer": "C"},
    "molecular_structure": {
        "answer_field": "product_smiles",
        "expected_smiles": "CCO",
        "connectivity_only": True,
        "require_single_fragment": True,
        "similarity_threshold": 0.8,
    },
    "refusal_vocab": {"refusal_vocab": ["REFUSE"], "expected_decision": "REFUSE"},
    "predicate_leaf": {
        "predicate": {"op": "equals", "arg": 1},
        "role": "gate",
        "answer_field": "x",
    },
    "all_of": {
        "children": [
            {
                "name": "c1",
                "predicate": {"op": "equals", "arg": 1},
                "role": "gate",
                "answer_field": "x",
            }
        ]
    },
    "composite": {
        "children": [
            {
                "name": "c1",
                "predicate": {"op": "equals", "arg": 1},
                "role": "gate",
                "answer_field": "x",
            }
        ]
    },
    "average_of": {
        "children": [
            {
                "name": "c1",
                "predicate": {"op": "equals", "arg": 1},
                "role": "gate",
                "answer_field": "x",
            }
        ]
    },
    "list_match": {
        "answer_field": "rows",
        "match_key": "id",
        "ground_truth": [
            {
                "id": "a",
                "fields": {
                    "v": {"predicate": {"op": "equals", "arg": 1}, "role": "additive"}
                },
            }
        ],
    },
    "dict_match": {
        "answer_field": "obj",
        "ground_truth": {
            "k": {"predicate": {"op": "equals", "arg": 1}, "role": "gate"}
        },
    },
    "longest_subsequence": {"answer_field": "seq", "ground_truth": [["a"]]},
    "finished_file": {"expected": "finished"},
}

CORRECT_ANSWERS: dict[str, dict] = {
    "numeric_tolerance": {"x": 1.0},
    "numeric_range": {"x": 1.0},
    "label_set_jaccard": {"labels": ["A", "B"]},
    "jaccard_label_set": {"labels": ["A", "B"]},
    "distribution_comparison": {
        "total_cells": 10,
        "cell_type_distribution": {"T": 50.0},
    },
    "marker_gene_precision_recall": {"genes": ["CD3D"]},
    "marker_gene_separation": {
        "per_gene_stats": [{"gene": "A", "auroc": 0.95}],
        "mean_auroc": 0.95,
    },
    "spatial_adjacency": {
        "median_ic_to_pc_um": 10.0,
        "p90_ic_to_pc_um": 50.0,
        "pct_ic_within_15um": 70.0,
        "pct_ic_mixed_within_55um": 70.0,
        "adjacency_pass": True,
    },
    "multiple_choice": {"answer": "C"},
    "molecular_structure": {"product_smiles": "CCO"},
    "refusal_vocab": {"decision": "REFUSE"},
    "predicate_leaf": {"x": 1},
    "all_of": {"x": 1},
    "composite": {"x": 1},
    "average_of": {"x": 1},
    "list_match": {"rows": [{"id": "a", "v": 1}]},
    "dict_match": {"obj": {"k": 1}},
    "longest_subsequence": {"seq": [["a"]]},
    "finished_file": {"finished_file_contents": "finished"},
}


def test_every_registered_grader_is_covered() -> None:
    """A new grader must be added here, so it cannot skip these guarantees."""
    assert set(GRADER_REGISTRY) == set(CONFIGS) == set(CORRECT_ANSWERS)


@pytest.mark.parametrize(
    ("grader_type", "config"),
    [
        (
            "predicate_leaf",
            {
                "role": [],
                "answer_field": "x",
                "predicate": {"op": "equals", "arg": 1},
            },
        ),
        (
            "average_of",
            {
                "children": [
                    {
                        "role": [],
                        "answer_field": "x",
                        "predicate": {"op": "equals", "arg": 1},
                    }
                ]
            },
        ),
        (
            "list_match",
            {
                "answer_field": "rows",
                "match_key": "id",
                "ground_truth": [
                    {
                        "id": "a",
                        "fields": {
                            "v": {
                                "role": [],
                                "predicate": {"op": "equals", "arg": 1},
                            }
                        },
                    }
                ],
            },
        ),
        (
            "dict_match",
            {
                "answer_field": "obj",
                "ground_truth": {
                    "k": {
                        "role": [],
                        "predicate": {"op": "equals", "arg": 1},
                    }
                },
            },
        ),
        (
            "dict_match",
            {
                "answer_field": "obj",
                "ground_truth": {
                    "k": {
                        "role": "gate",
                        "predicate": {"op": []},
                    }
                },
            },
        ),
    ],
)
def test_unhashable_enum_configuration_fails_cleanly(
    grader_type: str, config: dict
) -> None:
    result = get_grader(grader_type).evaluate_answer({}, config)

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics.get("configuration_error")


def test_list_match_malformed_nested_match_key_is_an_ordinary_miss() -> None:
    result = get_grader("list_match").evaluate_answer(
        {"rows": [{"id": {"a": 1, 2: [3]}, "v": 1}]},
        CONFIGS["list_match"],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "configuration_error" not in result.metrics


@pytest.mark.parametrize("grader_type", sorted(GRADER_REGISTRY))
def test_empty_answer_scores_zero(grader_type: str) -> None:
    result = get_grader(grader_type).evaluate_answer({}, CONFIGS[grader_type])
    assert not result.passed
    assert result.score == 0.0, (
        f"{grader_type} awards {result.score} for an empty answer; reward is read "
        "from score, so this is full credit for a non-answer"
    )


@pytest.mark.parametrize("grader_type", sorted(GRADER_REGISTRY))
def test_missing_graded_field_scores_zero(grader_type: str) -> None:
    result = get_grader(grader_type).evaluate_answer(
        {"unrelated_key": "unrelated"}, CONFIGS[grader_type]
    )
    assert not result.passed
    assert result.score == 0.0


@pytest.mark.parametrize("grader_type", sorted(GRADER_REGISTRY))
def test_correct_answer_scores_full_credit(grader_type: str) -> None:
    """Guards the other direction: fail-closed must not break real answers."""
    result = get_grader(grader_type).evaluate_answer(
        CORRECT_ANSWERS[grader_type], CONFIGS[grader_type]
    )
    assert result.passed
    assert result.score == 1.0


def test_grader_result_score_default_is_fail_closed() -> None:
    assert (
        GraderResult(passed=False, metrics={}, reasoning="", agent_answer={}).score
        == 0.0
    )


def test_no_grader_result_omits_an_explicit_score() -> None:
    """Belt and braces: the default is safe, but intent should still be explicit."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "latch_eval_tools"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "GraderResult"
                and "score" not in {kw.arg for kw in node.keywords if kw.arg}
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize(
    ("grader_type", "config", "answer"),
    [
        (
            "predicate_leaf",
            {
                "role": "gate",
                "answer_field": "quality",
                "predicate": {
                    "op": "weighted_label",
                    "table": {"best": OVERFLOWING_INTEGER},
                },
            },
            {"quality": "best"},
        ),
        (
            "average_of",
            {
                "pass_rule": "score_threshold",
                "score_threshold": OVERFLOWING_INTEGER,
                "children": [
                    {
                        "role": "gate",
                        "answer_field": "x",
                        "predicate": {"op": "equals", "arg": 1},
                    }
                ],
            },
            {"x": 1},
        ),
        (
            "list_match",
            {
                "answer_field": "rows",
                "match_key": "id",
                "additive_score_min": OVERFLOWING_INTEGER,
                "ground_truth": [
                    {
                        "id": "a",
                        "fields": {
                            "value": {
                                "role": "additive",
                                "predicate": {"op": "equals", "arg": 1},
                            }
                        },
                    }
                ],
            },
            {"rows": [{"id": "a", "value": 1}]},
        ),
    ],
)
def test_unrepresentable_configuration_numbers_fail_cleanly(
    grader_type: str, config: dict, answer: dict
) -> None:
    result = get_grader(grader_type).evaluate_answer(answer, config)

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics.get("configuration_error")


class TestWrongAnswersDoNotEarnCredit:
    def test_finished_file_wrong_contents(self) -> None:
        result = get_grader("finished_file").evaluate_answer(
            {"finished_file_contents": "not-finished"}, {"expected": "finished"}
        )
        assert not result.passed
        assert result.score == 0.0

    def test_multiple_choice_wrong_letter(self) -> None:
        result = get_grader("multiple_choice").evaluate_answer(
            {"answer": "A"}, {"correct_answer": "C"}
        )
        assert not result.passed
        assert result.score == 0.0

    def test_marker_gene_non_list_payload(self) -> None:
        result = get_grader("marker_gene_precision_recall").evaluate_answer(
            {"genes": "CD3D"}, CONFIGS["marker_gene_precision_recall"]
        )
        assert not result.passed
        assert result.score == 0.0


class TestHardFailVetoZeroesTheScore:
    """A triggered ``hard_fail`` must zero the reward, not just flip ``passed``."""

    ALL_OF = {
        "children": [
            {
                "name": "scoring",
                "predicate": {"op": "equals", "arg": 1},
                "role": "gate",
                "answer_field": "x",
            },
            {
                "name": "veto",
                "predicate": {"op": "equals", "arg": True},
                "role": "hard_fail",
                "answer_field": "cheated",
            },
        ]
    }

    def test_all_of_veto_triggered(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"x": 1, "cheated": True}, self.ALL_OF
        )
        assert not result.passed
        assert result.score == 0.0

    def test_all_of_veto_clean(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"x": 1, "cheated": False}, self.ALL_OF
        )
        assert result.passed
        assert result.score == 1.0

    def test_typed_predicate_leaf_hard_fail_vetoes(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"answer": "A", "flag": "forbidden"},
            {
                "children": [
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "A"},
                    },
                    {
                        "type": "predicate_leaf",
                        "config": {
                            "name": "forbidden flag",
                            "role": "hard_fail",
                            "answer_field": "flag",
                            "predicate": {"op": "equals", "arg": "forbidden"},
                        },
                    },
                ]
            },
        )

        assert not result.passed
        assert result.score == 0.0
        assert result.metrics["hard_fail_triggered"] == ["forbidden flag"]

    def test_typed_predicate_leaf_additive_is_invalid(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"quality": "best"},
            {
                "children": [
                    {
                        "type": "predicate_leaf",
                        "config": {
                            "name": "quality score",
                            "role": "additive",
                            "answer_field": "quality",
                            "predicate": {
                                "op": "weighted_label",
                                "table": {"partial": 1.0, "best": 2.0},
                                "default": 0.0,
                            },
                        },
                    }
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_all_of_with_only_hard_fail_children_fails_closed(self) -> None:
        config = {
            "children": [
                {
                    "name": "veto",
                    "predicate": {"op": "equals", "arg": True},
                    "role": "hard_fail",
                    "answer_field": "cheated",
                }
            ]
        }
        # Such a composite can express a veto but never a reward, so an empty
        # answer would otherwise earn full credit for ungraded work.
        for answer in ({}, {"cheated": False}, {"cheated": True}):
            result = get_grader("all_of").evaluate_answer(answer, config)
            assert not result.passed
            assert result.score == 0.0
            assert result.metrics.get("configuration_error")

    def test_list_match_hard_fail_vetoes(self) -> None:
        config = {
            "answer_field": "rows",
            "match_key": "id",
            "ground_truth": [
                {
                    "id": "a",
                    "fields": {
                        "bad": {
                            "predicate": {"op": "equals", "arg": True},
                            "role": "hard_fail",
                        },
                        "good": {
                            "predicate": {"op": "equals", "arg": 1},
                            "role": "additive",
                        },
                    },
                }
            ],
        }
        tripped = get_grader("list_match").evaluate_answer(
            {"rows": [{"id": "a", "bad": True, "good": 1}]}, config
        )
        assert not tripped.passed
        assert tripped.score == 0.0
        assert tripped.metrics["hard_fail_triggered"] == ["a.bad"]

        clean = get_grader("list_match").evaluate_answer(
            {"rows": [{"id": "a", "bad": False, "good": 1}]}, config
        )
        assert clean.passed
        assert clean.score == 1.0

        missing = get_grader("list_match").evaluate_answer(
            {"rows": [{"id": "a", "good": 1}]}, config
        )
        assert not missing.passed
        assert missing.score == 0.0
        assert missing.metrics["hard_fail_triggered"] == ["a.bad"]

    def test_dict_match_hard_fail_vetoes(self) -> None:
        config = {
            "answer_field": "obj",
            "ground_truth": {
                "e": {
                    "fields": {
                        "bad": {
                            "predicate": {"op": "equals", "arg": True},
                            "role": "hard_fail",
                        },
                        "good": {
                            "predicate": {"op": "equals", "arg": 1},
                            "role": "gate",
                        },
                    }
                }
            },
        }
        tripped = get_grader("dict_match").evaluate_answer(
            {"obj": {"e": {"bad": True, "good": 1}}}, config
        )
        assert not tripped.passed
        assert tripped.score == 0.0

        clean = get_grader("dict_match").evaluate_answer(
            {"obj": {"e": {"bad": False, "good": 1}}}, config
        )
        assert clean.passed
        assert clean.score == 1.0

        missing = get_grader("dict_match").evaluate_answer(
            {"obj": {"e": {"good": 1}}}, config
        )
        assert not missing.passed
        assert missing.score == 0.0
        assert missing.metrics["hard_fail_triggered"] == ["e.bad"]


class TestAllOfStrictBinary:
    REQUIRED_CHILDREN = [
        {
            "name": "c1",
            "predicate": {"op": "equals", "arg": 1},
            "role": "gate",
            "answer_field": "a",
        },
        {
            "name": "c2",
            "predicate": {"op": "equals", "arg": 2},
            "role": "gate",
            "answer_field": "b",
        },
    ]

    def test_one_failed_child_zeroes_composite(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "b": 999}, {"children": self.REQUIRED_CHILDREN}
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["scoring_passed"] == 1
        assert result.metrics["failed_children"] == ["c2"]

    def test_all_children_pass_for_full_binary_credit(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "b": 2},
            {"pass_rule": "all", "children": self.REQUIRED_CHILDREN},
        )

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics["type"] == "all_of"

    @pytest.mark.parametrize("grader_type", ["all_of", "average_of"])
    def test_duplicate_child_names_are_configuration_errors(
        self, grader_type: str
    ) -> None:
        children = [
            {
                "name": "same",
                "role": "gate",
                "answer_field": field,
                "predicate": {"op": "equals", "arg": 1},
            }
            for field in ("a", "b")
        ]
        result = get_grader(grader_type).evaluate_answer(
            {"a": 1, "b": 1}, {"children": children}
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_partial_inner_score_is_diagnostic_only(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 1.0, "b": 99.0},
            {
                "children": [
                    {
                        "type": "numeric_range",
                        "config": {
                            "ground_truth": {"a": 1.0, "b": 2.0},
                            "ranges": {
                                "a": {"min": 0.5, "max": 1.5},
                                "b": {"min": 1.5, "max": 2.5},
                            },
                        },
                    }
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["scoring_total_score"] == 0.5
        assert result.field_scores == {"children[0].numeric_range": 0.5}

    def test_child_pass_controls_binary_result_not_child_score(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"quality": "partial"},
            {
                "children": [
                    {
                        "name": "quality",
                        "role": "gate",
                        "answer_field": "quality",
                        "threshold": 1.0,
                        "predicate": {
                            "op": "weighted_label",
                            "table": {"partial": 1.0, "best": 2.0},
                            "default": 0.0,
                        },
                    }
                ]
            },
        )

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics["scoring_total_score"] == 1.0
        assert result.metrics["score_denominator"] == 2.0

    @pytest.mark.parametrize(
        "extra_config",
        [
            {"pass_rule": "min_passing", "min_passing_children": 1},
            {"pass_rule": "score_threshold", "score_threshold": 1.0},
            {"min_passing_children": 1},
        ],
    )
    def test_partial_pass_rules_are_configuration_errors(
        self, extra_config: dict
    ) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "b": 2},
            {**extra_config, "children": self.REQUIRED_CHILDREN},
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_additive_predicate_child_is_invalid(self) -> None:
        child = {**self.REQUIRED_CHILDREN[0], "role": "additive"}
        result = get_grader("all_of").evaluate_answer({"a": 1}, {"children": [child]})

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_typed_child_outer_role_is_invalid(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"answer": "A"},
            {
                "children": [
                    {
                        "type": "multiple_choice",
                        "role": "gate",
                        "config": {"correct_answer": "A"},
                    }
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_nested_failure_zeroes_outer_composite(self) -> None:
        nested = {"type": "all_of", "config": {"children": self.REQUIRED_CHILDREN}}
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "b": 999, "outer": True},
            {
                "children": [
                    nested,
                    {
                        "name": "outer",
                        "role": "gate",
                        "answer_field": "outer",
                        "predicate": {"op": "equals", "arg": True},
                    },
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_composite_alias_is_also_strict_binary(self) -> None:
        result = get_grader("composite").evaluate_answer(
            {"a": 1, "b": 999}, {"children": self.REQUIRED_CHILDREN}
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_unnamed_typed_children_keep_distinct_diagnostics(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "b": 2},
            {
                "children": [
                    {
                        "type": "numeric_range",
                        "config": {
                            "ground_truth": {"a": 1},
                            "ranges": {"a": {"min": 0, "max": 2}},
                        },
                    },
                    {
                        "type": "numeric_range",
                        "config": {
                            "ground_truth": {"b": 2},
                            "ranges": {"b": {"min": 1, "max": 3}},
                        },
                    },
                ]
            },
        )

        assert result.field_scores == {
            "children[0].numeric_range": 1.0,
            "children[1].numeric_range": 1.0,
        }


class TestAverageOfPartialCredit:
    TYPED_CHILDREN = [
        {
            "type": "numeric_range",
            "config": {
                "ground_truth": {field: expected},
                "ranges": {field: {"min": expected - 0.5, "max": expected + 0.5}},
            },
        }
        for field, expected in (("a", 1), ("b", 2), ("c", 3), ("d", 4))
    ]
    THREE_OF_FOUR = {"a": 1, "b": 2, "c": 3, "d": 999}

    def test_min_passing_three_of_four_keeps_partial_credit(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            self.THREE_OF_FOUR,
            {
                "pass_rule": "min_passing",
                "min_passing_children": 3,
                "children": self.TYPED_CHILDREN,
            },
        )

        assert result.passed is True
        assert result.score == pytest.approx(0.75)
        assert result.metrics["type"] == "average_of"
        assert result.metrics["scoring_passed"] == 3
        assert result.metrics["failed_children"] == ["children[3].numeric_range"]
        assert result.field_scores == {
            "children[0].numeric_range": 1.0,
            "children[1].numeric_range": 1.0,
            "children[2].numeric_range": 1.0,
            "children[3].numeric_range": 0.0,
        }

    def test_failed_score_threshold_keeps_normalized_score(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            self.THREE_OF_FOUR,
            {
                "pass_rule": "score_threshold",
                "score_threshold": 3.5,
                "children": self.TYPED_CHILDREN,
            },
        )

        assert result.passed is False
        assert result.metrics["scoring_total_score"] == 3.0
        assert result.metrics["score_denominator"] == 4.0
        assert result.score == pytest.approx(0.75)

    def test_default_all_rule_keeps_score_when_binary_pass_is_false(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            self.THREE_OF_FOUR, {"children": self.TYPED_CHILDREN}
        )

        assert result.passed is False
        assert result.score == pytest.approx(0.75)

    def test_hard_fail_veto_zeroes_partial_credit(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {**self.THREE_OF_FOUR, "cheated": True},
            {
                "pass_rule": "min_passing",
                "min_passing_children": 3,
                "children": [
                    *self.TYPED_CHILDREN,
                    {
                        "name": "forbidden shortcut",
                        "role": "hard_fail",
                        "answer_field": "cheated",
                        "predicate": {"op": "equals", "arg": True},
                    },
                ],
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["hard_fail_triggered"] == ["forbidden shortcut"]

    def test_unavailable_hard_fail_zeroes_partial_credit(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"a": 1, "flag": True},
            {
                "children": [
                    self.TYPED_CHILDREN[0],
                    {
                        "name": "broken veto",
                        "role": "hard_fail",
                        "answer_field": "flag",
                        "predicate": {"op": "not_a_real_op"},
                    },
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["hard_fail_unavailable"] == ["broken veto"]

    def test_nested_average_can_satisfy_strict_all_of(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {**self.THREE_OF_FOUR, "reported": True},
            {
                "children": [
                    {
                        "type": "average_of",
                        "config": {
                            "pass_rule": "min_passing",
                            "min_passing_children": 3,
                            "children": self.TYPED_CHILDREN,
                        },
                    },
                    {
                        "name": "reported",
                        "role": "gate",
                        "answer_field": "reported",
                        "predicate": {"op": "equals", "arg": True},
                    },
                ]
            },
        )

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics["scoring_total_score"] == pytest.approx(1.75)

    def test_nested_hard_fail_bubbles_into_strict_all_of(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "cheated": True},
            {
                "children": [
                    {
                        "type": "average_of",
                        "config": {
                            "children": [
                                self.TYPED_CHILDREN[0],
                                {
                                    "name": "forbidden shortcut",
                                    "role": "hard_fail",
                                    "answer_field": "cheated",
                                    "predicate": {"op": "equals", "arg": True},
                                },
                            ]
                        },
                    }
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["hard_fail_triggered"] == [
            {
                "child": "children[0].average_of",
                "children": ["forbidden shortcut"],
            }
        ]

    def test_additive_predicate_leaf_uses_its_declared_score_max(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"quality": "partial"},
            {
                "children": [
                    {
                        "type": "predicate_leaf",
                        "config": {
                            "name": "quality",
                            "role": "additive",
                            "answer_field": "quality",
                            "predicate": {
                                "op": "weighted_label",
                                "table": {"partial": 1.0, "best": 2.0},
                                "default": 0.0,
                            },
                        },
                    }
                ]
            },
        )

        assert result.passed is True
        assert result.score == pytest.approx(0.5)
        assert result.metrics["score_denominator"] == 2.0

    @pytest.mark.parametrize(
        "config",
        [
            {"children": []},
            {
                "children": [
                    {
                        "role": "hard_fail",
                        "predicate": {"op": "equals", "arg": True},
                    }
                ]
            },
            {
                "pass_rule": "min_passing",
                "children": TYPED_CHILDREN,
            },
            {
                "pass_rule": "all",
                "min_passing_children": 1,
                "children": TYPED_CHILDREN,
            },
            {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "score_threshold": 1.0,
                "children": TYPED_CHILDREN,
            },
            {
                "pass_rule": "score_threshold",
                "score_threshold": float("nan"),
                "children": TYPED_CHILDREN,
            },
            {
                "pass_rule": "score_threshold",
                "score_threshold": -1.0,
                "children": TYPED_CHILDREN,
            },
            {
                "children": [
                    {
                        "type": "numeric_range",
                        "role": "gate",
                        "config": TYPED_CHILDREN[0]["config"],
                    }
                ]
            },
            {
                "children": [
                    {
                        "role": "score",
                        "predicate": {"op": "equals", "arg": 1},
                        "answer_field": "a",
                    }
                ]
            },
        ],
    )
    def test_invalid_configurations_fail_closed(self, config: dict) -> None:
        result = get_grader("average_of").evaluate_answer({"a": 1}, config)

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_child_configuration_error_zeroes_other_credit(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"a": 1, "answer": "A"},
            {
                "children": [
                    self.TYPED_CHILDREN[0],
                    {"type": "multiple_choice", "config": {}},
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["misconfigured_children"] == [
            "children[1].multiple_choice"
        ]

    def test_invalid_predicate_zeroes_other_credit(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"a": 1, "quality": "good"},
            {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "children": [
                    self.TYPED_CHILDREN[0],
                    {
                        "name": "broken predicate",
                        "role": "gate",
                        "answer_field": "quality",
                        "predicate": {"op": "not_a_real_op"},
                    },
                ],
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]
        assert result.metrics["misconfigured_children"] == ["broken predicate"]

    def test_child_system_error_zeroes_other_credit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class RaisingGrader(BinaryGrader):
            def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
                raise ValueError("synthetic grader failure")

        monkeypatch.setitem(GRADER_REGISTRY, "raising_for_test", RaisingGrader)
        result = get_grader("average_of").evaluate_answer(
            {"a": 1},
            {
                "children": [
                    self.TYPED_CHILDREN[0],
                    {"type": "raising_for_test", "config": {}},
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["grader_system_error"] is True
        assert result.metrics["system_error_children"] == [
            "children[1].raising_for_test"
        ]

    @pytest.mark.parametrize("grader_type", ["all_of", "average_of"])
    def test_malformed_child_metrics_are_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, grader_type: str
    ) -> None:
        class MalformedMetricsGrader(BinaryGrader):
            def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
                return GraderResult(
                    passed=True,
                    metrics=None,  # type: ignore[arg-type]
                    reasoning="synthetic malformed result",
                    agent_answer=agent_answer,
                    score=1.0,
                )

        monkeypatch.setitem(
            GRADER_REGISTRY, "malformed_metrics_for_test", MalformedMetricsGrader
        )
        result = get_grader(grader_type).evaluate_answer(
            {"answer": "A"},
            {"children": [{"type": "malformed_metrics_for_test", "config": {}}]},
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]
        assert result.metrics["misconfigured_children"] == [
            "children[0].malformed_metrics_for_test"
        ]

    @pytest.mark.parametrize("grader_type", ["all_of", "average_of"])
    def test_child_grader_error_metric_is_a_system_error(
        self, monkeypatch: pytest.MonkeyPatch, grader_type: str
    ) -> None:
        class ErrorMetricGrader(BinaryGrader):
            def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
                return GraderResult(
                    passed=True,
                    metrics={"grader_error": "synthetic failure"},
                    reasoning="synthetic error result",
                    agent_answer=agent_answer,
                    score=1.0,
                )

        monkeypatch.setitem(GRADER_REGISTRY, "error_metric_for_test", ErrorMetricGrader)
        result = get_grader(grader_type).evaluate_answer(
            {"answer": "A"},
            {"children": [{"type": "error_metric_for_test", "config": {}}]},
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["grader_system_error"] is True
        assert result.metrics["system_error_children"] == [
            "children[0].error_metric_for_test"
        ]

    def test_nested_all_of_configuration_error_cannot_be_outvoted(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"answer": "A", "value": 1},
            {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "children": [
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "A"},
                    },
                    {
                        "type": "all_of",
                        "config": {
                            "children": [
                                {
                                    "name": "broken predicate",
                                    "role": "gate",
                                    "answer_field": "value",
                                    "predicate": {"op": "not_a_real_op"},
                                }
                            ]
                        },
                    },
                ],
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]
        assert result.metrics["misconfigured_children"] == ["children[1].all_of"]

    def test_malformed_predicate_answer_remains_an_ordinary_miss(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"labels": 123, "answer": "A"},
            {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "children": [
                    {
                        "name": "label overlap",
                        "role": "gate",
                        "answer_field": "labels",
                        "predicate": {"op": "f1", "expected": ["A"]},
                    },
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "A"},
                    },
                ],
            },
        )

        assert result.passed is True
        assert result.score == pytest.approx(0.5)
        assert "configuration_error" not in result.metrics
        assert "grader_system_error" not in result.metrics

    def test_invalid_predicate_answer_field_cannot_be_outvoted(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"answer": "A"},
            {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "children": [
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "A"},
                    },
                    {
                        "name": "broken binding",
                        "role": "gate",
                        "answer_field": [],
                        "predicate": {"op": "equals", "arg": 1},
                    },
                ],
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]
        assert result.metrics["misconfigured_children"] == ["broken binding"]

    @pytest.mark.parametrize("grader_type", ["all_of", "average_of"])
    def test_misconfigured_hard_fail_is_unavailable_not_triggered(
        self, grader_type: str
    ) -> None:
        result = get_grader(grader_type).evaluate_answer(
            {"answer": "A", "shortcut": True},
            {
                "children": [
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "A"},
                    },
                    {
                        "name": "broken veto",
                        "role": "hard_fail",
                        "answer_field": "shortcut",
                        "predicate": {"op": "not_a_real_op"},
                    },
                ]
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]
        assert result.metrics["misconfigured_children"] == ["broken veto"]
        assert result.metrics["hard_fail_unavailable"] == ["broken veto"]
        assert result.metrics["hard_fail_triggered"] == []

    @pytest.mark.parametrize("grader_type", ["all_of", "average_of"])
    def test_non_object_composite_config_fails_cleanly(self, grader_type: str) -> None:
        result = get_grader(grader_type).evaluate_answer(
            {"answer": "A"},
            [],  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["configuration_error"]

    def test_child_configuration_error_cannot_be_outvoted(self) -> None:
        result = get_grader("average_of").evaluate_answer(
            {"answer": "A"},
            {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "children": [
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "A"},
                    },
                    {"type": "numeric_range", "config": {}},
                ],
            },
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["scoring_count"] == 1
        assert result.metrics["misconfigured_children"] == ["children[1].numeric_range"]

    def test_zero_capacity_scoring_child_is_a_configuration_error(self) -> None:
        average = {
            "type": "average_of",
            "config": {
                "children": [
                    {
                        "name": "zero-capacity score",
                        "role": "additive",
                        "answer_field": "quality",
                        "predicate": {
                            "op": "weighted_label",
                            "table": {"none": 0.0},
                            "default": 0.0,
                        },
                    }
                ]
            },
        }

        inner = get_grader("average_of").evaluate_answer(
            {"quality": "none"}, average["config"]
        )
        outer = get_grader("all_of").evaluate_answer(
            {"quality": "none"}, {"children": [average]}
        )

        assert inner.passed is False
        assert inner.score == 0.0
        assert inner.metrics["configuration_error"]
        assert outer.passed is False
        assert outer.score == 0.0

    def test_gate_only_list_match_cannot_satisfy_average_min_passing(self) -> None:
        gate_only_list_match = {
            "type": "list_match",
            "config": {
                "answer_field": "rows",
                "match_key": "id",
                "tuple_pass_min": 1,
                "ground_truth": [
                    {
                        "id": "expected",
                        "fields": {
                            "value": {
                                "role": "gate",
                                "predicate": {"op": "equals", "arg": 1},
                            }
                        },
                    }
                ],
            },
        }
        average = {
            "type": "average_of",
            "config": {
                "pass_rule": "min_passing",
                "min_passing_children": 1,
                "children": [
                    gate_only_list_match,
                    {
                        "type": "multiple_choice",
                        "config": {"correct_answer": "B"},
                    },
                ],
            },
        }
        answer = {
            "rows": [{"id": "expected", "value": 1}],
            "answer": "A",
        }

        strict_gate = get_grader("all_of").evaluate_answer(
            answer, {"children": [gate_only_list_match]}
        )
        inner = get_grader("average_of").evaluate_answer(answer, average["config"])
        outer = get_grader("all_of").evaluate_answer(answer, {"children": [average]})

        assert strict_gate.passed is True
        assert strict_gate.score == 1.0
        assert strict_gate.metrics["score_denominator"] == 0.0
        assert "configuration_error" not in strict_gate.metrics
        assert inner.passed is False
        assert inner.score == 0.0
        assert inner.metrics["configuration_error"]
        assert inner.metrics["misconfigured_children"] == ["children[0].list_match"]
        assert outer.passed is False
        assert outer.score == 0.0
        assert outer.metrics["configuration_error"]


class TestDictMatchOptionalKeys:
    CONFIG = {
        "answer_field": "obj",
        "all_keys_required": False,
        "ground_truth": {
            "k1": {"predicate": {"op": "equals", "arg": 1}, "role": "gate"},
            "k2": {"predicate": {"op": "equals", "arg": 2}, "role": "gate"},
        },
    }

    def test_omitting_every_key_is_not_full_credit(self) -> None:
        result = get_grader("dict_match").evaluate_answer({"obj": {}}, self.CONFIG)
        assert not result.passed
        assert result.score == 0.0

    def test_omitting_a_key_does_not_beat_answering_it_wrong(self) -> None:
        omitted = get_grader("dict_match").evaluate_answer(
            {"obj": {"k1": 1}}, self.CONFIG
        )
        wrong = get_grader("dict_match").evaluate_answer(
            {"obj": {"k1": 1, "k2": 999}}, self.CONFIG
        )
        # Answering wrongly must never score below simply omitting the key, and
        # omitting must not be worth more either -- the two are the same failure.
        assert wrong.score == omitted.score
        assert wrong.score < 1.0


# Predicate shapes that are satisfied by a bare `None`, so a field the agent
# never supplied would earn credit if absence were passed to the predicate.
VACUOUS_PREDICATES: dict[str, dict] = {
    "not": {"op": "not", "arg": {"op": "equals", "arg": "bad"}},
    "every": {"op": "every", "path": "$[*]", "body": {"op": "equals", "arg": 1}},
    "none": {"op": "none", "path": "$[*]", "body": {"op": "equals", "arg": 1}},
    "weighted_label_positive_default": {
        "op": "weighted_label",
        "table": {"good": 1.0},
        "default": 0.5,
    },
}


def _leaf(predicate: dict, **overrides: object) -> dict:
    return {
        "predicate": predicate,
        "role": "gate",
        "answer_field": "x",
        "threshold": 0.4,
        **overrides,
    }


class TestAbsentFieldCannotSatisfyAPredicate:
    """An omitted field is graded as a failure, never handed to the predicate."""

    @pytest.mark.parametrize("shape", sorted(VACUOUS_PREDICATES))
    def test_predicate_leaf(self, shape: str) -> None:
        result = get_grader("predicate_leaf").evaluate_answer(
            {}, _leaf(VACUOUS_PREDICATES[shape])
        )
        assert not result.passed
        assert result.score == 0.0
        assert result.metrics["missing_answer_field"] is True

    @pytest.mark.parametrize("shape", sorted(VACUOUS_PREDICATES))
    def test_all_of_child(self, shape: str) -> None:
        config = {"children": [_leaf(VACUOUS_PREDICATES[shape], name=shape)]}
        result = get_grader("all_of").evaluate_answer({}, config)
        assert not result.passed
        assert result.score == 0.0

    @pytest.mark.parametrize("shape", sorted(VACUOUS_PREDICATES))
    def test_list_match_field(self, shape: str) -> None:
        config = {
            "answer_field": "rows",
            "match_key": "id",
            "tuple_pass_min": 1,
            "ground_truth": [
                {
                    "id": "a",
                    "fields": {
                        "v": {
                            "predicate": VACUOUS_PREDICATES[shape],
                            "role": "gate",
                            "threshold": 0.4,
                        }
                    },
                }
            ],
        }
        # The row is present and matches the ground truth, but omits the graded
        # field, so there is nothing to pay for.
        result = get_grader("list_match").evaluate_answer(
            {"rows": [{"id": "a"}]}, config
        )
        assert not result.passed
        assert result.score == 0.0

    @pytest.mark.parametrize("shape", sorted(VACUOUS_PREDICATES))
    def test_dict_match_field(self, shape: str) -> None:
        config = {
            "answer_field": "obj",
            "ground_truth": {
                "e": {
                    "fields": {
                        "v": {
                            "predicate": VACUOUS_PREDICATES[shape],
                            "role": "gate",
                            "threshold": 0.4,
                        }
                    }
                }
            },
        }
        result = get_grader("dict_match").evaluate_answer({"obj": {"e": {}}}, config)
        assert not result.passed
        assert result.score == 0.0

    def test_an_explicit_null_is_still_a_real_answer(self) -> None:
        """`null` was submitted, so it is graded rather than treated as absent."""
        config = _leaf({"op": "equals", "arg": None})
        supplied = get_grader("predicate_leaf").evaluate_answer({"x": None}, config)
        assert supplied.passed
        assert supplied.score == 1.0

        absent = get_grader("predicate_leaf").evaluate_answer({}, config)
        assert not absent.passed
        assert absent.score == 0.0


class TestQuantifiersAreNotVacuouslyTrue:
    EVERY = _leaf({"op": "every", "path": "$[*]", "body": {"op": "equals", "arg": 1}})
    NONE = _leaf({"op": "none", "path": "$[*]", "body": {"op": "equals", "arg": 9}})

    @pytest.mark.parametrize("config", [EVERY, NONE])
    def test_empty_list_earns_nothing(self, config: dict) -> None:
        result = get_grader("predicate_leaf").evaluate_answer({"x": []}, config)
        assert not result.passed
        assert result.score == 0.0

    def test_every_still_passes_over_a_non_empty_list(self) -> None:
        result = get_grader("predicate_leaf").evaluate_answer({"x": [1, 1]}, self.EVERY)
        assert result.passed
        assert result.score == 1.0

    def test_every_still_fails_when_an_element_is_wrong(self) -> None:
        result = get_grader("predicate_leaf").evaluate_answer({"x": [1, 2]}, self.EVERY)
        assert not result.passed
        assert result.score == 0.0

    def test_none_still_passes_over_a_clean_non_empty_list(self) -> None:
        result = get_grader("predicate_leaf").evaluate_answer({"x": [1, 2]}, self.NONE)
        assert result.passed
        assert result.score == 1.0


class TestRootHardFailCannotBeCollectedByAbstaining:
    """A root `hard_fail` leaf pays 1.0 when the veto does not fire.

    That is a legitimate "did the agent avoid X" eval, but the reward has to
    require actually answering: otherwise an empty submission scores the same as
    genuinely avoiding the vetoed behaviour.
    """

    CONFIG = {
        "predicate": {"op": "equals", "arg": True},
        "role": "hard_fail",
        "answer_field": "cheated",
    }

    def test_omitting_the_field_earns_nothing(self) -> None:
        result = get_grader("predicate_leaf").evaluate_answer({}, self.CONFIG)
        assert not result.passed
        assert result.score == 0.0

    def test_answering_cleanly_still_earns_full_credit(self) -> None:
        result = get_grader("predicate_leaf").evaluate_answer(
            {"cheated": False}, self.CONFIG
        )
        assert result.passed
        assert result.score == 1.0

    def test_triggering_the_veto_still_scores_zero(self) -> None:
        result = get_grader("predicate_leaf").evaluate_answer(
            {"cheated": True}, self.CONFIG
        )
        assert not result.passed
        assert result.score == 0.0

    def test_a_composite_bound_hard_fail_field_is_required(self) -> None:
        config = {
            "children": [
                {
                    "name": "scoring",
                    "predicate": {"op": "equals", "arg": 1},
                    "role": "gate",
                    "answer_field": "x",
                },
                {
                    "name": "veto",
                    "predicate": {"op": "equals", "arg": True},
                    "role": "hard_fail",
                    "answer_field": "cheated",
                },
            ]
        }
        result = get_grader("all_of").evaluate_answer({"x": 1}, config)
        assert not result.passed
        assert result.score == 0.0
        assert result.metrics["hard_fail_triggered"] == ["veto"]

    def test_a_composite_whole_answer_veto_can_be_optional(self) -> None:
        config = {
            "children": [
                {
                    "name": "scoring",
                    "predicate": {"op": "equals", "arg": 1},
                    "role": "gate",
                    "answer_field": "x",
                },
                {
                    "name": "optional-cheating-veto",
                    "predicate": {
                        "op": "field",
                        "name": "cheated",
                        "body": {"op": "equals", "arg": True},
                    },
                    "role": "hard_fail",
                },
            ]
        }

        clean = get_grader("all_of").evaluate_answer({"x": 1}, config)
        assert clean.passed
        assert clean.score == 1.0
        assert clean.metrics["hard_fail_triggered"] == []

        tripped = get_grader("all_of").evaluate_answer(
            {"x": 1, "cheated": True}, config
        )
        assert not tripped.passed
        assert tripped.score == 0.0
        assert tripped.metrics["hard_fail_triggered"] == ["optional-cheating-veto"]


class TestListMatchEmptyAnswerDoesNotPass:
    CONFIG = {
        "answer_field": "rows",
        "match_key": "id",
        "tuple_pass_min": 0,
        "additive_score_min": 0,
        "ground_truth": [
            {
                "id": "a",
                "fields": {
                    "v": {"predicate": {"op": "equals", "arg": 1}, "role": "additive"}
                },
            }
        ],
    }

    def test_empty_list_does_not_report_a_pass(self) -> None:
        # Permissive minimums otherwise mark a non-answer as passing, and that
        # verdict propagates when list_match is nested as an `all_of` child.
        result = get_grader("list_match").evaluate_answer({"rows": []}, self.CONFIG)
        assert not result.passed
        assert result.score == 0.0
        assert result.metrics["composite_error"]

    def test_unmatched_rows_do_not_report_a_pass(self) -> None:
        result = get_grader("list_match").evaluate_answer(
            {"rows": [{"id": "not-in-gt", "v": 1}]}, self.CONFIG
        )
        assert not result.passed
        assert result.score == 0.0

    def test_a_matching_row_still_passes(self) -> None:
        result = get_grader("list_match").evaluate_answer(
            {"rows": [{"id": "a", "v": 1}]}, self.CONFIG
        )
        assert result.passed
        assert result.score == 1.0


class TestPartialAnswersScoreTheSameSkippedOrWrong:
    """Per grader with partial credit: skip N fields == answer N fields wrongly."""

    def test_numeric_tolerance(self) -> None:
        config = {
            "ground_truth": {"a": 1.0, "b": 2.0, "c": 3.0},
            "tolerances": {k: {"type": "absolute", "value": 0.1} for k in "abc"},
        }
        grader = get_grader("numeric_tolerance")
        skipped = grader.evaluate_answer({"a": 1.0}, config)
        wrong = grader.evaluate_answer({"a": 1.0, "b": 99.0, "c": 99.0}, config)
        assert skipped.score == wrong.score == pytest.approx(1 / 3)

    def test_numeric_range(self) -> None:
        config = {
            "ground_truth": {"a": 1.0, "b": 2.0, "c": 3.0},
            "ranges": {k: {"min": 0.0, "max": 10.0} for k in "abc"},
        }
        grader = get_grader("numeric_range")
        skipped = grader.evaluate_answer({"a": 1.0}, config)
        wrong = grader.evaluate_answer({"a": 1.0, "b": 99.0, "c": 99.0}, config)
        assert skipped.score == wrong.score == pytest.approx(1 / 3)

    def test_dict_match(self) -> None:
        config = {
            "answer_field": "obj",
            "ground_truth": {
                k: {"predicate": {"op": "equals", "arg": v}, "role": "gate"}
                for k, v in (("a", 1), ("b", 2), ("c", 3))
            },
        }
        grader = get_grader("dict_match")
        skipped = grader.evaluate_answer({"obj": {"a": 1}}, config)
        wrong = grader.evaluate_answer({"obj": {"a": 1, "b": 9, "c": 9}}, config)
        assert skipped.score == wrong.score == pytest.approx(1 / 3)

    def test_list_match(self) -> None:
        config = {
            "answer_field": "rows",
            "match_key": "id",
            "ground_truth": [
                {
                    "id": key,
                    "fields": {
                        "v": {
                            "predicate": {"op": "equals", "arg": 1},
                            "role": "additive",
                        }
                    },
                }
                for key in ("a", "b", "c")
            ],
        }
        grader = get_grader("list_match")
        skipped = grader.evaluate_answer({"rows": [{"id": "a", "v": 1}]}, config)
        wrong = grader.evaluate_answer(
            {
                "rows": [
                    {"id": "a", "v": 1},
                    {"id": "b", "v": 9},
                    {"id": "c", "v": 9},
                ]
            },
            config,
        )
        assert skipped.score == wrong.score == pytest.approx(1 / 3)

    def test_all_of_omission_and_wrong_answer_both_score_zero(self) -> None:
        config = {
            "children": [
                {
                    "name": name,
                    "predicate": {"op": "equals", "arg": 1},
                    "role": "gate",
                    "answer_field": name,
                }
                for name in ("a", "b", "c")
            ],
        }
        grader = get_grader("all_of")
        skipped = grader.evaluate_answer({"a": 1}, config)
        wrong = grader.evaluate_answer({"a": 1, "b": 9, "c": 9}, config)
        assert skipped.score == wrong.score == 0.0

    def test_longest_subsequence(self) -> None:
        config = {"answer_field": "seq", "ground_truth": [["a"], ["b"], ["c"]]}
        grader = get_grader("longest_subsequence")
        skipped = grader.evaluate_answer({"seq": [["a"]]}, config)
        wrong = grader.evaluate_answer({"seq": [["a"], ["x"], ["y"]]}, config)
        assert skipped.score == wrong.score == pytest.approx(1 / 3)
