"""Regression tests for grader fail-closed behaviour.

Reward is read from ``GraderResult.score``, not ``passed`` (the Taiga harness
weights ``score`` at 1.0 and ``passed`` at 0.0). Any code path that returns
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

from latch_eval_tools.graders import GRADER_REGISTRY, GraderResult, get_grader

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
    "refusal_vocab": {"decision": "REFUSE"},
    "predicate_leaf": {"x": 1},
    "all_of": {"x": 1},
    "list_match": {"rows": [{"id": "a", "v": 1}]},
    "dict_match": {"obj": {"k": 1}},
    "longest_subsequence": {"seq": [["a"]]},
    "finished_file": {"finished_file_contents": "finished"},
}


def test_every_registered_grader_is_covered() -> None:
    """A new grader must be added here, so it cannot skip these guarantees."""
    assert set(GRADER_REGISTRY) == set(CONFIGS) == set(CORRECT_ANSWERS)


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
    assert GraderResult(passed=False, metrics={}, reasoning="", agent_answer={}).score == 0.0


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
            assert result.metrics.get("composite_error")

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


class TestAllOfPassRuleAllGatesScore:
    """`pass_rule="all"` must pay no partial credit unless every child passes."""

    CONFIG = {
        "pass_rule": "all",
        "children": [
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
        ],
    }

    def test_partial_pass_scores_zero(self) -> None:
        # One of two children passes: the mean-of-children would be 0.5, but a
        # strict AND gate must pay nothing until the composite passes.
        result = get_grader("all_of").evaluate_answer(
            {"a": 1, "b": 999}, self.CONFIG
        )
        assert not result.passed
        assert result.score == 0.0

    def test_zero_pass_scores_zero(self) -> None:
        result = get_grader("all_of").evaluate_answer(
            {"a": 999, "b": 999}, self.CONFIG
        )
        assert not result.passed
        assert result.score == 0.0

    def test_all_pass_scores_full(self) -> None:
        result = get_grader("all_of").evaluate_answer({"a": 1, "b": 2}, self.CONFIG)
        assert result.passed
        assert result.score == 1.0

    def test_pass_rule_all_is_the_default(self) -> None:
        config = {k: v for k, v in self.CONFIG.items() if k != "pass_rule"}
        result = get_grader("all_of").evaluate_answer({"a": 1, "b": 999}, config)
        assert not result.passed
        assert result.score == 0.0

    def test_score_threshold_still_pays_partial_credit(self) -> None:
        # Other pass rules are unaffected: they keep their mean-of-children score.
        config = {**self.CONFIG, "pass_rule": "score_threshold", "score_threshold": 5.0}
        result = get_grader("all_of").evaluate_answer({"a": 1, "b": 999}, config)
        assert not result.passed
        assert result.score == 0.5


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

    def test_all_of_score_threshold(self) -> None:
        config = {
            "pass_rule": "score_threshold",
            "score_threshold": 3.0,
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
        assert skipped.score == wrong.score == pytest.approx(1 / 3)

    def test_longest_subsequence(self) -> None:
        config = {"answer_field": "seq", "ground_truth": [["a"], ["b"], ["c"]]}
        grader = get_grader("longest_subsequence")
        skipped = grader.evaluate_answer({"seq": [["a"]]}, config)
        wrong = grader.evaluate_answer({"seq": [["a"], ["x"], ["y"]]}, config)
        assert skipped.score == wrong.score == pytest.approx(1 / 3)
