import pytest

from latch_eval_tools.graders.composite import ListMatchGrader
from latch_eval_tools.graders.label_set import LabelSetJaccardGrader
from latch_eval_tools.graders.marker_gene import MarkerGenePrecisionRecallGrader


def test_label_set_expected_count_rejects_short_high_overlap_answer() -> None:
    result = LabelSetJaccardGrader().evaluate_answer(
        {"labels": ["A", "B", "C"]},
        {
            "answer_field": "labels",
            "ground_truth_labels": ["A", "B", "C", "D"],
            "expected_count": 4,
            "scoring": {"pass_threshold": 0.70},
        },
    )

    assert result.metrics["jaccard_index"] == pytest.approx(0.75)
    assert not result.metrics["cardinality_pass"]
    assert not result.passed
    assert result.score == 0.0


def test_label_set_expected_count_requires_unique_items() -> None:
    result = LabelSetJaccardGrader().evaluate_answer(
        {"labels": ["A", "B", "C", "C"]},
        {
            "answer_field": "labels",
            "ground_truth_labels": ["A", "B", "C"],
            "expected_count": 4,
            "scoring": {"pass_threshold": 0.70},
        },
    )

    assert result.metrics["submitted_count"] == 4
    assert result.metrics["unique_count"] == 3
    assert not result.passed


def test_label_set_without_expected_count_preserves_variable_length_behavior() -> None:
    result = LabelSetJaccardGrader().evaluate_answer(
        {"labels": ["A", "B", "C"]},
        {
            "answer_field": "labels",
            "ground_truth_labels": ["A", "B", "C", "D"],
            "scoring": {"pass_threshold": 0.70},
        },
    )

    assert result.passed
    assert result.score == 1.0
    assert result.metrics["expected_count"] is None
    assert result.metrics["predicted_count"] == 3


@pytest.mark.parametrize("expected_count", [True, -1, 2.5, "10"])
def test_label_set_invalid_expected_count_fails_closed(
    expected_count: object,
) -> None:
    result = LabelSetJaccardGrader().evaluate_answer(
        {"labels": ["A"]},
        {
            "answer_field": "labels",
            "ground_truth_labels": ["A"],
            "expected_count": expected_count,
        },
    )

    assert not result.passed
    assert result.score == 0.0
    assert result.metrics["configuration_error"]


def _marker_config(*, expected_count: object | None = None) -> dict:
    config = {
        "answer_field": "genes",
        "canonical_markers": [f"G{i}" for i in range(10)],
        "scoring": {"pass_thresholds": {"precision_at_k": 0.1, "recall_at_k": 0.1}},
    }
    if expected_count is not None:
        config["expected_count"] = expected_count
    return config


def test_marker_gene_expected_count_fixes_k_and_rejects_short_answer() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": ["G0"]}, _marker_config(expected_count=10)
    )

    assert result.metrics["k"] == 10
    assert result.metrics["precision_at_k"] == pytest.approx(0.1)
    assert result.metrics["recall_at_k"] == pytest.approx(0.1)
    assert not result.metrics["cardinality_pass"]
    assert not result.passed


def test_marker_gene_expected_count_accepts_exact_unique_answer() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": [f"G{i}" for i in range(10)]},
        _marker_config(expected_count=10),
    )

    assert result.metrics["k"] == 10
    assert result.metrics["cardinality_pass"]
    assert result.passed
    assert result.score == 1.0


def test_marker_gene_expected_count_rejects_case_insensitive_duplicates() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": ["G0", "g0", *[f"G{i}" for i in range(1, 9)]]},
        _marker_config(expected_count=10),
    )

    assert result.metrics["submitted_count"] == 10
    assert result.metrics["unique_count"] == 9
    assert not result.passed


def test_marker_gene_does_not_fall_back_to_unrelated_list_field() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"unrelated": [f"G{i}" for i in range(10)]},
        _marker_config(expected_count=10),
    )

    assert not result.passed
    assert result.score == 0.0
    assert "missing required field: genes" in result.reasoning.lower()


def test_marker_gene_expected_count_failure_counts_as_failed_celltype() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": {"T": ["CD3D"], "B": ["MS4A1", "CD79A"]}},
        {
            "answer_field": "genes",
            "canonical_markers": {
                "T": ["CD3D", "TRAC"],
                "B": ["MS4A1", "CD79A"],
            },
            "expected_count": 2,
            "scoring": {
                "pass_thresholds": {
                    "min_recall_per_celltype": 0.5,
                    "min_celltypes_passing": 1,
                }
            },
        },
    )

    assert not result.metrics["per_celltype"]["T"]["cardinality_pass"]
    assert result.metrics["per_celltype"]["B"]["cardinality_pass"]
    assert result.metrics["celltypes_passing"] == 1
    assert result.passed
    assert result.score == 1.0


def test_marker_gene_without_expected_count_honors_min_celltypes_passing() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": {"T": ["CD3D"], "B": "not-a-list"}},
        {
            "answer_field": "genes",
            "canonical_markers": {
                "T": ["CD3D"],
                "B": ["MS4A1"],
            },
            "scoring": {
                "pass_thresholds": {
                    "min_recall_per_celltype": 1.0,
                    "min_celltypes_passing": 1,
                }
            },
        },
    )

    assert result.metrics["per_celltype"]["T"]["pass"] is True
    assert result.metrics["per_celltype"]["B"]["pass"] is False
    assert result.passed is True
    assert result.score == 1.0


def test_marker_gene_rejects_empty_per_celltype_ground_truth() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": {}},
        {
            "answer_field": "genes",
            "canonical_markers": {},
            "scoring": {"pass_thresholds": {}},
        },
    )

    assert not result.passed
    assert result.score == 0.0
    assert result.metrics["configuration_error"]


@pytest.mark.parametrize("canonical_genes", [[], "CD3D"])
def test_marker_gene_rejects_invalid_per_celltype_ground_truth(
    canonical_genes: object,
) -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": {"T": ["CD3D"]}},
        {
            "answer_field": "genes",
            "canonical_markers": {"T": canonical_genes},
            "scoring": {"pass_thresholds": {}},
        },
    )

    assert not result.passed
    assert result.score == 0.0
    assert result.metrics["configuration_error"]


def test_marker_gene_preserves_legacy_flat_wrapper_mode() -> None:
    result = MarkerGenePrecisionRecallGrader().evaluate_answer(
        {"genes": ["CD3D"]},
        {
            "answer_field": "genes",
            "canonical_markers": {"genes": ["CD3D"]},
            "scoring": {"pass_thresholds": {"precision_at_k": 1.0, "recall_at_k": 1.0}},
        },
    )

    assert result.passed
    assert result.score == 1.0
    assert result.metrics["answer_field_used"] == "genes"


def _list_match_config(
    *, k: object | None = None, weights: list[float] | None = None
) -> dict:
    """``n`` ground-truth rows, each with one gate and one additive leaf."""
    weights = [1.0] * 9 if weights is None else weights
    config = {
        "answer_field": "rows",
        "match_key": "cell_type",
        "ground_truth": [
            {
                "cell_type": f"C{i}",
                "fields": {
                    "direction": {
                        "role": "gate",
                        "predicate": {"op": "equals", "arg": "decrease"},
                    },
                    "robustness": {
                        "role": "additive",
                        "predicate": {
                            "op": "weighted_label",
                            "table": {"robust": weight},
                            "default": 0.0,
                        },
                    },
                },
            }
            for i, weight in enumerate(weights)
        ],
    }
    if k is not None:
        config["k"] = k
    return config


def _rows(*indices: int) -> dict:
    """A fully correct submitted row for each named ground-truth index."""
    return {
        "rows": [
            {"cell_type": f"C{i}", "direction": "decrease", "robustness": "robust"}
            for i in indices
        ]
    }


def test_list_match_k_normalizes_against_k_rows_not_the_whole_gt_pool() -> None:
    """Answering the maximum k rows correctly is a full score, not k/len(gt)."""
    result = ListMatchGrader().evaluate_answer(_rows(0, 1, 2), _list_match_config(k=3))

    assert result.metrics["additive_score"] == pytest.approx(3.0)
    assert result.metrics["additive_score_denominator"] == pytest.approx(3.0)
    assert result.score == pytest.approx(1.0)


def test_list_match_k_keeps_partial_credit_proportional_within_k() -> None:
    answer = _rows(0, 1, 2)
    answer["rows"][2]["robustness"] = "not-a-listed-label"

    result = ListMatchGrader().evaluate_answer(answer, _list_match_config(k=3))

    assert result.metrics["additive_score"] == pytest.approx(2.0)
    assert result.score == pytest.approx(2 / 3)


def test_list_match_k_denominator_uses_the_highest_capacity_rows() -> None:
    """The reachable maximum is the richest k rows, not the first or cheapest k."""
    result = ListMatchGrader().evaluate_answer(
        _rows(0, 1), _list_match_config(k=2, weights=[1.0, 2.0, 3.0])
    )

    assert result.metrics["additive_score"] == pytest.approx(3.0)
    assert result.metrics["additive_score_denominator"] == pytest.approx(5.0)
    assert result.score == pytest.approx(0.6)


def test_list_match_without_k_still_normalizes_against_every_gt_row() -> None:
    result = ListMatchGrader().evaluate_answer(_rows(0, 1, 2), _list_match_config())

    assert result.metrics["additive_score_denominator"] == pytest.approx(9.0)
    assert result.score == pytest.approx(1 / 3)


def test_list_match_k_at_or_above_gt_size_leaves_the_denominator_whole() -> None:
    result = ListMatchGrader().evaluate_answer(
        _rows(*range(9)), _list_match_config(k=20)
    )

    assert result.metrics["additive_score_denominator"] == pytest.approx(9.0)
    assert result.score == pytest.approx(1.0)
