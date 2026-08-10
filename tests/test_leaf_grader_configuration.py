"""Behavioral contract for deterministic leaf-grader configuration failures."""

import math

import pytest

from latch_eval_tools.graders import get_grader

OVERFLOWING_INTEGER = int("9" * 400)

VALID_CONFIGS: dict[str, dict] = {
    "numeric_tolerance": {
        "ground_truth": {"x": 1.0},
        "tolerances": {"x": {"type": "absolute", "value": 0.1}},
    },
    "numeric_range": {
        "ground_truth": {"x": 1.0},
        "ranges": {"x": {"min": 0.0, "max": 2.0}},
    },
    "distribution_comparison": {
        "ground_truth": {
            "total_cells": 10,
            "cell_type_distribution": {"T": 50.0},
        }
    },
    "refusal_vocab": {
        "refusal_vocab": ["REFUSE"],
        "expected_decision": "REFUSE",
    },
    "longest_subsequence": {
        "answer_field": "sequence",
        "ground_truth": [["A"]],
    },
    "marker_gene_precision_recall": {
        "answer_field": "genes",
        "canonical_markers": ["CD3D"],
    },
    "marker_gene_separation": {},
    "spatial_adjacency": {},
    "label_set_jaccard": {
        "answer_field": "labels",
        "ground_truth_labels": ["T"],
    },
    "multiple_choice": {"correct_answer": "A"},
    "finished_file": {},
}


VALID_ANSWERS: dict[str, dict] = {
    "numeric_tolerance": {"x": 1.0},
    "numeric_range": {"x": 1.0},
    "distribution_comparison": {
        "total_cells": 10,
        "cell_type_distribution": {"T": 50.0},
    },
    "refusal_vocab": {"decision": "REFUSE"},
    "longest_subsequence": {"sequence": [["A"]]},
    "marker_gene_precision_recall": {"genes": ["CD3D"]},
    "marker_gene_separation": {
        "per_gene_stats": [{"gene": "CD3D", "auroc": 0.95}],
        "mean_auroc": 0.95,
    },
    "spatial_adjacency": {
        "median_ic_to_pc_um": 10.0,
        "p90_ic_to_pc_um": 50.0,
        "pct_ic_within_15um": 70.0,
        "pct_ic_mixed_within_55um": 70.0,
        "adjacency_pass": True,
    },
    "label_set_jaccard": {"labels": ["T"]},
    "multiple_choice": {"answer": "A"},
    "finished_file": {"finished_file_contents": "finished"},
}


@pytest.mark.parametrize(
    ("grader_type", "config"),
    [
        ("numeric_tolerance", {}),
        (
            "numeric_tolerance",
            {"ground_truth": {"x": 1.0}, "tolerances": {"x": []}},
        ),
        (
            "numeric_tolerance",
            {
                "ground_truth": {"x": 1.0},
                "tolerances": {"x": {"type": [], "value": 0.1}},
            },
        ),
        (
            "numeric_tolerance",
            {"ground_truth": {"x": OVERFLOWING_INTEGER}, "tolerances": {}},
        ),
        ("numeric_range", {}),
        (
            "numeric_range",
            {
                "ground_truth": {"x": 1.0},
                "ranges": {"x": {"min": 2.0, "max": 0.0}},
            },
        ),
        ("distribution_comparison", {}),
        (
            "distribution_comparison",
            {
                "ground_truth": {"cell_type_distribution": {"T": 50.0}},
                "tolerances": {"total_cells": None},
            },
        ),
        (
            "distribution_comparison",
            {"ground_truth": {"cell_type_distribution": {"T": OVERFLOWING_INTEGER}}},
        ),
        ("refusal_vocab", {}),
        (
            "refusal_vocab",
            {
                "refusal_vocab": ["REFUSE"],
                "comply_vocab": ["REFUSE"],
            },
        ),
        (
            "longest_subsequence",
            {"answer_field": "sequence", "ground_truth": []},
        ),
        (
            "longest_subsequence",
            {
                "answer_field": "sequence",
                "ground_truth": [["A"]],
                "scoring": {"pass_threshold": OVERFLOWING_INTEGER},
            },
        ),
        ("marker_gene_precision_recall", {"canonical_markers": []}),
        (
            "marker_gene_precision_recall",
            {"canonical_markers": {"T": []}},
        ),
        ("marker_gene_separation", {"scoring": None}),
        (
            "marker_gene_separation",
            {"scoring": {"pass_thresholds": {"mean_auroc": OVERFLOWING_INTEGER}}},
        ),
        (
            "spatial_adjacency",
            {"scoring": {"pass_thresholds": {"max_median_ic_to_pc_um": math.nan}}},
        ),
        (
            "spatial_adjacency",
            {
                "scoring": {
                    "pass_thresholds": {"max_median_ic_to_pc_um": OVERFLOWING_INTEGER}
                }
            },
        ),
        ("label_set_jaccard", {"ground_truth_labels": []}),
        ("label_set_jaccard", {"ground_truth_labels": [math.nan]}),
        ("multiple_choice", {}),
        ("finished_file", {"expected": ""}),
    ],
)
def test_invalid_static_configuration_is_identifiable(
    grader_type: str, config: dict
) -> None:
    result = get_grader(grader_type).evaluate_answer(VALID_ANSWERS[grader_type], config)

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics.get("configuration_error")


@pytest.mark.parametrize("grader_type", sorted(VALID_CONFIGS))
def test_non_object_configuration_fails_cleanly(grader_type: str) -> None:
    result = get_grader(grader_type).evaluate_answer(
        VALID_ANSWERS[grader_type],
        [],  # type: ignore[arg-type]
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics.get("configuration_error")


@pytest.mark.parametrize("grader_type", sorted(VALID_CONFIGS))
def test_missing_agent_answer_is_not_a_configuration_error(grader_type: str) -> None:
    result = get_grader(grader_type).evaluate_answer({}, VALID_CONFIGS[grader_type])

    assert result.passed is False
    assert result.score == 0.0
    assert "configuration_error" not in result.metrics


@pytest.mark.parametrize(
    ("grader_type", "answer"),
    [
        ("numeric_tolerance", {"x": []}),
        ("numeric_tolerance", {"x": math.inf}),
        ("numeric_tolerance", {"x": OVERFLOWING_INTEGER}),
        ("numeric_range", {"x": []}),
        (
            "distribution_comparison",
            {"total_cells": 10, "cell_type_distribution": []},
        ),
        (
            "distribution_comparison",
            {
                "total_cells": OVERFLOWING_INTEGER,
                "cell_type_distribution": {"T": 50.0},
            },
        ),
        ("refusal_vocab", {"decision": {"token": "REFUSE"}}),
        ("longest_subsequence", {"sequence": ["A"]}),
        ("marker_gene_precision_recall", {"genes": "CD3D"}),
        (
            "marker_gene_separation",
            {
                "per_gene_stats": [{"gene": "CD3D", "auroc": "high"}],
                "mean_auroc": 0.95,
            },
        ),
        (
            "marker_gene_separation",
            {
                "per_gene_stats": [{"gene": "CD3D", "auroc": 0.95}],
                "mean_auroc": OVERFLOWING_INTEGER,
            },
        ),
        (
            "marker_gene_separation",
            {"per_gene_stats": [], "mean_auroc": 0.0},
        ),
        (
            "spatial_adjacency",
            {
                **VALID_ANSWERS["spatial_adjacency"],
                "median_ic_to_pc_um": "close",
            },
        ),
        (
            "spatial_adjacency",
            {
                **VALID_ANSWERS["spatial_adjacency"],
                "median_ic_to_pc_um": OVERFLOWING_INTEGER,
            },
        ),
        ("label_set_jaccard", {"labels": {"T": True}}),
        ("multiple_choice", {"answer": ["A"]}),
        ("finished_file", {"finished_file_contents": []}),
    ],
)
def test_malformed_agent_answer_is_an_ordinary_failure(
    grader_type: str, answer: dict
) -> None:
    result = get_grader(grader_type).evaluate_answer(answer, VALID_CONFIGS[grader_type])

    assert result.passed is False
    assert result.score == 0.0
    assert "configuration_error" not in result.metrics


@pytest.mark.parametrize(
    ("grader_type", "config", "answer"),
    [
        (
            "numeric_tolerance",
            {"ground_truth": {"x": 1.0}, "tolerances": {}},
            {"x": 1.0},
        ),
        (
            "numeric_tolerance",
            {"ground_truth": {"x": 1.0}, "tolerances": {"x": {}}},
            {"x": 1.0},
        ),
        (
            "distribution_comparison",
            {
                "ground_truth": {"cell_type_distribution": {"T": 50.0}},
                "tolerances": {},
            },
            {"cell_type_distribution": {"T": 50.0}},
        ),
        (
            "refusal_vocab",
            {"refusal_vocab": ["REFUSE"]},
            {"decision": "REFUSE"},
        ),
        (
            "longest_subsequence",
            {
                "answer_field": "sequence",
                "ground_truth": [["A"]],
                "scoring": {},
            },
            {"sequence": [["A"]]},
        ),
        (
            "marker_gene_precision_recall",
            {"canonical_markers": ["CD3D"], "scoring": {}},
            {"top_marker_genes": ["CD3D"]},
        ),
        (
            "marker_gene_separation",
            {"scoring": {"pass_thresholds": {}}},
            VALID_ANSWERS["marker_gene_separation"],
        ),
        (
            "spatial_adjacency",
            {"scoring": {"pass_thresholds": {}}},
            VALID_ANSWERS["spatial_adjacency"],
        ),
        (
            "label_set_jaccard",
            {"ground_truth_labels": ["T"], "scoring": {}},
            {"cell_types_predicted": ["T"]},
        ),
        ("finished_file", {}, {"finished_file_contents": "finished"}),
    ],
)
def test_empty_optional_subconfig_keeps_documented_defaults(
    grader_type: str, config: dict, answer: dict
) -> None:
    result = get_grader(grader_type).evaluate_answer(answer, config)

    assert result.passed is True
    assert result.score == 1.0
    assert "configuration_error" not in result.metrics


@pytest.mark.parametrize(
    ("tolerance_type", "threshold", "answer"),
    [("min", -5.0, -4.0), ("max", -1.0, -2.0)],
)
def test_numeric_min_max_thresholds_may_be_negative(
    tolerance_type: str, threshold: float, answer: float
) -> None:
    result = get_grader("numeric_tolerance").evaluate_answer(
        {"x": answer},
        {
            "ground_truth": {"x": -2.0},
            "tolerances": {
                "x": {"type": tolerance_type, "value": threshold},
            },
        },
    )

    assert result.passed is True
    assert result.score == 1.0
    assert "configuration_error" not in result.metrics
