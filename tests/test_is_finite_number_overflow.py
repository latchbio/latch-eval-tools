"""Regression test: a huge-magnitude JSON integer must fail gracefully.

``float(value)`` raises ``OverflowError`` for ints whose magnitude exceeds
~1.8e308 (still well within CPython's int-parsing limits). ``_is_finite_number``
used to let that exception escape, which crashed grading instead of scoring
the field as failed.
"""

from latch_eval_tools.graders import get_grader

HUGE_INT = int("9" * 400)


def test_numeric_tolerance_grader_rejects_huge_int_without_crashing():
    grader = get_grader("numeric_tolerance")
    result = grader.evaluate_answer(
        {"x": HUGE_INT},
        {"ground_truth": {"x": 1.0}, "tolerances": {}},
    )
    assert result.passed is False
    assert result.score == 0.0


def test_numeric_range_grader_rejects_huge_int_without_crashing():
    grader = get_grader("numeric_range")
    result = grader.evaluate_answer(
        {"x": HUGE_INT},
        {"ground_truth": {"x": 1.0}, "ranges": {"x": {"min": 0.0, "max": 2.0}}},
    )
    assert result.passed is False
    assert result.score == 0.0


def test_distribution_comparison_grader_rejects_huge_int_without_crashing():
    grader = get_grader("distribution_comparison")
    result = grader.evaluate_answer(
        {"cell_type_distribution": {"T": HUGE_INT}},
        {"ground_truth": {"cell_type_distribution": {"T": 50.0}}},
    )
    assert result.passed is False
    assert result.score == 0.0


def test_spatial_adjacency_grader_rejects_huge_int_without_crashing():
    grader = get_grader("spatial_adjacency")
    result = grader.evaluate_answer(
        {
            "median_ic_to_pc_um": HUGE_INT,
            "p90_ic_to_pc_um": 10.0,
            "pct_ic_within_15um": 80.0,
            "pct_ic_mixed_within_55um": 80.0,
            "adjacency_pass": True,
        },
        {"scoring": {"pass_thresholds": {}}},
    )
    assert result.passed is False
    assert result.score == 0.0


def test_marker_gene_separation_grader_rejects_huge_int_without_crashing():
    grader = get_grader("marker_gene_separation")
    result = grader.evaluate_answer(
        {
            "mean_auroc": HUGE_INT,
            "per_gene_stats": [{"gene": "CD3D", "auroc": 0.9}],
        },
        {"scoring": {"pass_thresholds": {}}},
    )
    assert result.passed is False
    assert result.score == 0.0
