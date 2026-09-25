import math

import pytest

from latch_eval_tools.graders.numeric import NumericToleranceGrader


MAX_INTEROPERABLE_MAGNITUDE = float((1 << 53) - 1)


def _assert_json_safe(metrics: dict) -> None:
    for key, value in metrics.items():
        assert not (isinstance(value, float) and not math.isfinite(value)), (
            f"metrics[{key!r}] = {value!r} is not JSON-interoperable"
        )
        assert not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and abs(value) > MAX_INTEROPERABLE_MAGNITUDE
        ), f"metrics[{key!r}] = {value!r} is not JSON-interoperable"


def test_relative_tolerance_against_zero_ground_truth_has_json_safe_metrics() -> None:
    # A relative-tolerance comparison against a zero ground truth is an
    # undefined ratio; the grader tracks this with an `inf` sentinel
    # internally but must not leak it into the persisted metrics.
    result = NumericToleranceGrader().evaluate_answer(
        {"delta": 5.0},
        {
            "ground_truth": {"delta": 0.0},
            "tolerances": {"delta": {"type": "relative", "value": 0.1}},
        },
    )

    assert result.passed is False
    assert result.metrics["delta_error"] is None
    _assert_json_safe(result.metrics)
    assert "inf" in result.reasoning


def test_non_finite_agent_answer_has_json_safe_metrics() -> None:
    result = NumericToleranceGrader().evaluate_answer(
        {"delta": float("nan")},
        {
            "ground_truth": {"delta": 1.0},
            "tolerances": {"delta": {"type": "absolute", "value": 0.1}},
        },
    )

    assert result.passed is False
    assert result.metrics["delta_actual"] is None
    assert result.metrics["delta_error"] is None
    _assert_json_safe(result.metrics)


def test_finite_comparison_metrics_are_unaffected() -> None:
    result = NumericToleranceGrader().evaluate_answer(
        {"delta": 5.0},
        {
            "ground_truth": {"delta": 1.0},
            "tolerances": {"delta": {"type": "absolute", "value": 0.1}},
        },
    )

    assert result.metrics["delta_actual"] == 5.0
    assert result.metrics["delta_error"] == 4.0


def test_relative_error_against_tiny_ground_truth_has_json_safe_metrics() -> None:
    # A p-value ground truth makes the relative error a ratio against ~1e-25, so
    # any ordinary miss lands far above the IEEE-754 safe integer range. The
    # error term is finite but unpersistable, and it must not take the whole
    # grader result down with it.
    result = NumericToleranceGrader().evaluate_answer(
        {"padj": 1.0},
        {
            "ground_truth": {"padj": 7.197676794958261e-25},
            "tolerances": {"padj": {"type": "relative", "value": 0.35}},
        },
    )

    assert result.passed is False
    assert result.metrics["padj_pass"] is False
    assert result.metrics["padj_actual"] == 1.0
    assert result.metrics["padj_error"] is None
    _assert_json_safe(result.metrics)
    assert "1.0 vs 7.197676794958261e-25" in result.reasoning


def test_unsafe_integer_answer_has_json_safe_metrics() -> None:
    result = NumericToleranceGrader().evaluate_answer(
        {"count": 10**20},
        {
            "ground_truth": {"count": 12.0},
            "tolerances": {"count": {"type": "absolute", "value": 1.0}},
        },
    )

    assert result.passed is False
    assert result.metrics["count_actual"] is None
    assert result.metrics["count_error"] is None
    _assert_json_safe(result.metrics)
