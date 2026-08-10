import math

import pytest

from latch_eval_tools.graders.numeric import NumericToleranceGrader


def _assert_json_safe(metrics: dict) -> None:
    for key, value in metrics.items():
        assert not (isinstance(value, float) and not math.isfinite(value)), (
            f"metrics[{key!r}] = {value!r} is not JSON-interoperable"
        )


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
