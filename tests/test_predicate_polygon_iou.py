import math

import pytest  # pyright: ignore[reportMissingImports]

from latch_eval_tools.graders.predicate import PredicateLeafGrader, evaluate_predicate

REFERENCE_POLYGON = [[0, 0], [2, 0], [2, 2], [0, 2]]


def test_polygon_iou_predicate_returns_raw_iou() -> None:
    predicate = {"op": "polygon_iou", "reference_polygon": REFERENCE_POLYGON}

    assert evaluate_predicate(predicate, REFERENCE_POLYGON) == 1.0
    assert math.isclose(
        evaluate_predicate(predicate, [[1, 0], [3, 0], [3, 2], [1, 2]]),
        1 / 3,
    )
    assert evaluate_predicate(predicate, [[3, 0], [4, 0], [4, 1], [3, 1]]) == 0.0


def test_polygon_iou_leaf_uses_scalar_threshold() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"polygon": [[1, 0], [3, 0], [3, 2], [1, 2]]},
        {
            "role": "gate",
            "answer_field": "polygon",
            "threshold": 0.3,
            "predicate": {
                "op": "polygon_iou",
                "reference_polygon": REFERENCE_POLYGON,
            },
        },
    )

    assert result.passed is True
    assert math.isclose(result.score, 1 / 3)
    assert result.metrics["is_scalar"] is True


@pytest.mark.parametrize("threshold", [0, 1.1])
def test_polygon_iou_rejects_invalid_threshold(threshold: float) -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"polygon": REFERENCE_POLYGON},
        {
            "role": "gate",
            "threshold": threshold,
            "predicate": {
                "op": "polygon_iou",
                "reference_polygon": REFERENCE_POLYGON,
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_polygon_iou_hard_fail_uses_binary_reward() -> None:
    config = {
        "role": "hard_fail",
        "answer_field": "polygon",
        "threshold": 0.5,
        "predicate": {
            "op": "polygon_iou",
            "reference_polygon": REFERENCE_POLYGON,
        },
    }

    triggered = PredicateLeafGrader().evaluate_answer(
        {"polygon": REFERENCE_POLYGON}, config
    )
    clean = PredicateLeafGrader().evaluate_answer(
        {"polygon": [[3, 0], [4, 0], [4, 1], [3, 1]]}, config
    )

    assert triggered.passed is False
    assert triggered.score == 0.0
    assert clean.passed is True
    assert clean.score == 1.0


def test_polygon_iou_rejects_invalid_reference_polygon() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"polygon": REFERENCE_POLYGON},
        {
            "role": "gate",
            "threshold": 0.5,
            "predicate": {
                "op": "polygon_iou",
                "reference_polygon": [[0, 0], [1, 1]],
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_polygon_iou_rejects_invalid_submitted_polygon() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"polygon": [[0, 0], [1, 1]]},
        {
            "role": "gate",
            "answer_field": "polygon",
            "threshold": 0.5,
            "predicate": {
                "op": "polygon_iou",
                "reference_polygon": REFERENCE_POLYGON,
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("invalid_answer")
