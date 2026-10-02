import math

import pytest  # pyright: ignore[reportMissingImports]

from latch_eval_tools.graders import geometry, get_grader
from latch_eval_tools.graders.predicate import PredicateLeafGrader, evaluate_predicate

REFERENCE_VOLUME = [
    [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
    [[0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]],
]
OVERLAPPING_VOLUME = [
    [[1, 0, 0], [3, 0, 0], [3, 2, 0], [1, 2, 0]],
    [[1, 0, 2], [3, 0, 2], [3, 2, 2], [1, 2, 2]],
]


def test_volume_iou_predicate_returns_raw_iou() -> None:
    predicate = {"op": "volume_iou", "reference_volume": REFERENCE_VOLUME}

    assert evaluate_predicate(predicate, REFERENCE_VOLUME) == 1.0
    assert math.isclose(
        evaluate_predicate(predicate, OVERLAPPING_VOLUME),
        1 / 3,
        rel_tol=1e-6,
    )
    assert (
        evaluate_predicate(
            predicate,
            [
                [[3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]],
                [[3, 0, 1], [4, 0, 1], [4, 1, 1], [3, 1, 1]],
            ],
        )
        == 0.0
    )


def test_volume_iou_is_translation_invariant() -> None:
    translated = [
        [[coordinate + 1e11 for coordinate in point] for point in ring]
        for ring in REFERENCE_VOLUME
    ]
    translated_overlap = [
        [[coordinate + 1e11 for coordinate in point] for point in ring]
        for ring in OVERLAPPING_VOLUME
    ]
    predicate = {"op": "volume_iou", "reference_volume": translated}

    assert evaluate_predicate(predicate, translated) == 1.0
    assert math.isclose(
        evaluate_predicate(predicate, translated_overlap),
        1 / 3,
        rel_tol=1e-6,
    )


def test_volume_iou_leaf_uses_scalar_threshold() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"volume": OVERLAPPING_VOLUME},
        {
            "role": "gate",
            "answer_field": "volume",
            "threshold": 0.3,
            "predicate": {
                "op": "volume_iou",
                "reference_volume": REFERENCE_VOLUME,
            },
        },
    )

    assert result.passed is True
    assert math.isclose(result.score, 1 / 3, rel_tol=1e-6)
    assert result.metrics["is_scalar"] is True


@pytest.mark.parametrize("threshold", [0, 1.1])
def test_volume_iou_rejects_invalid_threshold(threshold: float) -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"volume": REFERENCE_VOLUME},
        {
            "role": "gate",
            "threshold": threshold,
            "predicate": {
                "op": "volume_iou",
                "reference_volume": REFERENCE_VOLUME,
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_volume_iou_rejects_invalid_reference_volume() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"volume": REFERENCE_VOLUME},
        {
            "role": "gate",
            "threshold": 0.5,
            "predicate": {
                "op": "volume_iou",
                "reference_volume": REFERENCE_VOLUME[:1],
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_volume_iou_backend_failure_is_a_structured_system_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("manifold failed")

    monkeypatch.setattr(geometry.trimesh.boolean, "intersection", fail)
    result = PredicateLeafGrader().evaluate_answer(
        {"volume": REFERENCE_VOLUME},
        {
            "role": "gate",
            "answer_field": "volume",
            "threshold": 0.5,
            "predicate": {
                "op": "volume_iou",
                "reference_volume": REFERENCE_VOLUME,
            },
        },
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["grader_system_error"] is True
    assert "manifold failed" in result.metrics["grader_error"]


@pytest.mark.parametrize("grader_type", ["list_match", "dict_match"])
def test_volume_backend_failure_propagates_through_match_graders(
    grader_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("manifold failed")

    monkeypatch.setattr(geometry.trimesh.boolean, "intersection", fail)
    leaf = {
        "role": "additive",
        "predicate": {
            "op": "volume_iou",
            "reference_volume": REFERENCE_VOLUME,
        },
    }
    if grader_type == "list_match":
        answer = {"rows": [{"id": "a", "volume": REFERENCE_VOLUME}]}
        config = {
            "answer_field": "rows",
            "match_key": "id",
            "ground_truth": [
                {"id": "a", "fields": {"volume": leaf}},
            ],
        }
    else:
        answer = {"values": {"volume": REFERENCE_VOLUME}}
        config = {
            "answer_field": "values",
            "ground_truth": {"volume": leaf},
        }

    result = get_grader(grader_type).evaluate_answer(answer, config)

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["grader_system_error"] is True
    assert result.metrics["system_error_fields"]
    assert "manifold failed" in result.metrics["grader_error"]


def test_volume_iou_rejects_invalid_submitted_volume() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"volume": REFERENCE_VOLUME[:1]},
        {
            "role": "gate",
            "answer_field": "volume",
            "threshold": 0.5,
            "predicate": {
                "op": "volume_iou",
                "reference_volume": REFERENCE_VOLUME,
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("invalid_answer")
