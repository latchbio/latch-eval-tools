import pytest  # pyright: ignore[reportMissingImports]

from latch_eval_tools.graders.predicate import PredicateLeafGrader, evaluate_predicate

REFERENCE_PATH = [[0, 0], [10, 0]]


def test_path_within_radius_predicate() -> None:
    predicate = {
        "op": "path_within_radius",
        "reference_path": REFERENCE_PATH,
        "tolerance_radius": 1.0,
    }

    assert evaluate_predicate(predicate, [[0, 1], [10, 1]]) is True
    assert evaluate_predicate(predicate, [[0, 2], [10, 2]]) is False


def test_path_within_radius_predicate_supports_scaled_4d_time() -> None:
    reference = [[0, 0, 0, 0], [10, 0, 0, 10]]
    submitted = [[0, 0, 0, 2], [10, 0, 0, 12]]
    result = PredicateLeafGrader().evaluate_answer(
        {"path": submitted},
        {
            "role": "gate",
            "answer_field": "path",
            "predicate": {
                "op": "path_within_radius",
                "reference_path": reference,
                "tolerance_radius": 0.5,
                "component_scales": [1, 1, 1, 0.25],
            },
        },
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics["is_scalar"] is False


@pytest.mark.parametrize(
    "predicate",
    [
        {
            "op": "path_within_radius",
            "reference_path": [[0, 0]],
            "tolerance_radius": 1.0,
        },
        {
            "op": "path_within_radius",
            "reference_path": REFERENCE_PATH,
            "tolerance_radius": 1.0,
            "component_scales": [1, 1, 1],
        },
        {
            "op": "path_within_radius",
            "reference_path": REFERENCE_PATH,
            "tolerance_radius": -1.0,
        },
        {
            "op": "path_within_radius",
            "reference_path": [[0, 0], [1e-200, 0], [1, 0]],
            "tolerance_radius": 1.0,
        },
    ],
)
def test_path_within_radius_rejects_invalid_configuration(predicate: dict) -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"path": REFERENCE_PATH},
        {
            "role": "gate",
            "answer_field": "path",
            "predicate": predicate,
        },
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")


def test_path_within_radius_rejects_invalid_submitted_path() -> None:
    result = PredicateLeafGrader().evaluate_answer(
        {"path": [[0, 0]]},
        {
            "role": "gate",
            "answer_field": "path",
            "predicate": {
                "op": "path_within_radius",
                "reference_path": REFERENCE_PATH,
                "tolerance_radius": 1.0,
            },
        },
    )

    assert result.passed is False
    assert result.metrics.get("invalid_answer")
