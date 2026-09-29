import math

import pytest  # pyright: ignore[reportMissingImports]

from latch_eval_tools.graders import geometry, get_grader

LOCATION_CONFIG = {
    "reference_locations": [[0, 0], [10, 0]],
    "tolerance_radius": 1,
    "answer_field": "locations",
    "pass_threshold": 0.75,
}

SQUARE_A = [[0, 0], [2, 0], [2, 2], [0, 2]]
SQUARE_B = [[10, 0], [12, 0], [12, 2], [10, 2]]
POLYGON_CONFIG = {
    "reference_polygons": [SQUARE_A, SQUARE_B],
    "iou_threshold": 0.9,
    "answer_field": "polygons",
    "pass_threshold": 0.75,
}


def test_location_radius_list_grader_matches_one_to_one() -> None:
    grader = get_grader("location_radius")
    result = grader.evaluate_answer(
        {"locations": [[0, 0], [0, 0], [10, 0]]}, LOCATION_CONFIG
    )

    assert result.passed is True
    assert math.isclose(result.score, 0.8)
    assert result.metrics["matched_count"] == 2
    assert math.isclose(result.metrics["precision"], 2 / 3)
    assert result.metrics["unmatched_submitted_indices"] == [1]


def test_location_radius_list_grader_fails_below_f1_threshold() -> None:
    result = get_grader("location_radius").evaluate_answer(
        {"locations": [[0, 0]]}, LOCATION_CONFIG
    )

    assert result.passed is False
    assert math.isclose(result.score, 2 / 3)
    assert result.metrics["unmatched_reference_indices"] == [1]


def test_polygon_iou_list_grader_matches_one_to_one() -> None:
    grader = get_grader("polygon_iou_list")
    result = grader.evaluate_answer(
        {"polygons": [SQUARE_A, SQUARE_A, SQUARE_B]}, POLYGON_CONFIG
    )

    assert result.passed is True
    assert math.isclose(result.score, 0.8)
    assert result.metrics["matched_count"] == 2
    assert math.isclose(result.metrics["precision"], 2 / 3)
    assert result.metrics["unmatched_submitted_indices"] == [1]


def test_polygon_iou_list_grader_fails_below_f1_threshold() -> None:
    result = get_grader("polygon_iou_list").evaluate_answer(
        {"polygons": [SQUARE_A]}, POLYGON_CONFIG
    )

    assert result.passed is False
    assert math.isclose(result.score, 2 / 3)
    assert result.metrics["unmatched_reference_indices"] == [1]


def test_visual_list_graders_reject_zero_pass_threshold() -> None:
    location_config = {**LOCATION_CONFIG, "pass_threshold": 0}
    polygon_config = {**POLYGON_CONFIG, "pass_threshold": 0}

    location_result = get_grader("location_radius").evaluate_answer(
        {"locations": []}, location_config
    )
    polygon_result = get_grader("polygon_iou_list").evaluate_answer(
        {"polygons": []}, polygon_config
    )

    assert location_result.passed is False
    assert location_result.metrics.get("configuration_error")
    assert polygon_result.passed is False
    assert polygon_result.metrics.get("configuration_error")


def test_polygon_list_backend_failure_is_a_structured_system_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise geometry.GeometryBackendError("GEOS failed")

    monkeypatch.setattr(geometry, "_polygon_iou", fail)
    result = get_grader("polygon_iou_list").evaluate_answer(
        {"polygons": [SQUARE_A, SQUARE_B]},
        POLYGON_CONFIG,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["grader_system_error"] is True
    assert "GEOS failed" in result.metrics["grader_error"]


def test_polygon_list_grader_rejects_zero_iou_threshold() -> None:
    result = get_grader("polygon_iou_list").evaluate_answer(
        {"polygons": [SQUARE_A, SQUARE_B]},
        {**POLYGON_CONFIG, "iou_threshold": 0},
    )

    assert result.passed is False
    assert result.metrics.get("configuration_error")
