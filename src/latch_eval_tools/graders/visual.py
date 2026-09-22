from __future__ import annotations

from typing import cast

from .base import (
    BinaryGrader,
    GraderResult,
    configuration_error_result,
    get_nested_value,
)
from .geometry import (
    locations_list_to_locations_list_match,
    normalize_coords_list,
    polygons_list_to_polygons_list_match,
)
from .number_contract import is_finite_number

_GRADER_NAME = "Location Radius List"
_POLYGON_GRADER_NAME = "Polygon IoU List"


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not is_finite_number(value):
        return None
    try:
        parsed = float(value)
        if isinstance(value, int) and int(parsed) != value:
            return None
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed


def _failure(
    agent_answer: object,
    reason: str,
    grader_name: str = _GRADER_NAME,
) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={},
        reasoning=f"{grader_name}: FAIL\n\n  x {reason}",
        agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
        score=0.0,
    )


class LocationRadiusListGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return configuration_error_result(
                agent_answer, _GRADER_NAME, "config must be an object"
            )

        try:
            references = normalize_coords_list(
                config.get("reference_locations"), "reference_locations"
            )
        except ValueError as exc:
            return configuration_error_result(agent_answer, _GRADER_NAME, str(exc))

        if not references:
            return configuration_error_result(
                agent_answer,
                _GRADER_NAME,
                "reference_locations must not be empty",
            )

        dimensions = len(references[0])
        if any(len(location) != dimensions for location in references):
            return configuration_error_result(
                agent_answer,
                _GRADER_NAME,
                "reference locations must have matching dimensions",
            )

        radius = _finite_float(config.get("tolerance_radius"))
        if radius is None or radius < 0:
            return configuration_error_result(
                agent_answer,
                _GRADER_NAME,
                "tolerance_radius must be a finite non-negative number",
            )

        pass_threshold = _finite_float(config.get("pass_threshold", 1.0))
        if pass_threshold is None or not 0 <= pass_threshold <= 1:
            return configuration_error_result(
                agent_answer,
                _GRADER_NAME,
                "pass_threshold must be a finite number in [0, 1]",
            )

        field = config.get("answer_field")
        if not isinstance(field, str) or not field:
            return configuration_error_result(
                agent_answer,
                _GRADER_NAME,
                "answer_field must be a non-empty string",
            )

        if not isinstance(agent_answer, dict):
            return _failure(agent_answer, "agent answer must be an object")

        submitted, found = get_nested_value(agent_answer, field)
        if not found:
            return _failure(agent_answer, f"missing required field: {field}")

        try:
            submissions = normalize_coords_list(submitted, f"agent_answer.{field}")
        except ValueError as exc:
            return _failure(agent_answer, str(exc))

        if any(len(location) != dimensions for location in submissions):
            return _failure(
                agent_answer,
                f"agent_answer.{field} locations must have {dimensions} coordinates",
            )

        stats = locations_list_to_locations_list_match(references, submissions, radius)
        score = cast(float, stats["recall"])
        passed = score >= pass_threshold
        metrics = {
            **stats,
            "answer_field": field,
            "pass_threshold": pass_threshold,
        }

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning=(
                f"{_GRADER_NAME}: {'PASS' if passed else 'FAIL'}\n\n"
                f"  Matched {stats['matched_count']}/{stats['reference_count']} "
                f"reference locations"
            ),
            agent_answer=agent_answer,
            score=score,
        )


class PolygonIoUListGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        if not isinstance(config, dict):
            return configuration_error_result(
                agent_answer, _POLYGON_GRADER_NAME, "config must be an object"
            )

        iou_threshold = _finite_float(config.get("iou_threshold"))
        if iou_threshold is None or not 0 <= iou_threshold <= 1:
            return configuration_error_result(
                agent_answer,
                _POLYGON_GRADER_NAME,
                "iou_threshold must be a finite number in [0, 1]",
            )

        references = config.get("reference_polygons")
        try:
            polygons_list_to_polygons_list_match(references, [], iou_threshold)
        except ValueError as exc:
            return configuration_error_result(
                agent_answer, _POLYGON_GRADER_NAME, str(exc)
            )

        pass_threshold = _finite_float(config.get("pass_threshold", 1.0))
        if pass_threshold is None or not 0 <= pass_threshold <= 1:
            return configuration_error_result(
                agent_answer,
                _POLYGON_GRADER_NAME,
                "pass_threshold must be a finite number in [0, 1]",
            )

        field = config.get("answer_field")
        if not isinstance(field, str) or not field:
            return configuration_error_result(
                agent_answer,
                _POLYGON_GRADER_NAME,
                "answer_field must be a non-empty string",
            )

        if not isinstance(agent_answer, dict):
            return _failure(
                agent_answer,
                "agent answer must be an object",
                _POLYGON_GRADER_NAME,
            )

        submitted, found = get_nested_value(agent_answer, field)
        if not found:
            return _failure(
                agent_answer,
                f"missing required field: {field}",
                _POLYGON_GRADER_NAME,
            )

        try:
            stats = polygons_list_to_polygons_list_match(
                references, submitted, iou_threshold
            )
        except ValueError as exc:
            return _failure(agent_answer, str(exc), _POLYGON_GRADER_NAME)

        score = cast(float, stats["recall"])
        passed = score >= pass_threshold
        metrics = {
            **stats,
            "answer_field": field,
            "pass_threshold": pass_threshold,
        }

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning=(
                f"{_POLYGON_GRADER_NAME}: {'PASS' if passed else 'FAIL'}\n\n"
                f"  Matched {stats['matched_count']}/{stats['reference_count']} "
                f"reference polygons"
            ),
            agent_answer=agent_answer,
            score=score,
        )
