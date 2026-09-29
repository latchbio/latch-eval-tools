import math

from latch_eval_tools.graders.geometry import (
    normalize_path_coords,
    path_hausdorff_distance,
    path_within_radius_match,
)


def test_path_distance_supports_2d_and_3d_paths() -> None:
    assert path_hausdorff_distance([[0, 0], [10, 0]], [[10, 0], [0, 0]]) == 0.0
    assert math.isclose(
        path_hausdorff_distance(
            [[0, 0, 0], [10, 0, 0]],
            [[0, 0, 1], [10, 0, 1]],
        ),
        1.0,
    )


def test_path_distance_supports_scaled_4d_time() -> None:
    reference = [[0, 0, 0, 0], [10, 0, 0, 10]]
    submitted = [[0, 0, 0, 2], [10, 0, 0, 12]]

    assert math.isclose(path_hausdorff_distance(reference, submitted), 2.0)
    assert math.isclose(
        path_hausdorff_distance(
            reference,
            submitted,
            component_scales=[1, 1, 1, 0.25],
        ),
        0.5,
    )
    assert (
        path_within_radius_match(
            reference,
            submitted,
            0.5,
            component_scales=[1, 1, 1, 0.25],
        )
        == 1.0
    )


def test_component_scales_match_explicit_coordinate_scaling() -> None:
    reference = [[0, 0, 0, 0], [2, 1, 3, 4], [4, 3, 5, 8]]
    submitted = [[0, 1, 0, 1], [2, 2, 2, 5], [5, 3, 6, 9]]
    scales = [2, 0.5, 3, 0.25]
    scaled_reference = [
        [value * scale for value, scale in zip(point, scales, strict=True)]
        for point in reference
    ]
    scaled_submitted = [
        [value * scale for value, scale in zip(point, scales, strict=True)]
        for point in submitted
    ]

    assert math.isclose(
        path_hausdorff_distance(
            reference,
            submitted,
            component_scales=scales,
        ),
        path_hausdorff_distance(scaled_reference, scaled_submitted),
    )


def test_path_radius_is_symmetric() -> None:
    reference = [[0, 0], [10, 0]]

    assert path_within_radius_match(reference, [[0, 1], [10, 1]], 1) == 1.0
    assert path_within_radius_match(reference, [[0, 0], [5, 0]], 1) == 0.0


def test_path_radius_checks_segments_between_vertices() -> None:
    reference = [[-1, 0], [0, 1], [1, 0]]
    submitted = [[-1, 0], [1, 0]]

    assert path_within_radius_match(reference, submitted, 0.5) == 0.0


def test_path_distance_finds_interior_maximum() -> None:
    reference = [[0, 0], [1, 0], [0, 1]]
    submitted = [[0, 0], [0, 1], [1, 0]]

    assert math.isclose(
        path_hausdorff_distance(reference, submitted),
        math.sqrt(2) - 1,
    )
    assert path_within_radius_match(reference, submitted, 0.41) == 0.0


def test_path_distance_handles_large_coordinates() -> None:
    extreme = [[-1e308, 0], [1e308, 0]]
    translated = [[1e15, 1e15], [1e15 + 10, 1e15]]
    shifted = [[1e15, 1e15 + 1], [1e15 + 10, 1e15 + 1]]

    assert path_hausdorff_distance(extreme, extreme) == 0.0
    assert path_hausdorff_distance(translated, shifted) == 1.0


def _assert_value_error(message: str, operation) -> None:
    try:
        operation()
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_path_validation_rejects_invalid_paths() -> None:
    _assert_value_error(
        "at least two points", lambda: normalize_path_coords([[0, 0]], "path")
    )
    _assert_value_error(
        "at most 128 points",
        lambda: normalize_path_coords([[index, 0] for index in range(129)], "path"),
    )
    _assert_value_error(
        "matching dimensions",
        lambda: normalize_path_coords([[0, 0], [1, 1, 1]], "path"),
    )
    _assert_value_error(
        "two, three, or four coordinates",
        lambda: normalize_path_coords([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], "path"),
    )
    _assert_value_error(
        "positive length", lambda: normalize_path_coords([[0, 0], [0, 0]], "path")
    )
    _assert_value_error(
        "dimensions must match",
        lambda: path_hausdorff_distance([[0, 0], [1, 0]], [[0, 0, 0], [1, 0, 0]]),
    )
    _assert_value_error(
        "numeric precision",
        lambda: path_hausdorff_distance(
            [[0, 0], [1e146, 0], [1e308, 1e308]],
            [[0, 1e146], [1e146, 1e146], [1e308, 1e308]],
        ),
    )
    _assert_value_error(
        "numeric precision",
        lambda: path_hausdorff_distance(
            [[0, 0], [0, 1e-320], [1e308, 0]],
            [[0, 0], [1e308, 0]],
        ),
    )
    _assert_value_error(
        "exactly 4 values",
        lambda: path_hausdorff_distance(
            [[0, 0, 0, 0], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [1, 1, 1, 1]],
            component_scales=[1, 1, 1],
        ),
    )
    _assert_value_error(
        "must be positive",
        lambda: path_hausdorff_distance(
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
            component_scales=[1, 0],
        ),
    )
    _assert_value_error(
        "numeric precision",
        lambda: path_hausdorff_distance(
            [[0, 0], [0, 1]],
            [[1, 0], [1, 1]],
            component_scales=[1, 1e200],
        ),
    )
