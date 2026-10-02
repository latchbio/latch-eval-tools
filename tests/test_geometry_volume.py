import math

import pytest  # pyright: ignore[reportMissingImports]

from latch_eval_tools.graders.geometry import (
    iou_volume_to_volume,
    normalize_volume_coords,
    volume_in_volume_gradient,
    volume_in_volume_match,
)


def _prism(ring: list[list[float]], height: float = 1.0):
    return [
        [[x, y, 0.0] for x, y in ring],
        [[x, y, height] for x, y in ring],
    ]


def test_loft_supports_triangular_caps() -> None:
    mesh = normalize_volume_coords(
        _prism([[0, 0], [1, 0], [0, 1]]),
        "volume",
    )

    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 0.5)


def test_loft_preserves_high_frequency_ring_vertices() -> None:
    vertex_count = 128
    star = []
    outer = []
    for index in range(vertex_count):
        angle = 2 * math.pi * index / vertex_count
        radius = 100.0 if index % 2 == 0 else 1.0
        star.append([radius * math.cos(angle), radius * math.sin(angle)])
        if index % 2 == 0:
            outer.append([100.0 * math.cos(angle), 100.0 * math.sin(angle)])

    iou = iou_volume_to_volume(_prism(star), _prism(outer))

    assert 0.009 < iou < 0.011


def test_loft_accepts_small_positive_ring_separation() -> None:
    volume = _prism([[0, 0], [1, 0], [1, 1], [0, 1]], height=5e-7)

    mesh = normalize_volume_coords(volume, "volume")

    assert mesh.is_volume
    assert math.isclose(mesh.volume, 5e-7)


def test_loft_preserves_tiny_cross_sections() -> None:
    volume = _prism([[0, 0], [1e-9, 0], [1e-9, 1e-9], [0, 1e-9]])

    assert iou_volume_to_volume(volume, volume) == 1.0


def test_loft_is_invariant_to_cyclic_ring_order() -> None:
    ring = [[0.0, 0.0], [4.0, 0.0], [0.0, 1.0]]
    rotated = ring[1:] + ring[:1]
    rotated_top = [_prism(ring)[0], _prism(rotated)[1]]

    assert math.isclose(
        iou_volume_to_volume(_prism(ring), rotated_top),
        1.0,
        rel_tol=1e-6,
    )


@pytest.mark.parametrize("threshold", [-1, 0, 1.1, math.nan, True, "0.5"])
def test_volume_match_rejects_invalid_thresholds(threshold) -> None:
    volume = _prism([[0, 0], [1, 0], [1, 1], [0, 1]])

    with pytest.raises(ValueError, match="threshold"):
        volume_in_volume_match(volume, volume, threshold)


@pytest.mark.parametrize(
    ("threshold_full", "threshold_null"),
    [(0.5, 0.5), (0.4, 0.5), (1.1, 0.5), (0.5, -0.1), (math.nan, 0.1)],
)
def test_volume_gradient_rejects_invalid_thresholds(
    threshold_full, threshold_null
) -> None:
    volume = _prism([[0, 0], [1, 0], [1, 1], [0, 1]])

    with pytest.raises(ValueError, match="threshold"):
        volume_in_volume_gradient(
            volume,
            volume,
            threshold_full,
            threshold_null,
        )
