from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from shapely import constrained_delaunay_triangles
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity

from .number_contract import is_finite_number

trimesh = importlib.import_module("trimesh")

# some def constants, maybe change
_VOL_PLAN_TOL = 1e-6
_VOL_MIN_RING_STEP = 0.0
_VOL_SAMPLES_PER_RING = 64
_MAX_PATH_POINTS = 128


class GeometryBackendError(RuntimeError):
    pass


@dataclass
class LFrame:
    origin: np.ndarray
    axis: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray


@dataclass
class PreppedRing:
    vertices_3d: np.ndarray
    vertices_2d: np.ndarray
    depth: float
    max_plan_err: float
    polygon_2d: Polygon


@dataclass
class _RingParameterCluster:
    parameter: float
    ring_values: dict[int, float]


def normalize_coords(loc: object, err_label: str) -> list[float]:
    if not isinstance(loc, list):
        raise ValueError(f"{err_label} must be an array")

    if len(loc) < 2:
        raise ValueError(f"{err_label} coordinate needs to have more than one axis")

    normed: list[float] = []

    for index, coordinate in enumerate(loc):
        if not is_finite_number(coordinate):
            raise ValueError(f"{err_label}[{index}] must be a finite number")

        try:
            parsed = float(coordinate)
            int_is_ex = not isinstance(coordinate, int) or int(parsed) == coordinate
        except (OverflowError, TypeError, ValueError):
            raise ValueError(
                f"{err_label}[{index}] cannot be represented as a float"
            ) from None

        if not int_is_ex:
            raise ValueError(
                f"{err_label}[{index}] cannot be represented exactly as a float"
            )

        normed.append(parsed)
    return normed


def normalize_coords_list(locs: object, err_label: str) -> list[list[float]]:

    if not isinstance(locs, list):
        raise ValueError(f"{err_label} location list has to be a list")

    normed_cords: list[list[float]] = []

    for i, v in enumerate(locs):
        cname = f"{err_label}[{i}]"
        cor = normalize_coords(v, cname)

        normed_cords.append(cor)

    return normed_cords


def normalize_path_coords(path: object, err_label: str) -> list[list[float]]:
    points = normalize_coords_list(path, err_label)
    if len(points) < 2:
        raise ValueError(f"{err_label} must contain at least two points")
    if len(points) > _MAX_PATH_POINTS:
        raise ValueError(f"{err_label} must contain at most {_MAX_PATH_POINTS} points")

    dimension = len(points[0])
    if dimension not in (2, 3):
        raise ValueError(
            f"{err_label} points must have exactly two or three coordinates"
        )
    if any(len(point) != dimension for point in points[1:]):
        raise ValueError(f"{err_label} points must have matching dimensions")
    if not any(start != end for start, end in pairwise(points)):
        raise ValueError(f"{err_label} must have positive length")

    return points


def _point_to_path_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    starts = path[:-1]
    segments = path[1:] - starts
    lengths_squared = np.einsum("ij,ij->i", segments, segments)
    offsets = points[:, np.newaxis, :] - starts[np.newaxis, :, :]
    projections = np.einsum("pij,ij->pi", offsets, segments)
    parameters = np.divide(
        projections,
        lengths_squared,
        out=np.zeros_like(projections),
        where=lengths_squared > 0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts + parameters[:, :, np.newaxis] * segments
    return np.min(np.linalg.norm(points[:, np.newaxis, :] - closest, axis=2), axis=1)


def _distance_piece(
    source_start: np.ndarray,
    source_delta: np.ndarray,
    target_start: np.ndarray,
    target_delta: np.ndarray,
    zone: int,
) -> tuple[float, float, float]:
    if zone < 0:
        offset = source_start - target_start
        return (
            np.dot(source_delta, source_delta).item(),
            (2 * np.dot(offset, source_delta)).item(),
            np.dot(offset, offset).item(),
        )
    if zone > 0:
        offset = source_start - (target_start + target_delta)
        return (
            np.dot(source_delta, source_delta).item(),
            (2 * np.dot(offset, source_delta)).item(),
            np.dot(offset, offset).item(),
        )

    offset = source_start - target_start
    target_length_squared = np.dot(target_delta, target_delta).item()
    offset_projection = np.dot(offset, target_delta).item()
    delta_projection = np.dot(source_delta, target_delta).item()
    return (
        (
            np.dot(source_delta, source_delta)
            - (delta_projection * delta_projection / target_length_squared)
        ).item(),
        (
            2
            * (
                np.dot(offset, source_delta)
                - (offset_projection * delta_projection / target_length_squared)
            )
        ).item(),
        (
            np.dot(offset, offset)
            - (offset_projection * offset_projection / target_length_squared)
        ).item(),
    )


def _target_distance_pieces(
    source_start: np.ndarray,
    source_delta: np.ndarray,
    target_start: np.ndarray,
    target_end: np.ndarray,
) -> list[tuple[float, float, float, float, float]]:
    target_delta = target_end - target_start
    target_length_squared = np.dot(target_delta, target_delta).item()
    if target_length_squared == 0:
        a, b, c = _distance_piece(
            source_start,
            source_delta,
            target_start,
            target_delta,
            -1,
        )
        return [(0.0, 1.0, a, b, c)]

    offset = source_start - target_start
    alpha = np.dot(offset, target_delta).item() / target_length_squared
    beta = np.dot(source_delta, target_delta).item() / target_length_squared
    cuts = [0.0, 1.0]
    if beta != 0:
        cuts.extend(
            cut for cut in (-alpha / beta, (1.0 - alpha) / beta) if 0.0 < cut < 1.0
        )
    cuts = sorted(set(cuts))

    pieces = []
    for lower, upper in pairwise(cuts):
        projection = alpha + beta * ((lower + upper) / 2)
        zone = -1 if projection < 0 else 1 if projection > 1 else 0
        pieces.append(
            (
                lower,
                upper,
                *_distance_piece(
                    source_start,
                    source_delta,
                    target_start,
                    target_delta,
                    zone,
                ),
            )
        )
    return pieces


def _quadratic_intersections(
    first: tuple[float, float, float, float, float],
    second: tuple[float, float, float, float, float],
) -> list[float]:
    lower = max(first[0], second[0])
    upper = min(first[1], second[1])
    if lower > upper:
        return []

    a = first[2] - second[2]
    b = first[3] - second[3]
    c = first[4] - second[4]
    tolerance = np.finfo(float).eps * 64 * max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            return []
        roots = [-c / b]
    else:
        discriminant = (b * b) - (4 * a * c)
        if discriminant < -tolerance:
            return []
        root = math.sqrt(max(0.0, discriminant))
        roots = [(-b - root) / (2 * a), (-b + root) / (2 * a)]

    return [
        min(upper, max(lower, root))
        for root in roots
        if lower - tolerance <= root <= upper + tolerance
    ]


def _directed_path_hausdorff(source: np.ndarray, target: np.ndarray) -> float:
    maximum = 0.0
    for source_start, source_end in pairwise(source):
        source_delta = source_end - source_start
        pieces = [
            piece
            for target_start, target_end in pairwise(target)
            for piece in _target_distance_pieces(
                source_start,
                source_delta,
                target_start,
                target_end,
            )
        ]
        candidates = {0.0, 1.0}
        for piece in pieces:
            candidates.update((piece[0], piece[1]))
        for index, first in enumerate(pieces):
            for second in pieces[index + 1 :]:
                candidates.update(_quadratic_intersections(first, second))

        parameters = np.asarray(sorted(candidates), dtype=float)
        for offset in range(0, len(parameters), 4096):
            batch = parameters[offset : offset + 4096]
            points = source_start + batch[:, np.newaxis] * source_delta
            maximum = max(
                maximum,
                np.max(_point_to_path_distances(points, target)).item(),
            )
    return maximum


def path_hausdorff_distance(reference_path: object, submitted_path: object) -> float:
    reference = np.asarray(
        normalize_path_coords(reference_path, "reference path"), dtype=float
    )
    submitted = np.asarray(
        normalize_path_coords(submitted_path, "submitted path"), dtype=float
    )
    if reference.shape[1] != submitted.shape[1]:
        raise ValueError("reference path and submitted path dimensions must match")

    nonzero_segments = [
        [not np.array_equal(start, end) for start, end in pairwise(path)]
        for path in (reference, submitted)
    ]
    origin = reference[0].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        translated_reference = reference - origin
        translated_submitted = submitted - origin

    if (
        np.isfinite(translated_reference).all()
        and np.isfinite(translated_submitted).all()
    ):
        scale = np.max(
            np.abs(np.vstack([translated_reference, translated_submitted]))
        ).item()
        reference = translated_reference / scale
        submitted = translated_submitted / scale
    else:
        scale = np.max(np.abs(np.vstack([reference, submitted]))).item()
        reference = reference / scale
        submitted = submitted / scale
        scaled_origin = reference[0].copy()
        reference -= scaled_origin
        submitted -= scaled_origin

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("path scale must be finite and positive")

    segment_lengths = []
    for path, path_nonzero_segments in zip(
        (reference, submitted), nonzero_segments, strict=True
    ):
        for (start, end), was_nonzero in zip(
            pairwise(path), path_nonzero_segments, strict=True
        ):
            if not was_nonzero:
                continue
            length = math.dist(start, end)
            if length == 0:
                raise ValueError(
                    "path segment scales exceed supported numeric precision"
                )
            segment_lengths.append(length)

    longest_segment = max(segment_lengths)
    if min(segment_lengths) < longest_segment * 1e-150:
        raise ValueError("path segment scales exceed supported numeric precision")

    normalized_distance = max(
        _directed_path_hausdorff(reference, submitted),
        _directed_path_hausdorff(submitted, reference),
    )
    distance = normalized_distance * scale
    if not math.isfinite(distance):
        raise ValueError("path distance must be finite")
    return distance


def path_within_radius_match(
    reference_path: object, submitted_path: object, radius: float
) -> float:
    if not is_finite_number(radius) or radius < 0:
        raise ValueError("radius must be a finite non-negative number")

    return (
        1.0
        if path_hausdorff_distance(reference_path, submitted_path) <= radius
        else 0.0
    )


def normalize_polygon_coords(polygon: object, err_label: str) -> Polygon:

    if not isinstance(polygon, list):
        raise ValueError(f"{err_label} polygons need to be a list of list of floats!")

    if len(polygon) < 3:
        raise ValueError(f"{err_label} a polygon needs at least three points")

    normed_vecs = []

    for ind, v in enumerate(polygon):
        norm = normalize_coords(v, f"{err_label}[{ind}]")
        normed_vecs.append(norm)

        if len(norm) != 2:
            raise ValueError(f"{err_label}[{ind}] must contain exactly two coordinates")

    unique_vert = normed_vecs
    # kick out if First is last
    if normed_vecs[0] == normed_vecs[-1]:
        unique_vert = normed_vecs[:-1]

    if len({tuple(vertex) for vertex in unique_vert}) < 3:
        raise ValueError(f"{err_label} must contain at least three unique vertices")

    try:
        result = Polygon(normed_vecs)
    except (GEOSException, TypeError, ValueError) as exc:
        raise ValueError(f"{err_label} could not be constructed: {exc}") from None

    if result.is_empty:
        raise ValueError(f"{err_label} must be not empty")

    if not result.is_valid:
        reas = explain_validity(result)
        raise ValueError(f"polygon {err_label} is not a valid polygon, reason: {reas}")

    if not math.isfinite(result.area) or result.area <= 0:
        raise ValueError(f"{err_label} must have a finite positive area")

    return result


def normalize_polygon_list(polygons: object, err_label: str) -> list[Polygon]:
    if not isinstance(polygons, list):
        raise ValueError(f"{err_label} must be a list of polygons")

    return [
        normalize_polygon_coords(polygon, f"{err_label}[{index}]")
        for index, polygon in enumerate(polygons)
    ]


def _polygon_iou(ref_p: Polygon, sub_p: Polygon) -> float:
    try:
        interse_ar = ref_p.intersection(sub_p).area
    except GEOSException as exc:
        raise GeometryBackendError(f"polygon intersection failed: {exc}") from exc

    union_area = ref_p.area + sub_p.area - interse_ar

    if not math.isfinite(interse_ar):
        raise ValueError("polygon intersection produced a non finite area")

    if not math.isfinite(union_area) or union_area <= 0:
        raise ValueError("polygon union must have a positive, finite area!!!")

    iou = interse_ar / union_area
    return min(1.0, max(0.0, iou))


def iou_polygon_to_polygon(ref_polygon: object, sub_polygon: object) -> float:
    ref_p = normalize_polygon_coords(ref_polygon, "reference polygon")
    sub_p = normalize_polygon_coords(sub_polygon, "submitted polygon")
    return _polygon_iou(ref_p, sub_p)


# --------------------------------- VOLUMETRIC STUFF \/ \/ \/


def normalize_volume_coords(vol: object, err_label: str):

    if not isinstance(vol, list):
        raise ValueError(f"{err_label} must be a list of rings!")

    if len(vol) < 2:
        raise ValueError(f"{err_label} must have at least two rings")

    normed_rings: list[list[list[float]]] = []
    open_rings: list[np.ndarray] = []

    for ri, r in enumerate(vol):
        ring_label = f"{err_label}[{ri}]"

        if not isinstance(r, list):
            raise ValueError(f"{ring_label} is not a polygon, ie list of vectors")

        if len(r) < 3:
            raise ValueError(
                f"{ring_label} each ring polygon needs to have at least 3 vectors"
            )

        normed_ring: list[list[float]] = []

        for vi, v in enumerate(r):
            v_label = f"{ring_label}[{vi}]"

            point = normalize_coords(v, v_label)

            if len(point) != 3:
                raise ValueError(f"{v_label} must contain 3 components")

            normed_ring.append(point)

        points = np.asarray(normed_ring, dtype=float)

        if len(points) > 1 and np.array_equal(points[0], points[-1]):
            points = points[:-1]

        if len(points) < 3 or len(np.unique(points, axis=0)) < 3:
            raise ValueError(f"{ring_label} must have at least 3 unique vecs")

        normed_rings.append(normed_ring)
        open_rings.append(points)

    first_ring = open_rings[0]
    first_anchor = first_ring[0]
    first_centroid = first_anchor + np.mean(first_ring - first_anchor, axis=0)
    centered_first_ring = first_ring - first_centroid
    centered_scale = np.max(np.abs(centered_first_ring)).item()
    if not math.isfinite(centered_scale) or centered_scale <= 0:
        raise ValueError(f"{err_label}[0] defines a degenerate plane")

    try:
        _, singular_vals, right_vecs = np.linalg.svd(
            centered_first_ring / centered_scale,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{err_label} could not build loft axis!, err {exc}") from None

    if len(singular_vals) < 2 or singular_vals[1] <= 0:
        raise ValueError(f"{err_label}[0] defines a not degenerate plane!!!!!!")

    loft_axis = right_vecs[-1]

    last_ring = open_rings[-1]
    last_anchor = last_ring[0]
    last_centroid = last_anchor + np.mean(last_ring - last_anchor, axis=0)
    try:
        total_progress = float(np.dot(last_centroid - first_centroid, loft_axis))
    except (TypeError, ValueError):
        raise ValueError(f"{err_label} loft progress could not be computed") from None

    if not math.isfinite(total_progress) or total_progress == 0:
        raise ValueError(
            f"{err_label} fors not progress away from the plane defined by the first ring!!!!!"
        )

    if total_progress < 0:
        loft_axis = -loft_axis

    frame = build_lframe(axis=loft_axis.tolist(), origin=first_centroid.tolist())

    try:
        prepped_rings = prep_loft(
            normed_rings,
            frame=frame,
            plan_tol=_VOL_PLAN_TOL,
            min_ring_sep=_VOL_MIN_RING_STEP,
        )

        return build_loft_mesh(
            prepped_rings, frame=frame, smpls_per_ring=_VOL_SAMPLES_PER_RING
        )
    except ValueError as exc:
        raise ValueError(f"{err_label} is invalid because of > {exc}") from None


def build_lframe(axis: object, origin: object) -> LFrame:
    axis_array = np.asarray(normalize_coords(axis, "loft axis"), dtype=float)
    origin_array = np.asarray(normalize_coords(origin, "loft origin"), dtype=float)

    if axis_array.shape != (3,):
        raise ValueError("loft axis needs to be a vec with 3 components")

    if origin_array.shape != (3,):
        raise ValueError("loft origin must be a vec with 3 components")

    if not np.isfinite(axis_array).all():
        raise ValueError("loft axis must be finite")

    if not np.isfinite(origin_array).all():
        raise ValueError("origin axis must be finite")

    axis_length = np.linalg.norm(axis_array)
    if axis_length <= 0:
        raise ValueError("loft axis must be non 0")

    axis_array = axis_array / axis_length

    helper = np.eye(3)[np.argmin(np.abs(axis_array))]

    axis_u = np.cross(helper, axis_array)
    axis_u = axis_u / np.linalg.norm(axis_u)

    axis_v = np.cross(axis_array, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)

    return LFrame(origin=origin_array, axis=axis_array, axis_u=axis_u, axis_v=axis_v)


def prep_ring(
    ring: object, *, frame: LFrame, plane_tol: float, err_label: str
) -> PreppedRing:
    if not is_finite_number(plane_tol) or plane_tol < 0:
        raise ValueError("plane_tol must be a finite non-negative number")

    if not isinstance(ring, list):
        raise ValueError(f"{err_label} must be an array")

    if len(ring) < 3:
        raise ValueError(f"{err_label} must contain at least three vertices")

    normed: list[list[float]] = []

    for i, v in enumerate(ring):
        point = normalize_coords(v, f"{err_label}[{i}]")

        if len(point) != 3:
            raise ValueError(f"{err_label}[{i}] vector must have exactly 3 components")

        normed.append(point)

    points = np.asarray(normed, dtype=float)

    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]

    if len(points) < 3:
        raise ValueError(f"{err_label} must contain at least 3 vertices")

    if len(np.unique(points, axis=0)) < 3:
        raise ValueError(f"{err_label} must contain at least 3 unique vertices")

    relative = points - frame.origin

    local_u = relative @ frame.axis_u
    local_v = relative @ frame.axis_v
    depths = relative @ frame.axis

    try:
        ring_depth = float(np.mean(depths))
        max_error = float(np.max(np.abs(depths - ring_depth)))
    except (TypeError, ValueError):
        raise ValueError(f"{err_label} planarity error could not be computed") from None

    if max_error > plane_tol:
        raise ValueError(
            f"{err_label} exceeds plance tolerance \n{max_error} > {plane_tol}"
        )

    vertices_2d = np.column_stack([local_u, local_v])

    edges = np.roll(vertices_2d, -1, axis=0) - vertices_2d
    if np.any(np.all(edges == 0, axis=1)):
        raise ValueError(f"{err_label} contains consecutive duplicate vertices")

    planar_scale = np.max(np.abs(vertices_2d)).item()
    if not math.isfinite(planar_scale) or planar_scale <= 0:
        raise ValueError(f"{err_label} has invalid projected coordinates")
    normalized_vertices = vertices_2d / planar_scale
    x = normalized_vertices[:, 0]
    y = normalized_vertices[:, 1]

    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    if signed_area == 0:
        raise ValueError(f"{err_label} has 0 projected area")

    if signed_area < 0:
        vertices_2d = vertices_2d[::-1]
        normalized_vertices = normalized_vertices[::-1]
        points = points[::-1]

    polygon = Polygon(normalized_vertices)

    if polygon.is_empty:
        raise ValueError(f"{err_label} must not be empty")

    if not polygon.is_valid:
        raise ValueError(f"{err_label} invalid, reason: {explain_validity(polygon)}")

    if not np.isfinite(polygon.area) or polygon.area <= 0:
        raise ValueError(f"{err_label} must have a positive finite area")

    vertices_3d = (
        frame.origin
        + vertices_2d[:, 0, None] * frame.axis_u
        + vertices_2d[:, 1, None] * frame.axis_v
        + ring_depth * frame.axis
    )

    return PreppedRing(
        vertices_2d=vertices_2d,
        vertices_3d=vertices_3d,
        depth=ring_depth,
        max_plan_err=max_error,
        polygon_2d=polygon,
    )


def prep_loft(
    rings: object, *, frame: LFrame, plan_tol: float, min_ring_sep: float
) -> list[PreppedRing]:
    if not is_finite_number(plan_tol) or plan_tol < 0:
        raise ValueError("plan_tol must be a finite non-negative number")

    if not is_finite_number(min_ring_sep) or min_ring_sep < 0:
        raise ValueError("min_ring_sep must be a finite non-negative number")

    if not isinstance(rings, list):
        raise ValueError("rings must be an array")

    if len(rings) < 2:
        raise ValueError("construction of the volume requires at least two rings")

    prepp = [
        prep_ring(r, frame=frame, plane_tol=plan_tol, err_label=f"rings[{i}]")
        for i, r in enumerate(rings)
    ]

    depths = np.asarray([r.depth for r in prepp], dtype=float)

    seps = np.diff(depths)

    if np.any(seps <= 0) or np.any(seps < min_ring_sep):
        raise ValueError(
            f"rings must progress along loft axis with at least {min_ring_sep} separation"
        )

    return prepp


def _ring_breakpoints(vertices_2d: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices_2d, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("ring must have shape (n, 2)")

    edge_vectors = np.roll(points, -1, axis=0) - points
    edge_scales = np.max(np.abs(edge_vectors), axis=1)
    if np.any(edge_scales <= 0):
        raise ValueError("ring contains a zero length edge")
    edge_lens = edge_scales * np.linalg.norm(
        edge_vectors / edge_scales[:, np.newaxis],
        axis=1,
    )

    outline = np.sum(edge_lens).item()
    if not math.isfinite(outline) or outline <= 0:
        raise ValueError("ring perimeter must be finite and positive")

    return np.concatenate([[0.0], np.cumsum(edge_lens)[:-1]]) / outline


def _sample_closed_ring(vertices_2d: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices_2d, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    breakpoints = _ring_breakpoints(points)

    if (
        parameters.ndim != 1
        or len(parameters) < 3
        or not np.isfinite(parameters).all()
        or np.any(parameters < 0)
        or np.any(parameters >= 1)
    ):
        raise ValueError("ring parameters must be finite values in [0, 1)")

    edge_indices = np.searchsorted(breakpoints[1:], parameters, side="right")
    starts = breakpoints[edge_indices]
    ends = np.concatenate([breakpoints[1:], [1.0]])[edge_indices]
    fractions = (parameters - starts) / (ends - starts)
    next_points = np.roll(points, -1, axis=0)

    return points[edge_indices] + (
        (next_points[edge_indices] - points[edge_indices]) * fractions[:, None]
    )


def resample_closed_ring(vertices_2d: np.ndarray, *, count: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("count must be an integer of at least 3")

    return _sample_closed_ring(
        vertices_2d,
        np.linspace(0.0, 1.0, count, endpoint=False),
    )


def align_adjacent_ring(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)

    if previous.shape != current.shape:
        raise ValueError("adj resampled rings must have matching shapes")

    best_shift = 0
    best_cost = math.inf

    for shift in range(len(current)):
        shifted = np.roll(current, -shift, axis=0)
        cost = np.sum((previous - shifted) ** 2).item()

        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    return np.roll(current, -best_shift, axis=0)


def ring_samples_to_world(
    vertices_2d: np.ndarray, *, depth: float, frame: LFrame
) -> np.ndarray:
    points = np.asarray(vertices_2d, dtype=float)

    return (
        frame.origin
        + points[:, 0, None] * frame.axis_u
        + points[:, 1, None] * frame.axis_v
        + depth * frame.axis
    )


def triang_cap(vertices_2d: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices_2d, dtype=np.float64)
    minimum = np.min(points, axis=0)
    ranges = np.ptp(points, axis=0)
    if not np.isfinite(ranges).all() or np.any(ranges <= 0):
        raise ValueError("ring cap dimensions must be finite and positive")
    normalized_points = (points - minimum) / ranges
    point_indices = {
        tuple(point): index for index, point in enumerate(normalized_points)
    }

    try:
        triangles = constrained_delaunay_triangles(Polygon(normalized_points))
    except (GEOSException, TypeError, ValueError) as exc:
        raise ValueError(f"ring cap triangulation failed: {exc}") from None

    faces: list[list[int]] = []
    for triangle in triangles.geoms:
        if not isinstance(triangle, Polygon):
            raise ValueError("ring cap triangulation produced a non-polygon")
        coordinates = np.asarray(triangle.exterior.coords[:-1], dtype=float)
        if coordinates.shape != (3, 2):
            raise ValueError("ring cap triangulation produced a non-triangle")
        try:
            face = [point_indices[tuple(point)] for point in coordinates]
        except KeyError:
            raise ValueError(
                "ring cap triangulation introduced an unknown vertex"
            ) from None

        a, b, c = normalized_points[face]
        signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if signed_area == 0:
            raise ValueError("ring cap triangulation produced a zero-area face")
        if signed_area < 0:
            face[1], face[2] = face[2], face[1]
        faces.append(face)

    if not faces:
        raise ValueError("ring cap triangulation produced no faces")

    return np.asarray(faces, dtype=np.int64)


def _align_ring_origins(
    prepped_rings: list[PreppedRing], minimum_count: int
) -> list[PreppedRing]:
    aligned = [prepped_rings[0]]

    for ring in prepped_rings[1:]:
        previous = aligned[-1]
        count = max(
            minimum_count,
            len(previous.vertices_2d),
            len(ring.vertices_2d),
        )
        previous_samples = resample_closed_ring(previous.vertices_2d, count=count)
        alignment_ranges = np.ptp(
            np.vstack([previous.vertices_2d, ring.vertices_2d]),
            axis=0,
        )
        if not np.isfinite(alignment_ranges).all() or np.any(alignment_ranges <= 0):
            raise ValueError("ring alignment dimensions must be finite and positive")
        best_shift = 0
        best_cost = math.inf

        for shift in range(len(ring.vertices_2d)):
            shifted = np.roll(ring.vertices_2d, -shift, axis=0)
            samples = resample_closed_ring(shifted, count=count)
            cost = np.sum(((previous_samples - samples) / alignment_ranges) ** 2).item()
            if cost < best_cost:
                best_cost = cost
                best_shift = shift

        aligned.append(
            PreppedRing(
                vertices_2d=np.roll(ring.vertices_2d, -best_shift, axis=0),
                vertices_3d=np.roll(ring.vertices_3d, -best_shift, axis=0),
                depth=ring.depth,
                max_plan_err=ring.max_plan_err,
                polygon_2d=ring.polygon_2d,
            )
        )

    return aligned


def _shared_ring_parameters(
    prepped_rings: list[PreppedRing], minimum_count: int
) -> list[np.ndarray]:
    clusters: list[_RingParameterCluster] = []

    for ring_index, ring in enumerate(prepped_rings):
        for raw_parameter in _ring_breakpoints(ring.vertices_2d):
            parameter = raw_parameter.item()
            nearest = None
            nearest_distance = math.inf
            for index, cluster in enumerate(clusters):
                if ring_index in cluster.ring_values:
                    continue
                distance = abs(parameter - cluster.parameter)
                if distance <= 1e-12 and distance < nearest_distance:
                    nearest = index
                    nearest_distance = distance
            if nearest is None:
                clusters.append(
                    _RingParameterCluster(
                        parameter=parameter,
                        ring_values={ring_index: parameter},
                    )
                )
            else:
                clusters[nearest].ring_values[ring_index] = parameter

    for raw_parameter in np.linspace(0.0, 1.0, minimum_count, endpoint=False):
        parameter = raw_parameter.item()
        if not any(abs(parameter - cluster.parameter) <= 1e-12 for cluster in clusters):
            clusters.append(_RingParameterCluster(parameter, {}))

    clusters.sort(key=lambda cluster: cluster.parameter)
    parameters_by_ring: list[np.ndarray] = []
    for ring_index in range(len(prepped_rings)):
        parameters = np.asarray(
            [
                cluster.ring_values.get(ring_index, cluster.parameter)
                for cluster in clusters
            ],
            dtype=float,
        )
        if np.any(np.diff(parameters) <= 0):
            raise ValueError("shared ring parameters are not strictly increasing")
        parameters_by_ring.append(parameters)

    return parameters_by_ring


def build_loft_mesh(
    prepped_rings: list[PreppedRing], *, frame: LFrame, smpls_per_ring: int
) -> Any:
    if len(prepped_rings) < 2:
        raise ValueError("at least two rings are required")
    if (
        isinstance(smpls_per_ring, bool)
        or not isinstance(smpls_per_ring, int)
        or smpls_per_ring < 3
    ):
        raise ValueError("smpls_per_ring must be an integer of at least 3")

    prepped_rings = _align_ring_origins(prepped_rings, smpls_per_ring)
    parameters_by_ring = _shared_ring_parameters(prepped_rings, smpls_per_ring)
    sampled_2d = [
        _sample_closed_ring(ring.vertices_2d, parameters)
        for ring, parameters in zip(prepped_rings, parameters_by_ring, strict=True)
    ]

    local_vertices = np.vstack(
        [
            np.column_stack([points, np.full(len(points), ring.depth, dtype=float)])
            for points, ring in zip(sampled_2d, prepped_rings, strict=True)
        ]
    )
    local_minimum = np.min(local_vertices, axis=0)
    local_ranges = np.ptp(local_vertices, axis=0)
    if not np.isfinite(local_ranges).all() or np.any(local_ranges <= 0):
        raise ValueError("loft mesh dimensions must be finite and positive")
    normalized_vertices = (local_vertices - local_minimum) / local_ranges
    faces: list[list[int]] = []

    count = len(parameters_by_ring[0])

    for ring_index in range(len(prepped_rings) - 1):
        lower_offset = ring_index * count
        upper_offset = (ring_index + 1) * count

        for vertex_index in range(count):
            next_index = (vertex_index + 1) % count

            a = lower_offset + vertex_index
            b = lower_offset + next_index
            c = upper_offset + next_index
            d = upper_offset + vertex_index

            faces.append([a, b, c])
            faces.append([a, c, d])

    first_cap = triang_cap(sampled_2d[0])
    faces.extend(first_cap[:, ::-1].tolist())

    last_offset = (len(prepped_rings) - 1) * count
    last_cap = triang_cap(sampled_2d[-1])
    faces.extend((last_cap + last_offset).tolist())

    mesh = trimesh.Trimesh(
        vertices=normalized_vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )

    if not mesh.is_watertight:
        raise ValueError("loft mesh is not watertight")

    if not mesh.is_winding_consistent:
        raise ValueError("loft mesh winding is inconsistent")

    if not mesh.is_volume:
        raise ValueError("loft mesh is not a valid positive volume")

    restored_local = (np.asarray(mesh.vertices) * local_ranges) + local_minimum
    world_vertices = (
        frame.origin
        + restored_local[:, 0, None] * frame.axis_u
        + restored_local[:, 1, None] * frame.axis_v
        + restored_local[:, 2, None] * frame.axis
    )
    if not np.isfinite(world_vertices).all():
        raise ValueError("loft mesh vertices must be finite")
    mesh.vertices = world_vertices

    return mesh


def volume_iou(reference: Any, submitted: Any) -> float:
    reference_bounds = np.asarray(reference.bounds, dtype=float)
    submitted_bounds = np.asarray(submitted.bounds, dtype=float)
    if (
        reference_bounds.shape != (2, 3)
        or submitted_bounds.shape != (2, 3)
        or not np.isfinite(reference_bounds).all()
        or not np.isfinite(submitted_bounds).all()
    ):
        raise ValueError("volume bounds must be finite three-dimensional bounds")

    if np.any(reference_bounds[1] <= submitted_bounds[0]) or np.any(
        submitted_bounds[1] <= reference_bounds[0]
    ):
        return 0.0

    reference_local = reference.copy()
    submitted_local = submitted.copy()
    origin = np.minimum(reference_bounds[0], submitted_bounds[0])
    reference_local.apply_translation(-origin)
    submitted_local.apply_translation(-origin)

    combined_ranges = np.maximum(reference_bounds[1], submitted_bounds[1]) - origin
    if not np.isfinite(combined_ranges).all() or np.any(combined_ranges <= 0):
        raise ValueError("combined volume dimensions must be finite and positive")
    scale = 1.0 / combined_ranges
    reference_local.apply_scale(scale)
    submitted_local.apply_scale(scale)

    if (
        not np.isfinite(reference_local.vertices).all()
        or not np.isfinite(submitted_local.vertices).all()
    ):
        raise ValueError("normalized volume vertices must be finite")

    try:
        intersection = trimesh.boolean.intersection(
            [reference_local, submitted_local],
            engine="manifold",
            check_volume=True,
        )
    except Exception as exc:
        raise GeometryBackendError(f"volume intersection failed: {exc}") from exc

    intersection_volume = (
        0.0
        if intersection is None or intersection.is_empty
        else abs(intersection.volume)
    )
    reference_volume = abs(reference_local.volume)
    submitted_volume = abs(submitted_local.volume)
    union_volume = (reference_volume + submitted_volume) - intersection_volume

    if not np.isfinite(union_volume) or union_volume <= 0:
        raise ValueError("volume union must be finite and positive")

    result = min(1.0, max(0.0, intersection_volume / union_volume))
    return result.item() if isinstance(result, np.generic) else result


def iou_volume_to_volume(reference_volume: object, submitted_volume: object) -> float:
    reference = normalize_volume_coords(reference_volume, "reference volume")
    submitted = normalize_volume_coords(submitted_volume, "submitted volume")
    return volume_iou(reference, submitted)


# --------------------------------- VOLUMETRIC STUFF /\ /\ /\


def location_within_radius_match(
    reference_location: object, location: object, radius: float
) -> float:
    if not is_finite_number(radius) or radius < 0:
        raise ValueError("radius must be a finite non-negative number")

    ref_loc = normalize_coords(reference_location, err_label="reference location")
    sub_loc = normalize_coords(location, err_label="submitted location")

    if len(ref_loc) != len(sub_loc):
        raise ValueError(
            "dimensions of reference location and submitted location need to match"
        )

    dif = math.dist(ref_loc, sub_loc)

    if dif <= radius:
        return 1.0
    else:
        return 0.0


def location_within_radius_gradient(
    reference_location: object, location: object, radius_full: float, radius_none: float
) -> float:
    if (
        not is_finite_number(radius_full)
        or not is_finite_number(radius_none)
        or radius_full < 0
        or radius_full >= radius_none
    ):
        raise ValueError("radii must satisfy 0 <= radius_full < radius_none")

    ref_loc = normalize_coords(reference_location, err_label="reference location")
    sub_loc = normalize_coords(location, err_label="submitted location")

    if len(ref_loc) != len(sub_loc):
        raise ValueError(
            "dimensions of reference location and submitted location need to match"
        )

    dif = math.dist(ref_loc, sub_loc)

    if dif <= radius_full:
        return 1.0

    if dif >= radius_none:
        return 0.0

    return (radius_none - dif) / (radius_none - radius_full)


def location_within_polygon_match(reference_polygon: object, location: object) -> float:
    pol = normalize_polygon_coords(reference_polygon, "reference polygon")
    loc = normalize_coords(location, "submitted location")

    if len(loc) != 2:
        raise ValueError("location inside a polygon needs to have 2 coordinates")

    po = Point(loc)

    return 1.0 if pol.covers(po) else 0.0


def location_within_volume_match(
    reference_volume: object, location: object
) -> float:  # todo need to figure out how to build volumes from polygon rings
    raise ValueError("location-within-volume matching is not implemented")


def polygon_within_polygon_match(
    reference_polygon: object, polygon: object, threshold: float
) -> float:
    if not is_finite_number(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be a finite number in (0, 1]")

    iou = iou_polygon_to_polygon(reference_polygon, polygon)

    return 1.0 if iou >= threshold else 0.0


def polygon_within_polygon_gradient(
    reference_polygon: object, polygon: object, iou_full: float, iou_none: float
) -> float:
    if (
        not is_finite_number(iou_full)
        or not is_finite_number(iou_none)
        or not 0 <= iou_none < iou_full <= 1
    ):
        raise ValueError("IoU thresholds must satisfy 0 <= iou_none < iou_full <= 1")

    iou = iou_polygon_to_polygon(reference_polygon, polygon)

    if iou >= iou_full:
        return 1.0

    if iou <= iou_none:
        return 0.0

    return (iou - iou_none) / (iou_full - iou_none)


def volume_in_volume_match(
    reference_volume: object, submitted_volume: object, threshold: float
) -> float:
    if not is_finite_number(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be a finite number in (0, 1]")

    iou3d = iou_volume_to_volume(reference_volume, submitted_volume)

    if iou3d >= threshold:
        return 1.0
    else:
        return 0.0


def volume_in_volume_gradient(
    reference_volume: object,
    submitted_volume: object,
    threshold_full: float,
    threshold_null: float,
) -> float:
    if (
        not is_finite_number(threshold_full)
        or not is_finite_number(threshold_null)
        or not 0 <= threshold_null < threshold_full <= 1
    ):
        raise ValueError(
            "IoU thresholds must satisfy 0 <= threshold_null < threshold_full <= 1"
        )

    iou3d = iou_volume_to_volume(reference_volume, submitted_volume)

    if iou3d >= threshold_full:
        return 1.0

    if iou3d < threshold_null:
        return 0.0

    return (iou3d - threshold_null) / (threshold_full - threshold_null)


def locations_list_to_locations_list_match(
    reference_locations: object,
    submitted_locations: object,
    radius: float,
) -> dict[str, object]:
    if not is_finite_number(radius) or radius < 0:
        raise ValueError("radius must be a finite non-negative number")

    reference_locs = normalize_coords_list(
        reference_locations,
        "reference locations",
    )
    submitted_locs = normalize_coords_list(
        submitted_locations,
        "submitted locations",
    )

    if not reference_locs:
        raise ValueError("reference locations must not be empty")

    matched_reference_indices: set[int] = set()
    matched_submitted_indices: set[int] = set()
    matches: list[dict[str, object]] = []

    for submitted_index, submitted in enumerate(submitted_locs):
        for reference_index, reference in enumerate(reference_locs):
            if reference_index in matched_reference_indices:
                continue

            distance = math.dist(submitted, reference)

            if distance <= radius:
                matched_reference_indices.add(reference_index)
                matched_submitted_indices.add(submitted_index)
                matches.append(
                    {
                        "reference_index": reference_index,
                        "submitted_index": submitted_index,
                        "distance": distance,
                    }
                )
                break

    reference_count = len(reference_locs)
    submitted_count = len(submitted_locs)
    matched_count = len(matches)

    recall = matched_count / reference_count
    precision = matched_count / submitted_count if submitted_count > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return {
        "reference_count": reference_count,
        "submitted_count": submitted_count,
        "matched_count": matched_count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "matches": matches,
        "unmatched_reference_indices": sorted(
            set(range(reference_count)) - matched_reference_indices
        ),
        "unmatched_submitted_indices": sorted(
            set(range(submitted_count)) - matched_submitted_indices
        ),
        "tolerance_radius": radius,
    }


def polygons_list_to_polygons_list_match(
    reference_polygons: object,
    submitted_polygons: object,
    iou_threshold: float,
) -> dict[str, object]:
    if not is_finite_number(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be a finite number in (0, 1]")

    references = normalize_polygon_list(reference_polygons, "reference polygons")
    submissions = normalize_polygon_list(submitted_polygons, "submitted polygons")

    if not references:
        raise ValueError("reference polygons must not be empty")

    matched_reference_indices: set[int] = set()
    matched_submitted_indices: set[int] = set()
    matches: list[dict[str, object]] = []

    for submitted_index, submitted in enumerate(submissions):
        for reference_index, reference in enumerate(references):
            if reference_index in matched_reference_indices:
                continue

            iou = _polygon_iou(reference, submitted)
            if iou >= iou_threshold:
                matched_reference_indices.add(reference_index)
                matched_submitted_indices.add(submitted_index)
                matches.append(
                    {
                        "reference_index": reference_index,
                        "submitted_index": submitted_index,
                        "iou": iou,
                    }
                )
                break

    reference_count = len(references)
    submitted_count = len(submissions)
    matched_count = len(matches)
    recall = matched_count / reference_count
    precision = matched_count / submitted_count if submitted_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "reference_count": reference_count,
        "submitted_count": submitted_count,
        "matched_count": matched_count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "matches": matches,
        "unmatched_reference_indices": sorted(
            set(range(reference_count)) - matched_reference_indices
        ),
        "unmatched_submitted_indices": sorted(
            set(range(submitted_count)) - matched_submitted_indices
        ),
        "iou_threshold": iou_threshold,
    }
