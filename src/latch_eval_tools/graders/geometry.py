import math
from dataclasses import dataclass

import mapbox_earcut
import numpy as np
from numpy.random.mtrand import f
import trimesh
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity

from .number_contract import is_finite_number

# some def constants, maybe change
_VOL_PLAN_TOL = 1e-6
_VOL_MIN_RING_STEP = 1e-6
_VOL_SAMPLES_PER_RING = 64

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


def iou_polygon_to_polygon(ref_polygon: object, sub_polygon: object) -> float:
    ref_p = normalize_polygon_coords(ref_polygon, "reference polygon")
    sub_p = normalize_polygon_coords(sub_polygon, "submitted polygon")

    try:
        interse_ar = ref_p.intersection(sub_p).area
    except GEOSException as exc:
        raise ValueError(f"polygon intersection has failed {exc}") from None

    union_area = ref_p.area + sub_p.area - interse_ar

    if not math.isfinite(interse_ar):
        raise ValueError("polygon intersection produced a non finite area")

    if not math.isfinite(union_area) or union_area <= 0:
        raise ValueError("polygon union must have a positive, finite area!!!")

    iou = interse_ar / union_area
    return min(1.0, max(0.0, iou))

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
            raise ValueError(f"{ring_label} each ring polygon needs to have at least 3 vectors")

        normed_ring: list[list[float]] = []

        for vi, v in enumerate(r):
            v_label = f"{ring_label}[{ri}]"

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
    first_centroid = np.mean(first_ring, axis=0)
    centered_first_ring = first_ring - first_centroid

    try:
        _, singular_vals, right_vecs = np.linalg.svd(centered_first_ring, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{err_label} could not build loft axis!, err {exc}") from None

    if len(singular_vals) < 2 or singular_vals[1] <= 0:
        raise ValueError(f"{err_label}[0] defines a not degenerate plane!!!!!!")

    loft_axis = right_vecs[-1]

    last_centroid = np.mean(open_rings[-1], axis=0)
    total_progress = float(np.dot(last_centroid - first_centroid, loft_axis))

    if not math.isfinite(total_progress) or total_progress == 0:
        raise ValueError(f"{err_label} fors not progress away from the plane defined by the first ring!!!!!")

    if total_progress < 0:
        loft_axis = -loft_axis

    frame = build_lframe(axis = loft_axis.tolist(), origin=first_centroid.tolist())

    try:
        prepped_rings = prep_loft(normed_rings, frame=frame, plan_tol=_VOL_PLAN_TOL, min_ring_sep=_VOL_MIN_RING_STEP)

        return build_loft_mesh(prepped_rings, frame=frame, smpls_per_ring=_VOL_SAMPLES_PER_RING)
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
    if np.any(np.linalg.norm(edges, axis=1) <= 0):
        raise ValueError(f"{err_label} contains consecutive duplicate vertices")

    x = vertices_2d[:, 0]
    y = vertices_2d[:, 1]

    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    if signed_area == 0:
        raise ValueError(f"{err_label} has 0 projected area")

    if signed_area < 0:
        vertices_2d = vertices_2d[::-1]
        points = points[::-1]

    polygon = Polygon(vertices_2d)

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


def resample_closed_ring(vertices_2d: np.ndarray, *, count: int) -> np.ndarray:
    points = np.asarray(vertices_2d, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("ring must have shape (n, 2")

    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("count must be an integer of at least 3")

    next_points = np.roll(points, -1, axis=0)
    edge_vecs = next_points - points
    edge_lens = np.linalg.norm(edge_vecs, axis=1)

    if np.any(edge_lens <= 0):
        raise ValueError("ring contains a zero len edge")

    outline = np.sum(edge_lens).item()
    cumulative = np.concatenate([[0.0], np.cumsum(edge_lens)])

    targets = np.linspace(0.0, outline, count, endpoint=False)

    edge_indices = np.searchsorted(cumulative[1:], targets, side="right")

    edge_starts = cumulative[edge_indices]
    fracs = (targets - edge_starts) / edge_lens[edge_indices]

    return points[edge_indices] + edge_vecs[edge_indices] * fracs[:, None]


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

    ring_ends = np.asarray([len(points)], dtype=np.uint32)

    indices = mapbox_earcut.triangulate_float64(points, ring_ends)

    return indices.reshape((-1, 3))


def build_loft_mesh(
    prepped_rings: list[PreppedRing], *, frame: LFrame, smpls_per_ring: int
) -> trimesh.Trimesh:
    if len(prepped_rings) < 2:
        raise ValueError("at least two rings are required")

    sampled_2d: list[np.ndarray] = []

    for ring in prepped_rings:
        sampled = resample_closed_ring(ring.vertices_2d, count=smpls_per_ring)

        if sampled_2d:
            sampled = align_adjacent_ring(sampled_2d[-1], sampled)

        sampled_2d.append(sampled)

    sampled_3d = [
        ring_samples_to_world(points, depth=ring.depth, frame=frame)
        for points, ring in zip(sampled_2d, prepped_rings, strict=True)
    ]

    vertices = np.vstack(sampled_3d)
    faces: list[list[int]] = []

    count = smpls_per_ring

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
        vertices=vertices,
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

    return mesh


def volume_iou(reference: trimesh.Trimesh, submitted: trimesh.Trimesh) -> float:
    intersection = trimesh.boolean.intersection(
        [reference, submitted], engine="manifold", check_volume=True
    )

    intersection_volume = (
        0.0
        if intersection is None or intersection.is_empty
        else abs(intersection.volume)
    )

    reference_volume = abs(reference.volume)
    submitted_volume = abs(submitted.volume)

    union_volume = (reference_volume + submitted_volume) - intersection_volume

    if not np.isfinite(union_volume) or union_volume <= 0:
        raise ValueError("volume union must be finite and positive")

    return min(1.0, max(0.0, (intersection_volume / union_volume)))


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
    if not is_finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number in [0, 1]")

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

    ref_v = normalize_volume_coords(reference_volume, "reference volume")
    sub_v = normalize_volume_coords(submitted_volume, "submitted volume")
    iou3d = volume_iou(ref_v, sub_v)

    if iou3d >= threshold:
        return 1.0
    else:
        return 0.0

def volume_in_volume_gradient(
    reference_volume: object, submitted_volume: object, threshold_full: float, threshold_null
) -> float:

    ref_v = normalize_volume_coords(reference_volume, "reference volume")
    sub_v = normalize_volume_coords(submitted_volume, "submitted volume")
    iou3d = volume_iou(ref_v, sub_v)

    if iou3d >= threshold_full:
        return 1.0

    if iou3d < threshold_null:
        return 0.0

    return (iou3d - threshold_null) / (threshold_full - threshold_null)
