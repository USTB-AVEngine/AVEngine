"""Pure NumPy support-plane authority for generated quadrupeds.

The primary authority locates visible foot bottoms inside four mutually
exclusive corridors around the complete semantic leaf-bone segments.  A
second, independent authority assigns vertices by summed skin weight over the
distal two bones of each semantic limb.  Neither authority is a fallback: both
must independently define a valid plane and agree on every foot floor.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ASSET_TOOLS_ROOT = Path(__file__).resolve().parent
if str(ASSET_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSET_TOOLS_ROOT))

from generated_animal_support_plane_contract import (
    CAPTURE_RADIUS_RATIO,
    CONTACT_BAND_ABSOLUTE_FLOOR,
    CONTACT_BAND_DIAGONAL_RATIO,
    CROSSCHECK_METHOD,
    EVIDENCE_SCHEMA,
    MINIMUM_CAPTURE_VERTICES,
    MINIMUM_CONTACT_BAND_VERTICES,
    MINIMUM_WEIGHT_OWNER_SCORE,
    PRIMARY_METHOD,
    SupportPlaneContractError,
    contact_band_thickness,
    validate_dual_authority_evidence as validate_serialized_dual_authority_evidence,
)


def _finite_array(value, *, label: str, shape=None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SupportPlaneContractError(f"{label} is not numeric") from exc
    if shape is not None and result.shape != shape:
        raise SupportPlaneContractError(
            f"{label} has shape {result.shape}, expected {shape}"
        )
    if not np.isfinite(result).all():
        raise SupportPlaneContractError(f"{label} contains non-finite values")
    return result


def _positive_finite(value, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SupportPlaneContractError(f"{label} must be positive and finite")
    return float(value)


def rigid_transform_aabb_reference(
    world_vertices,
    rotation,
    vertical_translation,
) -> dict:
    """Apply one declared leveling transform and measure its expected AABB.

    An axis-aligned bounding-box diagonal is not invariant under rotation.
    The only valid AABB comparison after leveling is therefore between the
    serialized output and this transformed copy of the authenticated input,
    never between the pre- and post-rotation AABBs directly.
    """

    vertices = _finite_array(world_vertices, label="world vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise SupportPlaneContractError(
            "world vertices must have non-empty shape (N, 3)"
        )
    matrix = _finite_array(
        rotation, label="declared rigid rotation", shape=(3, 3)
    )
    if (
        not np.allclose(
            matrix.T @ matrix,
            np.eye(3, dtype=np.float64),
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        or not math.isclose(
            float(np.linalg.det(matrix)),
            1.0,
            rel_tol=1.0e-8,
            abs_tol=1.0e-10,
        )
    ):
        raise SupportPlaneContractError(
            "declared leveling rotation is not a proper rigid rotation"
        )
    if (
        isinstance(vertical_translation, bool)
        or not isinstance(
            vertical_translation,
            (int, float, np.integer, np.floating),
        )
        or not math.isfinite(float(vertical_translation))
    ):
        raise SupportPlaneContractError(
            "declared vertical translation must be finite"
        )
    transformed = vertices @ matrix.T
    transformed[:, 2] += float(vertical_translation)
    minimum = transformed.min(axis=0)
    extent = transformed.max(axis=0) - minimum
    diagonal = float(np.linalg.norm(extent))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise SupportPlaneContractError(
            "declared rigid transform has a degenerate AABB"
        )
    return {
        "vertices": transformed,
        "bbox_min": minimum,
        "bbox_extent": extent,
        "bbox_diagonal": diagonal,
    }


def projected_segment_horizontal_distances(
    world_vertices,
    segment_heads,
    segment_tails,
) -> np.ndarray:
    """Return N-by-4 XY distances to four complete closed bone segments."""

    vertices = _finite_array(world_vertices, label="world vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise SupportPlaneContractError(
            "world vertices must have non-empty shape (N, 3)"
        )
    heads = _finite_array(
        segment_heads, label="semantic leaf heads", shape=(4, 3)
    )
    tails = _finite_array(
        segment_tails, label="semantic leaf tails", shape=(4, 3)
    )
    if np.any(np.linalg.norm(tails - heads, axis=1) <= 1.0e-12):
        raise SupportPlaneContractError(
            "semantic leaf segments must have non-zero 3D length"
        )

    points = vertices[:, :2]
    starts = heads[:, :2]
    deltas = tails[:, :2] - starts
    denominators = np.sum(deltas * deltas, axis=1)
    distances = np.empty((len(vertices), 4), dtype=np.float64)
    for index in range(4):
        if denominators[index] <= 1.0e-24:
            projected = np.broadcast_to(starts[index], points.shape)
        else:
            parameter = (
                (points - starts[index]) @ deltas[index]
                / denominators[index]
            )
            parameter = np.clip(parameter, 0.0, 1.0)
            projected = starts[index] + parameter[:, None] * deltas[index]
        distances[:, index] = np.linalg.norm(points - projected, axis=1)
    return distances


def _capture_bottom(
    vertices: np.ndarray,
    mask: np.ndarray,
    *,
    band_thickness: float,
    label: str,
) -> tuple[dict, np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) < MINIMUM_CAPTURE_VERTICES:
        raise SupportPlaneContractError(
            f"{label} captured fewer than {MINIMUM_CAPTURE_VERTICES} vertices: "
            f"captured={len(indices)}"
        )
    capture = vertices[indices]
    z_floor = float(capture[:, 2].min())
    band_mask = capture[:, 2] <= z_floor + band_thickness
    band_indices = indices[band_mask]
    if len(band_indices) < MINIMUM_CONTACT_BAND_VERTICES:
        raise SupportPlaneContractError(
            f"{label} bottom band captured fewer than "
            f"{MINIMUM_CONTACT_BAND_VERTICES} vertices: "
            f"captured={len(band_indices)}"
        )
    band = vertices[band_indices]
    point = [
        float(band[:, 0].mean()),
        float(band[:, 1].mean()),
        z_floor,
    ]
    return {
        "capture_count": int(len(indices)),
        "contact_band_size": int(len(band_indices)),
        "foot_point": point,
    }, band_indices


def fit_support_plane(
    foot_points,
    *,
    mesh_diagonal: float,
    maximum_residual_ratio: float,
    maximum_tilt_deg: float,
    label: str,
) -> dict:
    points = _finite_array(foot_points, label=f"{label} foot points", shape=(4, 3))
    diagonal = _positive_finite(mesh_diagonal, label="mesh diagonal")
    maximum_residual_ratio = _positive_finite(
        maximum_residual_ratio, label="maximum residual ratio"
    )
    maximum_tilt_deg = _positive_finite(
        maximum_tilt_deg, label="maximum tilt"
    )

    design = np.column_stack((points[:, 0], points[:, 1], np.ones(4)))
    coefficients, _residuals, rank, singular = np.linalg.lstsq(
        design, points[:, 2], rcond=None
    )
    if int(rank) != 3:
        raise SupportPlaneContractError(
            f"{label} foot points do not span a support plane: rank={rank}"
        )
    predicted = design @ coefficients
    residuals = points[:, 2] - predicted
    maximum_residual = float(np.abs(residuals).max())
    residual_ratio = maximum_residual / diagonal
    slope = math.hypot(float(coefficients[0]), float(coefficients[1]))
    tilt_deg = math.degrees(math.atan(slope))
    normal = np.asarray(
        [-float(coefficients[0]), -float(coefficients[1]), 1.0],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)
    if not (
        np.isfinite(coefficients).all()
        and np.isfinite(residuals).all()
        and np.isfinite(singular).all()
        and np.isfinite(normal).all()
        and math.isfinite(tilt_deg)
        and math.isfinite(residual_ratio)
    ):
        raise SupportPlaneContractError(f"{label} plane fit is non-finite")
    if residual_ratio > maximum_residual_ratio:
        raise SupportPlaneContractError(
            f"{label} residual ratio {residual_ratio:.6f} exceeds "
            f"{maximum_residual_ratio:.6f}"
        )
    if tilt_deg > maximum_tilt_deg:
        raise SupportPlaneContractError(
            f"{label} tilt {tilt_deg:.6f} exceeds {maximum_tilt_deg:.6f}"
        )
    return {
        "z_equals_ax_plus_by_plus_c": coefficients.tolist(),
        "residual_z": residuals.tolist(),
        "maximum_residual": maximum_residual,
        "maximum_residual_ratio_of_mesh_diagonal": residual_ratio,
        "normal": normal.tolist(),
        "tilt_deg": tilt_deg,
        "rank": int(rank),
        "singular_values": singular.tolist(),
    }


def _authority_record(
    *,
    method: str,
    captures: list[dict],
    plane: dict,
    maximum_captured_segment_distances: list[float] | None = None,
    minimum_captured_owner_scores: list[float] | None = None,
) -> dict:
    result = {
        "method": method,
        "exclusive_vertex_assignment": True,
        "capture_counts": [item["capture_count"] for item in captures],
        "contact_band_sizes": [
            item["contact_band_size"] for item in captures
        ],
        "foot_points": [item["foot_point"] for item in captures],
        "plane": plane,
    }
    if maximum_captured_segment_distances is not None:
        result["maximum_captured_segment_distances"] = (
            maximum_captured_segment_distances
        )
    if minimum_captured_owner_scores is not None:
        result["minimum_captured_owner_scores"] = minimum_captured_owner_scores
    return result


def evaluate_dual_authority_support_plane(
    world_vertices,
    segment_heads,
    segment_tails,
    distal_two_weight_scores,
    *,
    mesh_diagonal: float,
    maximum_residual_ratio: float = 0.02,
    maximum_tilt_deg: float = 30.0,
) -> dict:
    """Evaluate both mandatory authorities and return JSON-safe evidence."""

    vertices = _finite_array(world_vertices, label="world vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise SupportPlaneContractError(
            "world vertices must have non-empty shape (N, 3)"
        )
    diagonal = _positive_finite(mesh_diagonal, label="mesh diagonal")
    maximum_residual_ratio = _positive_finite(
        maximum_residual_ratio, label="maximum residual ratio"
    )
    maximum_tilt_deg = _positive_finite(
        maximum_tilt_deg, label="maximum tilt"
    )
    scores = _finite_array(
        distal_two_weight_scores, label="distal-two weight scores"
    )
    if scores.shape != (len(vertices), 4):
        raise SupportPlaneContractError(
            "distal-two weight scores must have shape (N, 4)"
        )
    if np.any(scores < 0.0):
        raise SupportPlaneContractError(
            "distal-two weight scores contain negative values"
        )

    band_thickness = contact_band_thickness(diagonal)
    capture_radius = diagonal * CAPTURE_RADIUS_RATIO

    distances = projected_segment_horizontal_distances(
        vertices, segment_heads, segment_tails
    )
    nearest_segment = np.argmin(distances, axis=1)
    primary_captures = []
    primary_band_indices = []
    maximum_segment_distances = []
    for index in range(4):
        mask = (
            (nearest_segment == index)
            & (distances[:, index] < capture_radius)
        )
        capture, band_indices = _capture_bottom(
            vertices,
            mask,
            band_thickness=band_thickness,
            label=f"primary semantic foot {index}",
        )
        primary_captures.append(capture)
        primary_band_indices.append(band_indices)
        maximum_segment_distances.append(
            float(distances[mask, index].max())
        )

    score_owner = np.argmax(scores, axis=1)
    owner_scores = scores[np.arange(len(vertices)), score_owner]
    crosscheck_captures = []
    crosscheck_band_indices = []
    minimum_owner_scores = []
    for index in range(4):
        mask = (
            (score_owner == index)
            & (owner_scores >= MINIMUM_WEIGHT_OWNER_SCORE)
        )
        capture, band_indices = _capture_bottom(
            vertices,
            mask,
            band_thickness=band_thickness,
            label=f"crosscheck semantic foot {index}",
        )
        crosscheck_captures.append(capture)
        crosscheck_band_indices.append(band_indices)
        minimum_owner_scores.append(float(owner_scores[mask].min()))

    primary_points = [item["foot_point"] for item in primary_captures]
    crosscheck_points = [item["foot_point"] for item in crosscheck_captures]
    primary_plane = fit_support_plane(
        primary_points,
        mesh_diagonal=diagonal,
        maximum_residual_ratio=maximum_residual_ratio,
        maximum_tilt_deg=maximum_tilt_deg,
        label="primary support plane",
    )
    crosscheck_plane = fit_support_plane(
        crosscheck_points,
        mesh_diagonal=diagonal,
        maximum_residual_ratio=maximum_residual_ratio,
        maximum_tilt_deg=maximum_tilt_deg,
        label="crosscheck support plane",
    )
    floor_delta = np.abs(
        np.asarray(primary_points, dtype=np.float64)[:, 2]
        - np.asarray(crosscheck_points, dtype=np.float64)[:, 2]
    )
    centroid_xy_delta = np.linalg.norm(
        np.asarray(primary_points, dtype=np.float64)[:, :2]
        - np.asarray(crosscheck_points, dtype=np.float64)[:, :2],
        axis=1,
    )
    if np.any(floor_delta > band_thickness):
        raise SupportPlaneContractError(
            "primary and crosscheck foot floors disagree beyond contact band: "
            f"delta={floor_delta.tolist()} maximum={band_thickness}"
        )
    if np.any(centroid_xy_delta > capture_radius):
        raise SupportPlaneContractError(
            "primary and crosscheck contact centroids disagree beyond the "
            f"fixed corridor: delta={centroid_xy_delta.tolist()} "
            f"maximum={capture_radius}"
        )

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "mesh_diagonal": diagonal,
        "thresholds": {
            "capture_radius_ratio_of_mesh_diagonal": CAPTURE_RADIUS_RATIO,
            "capture_radius": capture_radius,
            "contact_band_absolute_floor": CONTACT_BAND_ABSOLUTE_FLOOR,
            "contact_band_ratio_of_mesh_diagonal": (
                CONTACT_BAND_DIAGONAL_RATIO
            ),
            "contact_band_thickness": band_thickness,
            "minimum_capture_vertices": MINIMUM_CAPTURE_VERTICES,
            "minimum_contact_band_vertices": (
                MINIMUM_CONTACT_BAND_VERTICES
            ),
            "minimum_weight_owner_score": MINIMUM_WEIGHT_OWNER_SCORE,
            "maximum_floor_delta_between_authorities": band_thickness,
            "maximum_contact_centroid_xy_delta_between_authorities": (
                capture_radius
            ),
            "maximum_plane_residual_ratio_of_mesh_diagonal": (
                maximum_residual_ratio
            ),
            "maximum_tilt_deg": maximum_tilt_deg,
        },
        "primary": _authority_record(
            method=PRIMARY_METHOD,
            captures=primary_captures,
            plane=primary_plane,
            maximum_captured_segment_distances=maximum_segment_distances,
        ),
        "crosscheck": _authority_record(
            method=CROSSCHECK_METHOD,
            captures=crosscheck_captures,
            plane=crosscheck_plane,
            minimum_captured_owner_scores=minimum_owner_scores,
        ),
        "agreement": {
            "per_foot_floor_z_absolute_delta": floor_delta.tolist(),
            "maximum_floor_z_absolute_delta": float(floor_delta.max()),
            "maximum_allowed_floor_z_absolute_delta": band_thickness,
            "per_foot_contact_centroid_xy_distance": (
                centroid_xy_delta.tolist()
            ),
            "maximum_contact_centroid_xy_distance": float(
                centroid_xy_delta.max()
            ),
            "maximum_allowed_contact_centroid_xy_distance": capture_radius,
            "passed": True,
        },
        "fallback_used": False,
    }
    validate_serialized_dual_authority_evidence(evidence)
    return evidence


def _count_list(value, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            for item in value
        )
    ):
        raise SupportPlaneContractError(f"{label} must contain four counts")
    return value


def _require_close(actual, expected, *, label: str, atol=1.0e-10) -> None:
    actual_array = _finite_array(actual, label=label)
    expected_array = _finite_array(expected, label=f"expected {label}")
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=1.0e-10,
        atol=atol,
    ):
        raise SupportPlaneContractError(f"{label} is internally inconsistent")


def _validate_authority(
    authority,
    *,
    method: str,
    mesh_diagonal: float,
    maximum_residual_ratio: float,
    maximum_tilt_deg: float,
    capture_radius: float,
    minimum_owner_score: float | None,
    label: str,
) -> dict:
    if not isinstance(authority, dict):
        raise SupportPlaneContractError(f"{label} authority is missing")
    if (
        authority.get("method") != method
        or authority.get("exclusive_vertex_assignment") is not True
    ):
        raise SupportPlaneContractError(f"{label} authority method changed")
    capture_counts = _count_list(
        authority.get("capture_counts"), label=f"{label} capture counts"
    )
    band_sizes = _count_list(
        authority.get("contact_band_sizes"),
        label=f"{label} contact band sizes",
    )
    if any(value < MINIMUM_CAPTURE_VERTICES for value in capture_counts):
        raise SupportPlaneContractError(f"{label} capture is sparse")
    if any(value < MINIMUM_CONTACT_BAND_VERTICES for value in band_sizes):
        raise SupportPlaneContractError(f"{label} contact band is sparse")
    points = _finite_array(
        authority.get("foot_points"),
        label=f"{label} foot points",
        shape=(4, 3),
    )
    expected_plane = fit_support_plane(
        points,
        mesh_diagonal=mesh_diagonal,
        maximum_residual_ratio=maximum_residual_ratio,
        maximum_tilt_deg=maximum_tilt_deg,
        label=f"{label} support plane",
    )
    plane = authority.get("plane")
    if not isinstance(plane, dict):
        raise SupportPlaneContractError(f"{label} plane is missing")
    for field in (
        "z_equals_ax_plus_by_plus_c",
        "residual_z",
        "maximum_residual",
        "maximum_residual_ratio_of_mesh_diagonal",
        "normal",
        "tilt_deg",
        "rank",
        "singular_values",
    ):
        _require_close(
            plane.get(field),
            expected_plane[field],
            label=f"{label} plane {field}",
        )
    if label == "primary":
        distances = _finite_array(
            authority.get("maximum_captured_segment_distances"),
            label="primary maximum captured segment distances",
            shape=(4,),
        )
        if np.any(distances < 0.0) or np.any(distances >= capture_radius):
            raise SupportPlaneContractError(
                "primary capture exceeded the fixed segment corridor"
            )
    else:
        owner_scores = _finite_array(
            authority.get("minimum_captured_owner_scores"),
            label="crosscheck minimum captured owner scores",
            shape=(4,),
        )
        if np.any(owner_scores < float(minimum_owner_score)):
            raise SupportPlaneContractError(
                "crosscheck owner score fell below the fixed minimum"
            )
    return {
        "capture_counts": capture_counts,
        "band_sizes": band_sizes,
        "points": points,
        "plane": expected_plane,
    }


def validate_dual_authority_evidence(evidence) -> dict:
    """Strictly revalidate serialized v2 evidence and all derived numbers."""

    if not isinstance(evidence, dict) or evidence.get("schema") != EVIDENCE_SCHEMA:
        raise SupportPlaneContractError(
            "dual-authority support-plane schema is missing or unsupported"
        )
    if evidence.get("fallback_used") is not False:
        raise SupportPlaneContractError("support-plane fallback is forbidden")
    diagonal = _positive_finite(
        evidence.get("mesh_diagonal"), label="mesh diagonal"
    )
    thresholds = evidence.get("thresholds")
    if not isinstance(thresholds, dict):
        raise SupportPlaneContractError("support-plane thresholds are missing")
    band_thickness = contact_band_thickness(diagonal)
    capture_radius = diagonal * CAPTURE_RADIUS_RATIO
    exact_thresholds = {
        "capture_radius_ratio_of_mesh_diagonal": CAPTURE_RADIUS_RATIO,
        "capture_radius": capture_radius,
        "contact_band_absolute_floor": CONTACT_BAND_ABSOLUTE_FLOOR,
        "contact_band_ratio_of_mesh_diagonal": CONTACT_BAND_DIAGONAL_RATIO,
        "contact_band_thickness": band_thickness,
        "minimum_capture_vertices": MINIMUM_CAPTURE_VERTICES,
        "minimum_contact_band_vertices": MINIMUM_CONTACT_BAND_VERTICES,
        "minimum_weight_owner_score": MINIMUM_WEIGHT_OWNER_SCORE,
        "maximum_floor_delta_between_authorities": band_thickness,
        "maximum_contact_centroid_xy_delta_between_authorities": (
            capture_radius
        ),
    }
    for field, expected in exact_thresholds.items():
        actual = thresholds.get(field)
        if (
            isinstance(expected, int)
            and (
                not isinstance(actual, int)
                or isinstance(actual, bool)
                or actual != expected
            )
        ):
            raise SupportPlaneContractError(
                f"support-plane threshold {field} changed"
            )
        if isinstance(expected, float):
            _require_close(
                actual, expected, label=f"support-plane threshold {field}"
            )
    maximum_residual_ratio = _positive_finite(
        thresholds.get("maximum_plane_residual_ratio_of_mesh_diagonal"),
        label="maximum plane residual ratio",
    )
    maximum_tilt_deg = _positive_finite(
        thresholds.get("maximum_tilt_deg"),
        label="maximum tilt",
    )

    primary = _validate_authority(
        evidence.get("primary"),
        method=PRIMARY_METHOD,
        mesh_diagonal=diagonal,
        maximum_residual_ratio=maximum_residual_ratio,
        maximum_tilt_deg=maximum_tilt_deg,
        capture_radius=capture_radius,
        minimum_owner_score=None,
        label="primary",
    )
    crosscheck = _validate_authority(
        evidence.get("crosscheck"),
        method=CROSSCHECK_METHOD,
        mesh_diagonal=diagonal,
        maximum_residual_ratio=maximum_residual_ratio,
        maximum_tilt_deg=maximum_tilt_deg,
        capture_radius=capture_radius,
        minimum_owner_score=MINIMUM_WEIGHT_OWNER_SCORE,
        label="crosscheck",
    )
    floor_delta = np.abs(
        primary["points"][:, 2] - crosscheck["points"][:, 2]
    )
    centroid_xy_delta = np.linalg.norm(
        primary["points"][:, :2] - crosscheck["points"][:, :2],
        axis=1,
    )
    agreement = evidence.get("agreement")
    if not isinstance(agreement, dict) or agreement.get("passed") is not True:
        raise SupportPlaneContractError("support-plane agreement did not pass")
    _require_close(
        agreement.get("per_foot_floor_z_absolute_delta"),
        floor_delta,
        label="per-foot floor agreement",
    )
    _require_close(
        agreement.get("maximum_floor_z_absolute_delta"),
        float(floor_delta.max()),
        label="maximum floor agreement",
    )
    _require_close(
        agreement.get("maximum_allowed_floor_z_absolute_delta"),
        band_thickness,
        label="maximum allowed floor agreement",
    )
    _require_close(
        agreement.get("per_foot_contact_centroid_xy_distance"),
        centroid_xy_delta,
        label="per-foot contact centroid agreement",
    )
    _require_close(
        agreement.get("maximum_contact_centroid_xy_distance"),
        float(centroid_xy_delta.max()),
        label="maximum contact centroid agreement",
    )
    _require_close(
        agreement.get("maximum_allowed_contact_centroid_xy_distance"),
        capture_radius,
        label="maximum allowed contact centroid agreement",
    )
    if np.any(floor_delta > band_thickness):
        raise SupportPlaneContractError(
            "primary and crosscheck foot floors disagree beyond contact band"
        )
    if np.any(centroid_xy_delta > capture_radius):
        raise SupportPlaneContractError(
            "primary and crosscheck contact centroids disagree beyond corridor"
        )
    return {
        "mesh_diagonal": diagonal,
        "band_thickness": band_thickness,
        "capture_radius": capture_radius,
        "maximum_residual_ratio": maximum_residual_ratio,
        "maximum_tilt_deg": maximum_tilt_deg,
        "primary": primary,
        "crosscheck": crosscheck,
        "floor_delta": floor_delta,
        "centroid_xy_delta": centroid_xy_delta,
    }
