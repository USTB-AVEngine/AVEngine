"""Stdlib-only validation contract for generated-animal support planes.

This module is intentionally safe to import from the generic review runner,
whose validate-only path must not require Blender's NumPy environment.
"""

from __future__ import annotations

import math
from numbers import Integral, Real


EVIDENCE_SCHEMA = "avengine_generated_animal_support_plane_dual_authority_v2"
PRIMARY_METHOD = "nearest_complete_leaf_segment_voronoi_v2"
CROSSCHECK_METHOD = "distal_two_bone_weight_owner_v2"
POLICY = "four_semantic_feet_dual_authority_rigid_leveling_v2"

CAPTURE_RADIUS_RATIO = 0.05
CONTACT_BAND_ABSOLUTE_FLOOR = 0.004
CONTACT_BAND_DIAGONAL_RATIO = 0.003
MINIMUM_CAPTURE_VERTICES = 10
MINIMUM_CONTACT_BAND_VERTICES = 10
MINIMUM_WEIGHT_OWNER_SCORE = 0.1
RIGID_TRANSFORM_ABSOLUTE_TOLERANCE_RATIO = 5.0e-7
RIGID_TRANSFORM_RELATIVE_TOLERANCE = 1.0e-7
LEGACY_OUTPUT_READBACK_SCHEMA = (
    "avengine_generated_animal_support_plane_output_readback_v1"
)
OUTPUT_READBACK_SCHEMA = (
    "avengine_generated_animal_support_plane_output_readback_v2"
)
MAXIMUM_RIGID_VERTEX_DELTA_RATIO = 5.0e-6
MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO = 5.0e-6
MAXIMUM_SKIN_WEIGHT_DELTA = 5.0e-6
MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG = 1.0
MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO = 0.01
MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA = 0.005


class SupportPlaneContractError(RuntimeError):
    """Fail-closed support-plane evidence error."""


def _number(value, *, label: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise SupportPlaneContractError(f"{label} must be {qualifier}")
    return float(value)


def _vector(value, *, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SupportPlaneContractError(
            f"{label} must contain {length} finite values"
        )
    return [_number(item, label=label) for item in value]


def _points(value, *, label: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SupportPlaneContractError(f"{label} must contain four points")
    return [
        _vector(point, length=3, label=f"{label} point")
        for point in value
    ]


def _counts(value, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, Integral)
            or int(item) < 0
            for item in value
        )
    ):
        raise SupportPlaneContractError(f"{label} must contain four counts")
    return [int(item) for item in value]


def _close(actual, expected, *, label: str, tolerance: float = 1.0e-10):
    if isinstance(expected, (list, tuple)):
        if (
            not isinstance(actual, (list, tuple))
            or len(actual) != len(expected)
        ):
            raise SupportPlaneContractError(f"{label} is internally inconsistent")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _close(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
                tolerance=tolerance,
            )
        return
    actual_number = _number(actual, label=label)
    expected_number = _number(expected, label=f"expected {label}")
    if not math.isclose(
        actual_number,
        expected_number,
        rel_tol=1.0e-10,
        abs_tol=tolerance,
    ):
        raise SupportPlaneContractError(f"{label} is internally inconsistent")


def _transform_close(
    actual,
    expected,
    *,
    label: str,
    absolute_tolerance: float,
) -> None:
    if isinstance(expected, (list, tuple)):
        if (
            not isinstance(actual, (list, tuple))
            or len(actual) != len(expected)
        ):
            raise SupportPlaneContractError(
                f"{label} is inconsistent with the rigid leveling transform"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _transform_close(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
                absolute_tolerance=absolute_tolerance,
            )
        return
    actual_number = _number(actual, label=label)
    expected_number = _number(expected, label=f"expected {label}")
    if not math.isclose(
        actual_number,
        expected_number,
        rel_tol=RIGID_TRANSFORM_RELATIVE_TOLERANCE,
        abs_tol=absolute_tolerance,
    ):
        raise SupportPlaneContractError(
            f"{label} is inconsistent with the rigid leveling transform"
        )


def _rotation_from_normal_to_positive_z(normal) -> list[list[float]]:
    """Return the unique shortest-arc rotation from ``normal`` to +Z.

    Blender's ``Vector.rotation_difference`` constructs the same quaternion.
    The fitted support-plane normal always has a positive Z component, so the
    antiparallel ambiguity cannot occur.
    """

    x, y, z = _vector(normal, length=3, label="primary support-plane normal")
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        raise SupportPlaneContractError("primary support-plane normal is zero")
    x, y, z = x / length, y / length, z / length
    if z <= 0.0:
        raise SupportPlaneContractError(
            "primary support-plane normal must point toward positive Z"
        )

    # Quaternion [w, x, y, z] rotating n onto +Z:
    # normalize([1 + dot(n, +Z), cross(n, +Z)]).
    quaternion = [1.0 + z, y, -x, 0.0]
    quaternion_length = math.sqrt(
        sum(value * value for value in quaternion)
    )
    if quaternion_length <= 0.0:
        raise SupportPlaneContractError(
            "primary support-plane leveling rotation is ambiguous"
        )
    w, qx, qy, qz = [
        value / quaternion_length for value in quaternion
    ]
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * w),
            2.0 * (qx * qz + qy * w),
        ],
        [
            2.0 * (qx * qy + qz * w),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * w),
        ],
        [
            2.0 * (qx * qz - qy * w),
            2.0 * (qy * qz + qx * w),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def _rotate_point(rotation, point) -> list[float]:
    return [
        sum(rotation[row][column] * point[column] for column in range(3))
        for row in range(3)
    ]


def contact_band_thickness(mesh_diagonal: float) -> float:
    diagonal = _number(
        mesh_diagonal, label="mesh diagonal", positive=True
    )
    return max(
        CONTACT_BAND_ABSOLUTE_FLOOR,
        diagonal * CONTACT_BAND_DIAGONAL_RATIO,
    )


def _solve_three_by_three(matrix, vector, *, label: str) -> list[float]:
    augmented = [
        [float(matrix[row][column]) for column in range(3)]
        + [float(vector[row])]
        for row in range(3)
    ]
    scale = max(abs(value) for row in matrix for value in row)
    if not math.isfinite(scale) or scale <= 0.0:
        raise SupportPlaneContractError(f"{label} has a singular design")
    tolerance = scale * 1.0e-12
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise SupportPlaneContractError(f"{label} has a singular design")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [augmented[row][3] for row in range(3)]


def _fit_plane(
    points,
    *,
    mesh_diagonal: float,
    maximum_residual_ratio: float,
    maximum_tilt_deg: float,
    label: str,
) -> dict:
    points = _points(points, label=f"{label} foot points")
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    design_normal = [
        [
            sum(point[0] * point[0] for point in points),
            sum(point[0] * point[1] for point in points),
            sum_x,
        ],
        [
            sum(point[0] * point[1] for point in points),
            sum(point[1] * point[1] for point in points),
            sum_y,
        ],
        [sum_x, sum_y, 4.0],
    ]
    target_normal = [
        sum(point[0] * point[2] for point in points),
        sum(point[1] * point[2] for point in points),
        sum(point[2] for point in points),
    ]
    coefficients = _solve_three_by_three(
        design_normal, target_normal, label=label
    )
    residuals = [
        point[2]
        - (
            coefficients[0] * point[0]
            + coefficients[1] * point[1]
            + coefficients[2]
        )
        for point in points
    ]
    maximum_residual = max(abs(value) for value in residuals)
    residual_ratio = maximum_residual / mesh_diagonal
    slope = math.hypot(coefficients[0], coefficients[1])
    tilt_deg = math.degrees(math.atan(slope))
    normal_scale = math.sqrt(slope * slope + 1.0)
    normal = [
        -coefficients[0] / normal_scale,
        -coefficients[1] / normal_scale,
        1.0 / normal_scale,
    ]
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
        "z_equals_ax_plus_by_plus_c": coefficients,
        "residual_z": residuals,
        "maximum_residual": maximum_residual,
        "maximum_residual_ratio_of_mesh_diagonal": residual_ratio,
        "normal": normal,
        "tilt_deg": tilt_deg,
    }


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
    capture_counts = _counts(
        authority.get("capture_counts"), label=f"{label} capture counts"
    )
    band_sizes = _counts(
        authority.get("contact_band_sizes"),
        label=f"{label} contact band sizes",
    )
    if any(value < MINIMUM_CAPTURE_VERTICES for value in capture_counts):
        raise SupportPlaneContractError(f"{label} capture is sparse")
    if any(value < MINIMUM_CONTACT_BAND_VERTICES for value in band_sizes):
        raise SupportPlaneContractError(f"{label} contact band is sparse")
    points = _points(
        authority.get("foot_points"), label=f"{label} foot points"
    )
    expected_plane = _fit_plane(
        points,
        mesh_diagonal=mesh_diagonal,
        maximum_residual_ratio=maximum_residual_ratio,
        maximum_tilt_deg=maximum_tilt_deg,
        label=f"{label} support plane",
    )
    plane = authority.get("plane")
    if not isinstance(plane, dict):
        raise SupportPlaneContractError(f"{label} plane is missing")
    for field, expected in expected_plane.items():
        _close(
            plane.get(field),
            expected,
            label=f"{label} plane {field}",
        )
    if (
        plane.get("rank") != 3
        or isinstance(plane.get("rank"), bool)
    ):
        raise SupportPlaneContractError(f"{label} plane rank changed")
    singular = _vector(
        plane.get("singular_values"),
        length=3,
        label=f"{label} plane singular values",
    )
    if (
        any(value <= 0.0 for value in singular)
        or any(left < right for left, right in zip(singular, singular[1:]))
    ):
        raise SupportPlaneContractError(
            f"{label} plane singular values are invalid"
        )
    if label == "primary":
        distances = _vector(
            authority.get("maximum_captured_segment_distances"),
            length=4,
            label="primary maximum captured segment distances",
        )
        if any(value < 0.0 or value >= capture_radius for value in distances):
            raise SupportPlaneContractError(
                "primary capture exceeded the fixed segment corridor"
            )
    else:
        scores = _vector(
            authority.get("minimum_captured_owner_scores"),
            length=4,
            label="crosscheck minimum captured owner scores",
        )
        if any(value < float(minimum_owner_score) for value in scores):
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
    """Strictly revalidate serialized v2 evidence without third-party imports."""

    if not isinstance(evidence, dict) or evidence.get("schema") != EVIDENCE_SCHEMA:
        raise SupportPlaneContractError(
            "dual-authority support-plane schema is missing or unsupported"
        )
    if evidence.get("fallback_used") is not False:
        raise SupportPlaneContractError("support-plane fallback is forbidden")
    diagonal = _number(
        evidence.get("mesh_diagonal"),
        label="mesh diagonal",
        positive=True,
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
        if isinstance(expected, int):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, Integral)
                or int(actual) != expected
            ):
                raise SupportPlaneContractError(
                    f"support-plane threshold {field} changed"
                )
        else:
            _close(
                actual,
                expected,
                label=f"support-plane threshold {field}",
            )
    maximum_residual_ratio = _number(
        thresholds.get("maximum_plane_residual_ratio_of_mesh_diagonal"),
        label="maximum plane residual ratio",
        positive=True,
    )
    maximum_tilt_deg = _number(
        thresholds.get("maximum_tilt_deg"),
        label="maximum tilt",
        positive=True,
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
    floor_delta = [
        abs(primary_point[2] - crosscheck_point[2])
        for primary_point, crosscheck_point in zip(
            primary["points"], crosscheck["points"]
        )
    ]
    centroid_xy_delta = [
        math.hypot(
            primary_point[0] - crosscheck_point[0],
            primary_point[1] - crosscheck_point[1],
        )
        for primary_point, crosscheck_point in zip(
            primary["points"], crosscheck["points"]
        )
    ]
    agreement = evidence.get("agreement")
    if not isinstance(agreement, dict) or agreement.get("passed") is not True:
        raise SupportPlaneContractError("support-plane agreement did not pass")
    _close(
        agreement.get("per_foot_floor_z_absolute_delta"),
        floor_delta,
        label="per-foot floor agreement",
    )
    _close(
        agreement.get("maximum_floor_z_absolute_delta"),
        max(floor_delta),
        label="maximum floor agreement",
    )
    _close(
        agreement.get("maximum_allowed_floor_z_absolute_delta"),
        band_thickness,
        label="maximum allowed floor agreement",
    )
    _close(
        agreement.get("per_foot_contact_centroid_xy_distance"),
        centroid_xy_delta,
        label="per-foot contact centroid agreement",
    )
    _close(
        agreement.get("maximum_contact_centroid_xy_distance"),
        max(centroid_xy_delta),
        label="maximum contact centroid agreement",
    )
    _close(
        agreement.get("maximum_allowed_contact_centroid_xy_distance"),
        capture_radius,
        label="maximum allowed contact centroid agreement",
    )
    if any(value > band_thickness for value in floor_delta):
        raise SupportPlaneContractError(
            "primary and crosscheck foot floors disagree beyond contact band"
        )
    if any(value > capture_radius for value in centroid_xy_delta):
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


def validate_rigid_leveling_transform(evidence, support_plane) -> dict:
    """Recompute all serialized post-leveling fields from pre-leveling evidence.

    The manifest is allowed to report the rigid transform, but it is never an
    authority for that transform.  This stdlib-only verifier derives the same
    shortest-arc rotation and minimum-foot translation used by Blender from
    the independently revalidated primary plane, applies that one transform to
    both authorities, and rejects any inconsistent readback.
    """

    validated = validate_dual_authority_evidence(evidence)
    if not isinstance(support_plane, dict):
        raise SupportPlaneContractError(
            "support-plane leveling measurements are missing"
        )

    primary_points = validated["primary"]["points"]
    crosscheck_points = validated["crosscheck"]["points"]
    rotation = _rotation_from_normal_to_positive_z(
        validated["primary"]["plane"]["normal"]
    )
    rotated_primary = [
        _rotate_point(rotation, point) for point in primary_points
    ]
    rotated_crosscheck = [
        _rotate_point(rotation, point) for point in crosscheck_points
    ]
    vertical_translation = -min(point[2] for point in rotated_primary)
    expected_primary_after = [
        [point[0], point[1], point[2] + vertical_translation]
        for point in rotated_primary
    ]
    expected_crosscheck_after = [
        [point[0], point[1], point[2] + vertical_translation]
        for point in rotated_crosscheck
    ]
    expected_minimum_z = min(point[2] for point in expected_primary_after)

    coordinate_scale = max(
        1.0,
        validated["mesh_diagonal"],
        *(
            abs(value)
            for point in primary_points + crosscheck_points
            for value in point
        ),
    )
    absolute_tolerance = (
        RIGID_TRANSFORM_ABSOLUTE_TOLERANCE_RATIO * coordinate_scale
    )
    _transform_close(
        support_plane.get("applied_vertical_translation"),
        vertical_translation,
        label="applied vertical translation",
        absolute_tolerance=absolute_tolerance,
    )
    _transform_close(
        support_plane.get("foot_points_after"),
        expected_primary_after,
        label="primary foot points after leveling",
        absolute_tolerance=absolute_tolerance,
    )
    _transform_close(
        support_plane.get("crosscheck_foot_points_after"),
        expected_crosscheck_after,
        label="crosscheck foot points after leveling",
        absolute_tolerance=absolute_tolerance,
    )
    _transform_close(
        support_plane.get("minimum_foot_z_after"),
        expected_minimum_z,
        label="minimum primary foot Z after leveling",
        absolute_tolerance=absolute_tolerance,
    )
    if abs(expected_minimum_z) > absolute_tolerance:
        raise SupportPlaneContractError(
            "recomputed rigid leveling did not place the lowest primary foot at Z=0"
        )
    return {
        **validated,
        "rotation_from_primary_normal_to_positive_z": rotation,
        "vertical_translation": vertical_translation,
        "primary_points_after": expected_primary_after,
        "crosscheck_points_after": expected_crosscheck_after,
        "minimum_primary_z_after": expected_minimum_z,
        "transform_absolute_tolerance": absolute_tolerance,
        "transform_relative_tolerance": RIGID_TRANSFORM_RELATIVE_TOLERANCE,
    }


def _maximum_point_distance(actual, expected, *, label: str) -> float:
    actual_points = _points(actual, label=label)
    expected_points = _points(expected, label=f"expected {label}")
    return max(
        math.sqrt(
            sum(
                (actual_value - expected_value) ** 2
                for actual_value, expected_value in zip(
                    actual_point, expected_point
                )
            )
        )
        for actual_point, expected_point in zip(
            actual_points, expected_points
        )
    )


def _scene_summary(value, *, label: str) -> dict:
    keys = {
        "mesh_count",
        "skinned_mesh_count",
        "armature_count",
        "bone_count",
        "material_count",
        "image_count",
        "action_count",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(
            isinstance(item, bool)
            or not isinstance(item, Integral)
            or int(item) < 0
            for item in value.values()
        )
    ):
        raise SupportPlaneContractError(f"{label} scene summary is invalid")
    return {key: int(value[key]) for key in keys}


def _matrix(value, *, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise SupportPlaneContractError(f"{label} must be a finite 4x4 matrix")
    return [
        _vector(row, length=4, label=f"{label} row")
        for row in value
    ]


def _sha256(value, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SupportPlaneContractError(f"{label} must be a SHA-256 digest")
    return value


def validate_output_glb_readback(
    readback,
    *,
    source_evidence,
    support_plane,
    expected_scene,
    expected_semantic_rig,
    expected_vertex_count: int,
    expected_polygon_count: int,
) -> dict:
    """Validate an independent Blender re-import of the leveled GLB bytes."""

    readback_schema = (
        readback.get("schema") if isinstance(readback, dict) else None
    )
    if (
        not isinstance(readback, dict)
        or readback_schema not in {
            LEGACY_OUTPUT_READBACK_SCHEMA,
            OUTPUT_READBACK_SCHEMA,
        }
        or readback.get("status") != "passed_independent_glb_reimport"
        or readback.get("formal_dataset_registration_authorized") is not False
    ):
        raise SupportPlaneContractError(
            "support-plane output GLB readback is missing or unsupported"
        )
    transform = validate_rigid_leveling_transform(
        source_evidence, support_plane
    )
    expected_scene = _scene_summary(
        expected_scene, label="expected support-plane"
    )
    pre_scene = _scene_summary(
        readback.get("pre_level_scene"), label="pre-level readback"
    )
    post_scene = _scene_summary(
        readback.get("post_level_scene"), label="post-level readback"
    )
    if pre_scene != expected_scene or post_scene != expected_scene:
        raise SupportPlaneContractError(
            "independently imported scene does not match the support manifest"
        )
    if (
        readback.get("pre_level_semantic_rig") != expected_semantic_rig
        or readback.get("post_level_semantic_rig") != expected_semantic_rig
    ):
        raise SupportPlaneContractError(
            "independently inferred semantic feet or distal owners changed"
        )

    pre_evidence = readback.get("pre_level_dual_authority")
    post_evidence = readback.get("post_level_dual_authority")
    pre_validated = validate_dual_authority_evidence(pre_evidence)
    post_validated = validate_dual_authority_evidence(post_evidence)
    tolerance = transform["transform_absolute_tolerance"]
    _transform_close(
        pre_validated["mesh_diagonal"],
        transform["mesh_diagonal"],
        label="pre-level readback mesh diagonal",
        absolute_tolerance=tolerance,
    )
    comparison = readback.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("passed") is not True:
        raise SupportPlaneContractError(
            "support-plane output comparison did not pass"
        )
    if readback_schema == OUTPUT_READBACK_SCHEMA:
        if (
            comparison.get("bbox_diagonal_reference_method")
            != "pre_level_vertices_after_declared_rigid_transform_v1"
            or "post_level_bbox_diagonal_ratio_delta" in comparison
        ):
            raise SupportPlaneContractError(
                "post-level bounding-box reference is missing or unsupported"
            )
        expected_post_diagonal = _number(
            comparison.get("expected_post_level_bbox_diagonal"),
            label="expected post-level bounding-box diagonal",
            positive=True,
        )
        declared_pre_diagonal = _number(
            comparison.get("pre_level_bbox_diagonal"),
            label="declared pre-level bounding-box diagonal",
            positive=True,
        )
        declared_actual_post_diagonal = _number(
            comparison.get("actual_post_level_bbox_diagonal"),
            label="declared actual post-level bounding-box diagonal",
            positive=True,
        )
        _transform_close(
            declared_pre_diagonal,
            pre_validated["mesh_diagonal"],
            label="declared pre-level bounding-box diagonal",
            absolute_tolerance=tolerance,
        )
        _transform_close(
            declared_actual_post_diagonal,
            post_validated["mesh_diagonal"],
            label="declared actual post-level bounding-box diagonal",
            absolute_tolerance=tolerance,
        )
        post_diagonal_ratio_delta = abs(
            post_validated["mesh_diagonal"] - expected_post_diagonal
        ) / expected_post_diagonal
        bbox_ratio_field = (
            "post_level_bbox_diagonal_ratio_delta_from_expected_rigid_transform"
        )
        bbox_threshold_field = (
            "maximum_post_level_bbox_diagonal_ratio_delta_from_"
            "expected_rigid_transform"
        )
    else:
        expected_post_diagonal = transform["mesh_diagonal"]
        post_diagonal_ratio_delta = abs(
            post_validated["mesh_diagonal"] - expected_post_diagonal
        ) / expected_post_diagonal
        bbox_ratio_field = "post_level_bbox_diagonal_ratio_delta"
        bbox_threshold_field = "maximum_post_level_bbox_diagonal_ratio_delta"
    if (
        post_diagonal_ratio_delta
        > MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA
    ):
        raise SupportPlaneContractError(
            "post-level GLB bounding-box diagonal changed beyond its "
            "declared rigid-transform reference"
        )
    source = validate_dual_authority_evidence(source_evidence)
    pre_primary_delta = _maximum_point_distance(
        pre_validated["primary"]["points"],
        source["primary"]["points"],
        label="pre-level primary foot readback",
    )
    pre_cross_delta = _maximum_point_distance(
        pre_validated["crosscheck"]["points"],
        source["crosscheck"]["points"],
        label="pre-level crosscheck foot readback",
    )
    post_primary_delta = _maximum_point_distance(
        post_validated["primary"]["points"],
        transform["primary_points_after"],
        label="post-level primary foot readback",
    )
    post_cross_delta = _maximum_point_distance(
        post_validated["crosscheck"]["points"],
        transform["crosscheck_points_after"],
        label="post-level crosscheck foot readback",
    )
    if max(pre_primary_delta, pre_cross_delta) > tolerance:
        raise SupportPlaneContractError(
            "pre-level GLB readback does not reproduce the admitted feet"
        )
    if max(post_primary_delta, post_cross_delta) > transform["capture_radius"]:
        raise SupportPlaneContractError(
            "post-level GLB semantic feet moved outside the fixed corridor"
        )
    actual_minimum_z = min(
        point[2] for point in post_validated["primary"]["points"]
    )
    if (
        abs(
            actual_minimum_z
            - _number(
                support_plane.get("minimum_foot_z_after"),
                label="declared minimum foot Z",
            )
        )
        > transform["band_thickness"]
    ):
        raise SupportPlaneContractError(
            "post-level GLB foot floor moved outside the fixed contact band"
        )
    post_tilt = post_validated["primary"]["plane"]["tilt_deg"]
    if post_tilt > MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG:
        raise SupportPlaneContractError(
            "post-level GLB primary support plane is not level"
        )

    thresholds = readback.get("thresholds")
    if not isinstance(thresholds, dict):
        raise SupportPlaneContractError(
            "support-plane output readback thresholds are missing"
        )
    exact_thresholds = {
        "maximum_rigid_vertex_delta_ratio_of_mesh_diagonal": (
            MAXIMUM_RIGID_VERTEX_DELTA_RATIO
        ),
        "maximum_rigid_vertex_delta": (
            transform["mesh_diagonal"] * MAXIMUM_RIGID_VERTEX_DELTA_RATIO
        ),
        "maximum_rigid_bone_endpoint_delta_ratio_of_mesh_diagonal": (
            MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO
        ),
        "maximum_rigid_bone_endpoint_delta": (
            transform["mesh_diagonal"]
            * MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO
        ),
        "maximum_skin_weight_delta": MAXIMUM_SKIN_WEIGHT_DELTA,
        "maximum_primary_post_level_tilt_deg": (
            MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG
        ),
        "maximum_serialization_vertex_expansion_ratio": (
            MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO
        ),
        "maximum_foot_readback_delta": tolerance,
        "maximum_post_level_semantic_reacquisition_delta": transform[
            "capture_radius"
        ],
        "maximum_post_level_floor_reacquisition_delta": transform[
            "band_thickness"
        ],
        bbox_threshold_field: MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA,
    }
    obsolete_bbox_threshold = (
        "maximum_post_level_bbox_diagonal_ratio_delta"
        if readback_schema == OUTPUT_READBACK_SCHEMA
        else (
            "maximum_post_level_bbox_diagonal_ratio_delta_from_"
            "expected_rigid_transform"
        )
    )
    if obsolete_bbox_threshold in thresholds:
        raise SupportPlaneContractError(
            "support-plane output readback mixes bounding-box threshold "
            "contracts"
        )
    for field, expected in exact_thresholds.items():
        _transform_close(
            thresholds.get(field),
            expected,
            label=f"output readback threshold {field}",
            absolute_tolerance=tolerance,
        )

    binding = readback.get("mesh_and_weight_binding")
    if not isinstance(binding, dict):
        raise SupportPlaneContractError(
            "output mesh and post-smoothing weight binding is missing"
        )
    vertex_count = binding.get("vertex_count")
    serialized_vertex_count = binding.get("serialized_vertex_count")
    polygon_count = binding.get("polygon_count")
    if (
        isinstance(vertex_count, bool)
        or not isinstance(vertex_count, Integral)
        or int(vertex_count) != expected_vertex_count
        or isinstance(serialized_vertex_count, bool)
        or not isinstance(serialized_vertex_count, Integral)
        or int(serialized_vertex_count) < int(vertex_count)
        or isinstance(polygon_count, bool)
        or not isinstance(polygon_count, Integral)
        or int(polygon_count) != expected_polygon_count
    ):
        raise SupportPlaneContractError(
            "output mesh vertex or polygon binding changed"
        )
    expansion = (
        int(serialized_vertex_count) - int(vertex_count)
    ) / int(vertex_count)
    _transform_close(
        binding.get("serialization_vertex_expansion_ratio"),
        expansion,
        label="serialization vertex expansion ratio",
        absolute_tolerance=1.0e-12,
    )
    if expansion > MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO:
        raise SupportPlaneContractError(
            "output GLB vertex splitting exceeded the fixed limit"
        )
    topology_before = _sha256(
        binding.get("topology_sha256_before"),
        label="pre-level topology digest",
    )
    topology_after = _sha256(
        binding.get("topology_sha256_after"),
        label="post-level topology digest",
    )
    if int(serialized_vertex_count) == int(vertex_count) and (
        topology_before != topology_after
    ):
        raise SupportPlaneContractError(
            "output topology changed without a serialization vertex split"
        )
    bone_names = binding.get("bone_group_names")
    owner_bones = {
        bone
        for owners in expected_semantic_rig["distal_owner_bones"]
        for bone in owners
    }
    if (
        not isinstance(bone_names, list)
        or len(bone_names) != expected_scene["bone_count"]
        or len(set(bone_names)) != len(bone_names)
        or not owner_bones.issubset(set(bone_names))
    ):
        raise SupportPlaneContractError(
            "post-smoothing weight groups do not bind every distal owner"
        )
    maximum_weight_delta = _number(
        binding.get("maximum_skin_weight_delta"),
        label="maximum skin-weight delta",
    )
    if (
        maximum_weight_delta < 0.0
        or maximum_weight_delta > MAXIMUM_SKIN_WEIGHT_DELTA
        or binding.get("weights_changed_above_tolerance") != 0
    ):
        raise SupportPlaneContractError(
            "post-smoothing weights changed during support-plane export"
        )
    for field in ("materials", "uv_layers"):
        value = binding.get(field)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise SupportPlaneContractError(
                f"output mesh {field} binding is incomplete"
            )

    recomputed_comparison = {
        "pre_level_primary_foot_delta": pre_primary_delta,
        "pre_level_crosscheck_foot_delta": pre_cross_delta,
        "post_level_primary_foot_delta": post_primary_delta,
        "post_level_crosscheck_foot_delta": post_cross_delta,
        "actual_minimum_primary_foot_z": actual_minimum_z,
        "primary_post_level_tilt_deg": post_tilt,
        bbox_ratio_field: post_diagonal_ratio_delta,
    }
    for field, expected in recomputed_comparison.items():
        _transform_close(
            comparison.get(field),
            expected,
            label=f"output readback comparison {field}",
            absolute_tolerance=tolerance,
        )
    maximum_vertex_delta = _number(
        comparison.get(
            "maximum_world_vertex_delta_from_declared_transform"
        ),
        label="maximum readback vertex delta",
    )
    maximum_bone_delta = _number(
        comparison.get(
            "maximum_bone_endpoint_delta_from_declared_transform"
        ),
        label="maximum readback bone endpoint delta",
    )
    if (
        maximum_vertex_delta < 0.0
        or maximum_vertex_delta
        > transform["mesh_diagonal"] * MAXIMUM_RIGID_VERTEX_DELTA_RATIO
        or maximum_bone_delta < 0.0
        or maximum_bone_delta
        > transform["mesh_diagonal"]
        * MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO
    ):
        raise SupportPlaneContractError(
            "output GLB does not contain the declared mesh/skeleton transform"
        )

    object_readback = readback.get("object_transform_readback")
    if not isinstance(object_readback, dict):
        raise SupportPlaneContractError(
            "output object-transform readback is missing"
        )
    for field in (
        "pre_mesh_world_matrix",
        "post_mesh_world_matrix",
        "pre_armature_world_matrix",
        "post_armature_world_matrix",
    ):
        _matrix(object_readback.get(field), label=field)
    for field in ("pre_root_objects", "post_root_objects"):
        roots = object_readback.get(field)
        if (
            not isinstance(roots, list)
            or not roots
            or any(
                not isinstance(root, dict)
                or not isinstance(root.get("name"), str)
                or not isinstance(root.get("type"), str)
                for root in roots
            )
        ):
            raise SupportPlaneContractError(
                f"output object-transform roots are invalid: {field}"
            )
        for root in roots:
            _matrix(root.get("world_matrix"), label=f"{field} root matrix")
    return {
        "transform": transform,
        "pre_level": pre_validated,
        "post_level": post_validated,
        "maximum_vertex_delta": maximum_vertex_delta,
        "maximum_bone_endpoint_delta": maximum_bone_delta,
        "maximum_skin_weight_delta": maximum_weight_delta,
        "serialization_vertex_expansion_ratio": expansion,
    }
