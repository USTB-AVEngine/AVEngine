"""Strict M2 forward kinematics, semantic anchors, and contact evidence.

The runtime action representation stores one absolute child-local xyzw
rotation for every non-root skin joint.  This module combines those rotations
with the authored local translations in :class:`HabitatAssetMapping` and the
explicit ``actor_from_skin_root`` transform.  It never advances a simulator or
infers a hidden root pose.

Contact phases are evidence derived from four declared paw-anchor trajectories.
Idle is required to remain planted.  A walking paw is called dynamic only when
its vertical excursion clears an explicit metric threshold.  In particular,
low-excursion hind paws stay marked in contact and emit a low-confidence
warning; this module does not pretend to repair an unsupported gait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np

from avengine.assets.actions import (
    SAMPLE_RATE_HZ,
    TIME_BASE_HZ,
    BakedActionSet,
    baked_actions_content_sha256,
)
from avengine.assets.habitat import HabitatAssetMapping, HabitatJointRest


CONTACT_ORDER: tuple[str, str, str, str] = (
    "paw_front_left",
    "paw_front_right",
    "paw_hind_left",
    "paw_hind_right",
)

_MINIMUM_CONTACT_SAMPLE_COUNT = 3
_QUATERNION_TOLERANCE = 1.0e-9
_RIGID_TOLERANCE = 1.0e-7
_HEMISPHERE_ZERO_TOLERANCE = 1.0e-15

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


class KinematicsError(ValueError):
    """An M2 pose, anchor, mapping, or contact input is not admissible."""


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _has_negative_zero(value: float) -> bool:
    return float(value) == 0.0 and math.copysign(1.0, float(value)) < 0.0


def _canonical_float(value: float) -> float:
    number = float(value)
    if number == 0.0:
        return 0.0
    nearest_integer = round(number)
    if math.isclose(
        number,
        float(nearest_integer),
        rel_tol=0.0,
        abs_tol=1.0e-15 * max(1.0, abs(number)),
    ):
        return float(nearest_integer)
    return number


def _strict_vector(
    value: Any,
    *,
    length: int,
    owner: str,
    require_tuple: bool,
) -> tuple[float, ...]:
    if require_tuple and not isinstance(value, tuple):
        raise KinematicsError(f"{owner} must be an immutable tuple")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise KinematicsError(f"{owner} must contain {length} finite numbers")
    if len(value) != length or not all(_is_number(component) for component in value):
        raise KinematicsError(f"{owner} must contain {length} finite numbers")
    result = tuple(float(component) for component in value)
    if any(_has_negative_zero(component) for component in result):
        raise KinematicsError(f"{owner} must use canonical positive signed zero")
    return result


def _quaternion_sign_component(quaternion: Sequence[float]) -> float:
    scalar = float(quaternion[3])
    if math.isclose(scalar, 0.0, rel_tol=0.0, abs_tol=_HEMISPHERE_ZERO_TOLERANCE):
        for component in quaternion[:3]:
            if not math.isclose(
                float(component),
                0.0,
                rel_tol=0.0,
                abs_tol=_HEMISPHERE_ZERO_TOLERANCE,
            ):
                return float(component)
    return scalar


def _strict_quaternion(
    value: Any,
    *,
    owner: str,
    require_tuple: bool,
) -> QuaternionXYZW:
    quaternion = _strict_vector(
        value, length=4, owner=owner, require_tuple=require_tuple
    )
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_TOLERANCE):
        raise KinematicsError(f"{owner} must already be unit normalized")
    if _quaternion_sign_component(quaternion) < 0.0:
        raise KinematicsError(f"{owner} must use the canonical quaternion hemisphere")
    return quaternion


def _canonical_quaternion(value: Sequence[float]) -> QuaternionXYZW:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise KinematicsError("computed quaternion is not finite xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise KinematicsError("computed quaternion is zero")
    quaternion /= norm
    if _quaternion_sign_component(quaternion) < 0.0:
        quaternion = -quaternion
    return tuple(_canonical_float(component) for component in quaternion)  # type: ignore[return-value]


def _multiply_quaternions(
    left: Sequence[float], right: Sequence[float]
) -> QuaternionXYZW:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _canonical_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> Vector3:
    q_vector = np.asarray(quaternion[:3], dtype=np.float64)
    point = np.asarray(vector, dtype=np.float64)
    uv = np.cross(q_vector, point)
    uuv = np.cross(q_vector, uv)
    rotated = point + 2.0 * (float(quaternion[3]) * uv + uuv)
    return tuple(_canonical_float(component) for component in rotated)  # type: ignore[return-value]


def _matrix_rotation_to_quaternion(rotation: np.ndarray) -> QuaternionXYZW:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = (
                math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            quaternion = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = (
                math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            quaternion = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = (
                math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            quaternion = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    return _canonical_quaternion(quaternion)


@dataclass(frozen=True)
class RigidTransform:
    """Canonical ``parent_from_child`` transform in meters and xyzw."""

    translation_m: Vector3
    rotation_xyzw: QuaternionXYZW

    def __post_init__(self) -> None:
        translation = _strict_vector(
            self.translation_m,
            length=3,
            owner="translation_m",
            require_tuple=True,
        )
        rotation = _strict_quaternion(
            self.rotation_xyzw,
            owner="rotation_xyzw",
            require_tuple=True,
        )
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "rotation_xyzw", rotation)

    @classmethod
    def identity(cls) -> RigidTransform:
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def to_json_data(self) -> dict[str, list[float]]:
        return {
            "translation_m": list(self.translation_m),
            "rotation_xyzw": list(self.rotation_xyzw),
        }


def _computed_transform(
    translation: Sequence[float], rotation: Sequence[float]
) -> RigidTransform:
    canonical_translation = tuple(
        _canonical_float(component) for component in translation
    )
    return RigidTransform(
        canonical_translation,  # type: ignore[arg-type]
        _canonical_quaternion(rotation),
    )


def _compose(left: RigidTransform, right: RigidTransform) -> RigidTransform:
    rotated = _rotate_vector(left.rotation_xyzw, right.translation_m)
    translation = tuple(
        left_component + right_component
        for left_component, right_component in zip(
            left.translation_m, rotated, strict=True
        )
    )
    rotation = _multiply_quaternions(left.rotation_xyzw, right.rotation_xyzw)
    return _computed_transform(translation, rotation)


def _rigid_transform_from_matrix(value: Any, *, owner: str) -> RigidTransform:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise KinematicsError(f"{owner} must be a finite proper rigid 4x4") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise KinematicsError(f"{owner} must be a finite proper rigid 4x4")
    if np.any((matrix == 0.0) & np.signbit(matrix)):
        raise KinematicsError(f"{owner} must use canonical positive signed zero")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=_RIGID_TOLERANCE
    ):
        raise KinematicsError(f"{owner} must be a finite proper rigid 4x4")
    rotation = matrix[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    determinant = float(np.linalg.det(rotation))
    if (
        orthogonality_error > _RIGID_TOLERANCE
        or abs(determinant - 1.0) > _RIGID_TOLERANCE
    ):
        raise KinematicsError(
            f"{owner} must be a proper rigid transform "
            f"(orthogonality={orthogonality_error:.9g}, determinant={determinant:.9g})"
        )
    return _computed_transform(matrix[:3, 3], _matrix_rotation_to_quaternion(rotation))


def _validate_joint_rest(joint: HabitatJointRest, *, ordinal: int) -> None:
    if not isinstance(joint, HabitatJointRest):
        raise KinematicsError("mapping.joints must contain HabitatJointRest values")
    if joint.joint_ordinal != ordinal:
        raise KinematicsError("joint ordinals must exactly follow joint_order")
    if (
        isinstance(joint.node_index, bool)
        or not isinstance(joint.node_index, int)
        or joint.node_index < 0
    ):
        raise KinematicsError("joint node_index must be a non-negative integer")
    if not isinstance(joint.joint_id, str) or not joint.joint_id:
        raise KinematicsError("every joint must have a non-empty joint_id")
    if joint.parent_joint_id is not None and (
        not isinstance(joint.parent_joint_id, str) or not joint.parent_joint_id
    ):
        raise KinematicsError("parent_joint_id must be null or a non-empty string")
    _strict_vector(
        joint.local_translation_m,
        length=3,
        owner=f"joint {joint.joint_id!r} local_translation_m",
        require_tuple=True,
    )
    _strict_quaternion(
        joint.rest_rotation_xyzw,
        owner=f"joint {joint.joint_id!r} rest_rotation_xyzw",
        require_tuple=True,
    )
    scale = _strict_vector(
        joint.local_scale,
        length=3,
        owner=f"joint {joint.joint_id!r} local_scale",
        require_tuple=True,
    )
    if any(
        not math.isclose(component, 1.0, rel_tol=0.0, abs_tol=_RIGID_TOLERANCE)
        for component in scale
    ):
        raise KinematicsError(f"joint {joint.joint_id!r} local_scale must be unit")


@dataclass(frozen=True)
class _ValidatedMapping:
    mapping: HabitatAssetMapping
    actor_from_skin_root: RigidTransform
    by_name: dict[str, HabitatJointRest] = field(compare=False, repr=False)


def _validate_mapping(mapping: HabitatAssetMapping) -> _ValidatedMapping:
    if not isinstance(mapping, HabitatAssetMapping):
        raise KinematicsError("mapping must be a HabitatAssetMapping")
    if not (
        isinstance(mapping.source_glb_sha256, str)
        and len(mapping.source_glb_sha256) == 64
        and all(
            character in "0123456789abcdef" for character in mapping.source_glb_sha256
        )
    ):
        raise KinematicsError("mapping source_glb_sha256 must be lowercase SHA-256")
    if not isinstance(mapping.joints, tuple) or not mapping.joints:
        raise KinematicsError("mapping.joints must be a non-empty immutable tuple")
    for ordinal, joint in enumerate(mapping.joints):
        _validate_joint_rest(joint, ordinal=ordinal)
    joint_ids = tuple(joint.joint_id for joint in mapping.joints)
    if not isinstance(mapping.joint_order, tuple) or mapping.joint_order != joint_ids:
        raise KinematicsError("mapping joint_order must exactly follow mapping.joints")
    if len(set(joint_ids)) != len(joint_ids):
        raise KinematicsError("mapping joint IDs must be unique")
    node_indices = tuple(joint.node_index for joint in mapping.joints)
    if len(set(node_indices)) != len(node_indices):
        raise KinematicsError("mapping node indices must be unique")
    roots = [joint for joint in mapping.joints if joint.parent_joint_id is None]
    if len(roots) != 1 or roots[0].joint_id != mapping.root_joint_id:
        raise KinematicsError(
            "mapping must contain exactly one root matching root_joint_id"
        )
    expected_runtime_order = tuple(
        joint_id for joint_id in joint_ids if joint_id != mapping.root_joint_id
    )
    if (
        not isinstance(mapping.runtime_joint_order, tuple)
        or mapping.runtime_joint_order != expected_runtime_order
    ):
        raise KinematicsError(
            "mapping runtime_joint_order must equal joint_order without the root"
        )
    by_name = {joint.joint_id: joint for joint in mapping.joints}
    for joint in mapping.joints:
        if joint.parent_joint_id is not None and joint.parent_joint_id not in by_name:
            raise KinematicsError(
                f"joint {joint.joint_id!r} has unknown parent {joint.parent_joint_id!r}"
            )
    states: dict[str, int] = {}

    def visit(joint_id: str) -> None:
        state = states.get(joint_id, 0)
        if state == 1:
            raise KinematicsError("mapping joint hierarchy contains a cycle")
        if state == 2:
            return
        states[joint_id] = 1
        parent = by_name[joint_id].parent_joint_id
        if parent is not None:
            visit(parent)
        states[joint_id] = 2

    for joint_id in joint_ids:
        visit(joint_id)
    root = roots[0]
    if root.local_translation_m != (0.0, 0.0, 0.0) or root.rest_rotation_xyzw != (
        0.0,
        0.0,
        0.0,
        1.0,
    ):
        raise KinematicsError(
            "mapping root rest transform must be identity; root placement is explicit"
        )
    if not isinstance(mapping.actor_from_skin_root_source, str) or not (
        mapping.actor_from_skin_root_source.strip()
    ):
        raise KinematicsError("actor_from_skin_root_source must be explicit")
    actor_from_skin_root = _rigid_transform_from_matrix(
        mapping.actor_from_skin_root, owner="actor_from_skin_root"
    )
    return _ValidatedMapping(mapping, actor_from_skin_root, by_name)


def _validate_pose(
    validated: _ValidatedMapping, pose_rotation_xyzw: Any
) -> dict[str, QuaternionXYZW]:
    try:
        raw_pose = np.asarray(pose_rotation_xyzw, dtype=object)
    except (TypeError, ValueError) as exc:
        raise KinematicsError(
            "pose_rotation_xyzw must be a finite (N, 4) numeric array"
        ) from exc
    expected_shape = (len(validated.mapping.runtime_joint_order), 4)
    if raw_pose.shape != expected_shape:
        raise KinematicsError(
            f"pose_rotation_xyzw must have shape {expected_shape}, got {raw_pose.shape}"
        )
    if not all(_is_number(component) for component in raw_pose.flat):
        raise KinematicsError(
            "pose_rotation_xyzw must be a finite (N, 4) numeric array"
        )
    pose = np.asarray(raw_pose, dtype=np.float64)
    if not np.all(np.isfinite(pose)):
        raise KinematicsError("pose_rotation_xyzw must contain only finite numbers")
    if np.any((pose == 0.0) & np.signbit(pose)):
        raise KinematicsError(
            "pose_rotation_xyzw must use canonical positive signed zero"
        )
    result: dict[str, QuaternionXYZW] = {}
    for index, joint_id in enumerate(validated.mapping.runtime_joint_order):
        result[joint_id] = _strict_quaternion(
            tuple(float(component) for component in pose[index]),
            owner=f"pose_rotation_xyzw[{index}] ({joint_id!r})",
            require_tuple=True,
        )
    return result


@dataclass(frozen=True)
class JointKinematicPose:
    joint_id: str
    actor_from_joint: RigidTransform


@dataclass(frozen=True)
class KinematicFrame:
    """One actor-space FK solution in the asset's declared joint order."""

    source_glb_sha256: str
    joint_order: tuple[str, ...]
    joints: tuple[JointKinematicPose, ...]

    def joint_transform(self, joint_id: str) -> RigidTransform:
        for joint in self.joints:
            if joint.joint_id == joint_id:
                return joint.actor_from_joint
        raise KeyError(f"unknown joint_id: {joint_id!r}")

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema": "avengine_m2_forward_kinematics_frame_v1",
            "source_glb_sha256": self.source_glb_sha256,
            "joint_order": list(self.joint_order),
            "joints": [
                {
                    "joint_id": joint.joint_id,
                    "actor_from_joint": joint.actor_from_joint.to_json_data(),
                }
                for joint in self.joints
            ],
        }


def forward_kinematics(
    mapping: HabitatAssetMapping, pose_rotation_xyzw: Any
) -> KinematicFrame:
    """Compute actor-space joints from absolute child-local xyzw rotations."""

    validated = _validate_mapping(mapping)
    pose_by_name = _validate_pose(validated, pose_rotation_xyzw)
    skin_from_joint: dict[str, RigidTransform] = {}

    def solve(joint_id: str) -> RigidTransform:
        if joint_id in skin_from_joint:
            return skin_from_joint[joint_id]
        joint = validated.by_name[joint_id]
        if joint.parent_joint_id is None:
            transform = RigidTransform.identity()
        else:
            parent = solve(joint.parent_joint_id)
            local = RigidTransform(
                joint.local_translation_m,
                pose_by_name[joint_id],
            )
            transform = _compose(parent, local)
        skin_from_joint[joint_id] = transform
        return transform

    joints = tuple(
        JointKinematicPose(
            joint_id=joint_id,
            actor_from_joint=_compose(validated.actor_from_skin_root, solve(joint_id)),
        )
        for joint_id in mapping.joint_order
    )
    return KinematicFrame(mapping.source_glb_sha256, mapping.joint_order, joints)


@dataclass(frozen=True)
class AnchorDefinition:
    """A semantic anchor rigidly attached to one named skin joint."""

    anchor_id: str
    joint_id: str
    joint_from_anchor: RigidTransform

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or not self.anchor_id:
            raise KinematicsError("anchor_id must be a non-empty string")
        if not isinstance(self.joint_id, str) or not self.joint_id:
            raise KinematicsError("joint_id must be a non-empty string")
        if not isinstance(self.joint_from_anchor, RigidTransform):
            raise KinematicsError("joint_from_anchor must be a RigidTransform")

    def to_json_data(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "joint_id": self.joint_id,
            "joint_from_anchor": self.joint_from_anchor.to_json_data(),
        }


def _validate_anchors(
    joint_order: tuple[str, ...],
    anchors: Any,
) -> tuple[AnchorDefinition, ...]:
    if not isinstance(anchors, tuple):
        raise KinematicsError("anchor definitions must be an immutable tuple")
    if not anchors or any(
        not isinstance(anchor, AnchorDefinition) for anchor in anchors
    ):
        raise KinematicsError("anchor definitions must contain AnchorDefinition values")
    anchor_ids = [anchor.anchor_id for anchor in anchors]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise KinematicsError("anchor_id is duplicated")
    known_joints = set(joint_order)
    for anchor in anchors:
        if anchor.joint_id not in known_joints:
            raise KinematicsError(
                f"anchor {anchor.anchor_id!r} references unknown joint "
                f"{anchor.joint_id!r}"
            )
    return anchors


@dataclass(frozen=True)
class AnchorPose:
    anchor_id: str
    joint_id: str
    actor_from_anchor: RigidTransform


@dataclass(frozen=True)
class AnchorFrame:
    """Resolved actor-space semantic anchors for one pose."""

    source_glb_sha256: str
    anchor_order: tuple[str, ...]
    anchors: tuple[AnchorPose, ...]

    def anchor_transform(self, anchor_id: str) -> RigidTransform:
        for anchor in self.anchors:
            if anchor.anchor_id == anchor_id:
                return anchor.actor_from_anchor
        raise KeyError(f"unknown anchor_id: {anchor_id!r}")

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema": "avengine_m2_actor_anchor_frame_v1",
            "source_glb_sha256": self.source_glb_sha256,
            "anchor_order": list(self.anchor_order),
            "anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "joint_id": anchor.joint_id,
                    "actor_from_anchor": anchor.actor_from_anchor.to_json_data(),
                }
                for anchor in self.anchors
            ],
        }


def resolve_actor_anchors(
    mapping: HabitatAssetMapping,
    pose_rotation_xyzw: Any,
    anchors: tuple[AnchorDefinition, ...],
) -> AnchorFrame:
    """Resolve declared ``joint_from_anchor`` transforms into actor space."""

    frame = forward_kinematics(mapping, pose_rotation_xyzw)
    definitions = _validate_anchors(frame.joint_order, anchors)
    poses = tuple(
        AnchorPose(
            anchor_id=anchor.anchor_id,
            joint_id=anchor.joint_id,
            actor_from_anchor=_compose(
                frame.joint_transform(anchor.joint_id),
                anchor.joint_from_anchor,
            ),
        )
        for anchor in definitions
    )
    return AnchorFrame(
        mapping.source_glb_sha256,
        tuple(anchor.anchor_id for anchor in definitions),
        poses,
    )


@dataclass(frozen=True)
class ContactInferenceThresholds:
    """Explicit metric thresholds used by deterministic contact inference."""

    minimum_dynamic_vertical_range_m: float = 0.005
    contact_height_fraction: float = 0.35
    maximum_idle_vertical_range_m: float = 0.003
    maximum_idle_step_displacement_m: float = 0.003
    maximum_contact_horizontal_step_m: float = 0.015

    def __post_init__(self) -> None:
        for name in (
            "minimum_dynamic_vertical_range_m",
            "maximum_idle_vertical_range_m",
            "maximum_idle_step_displacement_m",
            "maximum_contact_horizontal_step_m",
        ):
            value = getattr(self, name)
            if not _is_number(value):
                raise KinematicsError(f"{name} must be finite")
            if _has_negative_zero(float(value)):
                raise KinematicsError(f"{name} must use canonical signed zero")
            if float(value) <= 0.0:
                raise KinematicsError(f"{name} must be positive")
        fraction = self.contact_height_fraction
        if not _is_number(fraction):
            raise KinematicsError("contact_height_fraction must be finite")
        if _has_negative_zero(float(fraction)):
            raise KinematicsError(
                "contact_height_fraction must use canonical signed zero"
            )
        if not 0.0 < float(fraction) < 1.0:
            raise KinematicsError(
                "contact_height_fraction must be strictly between 0 and 1"
            )

    def to_json_data(self) -> dict[str, float]:
        return {
            "minimum_dynamic_vertical_range_m": float(
                self.minimum_dynamic_vertical_range_m
            ),
            "contact_height_fraction": float(self.contact_height_fraction),
            "maximum_idle_vertical_range_m": float(self.maximum_idle_vertical_range_m),
            "maximum_idle_step_displacement_m": float(
                self.maximum_idle_step_displacement_m
            ),
            "maximum_contact_horizontal_step_m": float(
                self.maximum_contact_horizontal_step_m
            ),
        }


@dataclass(frozen=True)
class ContactWarning:
    code: str
    semantic_action_id: str
    contact_id: str
    message: str

    def to_json_data(self) -> dict[str, str]:
        return {
            "code": self.code,
            "semantic_action_id": self.semantic_action_id,
            "contact_id": self.contact_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class ContactTrajectoryMetric:
    contact_id: str
    inference_mode: str
    confidence: str
    idle_reference_height_m: float
    contact_height_threshold_m: float
    minimum_height_m: float
    maximum_height_m: float
    vertical_range_m: float
    maximum_step_displacement_m: float
    maximum_horizontal_step_m: float
    maximum_contact_horizontal_step_m: float
    contact_frame_count: int
    swing_frame_count: int

    def to_json_data(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "inference_mode": self.inference_mode,
            "confidence": self.confidence,
            "idle_reference_height_m": _canonical_float(self.idle_reference_height_m),
            "contact_height_threshold_m": _canonical_float(
                self.contact_height_threshold_m
            ),
            "minimum_height_m": _canonical_float(self.minimum_height_m),
            "maximum_height_m": _canonical_float(self.maximum_height_m),
            "vertical_range_m": _canonical_float(self.vertical_range_m),
            "maximum_step_displacement_m": _canonical_float(
                self.maximum_step_displacement_m
            ),
            "maximum_horizontal_step_m": _canonical_float(
                self.maximum_horizontal_step_m
            ),
            "maximum_contact_horizontal_step_m": _canonical_float(
                self.maximum_contact_horizontal_step_m
            ),
            "contact_frame_count": self.contact_frame_count,
            "swing_frame_count": self.swing_frame_count,
        }


@dataclass(frozen=True)
class ContactFrame:
    sample_index: int
    sample_tick: int
    source_time_seconds: float
    in_contact: tuple[bool, bool, bool, bool]

    def to_json_data(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "sample_tick": self.sample_tick,
            "source_time_seconds": _canonical_float(self.source_time_seconds),
            "contacts": [
                {"contact_id": contact_id, "in_contact": state}
                for contact_id, state in zip(
                    CONTACT_ORDER, self.in_contact, strict=True
                )
            ],
        }


@dataclass(frozen=True)
class ContactActionPhases:
    semantic_action_id: str
    source_action_name: str
    frames: tuple[ContactFrame, ...]
    metrics: tuple[ContactTrajectoryMetric, ...]

    def metric(self, contact_id: str) -> ContactTrajectoryMetric:
        for metric in self.metrics:
            if metric.contact_id == contact_id:
                return metric
        raise KeyError(f"unknown contact_id: {contact_id!r}")

    def to_json_data(self) -> dict[str, Any]:
        return {
            "semantic_action_id": self.semantic_action_id,
            "source_action_name": self.source_action_name,
            "sample_count": len(self.frames),
            "frames": [frame.to_json_data() for frame in self.frames],
            "metrics": [metric.to_json_data() for metric in self.metrics],
        }


@dataclass(frozen=True)
class ContactPhaseReport:
    source_glb_sha256: str
    baked_actions_sha256: str
    runtime_joint_order: tuple[str, ...]
    contact_order: tuple[str, str, str, str]
    sample_rate_hz: int
    time_base_hz: int
    anchor_definitions: tuple[AnchorDefinition, ...]
    thresholds: ContactInferenceThresholds
    actions: tuple[ContactActionPhases, ContactActionPhases]
    warnings: tuple[ContactWarning, ...]
    qualification_state: str = field(default="research_candidate", init=False)
    qualification_claim: bool = field(default=False, init=False)

    def action(self, semantic_action_id: str) -> ContactActionPhases:
        for action in self.actions:
            if action.semantic_action_id == semantic_action_id:
                return action
        raise KeyError(f"unknown semantic action: {semantic_action_id!r}")

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema": "avengine_m2_contact_phases_v1",
            "source_glb_sha256": self.source_glb_sha256,
            "baked_actions_sha256": self.baked_actions_sha256,
            "runtime_joint_order": list(self.runtime_joint_order),
            "qualification_state": self.qualification_state,
            "qualification_claim": self.qualification_claim,
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "-Z",
                "linear_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "sample_rate_hz": self.sample_rate_hz,
            "time_base_hz": self.time_base_hz,
            "contact_order": list(self.contact_order),
            "anchor_definitions": [
                anchor.to_json_data() for anchor in self.anchor_definitions
            ],
            "thresholds": self.thresholds.to_json_data(),
            "actions": [action.to_json_data() for action in self.actions],
            "warnings": [warning.to_json_data() for warning in self.warnings],
            "notes": [
                "Contact phases are inferred from declared actor-space paw-anchor trajectories.",
                "Actor-space contact warnings are diagnostic; world-space foot-lock "
                "certification also requires a hash-bound root trajectory.",
            ],
        }

    def to_canonical_json(self) -> str:
        return (
            json.dumps(
                self.to_json_data(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


def _cyclic_step_metrics(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous = np.roll(positions, 1, axis=0)
    difference = positions - previous
    distance = np.linalg.norm(difference, axis=1)
    horizontal = np.linalg.norm(difference[:, [0, 2]], axis=1)
    return distance, horizontal


def _trajectory_metric(
    *,
    contact_id: str,
    positions: np.ndarray,
    states: tuple[bool, ...],
    idle_reference_height: float,
    contact_height_threshold: float,
    inference_mode: str,
    confidence: str,
) -> ContactTrajectoryMetric:
    heights = positions[:, 1]
    step_distance, horizontal_step = _cyclic_step_metrics(positions)
    contact_horizontal_steps = [
        float(horizontal_step[index])
        for index in range(len(states))
        if states[index] and states[index - 1]
    ]
    return ContactTrajectoryMetric(
        contact_id=contact_id,
        inference_mode=inference_mode,
        confidence=confidence,
        idle_reference_height_m=_canonical_float(idle_reference_height),
        contact_height_threshold_m=_canonical_float(contact_height_threshold),
        minimum_height_m=_canonical_float(float(np.min(heights))),
        maximum_height_m=_canonical_float(float(np.max(heights))),
        vertical_range_m=_canonical_float(float(np.ptp(heights))),
        maximum_step_displacement_m=_canonical_float(float(np.max(step_distance))),
        maximum_horizontal_step_m=_canonical_float(float(np.max(horizontal_step))),
        maximum_contact_horizontal_step_m=_canonical_float(
            max(contact_horizontal_steps, default=0.0)
        ),
        contact_frame_count=sum(states),
        swing_frame_count=len(states) - sum(states),
    )


def _anchor_trajectories(
    mapping: HabitatAssetMapping,
    actions: BakedActionSet,
    anchors: tuple[AnchorDefinition, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for action in actions.actions:
        frames: list[list[Vector3]] = []
        for pose in action.rotations_xyzw:
            resolved = resolve_actor_anchors(mapping, pose, anchors)
            frames.append(
                [
                    resolved.anchor_transform(contact_id).translation_m
                    for contact_id in CONTACT_ORDER
                ]
            )
        trajectory = np.asarray(frames, dtype=np.float64)
        expected_shape = (action.sample_count, len(CONTACT_ORDER), 3)
        if trajectory.shape != expected_shape:
            raise KinematicsError(
                f"{action.semantic_action_id} anchor trajectory has invalid shape "
                f"{trajectory.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(trajectory)):
            raise KinematicsError("anchor trajectory contains non-finite coordinates")
        result[action.semantic_action_id] = trajectory
    return result


def derive_contact_phases(
    mapping: HabitatAssetMapping,
    actions: BakedActionSet,
    contact_anchors: tuple[AnchorDefinition, ...],
    thresholds: ContactInferenceThresholds | None = None,
) -> ContactPhaseReport:
    """Infer deterministic Idle/Walk contact evidence for four declared paws."""

    _validate_mapping(mapping)
    if not isinstance(actions, BakedActionSet):
        raise KinematicsError("actions must be a BakedActionSet")
    if actions.source_glb_sha256 != mapping.source_glb_sha256:
        raise KinematicsError(
            "actions source_glb_sha256 must match the Habitat asset mapping"
        )
    if actions.runtime_joint_order != mapping.runtime_joint_order:
        raise KinematicsError(
            "actions runtime_joint_order must match the Habitat asset mapping"
        )
    if any(
        action.sample_count < _MINIMUM_CONTACT_SAMPLE_COUNT
        for action in actions.actions
    ):
        raise KinematicsError(
            "Idle and Walk contact inference requires at least three samples each"
        )
    definitions = _validate_anchors(mapping.joint_order, contact_anchors)
    if tuple(anchor.anchor_id for anchor in definitions) != CONTACT_ORDER:
        raise KinematicsError(
            f"contact anchor definitions must follow fixed CONTACT_ORDER {CONTACT_ORDER}"
        )
    if thresholds is None:
        thresholds = ContactInferenceThresholds()
    elif not isinstance(thresholds, ContactInferenceThresholds):
        raise KinematicsError("thresholds must be ContactInferenceThresholds")

    trajectories = _anchor_trajectories(mapping, actions, definitions)
    idle_positions = trajectories["idle"]
    walk_positions = trajectories["walk"]
    idle_reference_heights = tuple(
        _canonical_float(float(np.median(idle_positions[:, index, 1])))
        for index in range(len(CONTACT_ORDER))
    )
    warnings: list[ContactWarning] = []
    idle_states_by_contact: list[tuple[bool, ...]] = []
    idle_metrics: list[ContactTrajectoryMetric] = []
    for contact_index, contact_id in enumerate(CONTACT_ORDER):
        positions = idle_positions[:, contact_index]
        states = (True,) * len(positions)
        preliminary = _trajectory_metric(
            contact_id=contact_id,
            positions=positions,
            states=states,
            idle_reference_height=idle_reference_heights[contact_index],
            contact_height_threshold=idle_reference_heights[contact_index],
            inference_mode="forced_idle_contact",
            confidence="high",
        )
        confidence = "high"
        if (
            preliminary.vertical_range_m > thresholds.maximum_idle_vertical_range_m
            or preliminary.maximum_step_displacement_m
            > thresholds.maximum_idle_step_displacement_m
        ):
            confidence = "low"
            warnings.append(
                ContactWarning(
                    code="idle_anchor_motion",
                    semantic_action_id="idle",
                    contact_id=contact_id,
                    message=(
                        "Idle remains declared in contact, but the anchor moves: "
                        f"vertical_range_m={preliminary.vertical_range_m:.9g}, "
                        "maximum_step_displacement_m="
                        f"{preliminary.maximum_step_displacement_m:.9g}."
                    ),
                )
            )
        idle_metrics.append(
            _trajectory_metric(
                contact_id=contact_id,
                positions=positions,
                states=states,
                idle_reference_height=idle_reference_heights[contact_index],
                contact_height_threshold=idle_reference_heights[contact_index],
                inference_mode="forced_idle_contact",
                confidence=confidence,
            )
        )
        idle_states_by_contact.append(states)

    walk_states_by_contact: list[tuple[bool, ...]] = []
    walk_metrics: list[ContactTrajectoryMetric] = []
    for contact_index, contact_id in enumerate(CONTACT_ORDER):
        positions = walk_positions[:, contact_index]
        heights = positions[:, 1]
        vertical_range = float(np.ptp(heights))
        reference = idle_reference_heights[contact_index]
        height_threshold = reference + (
            thresholds.contact_height_fraction * vertical_range
        )
        if vertical_range < thresholds.minimum_dynamic_vertical_range_m:
            if contact_index < 2:
                raise KinematicsError(
                    f"front paw {contact_id!r} vertical excursion "
                    f"{vertical_range:.9g} m is below the required dynamic threshold"
                )
            states = (True,) * len(positions)
            inference_mode = "low_excursion_kept_contact"
            confidence = "low"
            warnings.append(
                ContactWarning(
                    code="low_vertical_excursion_kept_contact",
                    semantic_action_id="walk",
                    contact_id=contact_id,
                    message=(
                        f"Walk vertical_range_m={vertical_range:.9g} is below "
                        f"{thresholds.minimum_dynamic_vertical_range_m:.9g}; all "
                        "frames remain contact and no hind-paw swing is claimed."
                    ),
                )
            )
        else:
            states = tuple(bool(height <= height_threshold) for height in heights)
            if not any(states):
                raise KinematicsError(
                    f"paw {contact_id!r} has no contact frame near its Idle height"
                )
            if all(states):
                if contact_index < 2:
                    raise KinematicsError(
                        f"front paw {contact_id!r} has no supported swing frame"
                    )
                inference_mode = "height_threshold_no_swing"
                confidence = "low"
                warnings.append(
                    ContactWarning(
                        code="height_threshold_kept_contact",
                        semantic_action_id="walk",
                        contact_id=contact_id,
                        message=(
                            "Dynamic vertical range was measured, but no sample rose "
                            "above the Idle-bound contact threshold; no swing is claimed."
                        ),
                    )
                )
            else:
                inference_mode = "height_dynamic"
                confidence = "high"
        metric = _trajectory_metric(
            contact_id=contact_id,
            positions=positions,
            states=states,
            idle_reference_height=reference,
            contact_height_threshold=height_threshold,
            inference_mode=inference_mode,
            confidence=confidence,
        )
        if (
            metric.maximum_contact_horizontal_step_m
            > thresholds.maximum_contact_horizontal_step_m
        ):
            metric = ContactTrajectoryMetric(
                **{
                    **metric.__dict__,
                    "confidence": "low",
                }
            )
            warnings.append(
                ContactWarning(
                    code="contact_horizontal_sliding",
                    semantic_action_id="walk",
                    contact_id=contact_id,
                    message=(
                        "Contact-phase horizontal anchor step exceeds threshold: "
                        f"{metric.maximum_contact_horizontal_step_m:.9g} m > "
                        f"{thresholds.maximum_contact_horizontal_step_m:.9g} m."
                    ),
                )
            )
        walk_states_by_contact.append(states)
        walk_metrics.append(metric)

    def action_report(
        semantic_action_id: str,
        states_by_contact: Sequence[tuple[bool, ...]],
        metrics: Sequence[ContactTrajectoryMetric],
    ) -> ContactActionPhases:
        action = actions.action(semantic_action_id)
        frames = tuple(
            ContactFrame(
                sample_index=index,
                sample_tick=action.sample_ticks[index],
                source_time_seconds=action.source_times_seconds[index],
                in_contact=tuple(
                    states_by_contact[contact_index][index]
                    for contact_index in range(len(CONTACT_ORDER))
                ),  # type: ignore[arg-type]
            )
            for index in range(action.sample_count)
        )
        return ContactActionPhases(
            semantic_action_id=semantic_action_id,
            source_action_name=action.source_action_name,
            frames=frames,
            metrics=tuple(metrics),
        )

    return ContactPhaseReport(
        source_glb_sha256=mapping.source_glb_sha256,
        baked_actions_sha256=baked_actions_content_sha256(actions),
        runtime_joint_order=mapping.runtime_joint_order,
        contact_order=CONTACT_ORDER,
        sample_rate_hz=SAMPLE_RATE_HZ,
        time_base_hz=TIME_BASE_HZ,
        anchor_definitions=definitions,
        thresholds=thresholds,
        actions=(
            action_report("idle", idle_states_by_contact, idle_metrics),
            action_report("walk", walk_states_by_contact, walk_metrics),
        ),
        warnings=tuple(warnings),
    )


__all__ = [
    "CONTACT_ORDER",
    "AnchorDefinition",
    "AnchorFrame",
    "ContactActionPhases",
    "ContactFrame",
    "ContactInferenceThresholds",
    "ContactPhaseReport",
    "ContactTrajectoryMetric",
    "ContactWarning",
    "JointKinematicPose",
    "KinematicFrame",
    "KinematicsError",
    "RigidTransform",
    "derive_contact_phases",
    "forward_kinematics",
    "resolve_actor_anchors",
]
