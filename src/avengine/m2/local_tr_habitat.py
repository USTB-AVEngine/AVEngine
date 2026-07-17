"""Research-only mixed local-translation/rotation Habitat mapping.

This module is intentionally separate from :mod:`avengine.m2.habitat`.  The
formal M2 v1 contract has spherical joints only and must remain byte-for-byte
compatible with already admitted packages.  The research local-TR contract
expands only translation-driven non-root skin joints into the following URDF
chain::

    parent -> Px -> Py -> Pz -> same-named spherical skin link

The first prismatic joint carries the authored rest translation in its URDF
origin.  Runtime prismatic positions therefore encode ``T_absolute - T_rest``;
the final spherical link receives the absolute child-local xyzw rotation.  A
static-translation joint remains a direct spherical joint with its rest
translation in the origin.

The code is pure Python and does not import ``habitat_sim``.  Runtime link
offsets are discovered by a caller and then bound by exact link name through
:func:`bind_local_tr_habitat_layout`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence
from xml.sax.saxutils import quoteattr

import numpy as np

from avengine.m2.glb import GlbDocument
from avengine.m2.habitat import (
    HabitatJointRest,
    HabitatLinkJointBlock,
    HabitatMappingError,
    build_habitat_asset_mapping,
)


_DEFAULT_PRISMATIC_LIMIT_MARGIN_M = 1.0e-4
_REST_TRANSLATION_TOLERANCE_M = 1.0e-9
_UNIT_QUATERNION_TOLERANCE = 1.0e-9
_RUNTIME_LIMIT_TOLERANCE_M = 1.0e-12
_ZERO_TOLERANCE = 1.0e-15
_DUMMY_LINK_PREFIX = "__avengine_local_tr__"
_DUMMY_JOINT_PREFIX = "__avengine_local_tr_joint__"
_AXIS_NAMES = ("x", "y", "z")
_AXIS_VECTORS = ("1 0 0", "0 1 0", "0 0 1")

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class LocalTRHabitatMappingError(HabitatMappingError):
    """A research local-TR action set or runtime layout is inconsistent."""


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _number_text(value: float) -> str:
    return format(_canonical_float(value), ".17g")


def _ordered_names(value: Any, *, owner: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise LocalTRHabitatMappingError(
            f"{owner} must be an ordered sequence of names"
        )
    try:
        names = tuple(value)
    except TypeError as exc:
        raise LocalTRHabitatMappingError(
            f"{owner} must be an ordered sequence of names"
        ) from exc
    if any(not isinstance(name, str) or not name for name in names):
        raise LocalTRHabitatMappingError(f"{owner} must contain non-empty string names")
    if len(set(names)) != len(names):
        raise LocalTRHabitatMappingError(f"{owner} must contain unique names")
    return names


def _array(value: Any, *, shape_tail: tuple[int, ...], owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LocalTRHabitatMappingError(
            f"{owner} must be a finite numeric array ending in {shape_tail}"
        ) from exc
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise LocalTRHabitatMappingError(
            f"{owner} must have shape (frame_count, {', '.join(map(str, shape_tail))})"
        )
    if result.shape[0] == 0 or not np.all(np.isfinite(result)):
        raise LocalTRHabitatMappingError(
            f"{owner} must contain at least one frame of finite numbers"
        )
    return result


def _pose_array(
    value: Any, *, expected_shape: tuple[int, ...], owner: str
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LocalTRHabitatMappingError(
            f"{owner} must be a finite numeric array with shape {expected_shape}"
        ) from exc
    if result.shape != expected_shape or not np.all(np.isfinite(result)):
        raise LocalTRHabitatMappingError(
            f"{owner} must be a finite numeric array with shape {expected_shape}"
        )
    return result


def _quaternion_sign_component(quaternion: np.ndarray) -> float:
    component = float(quaternion[3])
    if math.isclose(component, 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE):
        for candidate in quaternion[:3]:
            if not math.isclose(
                float(candidate), 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE
            ):
                return float(candidate)
    return component


def _validated_quaternion(value: np.ndarray, *, owner: str) -> Quaternion:
    norm = float(np.linalg.norm(value))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_QUATERNION_TOLERANCE):
        raise LocalTRHabitatMappingError(f"{owner} must already be unit normalized")
    if _quaternion_sign_component(value) < 0.0:
        raise LocalTRHabitatMappingError(
            f"{owner} must use the canonical quaternion hemisphere"
        )
    return tuple(_canonical_float(component) for component in value)  # type: ignore[return-value]


def _dummy_link_names(joint_ordinal: int) -> tuple[str, str, str]:
    stem = f"{_DUMMY_LINK_PREFIX}{joint_ordinal:06d}"
    return (f"{stem}__x", f"{stem}__y", f"{stem}__z")


def _dummy_joint_names(joint_ordinal: int) -> tuple[str, str, str, str]:
    stem = f"{_DUMMY_JOINT_PREFIX}{joint_ordinal:06d}"
    return (f"{stem}__x", f"{stem}__y", f"{stem}__z", f"{stem}__rotation")


def _link_lines(name: str) -> list[str]:
    return [
        f"  <link name={quoteattr(name)}>",
        "    <inertial>",
        '      <mass value="0.001"/>',
        '      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>',
        "    </inertial>",
        "  </link>",
    ]


@dataclass(frozen=True)
class LocalTRDrivenJoint:
    """One translation-driven skin joint and its bounded dummy-link chain."""

    joint_id: str
    joint_ordinal: int
    runtime_joint_ordinal: int
    dummy_link_names: tuple[str, str, str]
    delta_min_m: Vector3
    delta_max_m: Vector3
    limit_lower_m: Vector3
    limit_upper_m: Vector3


@dataclass(frozen=True)
class LocalTRRuntimeLinkSpec:
    """Expected runtime position block for one actual or dummy Habitat link."""

    link_name: str
    joint_id: str
    component: str
    joint_position_count: int


@dataclass(frozen=True)
class LocalTRHabitatMapping:
    """Hash-bound research mapping from absolute local T/R to mixed AO joints."""

    source_glb_sha256: str
    root_joint_id: str
    joint_order: tuple[str, ...]
    runtime_joint_order: tuple[str, ...]
    translation_driven_joint_ids: tuple[str, ...]
    joints: tuple[HabitatJointRest, ...]
    rest_translations_m: tuple[Vector3, ...]
    driven_joints: tuple[LocalTRDrivenJoint, ...]
    actor_from_skin_root: Matrix4
    actor_from_skin_root_source: str
    prismatic_limit_margin_m: float

    @property
    def runtime_link_specs(self) -> tuple[LocalTRRuntimeLinkSpec, ...]:
        """Return the exact expected Habitat link set in asset-defined order."""

        driven = {record.joint_id: record for record in self.driven_joints}
        result: list[LocalTRRuntimeLinkSpec] = []
        for joint_id in self.runtime_joint_order:
            record = driven.get(joint_id)
            if record is not None:
                result.extend(
                    LocalTRRuntimeLinkSpec(
                        link_name=link_name,
                        joint_id=joint_id,
                        component=f"translation_delta_{axis_name}_m",
                        joint_position_count=1,
                    )
                    for axis_name, link_name in zip(
                        _AXIS_NAMES, record.dummy_link_names, strict=True
                    )
                )
            result.append(
                LocalTRRuntimeLinkSpec(
                    link_name=joint_id,
                    joint_id=joint_id,
                    component="rotation_xyzw",
                    joint_position_count=4,
                )
            )
        return tuple(result)

    @property
    def runtime_joint_position_count(self) -> int:
        """Return ``4*N + 3*D`` for the mixed joint-position vector."""

        return 4 * len(self.runtime_joint_order) + 3 * len(self.driven_joints)

    def render_urdf(self, *, robot_name: str = "avengine_m2_local_tr_research") -> str:
        """Render a deterministic URDF implementing ``T(rest + delta) @ R``."""

        if not isinstance(robot_name, str) or not robot_name:
            raise LocalTRHabitatMappingError("robot_name must be a non-empty string")
        driven = {record.joint_id: record for record in self.driven_joints}
        lines = [
            '<?xml version="1.0"?>',
            f"<robot name={quoteattr(robot_name)}>",
            "  <!-- Research local-TR: first P origin=T_rest; state=T_abs-T_rest. -->",
        ]
        for joint in self.joints:
            lines.extend(_link_lines(joint.joint_id))
        for record in self.driven_joints:
            for dummy_link_name in record.dummy_link_names:
                lines.extend(_link_lines(dummy_link_name))

        for joint in self.joints:
            if joint.parent_joint_id is None:
                continue
            xyz = " ".join(_number_text(value) for value in joint.local_translation_m)
            record = driven.get(joint.joint_id)
            if record is None:
                lines.extend(
                    [
                        f"  <joint name={quoteattr('joint_' + joint.joint_id)} "
                        'type="spherical">',
                        f"    <parent link={quoteattr(joint.parent_joint_id)}/>",
                        f"    <child link={quoteattr(joint.joint_id)}/>",
                        f'    <origin xyz="{xyz}" rpy="0 0 0"/>',
                        "  </joint>",
                    ]
                )
                continue

            dummy_joint_names = _dummy_joint_names(joint.joint_ordinal)
            parents = (joint.parent_joint_id, *record.dummy_link_names[:2])
            children = record.dummy_link_names
            for axis_index, (parent, child) in enumerate(
                zip(parents, children, strict=True)
            ):
                origin = xyz if axis_index == 0 else "0 0 0"
                lower = _number_text(record.limit_lower_m[axis_index])
                upper = _number_text(record.limit_upper_m[axis_index])
                lines.extend(
                    [
                        f"  <joint name={quoteattr(dummy_joint_names[axis_index])} "
                        'type="prismatic">',
                        f"    <parent link={quoteattr(parent)}/>",
                        f"    <child link={quoteattr(child)}/>",
                        f'    <origin xyz="{origin}" rpy="0 0 0"/>',
                        f'    <axis xyz="{_AXIS_VECTORS[axis_index]}"/>',
                        f'    <limit lower="{lower}" upper="{upper}" '
                        'effort="1000" velocity="1000"/>',
                        "  </joint>",
                    ]
                )
            lines.extend(
                [
                    f"  <joint name={quoteattr(dummy_joint_names[3])} "
                    'type="spherical">',
                    f"    <parent link={quoteattr(record.dummy_link_names[2])}/>",
                    f"    <child link={quoteattr(joint.joint_id)}/>",
                    '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                    "  </joint>",
                ]
            )
        lines.append("</robot>")
        return "\n".join(lines) + "\n"

    def joint_mapping_data(self) -> dict[str, Any]:
        """Return detached JSON data for the research local-TR mapping."""

        return {
            "schema": "avengine_m2_habitat_joint_mapping_local_tr_v2",
            "research_only": True,
            "qualification_claim": False,
            "source_glb_sha256": self.source_glb_sha256,
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "-Z",
                "linear_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "root_joint_id": self.root_joint_id,
            "joint_order": list(self.joint_order),
            "runtime_joint_order": list(self.runtime_joint_order),
            "translation_driven_joint_ids": list(self.translation_driven_joint_ids),
            "joint_pose_encoding": (
                "absolute_child_local_translation_m_plus_rotation_xyzw_float64"
            ),
            "local_transform_composition": "T(rest + delta) @ R",
            "prismatic_state_semantics": (
                "absolute_child_local_translation_m - rest_local_translation_m"
            ),
            "actor_from_skin_root": [list(row) for row in self.actor_from_skin_root],
            "actor_from_skin_root_source": self.actor_from_skin_root_source,
            "runtime_root_formula": (
                "world_from_skin_root = world_from_actor @ actor_from_skin_root"
            ),
            "rest_translations_m": [
                list(translation) for translation in self.rest_translations_m
            ],
            "prismatic_limit_margin_m": self.prismatic_limit_margin_m,
            "translation_driven_joints": [
                {
                    "joint_id": record.joint_id,
                    "joint_ordinal": record.joint_ordinal,
                    "runtime_joint_ordinal": record.runtime_joint_ordinal,
                    "dummy_link_names": list(record.dummy_link_names),
                    "delta_min_m": list(record.delta_min_m),
                    "delta_max_m": list(record.delta_max_m),
                    "limit_lower_m": list(record.limit_lower_m),
                    "limit_upper_m": list(record.limit_upper_m),
                }
                for record in self.driven_joints
            ],
            "habitat_layout": {
                "base_link": self.root_joint_id,
                "runtime_joint_types": "mixed_prismatic_and_spherical",
                "runtime_joint_position_count": self.runtime_joint_position_count,
                "runtime_joint_position_count_formula": "4*N + 3*D",
                "render_mode": "skin",
                "links": [
                    {
                        "link_name": spec.link_name,
                        "joint_id": spec.joint_id,
                        "component": spec.component,
                        "joint_position_count": spec.joint_position_count,
                    }
                    for spec in self.runtime_link_specs
                ],
            },
        }


@dataclass(frozen=True)
class LocalTRHabitatBinding:
    """Exact name/offset binding for an instantiated mixed-joint Habitat AO."""

    runtime_joint_order: tuple[str, ...]
    translation_driven_joint_ids: tuple[str, ...]
    rest_translations_m: tuple[Vector3, ...]
    driven_joints: tuple[LocalTRDrivenJoint, ...]
    joint_position_count: int
    link_specs: tuple[LocalTRRuntimeLinkSpec, ...]
    blocks: tuple[HabitatLinkJointBlock, ...]

    def map_pose(
        self, absolute_translations_m: Any, rotations_xyzw: Any
    ) -> tuple[float, ...]:
        """Map absolute child-local T/R arrays into the mixed Habitat vector."""

        joint_count = len(self.runtime_joint_order)
        translations = _pose_array(
            absolute_translations_m,
            expected_shape=(joint_count, 3),
            owner="absolute_translations_m",
        )
        rotations = _pose_array(
            rotations_xyzw,
            expected_shape=(joint_count, 4),
            owner="rotations_xyzw",
        )
        rest = np.asarray(self.rest_translations_m, dtype=np.float64)
        driven = {record.joint_id: record for record in self.driven_joints}
        blocks = {block.link_name: block for block in self.blocks}
        positions = [0.0] * self.joint_position_count

        for joint_index, joint_id in enumerate(self.runtime_joint_order):
            quaternion = _validated_quaternion(
                rotations[joint_index],
                owner=f"rotations_xyzw[{joint_index}] ({joint_id!r})",
            )
            rotation_block = blocks[joint_id]
            rotation_start = rotation_block.joint_position_offset
            positions[rotation_start : rotation_start + 4] = quaternion

            delta = translations[joint_index] - rest[joint_index]
            record = driven.get(joint_id)
            if record is None:
                maximum = float(np.max(np.abs(delta)))
                if maximum > _REST_TRANSLATION_TOLERANCE_M:
                    raise LocalTRHabitatMappingError(
                        f"absolute_translations_m[{joint_index}] ({joint_id!r}) "
                        "differs from rest but the joint has no prismatic chain "
                        f"(maximum delta {maximum:.9g} m)"
                    )
                continue

            for axis_index, dummy_link_name in enumerate(record.dummy_link_names):
                value = _canonical_float(delta[axis_index])
                lower = record.limit_lower_m[axis_index]
                upper = record.limit_upper_m[axis_index]
                if (
                    value < lower - _RUNTIME_LIMIT_TOLERANCE_M
                    or value > upper + _RUNTIME_LIMIT_TOLERANCE_M
                ):
                    raise LocalTRHabitatMappingError(
                        f"translation delta for joint {joint_id!r} axis "
                        f"{_AXIS_NAMES[axis_index]} is outside the compiled "
                        f"prismatic limit [{lower:.9g}, {upper:.9g}]"
                    )
                block = blocks[dummy_link_name]
                positions[block.joint_position_offset] = value
        return tuple(_canonical_float(value) for value in positions)

    def to_json_data(self) -> dict[str, Any]:
        """Return detached measured-layout data in asset-defined link order."""

        by_name = {block.link_name: block for block in self.blocks}
        return {
            "schema": "avengine_m2_habitat_runtime_binding_local_tr_v2",
            "research_only": True,
            "runtime_joint_order": list(self.runtime_joint_order),
            "translation_driven_joint_ids": list(self.translation_driven_joint_ids),
            "joint_position_count": self.joint_position_count,
            "links": [
                {
                    "link_name": spec.link_name,
                    "joint_id": spec.joint_id,
                    "component": spec.component,
                    "joint_position_offset": by_name[
                        spec.link_name
                    ].joint_position_offset,
                    "joint_position_count": spec.joint_position_count,
                }
                for spec in self.link_specs
            ],
        }


def _translation_driven_ids(actions: Any) -> tuple[str, ...]:
    """Read the final public field while tolerating the design-draft alias."""

    if hasattr(actions, "translation_driven_joint_ids"):
        value = actions.translation_driven_joint_ids
    elif hasattr(actions, "translation_driven_joint_order"):
        value = actions.translation_driven_joint_order
    else:
        raise LocalTRHabitatMappingError(
            "actions must declare translation_driven_joint_ids"
        )
    return _ordered_names(value, owner="actions.translation_driven_joint_ids")


def build_local_tr_habitat_mapping(
    document: GlbDocument,
    actions: Any,
    *,
    actor_from_skin_root: Sequence[Sequence[float]],
    actor_from_skin_root_source: str,
    prismatic_limit_margin_m: float = _DEFAULT_PRISMATIC_LIMIT_MARGIN_M,
) -> LocalTRHabitatMapping:
    """Compile one research local-TR action set into a mixed Habitat mapping."""

    try:
        base = build_habitat_asset_mapping(
            document,
            actor_from_skin_root=actor_from_skin_root,
            actor_from_skin_root_source=actor_from_skin_root_source,
        )
    except HabitatMappingError as exc:
        raise LocalTRHabitatMappingError(str(exc)) from exc

    if isinstance(prismatic_limit_margin_m, bool) or not isinstance(
        prismatic_limit_margin_m, (int, float)
    ):
        raise LocalTRHabitatMappingError(
            "prismatic_limit_margin_m must be a positive finite number"
        )
    margin = float(prismatic_limit_margin_m)
    if not math.isfinite(margin) or margin <= 0.0:
        raise LocalTRHabitatMappingError(
            "prismatic_limit_margin_m must be a positive finite number"
        )

    if getattr(actions, "source_glb_sha256", None) != document.sha256:
        raise LocalTRHabitatMappingError(
            "actions.source_glb_sha256 must match the parsed GLB"
        )
    runtime_joint_order = _ordered_names(
        getattr(actions, "runtime_joint_order", None),
        owner="actions.runtime_joint_order",
    )
    if runtime_joint_order != base.runtime_joint_order:
        raise LocalTRHabitatMappingError(
            "actions.runtime_joint_order must exactly match the parsed skin order"
        )
    driven_ids = _translation_driven_ids(actions)
    driven_set = set(driven_ids)
    if not driven_set.issubset(runtime_joint_order):
        raise LocalTRHabitatMappingError(
            "translation_driven_joint_ids must be a subset of runtime_joint_order"
        )
    expected_driven_order = tuple(
        joint_id for joint_id in runtime_joint_order if joint_id in driven_set
    )
    if driven_ids != expected_driven_order:
        raise LocalTRHabitatMappingError(
            "translation_driven_joint_ids must preserve runtime_joint_order"
        )

    rest = _pose_array(
        getattr(actions, "rest_translations_m", None),
        expected_shape=(len(runtime_joint_order), 3),
        owner="actions.rest_translations_m",
    )
    joint_by_id = {joint.joint_id: joint for joint in base.joints}
    document_rest = np.asarray(
        [joint_by_id[joint_id].local_translation_m for joint_id in runtime_joint_order],
        dtype=np.float64,
    )
    rest_error = float(np.max(np.abs(rest - document_rest)))
    if rest_error > _REST_TRANSLATION_TOLERANCE_M:
        raise LocalTRHabitatMappingError(
            "actions.rest_translations_m must match the parsed GLB rest pose "
            f"(maximum error {rest_error:.9g} m)"
        )

    source_joint_names = set(base.joint_order)
    generated_dummy_names = {
        name
        for joint_id in driven_ids
        for name in _dummy_link_names(joint_by_id[joint_id].joint_ordinal)
    }
    collisions = source_joint_names.intersection(generated_dummy_names)
    if collisions:
        raise LocalTRHabitatMappingError(
            "generated dummy link names collide with source skin joints: "
            f"{sorted(collisions)}"
        )

    try:
        clips = tuple(actions.actions)
    except (AttributeError, TypeError) as exc:
        raise LocalTRHabitatMappingError(
            "actions.actions must be a non-empty ordered sequence"
        ) from exc
    if not clips:
        raise LocalTRHabitatMappingError(
            "actions.actions must be a non-empty ordered sequence"
        )

    delta_min = np.full((len(runtime_joint_order), 3), np.inf, dtype=np.float64)
    delta_max = np.full((len(runtime_joint_order), 3), -np.inf, dtype=np.float64)
    maximum_static_delta = np.zeros(len(runtime_joint_order), dtype=np.float64)
    for clip_index, clip in enumerate(clips):
        owner = getattr(clip, "semantic_action_id", f"clip_{clip_index}")
        translations = _array(
            getattr(clip, "translations_m", None),
            shape_tail=(len(runtime_joint_order), 3),
            owner=f"actions.actions[{clip_index}] ({owner!r}).translations_m",
        )
        rotations = _array(
            getattr(clip, "rotations_xyzw", None),
            shape_tail=(len(runtime_joint_order), 4),
            owner=f"actions.actions[{clip_index}] ({owner!r}).rotations_xyzw",
        )
        if rotations.shape[0] != translations.shape[0]:
            raise LocalTRHabitatMappingError(
                f"actions.actions[{clip_index}] translation and rotation frame "
                "counts must match"
            )
        norms = np.linalg.norm(rotations, axis=2)
        norm_error = float(np.max(np.abs(norms - 1.0)))
        if norm_error > _UNIT_QUATERNION_TOLERANCE:
            raise LocalTRHabitatMappingError(
                f"actions.actions[{clip_index}] rotations must already be unit "
                f"normalized (maximum error {norm_error:.9g})"
            )
        deltas = translations - rest[np.newaxis, :, :]
        delta_min = np.minimum(delta_min, np.min(deltas, axis=0))
        delta_max = np.maximum(delta_max, np.max(deltas, axis=0))
        maximum_static_delta = np.maximum(
            maximum_static_delta, np.max(np.abs(deltas), axis=(0, 2))
        )

    for joint_index, joint_id in enumerate(runtime_joint_order):
        maximum = float(maximum_static_delta[joint_index])
        if joint_id not in driven_set and maximum > _REST_TRANSLATION_TOLERANCE_M:
            raise LocalTRHabitatMappingError(
                f"non-driven joint {joint_id!r} contains translation samples "
                f"away from rest (maximum delta {maximum:.9g} m)"
            )

    driven_records: list[LocalTRDrivenJoint] = []
    for joint_id in driven_ids:
        runtime_index = runtime_joint_order.index(joint_id)
        joint = joint_by_id[joint_id]
        minima = np.minimum(delta_min[runtime_index], 0.0)
        maxima = np.maximum(delta_max[runtime_index], 0.0)
        lower = minima - margin
        upper = maxima + margin
        driven_records.append(
            LocalTRDrivenJoint(
                joint_id=joint_id,
                joint_ordinal=joint.joint_ordinal,
                runtime_joint_ordinal=runtime_index,
                dummy_link_names=_dummy_link_names(joint.joint_ordinal),
                delta_min_m=tuple(
                    _canonical_float(value) for value in delta_min[runtime_index]
                ),  # type: ignore[arg-type]
                delta_max_m=tuple(
                    _canonical_float(value) for value in delta_max[runtime_index]
                ),  # type: ignore[arg-type]
                limit_lower_m=tuple(_canonical_float(value) for value in lower),  # type: ignore[arg-type]
                limit_upper_m=tuple(_canonical_float(value) for value in upper),  # type: ignore[arg-type]
            )
        )

    return LocalTRHabitatMapping(
        source_glb_sha256=document.sha256,
        root_joint_id=base.root_joint_id,
        joint_order=base.joint_order,
        runtime_joint_order=runtime_joint_order,
        translation_driven_joint_ids=driven_ids,
        joints=base.joints,
        rest_translations_m=tuple(
            tuple(_canonical_float(value) for value in translation)
            for translation in rest
        ),  # type: ignore[arg-type]
        driven_joints=tuple(driven_records),
        actor_from_skin_root=base.actor_from_skin_root,
        actor_from_skin_root_source=base.actor_from_skin_root_source,
        prismatic_limit_margin_m=margin,
    )


def bind_local_tr_habitat_layout(
    mapping: LocalTRHabitatMapping,
    link_blocks: Sequence[HabitatLinkJointBlock],
    *,
    joint_position_count: int,
) -> LocalTRHabitatBinding:
    """Bind a measured mixed AO layout by exact link name and dense offsets."""

    if not isinstance(mapping, LocalTRHabitatMapping):
        raise LocalTRHabitatMappingError("mapping must be a LocalTRHabitatMapping")
    if (
        isinstance(joint_position_count, bool)
        or not isinstance(joint_position_count, int)
        or joint_position_count < 0
    ):
        raise LocalTRHabitatMappingError(
            "joint_position_count must be a non-negative integer"
        )
    if isinstance(link_blocks, (str, bytes)):
        raise LocalTRHabitatMappingError(
            "link_blocks must be a sequence of HabitatLinkJointBlock values"
        )
    blocks = tuple(link_blocks)
    if any(not isinstance(block, HabitatLinkJointBlock) for block in blocks):
        raise LocalTRHabitatMappingError(
            "link_blocks must contain HabitatLinkJointBlock instances"
        )
    names = [block.link_name for block in blocks]
    if any(not isinstance(name, str) or not name for name in names):
        raise LocalTRHabitatMappingError(
            "every Habitat link block must have a non-empty name"
        )
    if len(set(names)) != len(names):
        raise LocalTRHabitatMappingError("Habitat link block names must be unique")

    specs = mapping.runtime_link_specs
    expected_by_name = {spec.link_name: spec for spec in specs}
    expected_names = set(expected_by_name)
    actual_names = set(names)
    if actual_names != expected_names:
        raise LocalTRHabitatMappingError(
            "Habitat link names must exactly match the mixed local-TR layout: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    occupied: set[int] = set()
    for block in blocks:
        offset = block.joint_position_offset
        count = block.joint_position_count
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise LocalTRHabitatMappingError(
                f"Habitat link {block.link_name!r} offset must be non-negative"
            )
        expected_count = expected_by_name[block.link_name].joint_position_count
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != expected_count
        ):
            raise LocalTRHabitatMappingError(
                f"Habitat link {block.link_name!r} must expose exactly "
                f"{expected_count} joint positions"
            )
        indices = set(range(offset, offset + count))
        if not indices or max(indices) >= joint_position_count:
            raise LocalTRHabitatMappingError(
                f"Habitat link {block.link_name!r} block exceeds joint_position_count"
            )
        if occupied.intersection(indices):
            raise LocalTRHabitatMappingError(
                "Habitat joint-position blocks must not overlap"
            )
        occupied.update(indices)

    expected_count = mapping.runtime_joint_position_count
    if joint_position_count != expected_count or occupied != set(
        range(joint_position_count)
    ):
        raise LocalTRHabitatMappingError(
            "Habitat mixed blocks must densely cover exactly 4*N + 3*D joint positions"
        )
    by_name = {block.link_name: block for block in blocks}
    return LocalTRHabitatBinding(
        runtime_joint_order=mapping.runtime_joint_order,
        translation_driven_joint_ids=mapping.translation_driven_joint_ids,
        rest_translations_m=mapping.rest_translations_m,
        driven_joints=mapping.driven_joints,
        joint_position_count=joint_position_count,
        link_specs=specs,
        blocks=tuple(by_name[spec.link_name] for spec in specs),
    )


__all__ = [
    "LocalTRDrivenJoint",
    "LocalTRHabitatBinding",
    "LocalTRHabitatMapping",
    "LocalTRHabitatMappingError",
    "LocalTRRuntimeLinkSpec",
    "bind_local_tr_habitat_layout",
    "build_local_tr_habitat_mapping",
]
