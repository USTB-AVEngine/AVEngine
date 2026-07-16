"""Pure-Python compilation of an M2 skin into Habitat AO descriptors.

This module deliberately does not import :mod:`habitat_sim`.  It separates the
hash-bound asset description from the link offsets discovered after Habitat
instantiates an articulated object:

* :func:`build_habitat_asset_mapping` validates one rebased skin and emits its
  deterministic URDF, AO config inputs, and ordered rest-pose mapping;
* :func:`bind_habitat_link_layout` validates the runtime link-name/offset
  layout without relying on Habitat's traversal order; and
* :class:`HabitatJointBinding` maps canonical ``(N, 4)`` xyzw poses into the
  flat joint-position vector expected by spherical Bullet joints.

The rebased skin-root frame is not silently identified with the AVEngine actor
frame.  ``actor_from_skin_root`` and a provenance label are mandatory.  At
runtime the root placement is therefore explicitly
``world_from_actor @ actor_from_skin_root``.  Both frames use the AVEngine /
Habitat right-handed ``+Y`` up and ``-Z`` forward convention, but an asset may
still require a fixed rigid offset between them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import quoteattr

import numpy as np

from avengine.m2.glb import GlbDocument, GlbError, extract_skins


_UNIT_QUATERNION_TOLERANCE = 1.0e-9
_REST_QUATERNION_TOLERANCE = 1.0e-5
_UNIT_SCALE_TOLERANCE = 1.0e-7
_RIGID_TRANSFORM_TOLERANCE = 1.0e-7
_ZERO_TOLERANCE = 1.0e-15

# The runtime fork preserves upstream skinned-GLB framing by default.  Only
# canonical AVEngine packages whose URDF and glTF share the same native bind
# frame may opt in to the bounded frame correction through this exact key.
AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY = "avengine_native_gltf_skin_frame"


class HabitatMappingError(ValueError):
    """A rebased asset or Habitat link layout violates the M2 boundary."""


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _quaternion_sign_component(quaternion: np.ndarray) -> float:
    component = float(quaternion[3])
    if math.isclose(component, 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE):
        for candidate in quaternion[:3]:
            if not math.isclose(
                float(candidate), 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE
            ):
                return float(candidate)
    return component


def _canonical_rest_quaternion(value: Sequence[float], *, owner: str) -> Quaternion:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise HabitatMappingError(f"{owner} must contain four finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_REST_QUATERNION_TOLERANCE):
        raise HabitatMappingError(f"{owner} must be a unit quaternion")
    quaternion /= norm
    if _quaternion_sign_component(quaternion) < 0.0:
        quaternion = -quaternion
    return tuple(_canonical_float(component) for component in quaternion)  # type: ignore[return-value]


def _validate_pose_quaternion(value: np.ndarray, *, owner: str) -> Quaternion:
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise HabitatMappingError(f"{owner} must contain four finite numbers")
    norm = float(np.linalg.norm(value))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_QUATERNION_TOLERANCE):
        raise HabitatMappingError(f"{owner} must already be unit normalized")
    if _quaternion_sign_component(value) < 0.0:
        raise HabitatMappingError(
            f"{owner} must use the canonical quaternion hemisphere"
        )
    return tuple(_canonical_float(component) for component in value)  # type: ignore[return-value]


def _validate_rigid_transform(value: Any, *, owner: str) -> Matrix4:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HabitatMappingError(f"{owner} must be a finite 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HabitatMappingError(f"{owner} must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=_RIGID_TRANSFORM_TOLERANCE
    ):
        raise HabitatMappingError(
            f"{owner} must have homogeneous final row [0, 0, 0, 1]"
        )
    rotation = matrix[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    determinant = float(np.linalg.det(rotation))
    if (
        orthogonality_error > _RIGID_TRANSFORM_TOLERANCE
        or abs(determinant - 1.0) > _RIGID_TRANSFORM_TOLERANCE
    ):
        raise HabitatMappingError(
            f"{owner} must be a proper rigid transform "
            f"(orthogonality={orthogonality_error:.9g}, determinant={determinant:.9g})"
        )
    rows = [tuple(_canonical_float(component) for component in row) for row in matrix]
    return tuple(rows)  # type: ignore[return-value]


def _ordered_names(value: Sequence[str], *, owner: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise HabitatMappingError(f"{owner} must be an ordered sequence of names")
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise HabitatMappingError(f"{owner} must contain non-empty string names")
    if len(set(names)) != len(names):
        raise HabitatMappingError(f"{owner} must contain unique names")
    return names


def _number_text(value: float) -> str:
    number = _canonical_float(value)
    return format(number, ".17g")


@dataclass(frozen=True)
class HabitatJointRest:
    """One skin joint in the authored skin order."""

    joint_ordinal: int
    node_index: int
    joint_id: str
    parent_joint_id: str | None
    local_translation_m: Vector3
    rest_rotation_xyzw: Quaternion
    local_scale: Vector3


@dataclass(frozen=True)
class HabitatAssetMapping:
    """Immutable, hash-bound description of one rebased M2 skin."""

    source_glb_sha256: str
    root_joint_id: str
    joint_order: tuple[str, ...]
    runtime_joint_order: tuple[str, ...]
    joints: tuple[HabitatJointRest, ...]
    actor_from_skin_root: Matrix4
    actor_from_skin_root_source: str

    def render_urdf(self, *, robot_name: str = "avengine_m2_animal") -> str:
        """Return deterministic URDF with the skin root as the free base.

        URDF joint origins carry the authored local translations.  The authored
        local rest rotations are deliberately supplied through spherical xyzw
        joint-position blocks, so baking them into ``origin rpy`` as well would
        apply each rotation twice.
        """

        if not isinstance(robot_name, str) or not robot_name:
            raise HabitatMappingError("robot_name must be a non-empty string")
        lines = [
            '<?xml version="1.0"?>',
            f"<robot name={quoteattr(robot_name)}>",
            "  <!-- Spherical joint positions use canonical xyzw quaternions. -->",
        ]
        for joint in self.joints:
            lines.extend(
                [
                    f"  <link name={quoteattr(joint.joint_id)}>",
                    "    <inertial>",
                    '      <mass value="0.001"/>',
                    '      <inertia ixx="1e-6" ixy="0" ixz="0" '
                    'iyy="1e-6" iyz="0" izz="1e-6"/>',
                    "    </inertial>",
                    "  </link>",
                ]
            )
        for joint in self.joints:
            if joint.parent_joint_id is None:
                continue
            xyz = " ".join(_number_text(value) for value in joint.local_translation_m)
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
        lines.append("</robot>")
        return "\n".join(lines) + "\n"

    def joint_mapping_data(self) -> dict[str, Any]:
        """Return detached JSON data for the M2 Habitat joint-mapping file."""

        return {
            "schema": "avengine_m2_habitat_joint_mapping_v1",
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
            "joint_pose_encoding": "ordered_local_rotation_xyzw_float64",
            "actor_from_skin_root": [list(row) for row in self.actor_from_skin_root],
            "actor_from_skin_root_source": self.actor_from_skin_root_source,
            "runtime_root_formula": (
                "world_from_skin_root = world_from_actor @ actor_from_skin_root"
            ),
            "joints": [
                {
                    "joint_ordinal": joint.joint_ordinal,
                    "node_index": joint.node_index,
                    "joint_id": joint.joint_id,
                    "parent_joint_id": joint.parent_joint_id,
                    "local_translation_m": list(joint.local_translation_m),
                    "rest_rotation_xyzw": list(joint.rest_rotation_xyzw),
                    "local_scale": list(joint.local_scale),
                }
                for joint in self.joints
            ],
            "habitat_layout": {
                "base_link": self.root_joint_id,
                "runtime_joint_type": "spherical",
                "runtime_joint_position_count": 4 * len(self.runtime_joint_order),
                "runtime_joint_position_encoding": "xyzw",
                "render_mode": "skin",
            },
        }


@dataclass(frozen=True)
class HabitatLinkJointBlock:
    """One measured Habitat link's block in ``ao.joint_positions``."""

    link_name: str
    joint_position_offset: int
    joint_position_count: int


@dataclass(frozen=True)
class HabitatJointBinding:
    """Validated name-to-offset binding for one instantiated Habitat AO."""

    runtime_joint_order: tuple[str, ...]
    joint_position_count: int
    blocks: tuple[HabitatLinkJointBlock, ...]

    def map_pose(self, pose_rotation_xyzw: Any) -> tuple[float, ...]:
        """Map a canonical ``(N, 4)`` pose into Habitat's flat vector."""

        try:
            pose = np.asarray(pose_rotation_xyzw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise HabitatMappingError(
                "pose_rotation_xyzw must be a finite (N, 4) numeric array"
            ) from exc
        expected_shape = (len(self.runtime_joint_order), 4)
        if pose.shape != expected_shape:
            raise HabitatMappingError(
                f"pose_rotation_xyzw must have shape {expected_shape}, got {pose.shape}"
            )
        by_name = {
            name: _validate_pose_quaternion(
                pose[index], owner=f"pose_rotation_xyzw[{index}] ({name!r})"
            )
            for index, name in enumerate(self.runtime_joint_order)
        }
        positions = [0.0] * self.joint_position_count
        for block in self.blocks:
            start = block.joint_position_offset
            positions[start : start + 4] = by_name[block.link_name]
        return tuple(positions)

    def to_json_data(self) -> dict[str, Any]:
        """Return detached JSON data ordered by the asset runtime order."""

        by_name = {block.link_name: block for block in self.blocks}
        return {
            "runtime_joint_order": list(self.runtime_joint_order),
            "joint_position_count": self.joint_position_count,
            "quaternion_order": "xyzw",
            "links": [
                {
                    "link_name": name,
                    "joint_position_offset": by_name[name].joint_position_offset,
                    "joint_position_count": by_name[name].joint_position_count,
                }
                for name in self.runtime_joint_order
            ],
        }


def build_habitat_asset_mapping(
    document: GlbDocument,
    *,
    actor_from_skin_root: Sequence[Sequence[float]],
    actor_from_skin_root_source: str,
) -> HabitatAssetMapping:
    """Compile one strictly rebased GLB skin into Habitat descriptors.

    ``actor_from_skin_root`` has no default, including for assets whose value is
    explicitly identity.  Callers must bind it to a rebase report or manifest
    through ``actor_from_skin_root_source``.
    """

    if not isinstance(document, GlbDocument):
        raise HabitatMappingError("document must be a parsed GlbDocument")
    if not isinstance(actor_from_skin_root_source, str) or not (
        actor_from_skin_root_source.strip()
    ):
        raise HabitatMappingError(
            "actor_from_skin_root_source must identify a rebase report or manifest"
        )
    transform = _validate_rigid_transform(
        actor_from_skin_root, owner="actor_from_skin_root"
    )
    try:
        skins = extract_skins(document)
    except GlbError as exc:
        raise HabitatMappingError(f"invalid GLB skin: {exc}") from exc
    if len(skins) != 1:
        raise HabitatMappingError(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    roots = [joint for joint in skin.joints if joint.parent_joint_node_index is None]
    if len(roots) != 1:
        raise HabitatMappingError(
            f"skin must contain exactly one joint root, found {len(roots)}"
        )
    root = roots[0]
    if any(
        joint.node_index != root.node_index and joint.parent_joint_node_index is None
        for joint in skin.joints
    ):
        raise HabitatMappingError("skin joints must form one connected joint tree")

    names = [joint.name for joint in skin.joints]
    if any(name is None or not name for name in names):
        raise HabitatMappingError("every skin joint must have a non-empty name")
    joint_order = tuple(name for name in names if name is not None)
    if len(set(joint_order)) != len(joint_order):
        raise HabitatMappingError("skin joint names must be unique")
    name_by_node = {
        joint.node_index: joint.name for joint in skin.joints if joint.name is not None
    }

    root_trs = root.local_trs
    if not np.allclose(
        root_trs.translation, [0.0, 0.0, 0.0], rtol=0.0, atol=_UNIT_SCALE_TOLERANCE
    ) or not np.allclose(
        root_trs.rotation_xyzw,
        [0.0, 0.0, 0.0, 1.0],
        rtol=0.0,
        atol=_UNIT_SCALE_TOLERANCE,
    ):
        raise HabitatMappingError(
            "rebased skin root must have identity local translation/rotation; "
            "its fixed placement belongs in actor_from_skin_root"
        )

    records: list[HabitatJointRest] = []
    for joint in skin.joints:
        assert joint.name is not None  # established above
        if joint.node_index != root.node_index and (
            joint.parent_joint_node_index not in name_by_node
        ):
            raise HabitatMappingError(
                f"joint {joint.name!r} is disconnected from the skin joint tree"
            )
        scale = np.asarray(joint.local_trs.scale, dtype=np.float64)
        scale_error = float(np.max(np.abs(scale - 1.0)))
        if scale_error > _UNIT_SCALE_TOLERANCE:
            raise HabitatMappingError(
                f"joint {joint.name!r} local scale must be exactly unit after rebase "
                f"(maximum error {scale_error:.9g})"
            )
        translation = tuple(
            _canonical_float(component) for component in joint.local_trs.translation
        )
        rotation = _canonical_rest_quaternion(
            joint.local_trs.rotation_xyzw,
            owner=f"joint {joint.name!r} rest_rotation_xyzw",
        )
        records.append(
            HabitatJointRest(
                joint_ordinal=joint.joint_ordinal,
                node_index=joint.node_index,
                joint_id=joint.name,
                parent_joint_id=name_by_node.get(joint.parent_joint_node_index),
                local_translation_m=translation,  # type: ignore[arg-type]
                rest_rotation_xyzw=rotation,
                local_scale=(1.0, 1.0, 1.0),
            )
        )
    runtime_joint_order = tuple(name for name in joint_order if name != root.name)
    if not runtime_joint_order:
        raise HabitatMappingError("skin must contain at least one non-root joint")
    return HabitatAssetMapping(
        source_glb_sha256=document.sha256,
        root_joint_id=root.name,
        joint_order=joint_order,
        runtime_joint_order=runtime_joint_order,
        joints=tuple(records),
        actor_from_skin_root=transform,
        actor_from_skin_root_source=actor_from_skin_root_source,
    )


def build_habitat_asset_mapping_from_rebase_report(
    document: GlbDocument,
    rebase_report: Mapping[str, Any],
) -> HabitatAssetMapping:
    """Build a mapping while binding its root transform to a rebase report."""

    if not isinstance(rebase_report, Mapping):
        raise HabitatMappingError("rebase_report must be a mapping")
    if rebase_report.get("schema") != "avengine_m2_skin_root_rebase_v1":
        raise HabitatMappingError("rebase_report has an unexpected schema")
    if rebase_report.get("status") != "pass":
        raise HabitatMappingError("rebase_report status must be 'pass'")
    output = rebase_report.get("output")
    skin = rebase_report.get("skin")
    if not isinstance(output, Mapping) or output.get("sha256") != document.sha256:
        raise HabitatMappingError(
            "rebase_report output sha256 must match the parsed GLB"
        )
    if not isinstance(skin, Mapping) or "actor_from_canonical_root" not in skin:
        raise HabitatMappingError(
            "rebase_report.skin must declare actor_from_canonical_root"
        )
    mapping = build_habitat_asset_mapping(
        document,
        actor_from_skin_root=skin["actor_from_canonical_root"],
        actor_from_skin_root_source=(
            "avengine_m2_skin_root_rebase_v1.skin.actor_from_canonical_root"
        ),
    )
    if skin.get("root_joint") != mapping.root_joint_id:
        raise HabitatMappingError(
            "rebase_report skin.root_joint must match the parsed GLB root"
        )
    return mapping


def build_habitat_ao_config_data(
    *,
    render_asset: str,
    urdf_filepath: str,
    semantic_id: int,
) -> dict[str, Any]:
    """Return deterministic Habitat AO config JSON data for skinned rendering."""

    if not isinstance(render_asset, str) or not render_asset:
        raise HabitatMappingError("render_asset must be a non-empty string")
    if not isinstance(urdf_filepath, str) or not urdf_filepath:
        raise HabitatMappingError("urdf_filepath must be a non-empty string")
    if (
        isinstance(semantic_id, bool)
        or not isinstance(semantic_id, int)
        or semantic_id < 0
    ):
        raise HabitatMappingError("semantic_id must be a non-negative integer")
    return {
        "urdf_filepath": urdf_filepath,
        "render_asset": render_asset,
        "uniform_scale": 1.0,
        "mass_scale": 1.0,
        "semantic_id": semantic_id,
        "base_type": "free",
        "inertia_source": "computed",
        "link_order": "tree_traversal",
        "render_mode": "skin",
        "shader_type": "phong",
        "user_defined": {AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY: True},
    }


def bind_habitat_link_layout(
    runtime_joint_order: Sequence[str],
    link_blocks: Sequence[HabitatLinkJointBlock],
    *,
    joint_position_count: int,
) -> HabitatJointBinding:
    """Validate measured Habitat link offsets and bind them by exact name."""

    names = _ordered_names(runtime_joint_order, owner="runtime_joint_order")
    if (
        isinstance(joint_position_count, bool)
        or not isinstance(joint_position_count, int)
        or joint_position_count < 0
    ):
        raise HabitatMappingError("joint_position_count must be a non-negative integer")
    if isinstance(link_blocks, (str, bytes)):
        raise HabitatMappingError("link_blocks must be a sequence of link blocks")
    blocks = tuple(link_blocks)
    if any(not isinstance(block, HabitatLinkJointBlock) for block in blocks):
        raise HabitatMappingError(
            "link_blocks must contain HabitatLinkJointBlock instances"
        )
    block_names = [block.link_name for block in blocks]
    if any(not isinstance(name, str) or not name for name in block_names):
        raise HabitatMappingError("every Habitat link block must have a name")
    if len(set(block_names)) != len(block_names):
        raise HabitatMappingError("Habitat link block names must be unique")
    expected = set(names)
    actual = set(block_names)
    if actual != expected:
        raise HabitatMappingError(
            "Habitat link names must exactly match runtime_joint_order: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    occupied: set[int] = set()
    for block in blocks:
        offset = block.joint_position_offset
        count = block.joint_position_count
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise HabitatMappingError(
                f"Habitat link {block.link_name!r} offset must be non-negative"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count != 4:
            raise HabitatMappingError(
                f"Habitat link {block.link_name!r} must expose exactly 4 positions"
            )
        indices = set(range(offset, offset + count))
        if max(indices) >= joint_position_count:
            raise HabitatMappingError(
                f"Habitat link {block.link_name!r} block exceeds joint_position_count"
            )
        if occupied.intersection(indices):
            raise HabitatMappingError("Habitat joint-position blocks must not overlap")
        occupied.update(indices)
    expected_indices = set(range(joint_position_count))
    if joint_position_count != 4 * len(names) or occupied != expected_indices:
        raise HabitatMappingError(
            "Habitat spherical blocks must densely cover exactly 4 positions per "
            "runtime joint"
        )
    by_name = {block.link_name: block for block in blocks}
    return HabitatJointBinding(
        runtime_joint_order=names,
        joint_position_count=joint_position_count,
        blocks=tuple(by_name[name] for name in names),
    )


def map_runtime_pose_to_habitat_joint_positions(
    runtime_joint_order: Sequence[str],
    pose_rotation_xyzw: Any,
    link_blocks: Sequence[HabitatLinkJointBlock],
    *,
    joint_position_count: int,
) -> tuple[float, ...]:
    """Convenience wrapper to bind a measured layout and map one M2 pose."""

    binding = bind_habitat_link_layout(
        runtime_joint_order,
        link_blocks,
        joint_position_count=joint_position_count,
    )
    return binding.map_pose(pose_rotation_xyzw)


__all__ = [
    "AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY",
    "HabitatAssetMapping",
    "HabitatJointBinding",
    "HabitatJointRest",
    "HabitatLinkJointBlock",
    "HabitatMappingError",
    "bind_habitat_link_layout",
    "build_habitat_ao_config_data",
    "build_habitat_asset_mapping",
    "build_habitat_asset_mapping_from_rebase_report",
    "map_runtime_pose_to_habitat_joint_positions",
]
