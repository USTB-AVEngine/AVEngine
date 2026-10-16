"""Fail-closed skin-root rebasing for Habitat-native articulated assets.

Habitat drives a GLB skin from URDF link nodes and removes the matched rig-root
transform before applying inverse-bind matrices.  A source GLB whose mesh is
authored in a scene/world basis while its skin root has a non-identity bind
frame can therefore deform outside its static drawable bounds.  This module
converts that narrowly defined input into a root-local GLB:

* mesh positions and normals are expressed in the skin-root bind basis;
* the common scene ancestors, mesh node, and skin-root node become identity;
* joint scales are normalized when they are demonstrably exporter noise;
* inverse-bind matrices are recomputed from the root-local joint hierarchy;
* constant translation/scale channels and the constant root channel are
  canonicalized so the runtime needs only root motion plus joint rotations.

The separate :func:`rebase_skin_root_preserving_local_tr` entry point retains
STEP/LINEAR non-root translations for the research-only local-TR v2 runtime.
It shares the geometric proof but never changes the spherical-only default.

The operation is a technical normalization, never an asset qualification.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from avengine.assets.glb import (
    AnimationChannel,
    GlbDocument,
    decode_accessor,
    extract_actions,
    extract_node_hierarchy,
    extract_skins,
    load_glb,
    parse_glb,
)
from avengine.assets.glb_write import build_glb


_FLOAT_COMPONENT_TYPE = 5126
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_MAX_SCALE_NOISE = 5.0e-5
_MAX_BIND_ERROR = 5.0e-5
_MAX_CONSTANT_CHANNEL_ERROR = 5.0e-5


class RebaseError(ValueError):
    """The candidate cannot be safely normalized by this bounded compiler."""


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create one output without following or replacing an existing leaf."""

    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _objects(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RebaseError(f"{name} must be an array of objects")
    return value


def _quaternion_matrix(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RebaseError("quaternion must contain four finite components")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise RebaseError("quaternion cannot be zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _node_matrix(
    node: Mapping[str, Any], *, force_unit_scale: bool = False
) -> np.ndarray:
    if "matrix" in node:
        raise RebaseError("matrix-authored nodes are outside the M2 rebase boundary")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = _quaternion_matrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if translation.shape != (3,) or scale.shape != (3,):
        raise RebaseError("node TRS has an invalid shape")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(scale)):
        raise RebaseError("node TRS must be finite")
    if force_unit_scale:
        scale = np.ones(3, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _parents(nodes: Sequence[Mapping[str, Any]]) -> list[int | None]:
    parents: list[int | None] = [None] * len(nodes)
    for parent_index, node in enumerate(nodes):
        children = node.get("children", [])
        if not isinstance(children, list):
            raise RebaseError(f"nodes[{parent_index}].children must be an array")
        for child in children:
            if isinstance(child, bool) or not isinstance(child, int):
                raise RebaseError("node child index must be an integer")
            if child < 0 or child >= len(nodes) or parents[child] is not None:
                raise RebaseError("node hierarchy has an invalid/multiple parent")
            parents[child] = parent_index
    return parents


def _global_matrix(
    node_index: int,
    nodes: Sequence[Mapping[str, Any]],
    parents: Sequence[int | None],
    *,
    force_unit_scale_nodes: set[int] | None = None,
) -> np.ndarray:
    chain: list[int] = []
    cursor: int | None = node_index
    seen: set[int] = set()
    while cursor is not None:
        if cursor in seen:
            raise RebaseError("node hierarchy contains a cycle")
        seen.add(cursor)
        chain.append(cursor)
        cursor = parents[cursor]
    result = np.eye(4, dtype=np.float64)
    forced = force_unit_scale_nodes or set()
    for index in reversed(chain):
        result = result @ _node_matrix(nodes[index], force_unit_scale=index in forced)
    return result


def _rigid_transform(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RebaseError("skin-root bind transform is invalid")
    linear = matrix[:3, :3]
    left, _, right = np.linalg.svd(linear)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    error = float(np.max(np.abs(linear - rotation)))
    if error > _MAX_SCALE_NOISE:
        raise RebaseError(
            f"skin-root path contains non-rigid scale/shear ({error:.9g})"
        )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = matrix[:3, 3]
    return result, error


def _accessor_layout(
    document: Mapping[str, Any], accessor_index: int
) -> tuple[int, int, int, int, str]:
    accessors = _objects(document.get("accessors", []), "accessors")
    views = _objects(document.get("bufferViews", []), "bufferViews")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise RebaseError(f"accessor index is out of range: {accessor_index}")
    accessor = accessors[accessor_index]
    if accessor.get("componentType") != _FLOAT_COMPONENT_TYPE:
        raise RebaseError(f"accessor {accessor_index} must use FLOAT")
    element_type = accessor.get("type")
    if element_type not in _COMPONENTS:
        raise RebaseError(f"unsupported accessor type: {element_type!r}")
    count = accessor.get("count")
    view_index = accessor.get("bufferView")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or view_index < 0
        or view_index >= len(views)
    ):
        raise RebaseError(f"accessor {accessor_index} has an invalid layout")
    view = views[view_index]
    if view.get("buffer") != 0:
        raise RebaseError("rebase supports only embedded buffer 0")
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (view_offset, accessor_offset)
    ):
        raise RebaseError("accessor offsets must be non-negative integers")
    components = _COMPONENTS[element_type]
    element_bytes = components * 4
    stride = view.get("byteStride", element_bytes)
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < element_bytes
        or stride % 4
    ):
        raise RebaseError("accessor stride is invalid")
    return view_offset + accessor_offset, stride, count, components, element_type


def _decode_joint_indices(source: GlbDocument, accessor_index: int) -> np.ndarray:
    """Decode the bounded JOINTS_0 formats accepted by the M2 compiler."""

    document = source.json
    accessors = _objects(document.get("accessors", []), "accessors")
    views = _objects(document.get("bufferViews", []), "bufferViews")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise RebaseError(f"JOINTS_0 accessor is out of range: {accessor_index}")
    accessor = accessors[accessor_index]
    if accessor.get("type") != "VEC4" or accessor.get("normalized", False):
        raise RebaseError("JOINTS_0 must be a non-normalized VEC4")
    if "sparse" in accessor:
        raise RebaseError("sparse JOINTS_0 is outside the M2 compiler boundary")
    component_type = accessor.get("componentType")
    format_by_type = {5121: "B", 5123: "H"}
    component_format = format_by_type.get(component_type)
    if component_format is None:
        raise RebaseError("JOINTS_0 must use UNSIGNED_BYTE or UNSIGNED_SHORT")
    count = accessor.get("count")
    view_index = accessor.get("bufferView")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or view_index < 0
        or view_index >= len(views)
    ):
        raise RebaseError("JOINTS_0 has an invalid accessor layout")
    view = views[view_index]
    if view.get("buffer") != 0:
        raise RebaseError("JOINTS_0 must reference embedded buffer 0")
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (view_offset, accessor_offset, view_length)
    ):
        raise RebaseError("JOINTS_0 offsets/length must be non-negative integers")
    packer = struct.Struct("<" + component_format * 4)
    stride = view.get("byteStride", packer.size)
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < packer.size
        or stride % (packer.size // 4)
    ):
        raise RebaseError("JOINTS_0 byteStride is invalid")
    required = accessor_offset + (count - 1) * stride + packer.size
    if required > view_length or view_offset + required > len(source.binary):
        raise RebaseError("JOINTS_0 extends beyond its bufferView")
    values = np.empty((count, 4), dtype=np.int64)
    start = view_offset + accessor_offset
    for index in range(count):
        values[index] = packer.unpack_from(source.binary, start + index * stride)
    return values


def _default_scene_nodes(document: Mapping[str, Any]) -> set[int]:
    nodes = _objects(document.get("nodes", []), "nodes")
    scenes = _objects(document.get("scenes", []), "scenes")
    default_scene = document.get("scene", 0 if len(scenes) == 1 else None)
    if (
        isinstance(default_scene, bool)
        or not isinstance(default_scene, int)
        or default_scene < 0
        or default_scene >= len(scenes)
    ):
        raise RebaseError("GLB must declare one valid default scene")
    roots = scenes[default_scene].get("nodes")
    if not isinstance(roots, list) or not roots:
        raise RebaseError("default scene must contain at least one root node")
    reachable: set[int] = set()
    stack = list(roots)
    while stack:
        node_index = stack.pop()
        if (
            isinstance(node_index, bool)
            or not isinstance(node_index, int)
            or node_index < 0
            or node_index >= len(nodes)
        ):
            raise RebaseError("default scene contains an invalid node index")
        if node_index in reachable:
            continue
        reachable.add(node_index)
        children = nodes[node_index].get("children", [])
        if not isinstance(children, list):
            raise RebaseError("scene node children must be an array")
        stack.extend(children)
    return reachable


def _write_accessor(
    document: dict[str, Any],
    binary: bytearray,
    accessor_index: int,
    values: np.ndarray,
) -> None:
    offset, stride, count, components, _ = _accessor_layout(document, accessor_index)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count, components) or not np.all(np.isfinite(array)):
        raise RebaseError(
            f"replacement accessor {accessor_index} has shape/value mismatch"
        )
    packer = struct.Struct("<" + "f" * components)
    for index, row in enumerate(array):
        packer.pack_into(binary, offset + index * stride, *row.tolist())


def _matrix_values(matrix: np.ndarray) -> np.ndarray:
    """Return one glTF MAT4 element in column-major component order."""

    return np.asarray(matrix, dtype=np.float64).T.reshape(16)


def _identity_node(node: dict[str, Any]) -> None:
    node.pop("matrix", None)
    node["translation"] = [0.0, 0.0, 0.0]
    node["rotation"] = [0.0, 0.0, 0.0, 1.0]
    node["scale"] = [1.0, 1.0, 1.0]


def _maximum_constant_error(values: np.ndarray) -> float:
    return float(np.max(np.abs(values - values[0])))


def _quaternion_equivalence_error(values: np.ndarray, reference: np.ndarray) -> float:
    reference = reference / np.linalg.norm(reference)
    maximum = 0.0
    for value in values:
        normalized = value / np.linalg.norm(value)
        maximum = max(
            maximum,
            min(
                float(np.max(np.abs(normalized - reference))),
                float(np.max(np.abs(normalized + reference))),
            ),
        )
    return maximum


def _set_channel_values(
    document: dict[str, Any],
    binary: bytearray,
    channel: AnimationChannel,
    value: Sequence[float],
) -> None:
    if channel.interpolation == "CUBICSPLINE":
        raise RebaseError("CUBICSPLINE channels are outside this rebase boundary")
    replacement = np.tile(
        np.asarray(value, dtype=np.float64), (len(channel.timestamps_seconds), 1)
    )
    _write_accessor(document, binary, channel.output_accessor_index, replacement)


def _canonicalize_channels(
    source: GlbDocument,
    document: dict[str, Any],
    binary: bytearray,
    *,
    root_node: int,
    joint_nodes: set[int],
    preserve_local_tr: bool,
) -> tuple[list[dict[str, Any]], set[int]]:
    node_defaults = {
        node.node_index: node.local_trs for node in extract_node_hierarchy(source)
    }
    reports: list[dict[str, Any]] = []
    translation_driven_nodes: set[int] = set()
    for action in extract_actions(source):
        for channel in action.channels:
            if channel.target_node_index not in joint_nodes:
                raise RebaseError(f"animation {action.name!r} targets a non-skin node")
            values = np.asarray(channel.values, dtype=np.float64)
            if preserve_local_tr and channel.interpolation == "CUBICSPLINE":
                raise RebaseError(
                    "CUBICSPLINE channels are outside this rebase boundary"
                )
            report = {
                "action": action.name,
                "node": channel.target_node_name,
                "path": channel.target_path,
                "interpolation": channel.interpolation,
                "sample_count": len(channel.timestamps_seconds),
            }
            if channel.target_path == "scale":
                error = float(np.max(np.abs(values - 1.0)))
                if error > _MAX_SCALE_NOISE:
                    raise RebaseError(
                        f"animation scale is not exporter noise: {action.name}/"
                        f"{channel.target_node_name} error={error:.9g}"
                    )
                _set_channel_values(document, binary, channel, [1.0, 1.0, 1.0])
                report["canonicalized"] = "identity_scale"
                report["maximum_input_error"] = error
            elif channel.target_path == "translation":
                temporal_error = _maximum_constant_error(values)
                default = np.asarray(
                    node_defaults[channel.target_node_index].translation,
                    dtype=np.float64,
                )
                default_error = max(
                    float(np.max(np.abs(values[0] - default))), temporal_error
                )
                if channel.target_node_index == root_node:
                    if default_error > _MAX_CONSTANT_CHANNEL_ERROR:
                        if preserve_local_tr:
                            raise RebaseError(
                                "skin-root translation is not constant: "
                                f"{action.name} error={default_error:.9g}"
                            )
                        raise RebaseError(
                            "per-bone dynamic/ambiguous translation cannot be "
                            f"expressed by stock Habitat: {action.name}/"
                            f"{channel.target_node_name} error={default_error:.9g}"
                        )
                    _set_channel_values(document, binary, channel, [0.0, 0.0, 0.0])
                    report["canonicalized"] = "root_zero_translation"
                elif preserve_local_tr:
                    if channel.interpolation not in {"STEP", "LINEAR"}:
                        raise RebaseError(
                            "local-TR translation interpolation must be STEP or LINEAR"
                        )
                    if float(np.max(np.abs(values - default))) > (
                        _MAX_CONSTANT_CHANNEL_ERROR
                    ):
                        translation_driven_nodes.add(channel.target_node_index)
                    report["canonicalized"] = (
                        "preserved_absolute_child_local_translation"
                    )
                elif default_error > _MAX_CONSTANT_CHANNEL_ERROR:
                    raise RebaseError(
                        "per-bone dynamic/ambiguous translation cannot be expressed "
                        f"by stock Habitat: {action.name}/{channel.target_node_name} "
                        f"error={default_error:.9g}"
                    )
                else:
                    _set_channel_values(document, binary, channel, default)
                    report["canonicalized"] = "fixed_bind_translation"
                report["maximum_input_error"] = default_error
            elif channel.target_path == "rotation":
                norms = np.linalg.norm(values, axis=1)
                if (
                    np.any(norms < 1.0e-12)
                    or float(np.max(np.abs(norms - 1.0))) > 1.0e-5
                ):
                    raise RebaseError("animation contains a non-unit quaternion")
                if channel.target_node_index == root_node:
                    reference = np.asarray(
                        node_defaults[root_node].rotation_xyzw, dtype=np.float64
                    )
                    error = _quaternion_equivalence_error(values, reference)
                    if error > _MAX_CONSTANT_CHANNEL_ERROR:
                        raise RebaseError(
                            f"skin-root animation is not constant: {action.name} "
                            f"error={error:.9g}"
                        )
                    _set_channel_values(document, binary, channel, [0.0, 0.0, 0.0, 1.0])
                    report["canonicalized"] = "root_identity_rotation"
                    report["maximum_input_error"] = error
                else:
                    report["canonicalized"] = "preserved_rotation"
            reports.append(report)
    return reports, translation_driven_nodes


def _ancestors(node_index: int, parents: Sequence[int | None]) -> list[int]:
    result: list[int] = []
    cursor = parents[node_index]
    while cursor is not None:
        result.append(cursor)
        cursor = parents[cursor]
    return result


def _unique(values: Iterable[int]) -> list[int]:
    return sorted(set(values))


def _rebase_skin_root(
    source_path: str | Path,
    output_path: str | Path,
    *,
    preserve_local_tr: bool,
) -> dict[str, Any]:
    """Implement the spherical-only and explicitly selected local-TR routes."""

    source_resolved = Path(source_path).resolve()
    output_argument = Path(output_path)
    if output_argument.exists() or output_argument.is_symlink():
        raise RebaseError(f"refusing to replace output: {output_argument}")
    output = output_argument.resolve()
    if output == source_resolved:
        raise RebaseError("output must not overwrite the source candidate")

    source = load_glb(source_resolved)
    source_json = source.json
    document = copy.deepcopy(dict(source_json))
    binary = bytearray(source.binary)
    nodes = _objects(document.get("nodes", []), "nodes")
    parents = _parents(nodes)
    skins = extract_skins(source)
    if len(skins) != 1:
        raise RebaseError(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    joint_nodes = {joint.node_index for joint in skin.joints}
    roots = [joint for joint in skin.joints if joint.parent_joint_node_index is None]
    joint_names = [joint.name for joint in skin.joints]
    if (
        len(roots) != 1
        or any(name is None or not name for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
    ):
        raise RebaseError("skin must be one named joint tree")
    root = roots[0]
    if any(
        joint.node_index != root.node_index and joint.parent_joint_node_index is None
        for joint in skin.joints
    ):
        raise RebaseError("skin joint hierarchy is disconnected")

    mesh_nodes = [
        index for index, node in enumerate(nodes) if node.get("skin") == skin.skin_index
    ]
    if len(mesh_nodes) != 1:
        raise RebaseError(
            f"expected exactly one mesh node for skin 0, found {mesh_nodes}"
        )
    mesh_node = mesh_nodes[0]
    mesh_index = nodes[mesh_node].get("mesh")
    meshes = _objects(document.get("meshes", []), "meshes")
    if (
        isinstance(mesh_index, bool)
        or not isinstance(mesh_index, int)
        or mesh_index < 0
        or mesh_index >= len(meshes)
    ):
        raise RebaseError("skinned node has an invalid mesh index")
    reachable_nodes = _default_scene_nodes(document)
    if root.node_index not in reachable_nodes or mesh_node not in reachable_nodes:
        raise RebaseError("skin root and skinned mesh must be in the default scene")

    scale_errors = {
        joint.name: float(
            np.max(np.abs(np.asarray(joint.local_trs.scale, dtype=np.float64) - 1.0))
        )
        for joint in skin.joints
    }
    maximum_joint_scale_error = max(scale_errors.values())
    if maximum_joint_scale_error > _MAX_SCALE_NOISE:
        raise RebaseError(
            f"joint scale is not bounded exporter noise: {maximum_joint_scale_error:.9g}"
        )

    root_global_raw = _global_matrix(root.node_index, nodes, parents)
    actor_from_root, root_rigid_error = _rigid_transform(root_global_raw)
    root_from_actor = np.linalg.inv(actor_from_root)

    if skin.inverse_bind_matrices is None:
        raise RebaseError("skin must declare inverseBindMatrices")
    source_bind_frames: list[np.ndarray] = []
    for joint, raw_inverse in zip(skin.joints, skin.inverse_bind_matrices, strict=True):
        inverse = np.asarray(raw_inverse, dtype=np.float64).reshape(4, 4).T
        global_bind = _global_matrix(joint.node_index, nodes, parents)
        source_bind_frames.append(global_bind @ inverse)
    source_bind_frame_raw = source_bind_frames[0]
    source_bind_consistency_errors = [
        float(np.max(np.abs(frame - source_bind_frame_raw)))
        for frame in source_bind_frames
    ]
    maximum_source_bind_error = max(source_bind_consistency_errors)
    if maximum_source_bind_error > _MAX_BIND_ERROR:
        raise RebaseError(
            "source joints disagree on the skin bind frame: "
            f"{maximum_source_bind_error:.9g}"
        )
    source_bind_frame, source_bind_rigid_error = _rigid_transform(source_bind_frame_raw)

    mesh_global_raw = _global_matrix(mesh_node, nodes, parents)
    mesh_global, mesh_rigid_error = _rigid_transform(mesh_global_raw)
    mesh_bind_frame_disagreement = float(
        np.max(np.abs(mesh_global - source_bind_frame))
    )
    canonical_from_source_bind = root_from_actor @ source_bind_frame
    canonical_from_source_linear = canonical_from_source_bind[:3, :3]
    orthogonality_error = float(
        np.max(
            np.abs(
                canonical_from_source_linear.T @ canonical_from_source_linear
                - np.eye(3)
            )
        )
    )
    canonical_from_source_determinant = float(
        np.linalg.det(canonical_from_source_linear)
    )
    if (
        orthogonality_error > _MAX_SCALE_NOISE
        or abs(canonical_from_source_determinant - 1.0) > _MAX_SCALE_NOISE
    ):
        raise RebaseError(
            "canonical-from-source bind transform must be proper rigid: "
            f"orthogonality={orthogonality_error:.9g}, "
            f"determinant={canonical_from_source_determinant:.9g}"
        )

    primitives = _objects(meshes[mesh_index].get("primitives", []), "mesh.primitives")
    transformed_positions: set[int] = set()
    transformed_normals: set[int] = set()
    transformed_tangents: set[int] = set()
    position_records: list[dict[str, Any]] = []
    influence_records: list[dict[str, Any]] = []
    for primitive_index, primitive in enumerate(primitives):
        if "targets" in primitive:
            raise RebaseError("morph targets are outside the M2 no-mouth boundary")
        attributes = primitive.get("attributes")
        if not isinstance(attributes, dict):
            raise RebaseError("mesh primitive attributes must be an object")
        if any(key in attributes for key in ("JOINTS_1", "WEIGHTS_1")):
            raise RebaseError(
                "more than four skin influences are outside this compiler"
            )
        for required in ("POSITION", "JOINTS_0", "WEIGHTS_0"):
            if required not in attributes:
                raise RebaseError(f"mesh primitive lacks {required}")

        position_index = attributes["POSITION"]
        source_position = decode_accessor(source, position_index)
        if source_position.element_type != "VEC3":
            raise RebaseError("POSITION must be FLOAT VEC3")
        position_count = source_position.count

        joint_index = attributes["JOINTS_0"]
        weight_index = attributes["WEIGHTS_0"]
        joint_values = _decode_joint_indices(source, joint_index)
        weight_accessor = decode_accessor(source, weight_index)
        if weight_accessor.element_type != "VEC4":
            raise RebaseError("WEIGHTS_0 must be FLOAT VEC4")
        weights = np.asarray(weight_accessor.values, dtype=np.float64)
        if joint_values.shape != (position_count, 4) or weights.shape != (
            position_count,
            4,
        ):
            raise RebaseError("POSITION/JOINTS_0/WEIGHTS_0 counts must match")
        if int(np.max(joint_values)) >= len(skin.joints):
            raise RebaseError("JOINTS_0 references a joint outside the skin")
        minimum_weight = float(np.min(weights))
        maximum_weight = float(np.max(weights))
        maximum_weight_sum_error = float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
        if minimum_weight < -1.0e-7 or maximum_weight_sum_error > 1.0e-5:
            raise RebaseError(
                "WEIGHTS_0 must be non-negative and sum to one: "
                f"minimum={minimum_weight:.9g}, "
                f"maximum_sum_error={maximum_weight_sum_error:.9g}"
            )
        influence_records.append(
            {
                "primitive": primitive_index,
                "joint_accessor": joint_index,
                "weight_accessor": weight_index,
                "vertex_count": position_count,
                "minimum_joint_ordinal": int(np.min(joint_values)),
                "maximum_joint_ordinal": int(np.max(joint_values)),
                "minimum_weight": minimum_weight,
                "maximum_weight": maximum_weight,
                "maximum_weight_sum_error": maximum_weight_sum_error,
            }
        )

        if position_index not in transformed_positions:
            positions = np.asarray(source_position.values, dtype=np.float64)
            homogeneous = np.concatenate(
                [positions, np.ones((len(positions), 1), dtype=np.float64)], axis=1
            )
            canonical = (canonical_from_source_bind @ homogeneous.T).T[:, :3]
            _write_accessor(document, binary, position_index, canonical)
            accessor = document["accessors"][position_index]
            accessor["min"] = np.min(canonical, axis=0).tolist()
            accessor["max"] = np.max(canonical, axis=0).tolist()
            transformed_positions.add(position_index)
            position_records.append(
                {
                    "primitive": primitive_index,
                    "accessor": position_index,
                    "count": len(canonical),
                    "canonical_min": accessor["min"],
                    "canonical_max": accessor["max"],
                }
            )

        normal_index = attributes.get("NORMAL")
        if normal_index is not None and normal_index not in transformed_normals:
            source_normal = decode_accessor(source, normal_index)
            if source_normal.element_type != "VEC3":
                raise RebaseError("NORMAL must be FLOAT VEC3")
            normals = np.asarray(source_normal.values, dtype=np.float64)
            normal_matrix = np.linalg.inv(canonical_from_source_linear).T
            canonical = (normal_matrix @ normals.T).T
            lengths = np.linalg.norm(canonical, axis=1)
            if np.any(lengths < 1.0e-12):
                raise RebaseError("NORMAL contains a zero vector")
            canonical /= lengths[:, None]
            _write_accessor(document, binary, normal_index, canonical)
            transformed_normals.add(normal_index)

        tangent_index = attributes.get("TANGENT")
        if tangent_index is not None and tangent_index not in transformed_tangents:
            source_tangent = decode_accessor(source, tangent_index)
            if source_tangent.element_type != "VEC4":
                raise RebaseError("TANGENT must be FLOAT VEC4")
            tangents = np.asarray(source_tangent.values, dtype=np.float64)
            tangent_matrix = np.linalg.inv(canonical_from_source_linear).T
            xyz = (tangent_matrix @ tangents[:, :3].T).T
            lengths = np.linalg.norm(xyz, axis=1)
            if np.any(lengths < 1.0e-12):
                raise RebaseError("TANGENT contains a zero vector")
            tangents[:, :3] = xyz / lengths[:, None]
            _write_accessor(document, binary, tangent_index, tangents)
            transformed_tangents.add(tangent_index)

    channel_report, translation_driven_nodes = _canonicalize_channels(
        source,
        document,
        binary,
        root_node=root.node_index,
        joint_nodes=joint_nodes,
        preserve_local_tr=preserve_local_tr,
    )

    # Canonical hierarchy: root/mesh/common ancestors are identity; children
    # retain their authored local translations and rotations but no scale.
    identity_nodes = set(_ancestors(root.node_index, parents))
    identity_nodes.update(_ancestors(mesh_node, parents))
    identity_nodes.update({root.node_index, mesh_node})
    for index in identity_nodes:
        _identity_node(nodes[index])
    for joint_index in joint_nodes:
        nodes[joint_index]["scale"] = [1.0, 1.0, 1.0]

    canonical_globals: dict[int, np.ndarray] = {}

    def canonical_global(node_index: int) -> np.ndarray:
        if node_index in canonical_globals:
            return canonical_globals[node_index]
        if node_index == root.node_index:
            value = np.eye(4, dtype=np.float64)
        else:
            parent = parents[node_index]
            if parent not in joint_nodes:
                raise RebaseError("canonical joint hierarchy is disconnected")
            value = canonical_global(parent) @ _node_matrix(
                nodes[node_index], force_unit_scale=True
            )
        canonical_globals[node_index] = value
        return value

    inverse_accessor = skin.inverse_bind_matrices_accessor_index
    if inverse_accessor is None:
        raise RebaseError("skin inverse-bind accessor disappeared")
    canonical_inverse = np.vstack(
        [
            _matrix_values(np.linalg.inv(canonical_global(joint.node_index)))
            for joint in skin.joints
        ]
    )
    _write_accessor(document, binary, inverse_accessor, canonical_inverse)

    payload = build_glb(document, binary)
    verified = parse_glb(payload)
    verified_skins = extract_skins(verified)
    if len(verified_skins) != 1:
        raise RebaseError("output skin readback failed")
    verified_skin = verified_skins[0]
    verified_nodes = _objects(verified.json.get("nodes", []), "nodes")
    verified_parents = _parents(verified_nodes)
    output_bind_errors: list[float] = []
    assert verified_skin.inverse_bind_matrices is not None
    for joint, raw_inverse in zip(
        verified_skin.joints, verified_skin.inverse_bind_matrices, strict=True
    ):
        inverse = np.asarray(raw_inverse, dtype=np.float64).reshape(4, 4).T
        global_bind = _global_matrix(joint.node_index, verified_nodes, verified_parents)
        output_bind_errors.append(
            float(np.max(np.abs(global_bind @ inverse - np.eye(4))))
        )
    maximum_output_bind_error = max(output_bind_errors)
    if maximum_output_bind_error > _MAX_BIND_ERROR:
        raise RebaseError(
            f"output inverse-bind closure failed: {maximum_output_bind_error:.9g}"
        )

    try:
        _write_exclusive(output, payload)
    except OSError as exc:
        raise RebaseError(
            f"failed to create output exclusively: {output}: {exc}"
        ) from exc

    runtime_joint_order = tuple(
        joint.name for joint in skin.joints if joint.node_index != root.node_index
    )
    translation_driven_joint_ids = tuple(
        joint.name
        for joint in skin.joints
        if joint.node_index in translation_driven_nodes
    )
    spherical_joint_count = len(skin.joints) - 1
    prismatic_joint_count = (
        3 * len(translation_driven_joint_ids) if preserve_local_tr else 0
    )
    report = {
        "schema": (
            "avengine_m2_skin_root_rebase_local_tr_v2"
            if preserve_local_tr
            else "avengine_m2_skin_root_rebase_v1"
        ),
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_resolved),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "skin": {
            "root_joint": root.name,
            "joint_count": len(skin.joints),
            "mesh_node": nodes[mesh_node].get("name"),
            "actor_from_canonical_root": actor_from_root.tolist(),
            "source_skin_bind_frame": source_bind_frame.tolist(),
            "source_mesh_global_bind_frame": mesh_global.tolist(),
            "source_mesh_vs_skin_bind_frame_max_abs": mesh_bind_frame_disagreement,
            "canonical_root_from_source_bind": canonical_from_source_bind.tolist(),
            "canonical_root_from_source_bind_orthogonality_error": orthogonality_error,
            "canonical_root_from_source_bind_determinant": canonical_from_source_determinant,
            "maximum_root_rigid_normalization_error": root_rigid_error,
            "maximum_source_bind_frame_rigid_normalization_error": source_bind_rigid_error,
            "maximum_source_mesh_rigid_normalization_error": mesh_rigid_error,
            "maximum_joint_scale_normalization_error": maximum_joint_scale_error,
            "maximum_source_bind_frame_consistency_error": maximum_source_bind_error,
            "maximum_output_bind_closure_error": maximum_output_bind_error,
        },
        "positions": position_records,
        "skin_influences": influence_records,
        "animation_channels": channel_report,
        "runtime_contract": {
            "base_link": root.name,
            "spherical_joint_count": spherical_joint_count,
            "joint_position_count": (4 * spherical_joint_count + prismatic_joint_count),
            "actor_root_transform_source": "actor_from_canonical_root",
            "per_bone_dynamic_translation": preserve_local_tr,
            "per_bone_dynamic_scale": False,
        },
        "notes": [
            "This technical rebase does not qualify mesh, weights, actions, contacts, provenance, or visual alignment.",
            "The actor root transform must be applied explicitly by the runtime; it is not a free-running GLB animation.",
        ],
    }
    if preserve_local_tr:
        report["runtime_contract"].update(
            {
                "schema": "avengine_m2_local_tr_runtime_v2",
                "coordinate_layout": "xyz_prismatic_then_xyzw_spherical",
                "translation_semantics": "absolute_child_local_meters",
                "rotation_semantics": "absolute_child_local_xyzw",
                "runtime_joint_order": list(runtime_joint_order),
                "translation_driven_joint_ids": list(translation_driven_joint_ids),
                "prismatic_joint_count": prismatic_joint_count,
            }
        )
        report["notes"].append(
            "This output requires the research-only local-TR v2 mixed prismatic/spherical runtime and is not compatible with the formal spherical-only v1 mapping."
        )
    return report


def rebase_skin_root(
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create one formal-compatible spherical-only root-local GLB."""

    return _rebase_skin_root(
        source_path,
        output_path,
        preserve_local_tr=False,
    )


def rebase_skin_root_preserving_local_tr(
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create one research-only local-TR v2 root-local GLB.

    Non-root STEP/LINEAR translation samples remain absolute child-local
    values.  Root motion, meaningful scale animation, and CUBICSPLINE remain
    outside the bounded compiler contract.
    """

    return _rebase_skin_root(
        source_path,
        output_path,
        preserve_local_tr=True,
    )
