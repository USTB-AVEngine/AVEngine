"""Hash-closed reuse of qualified M2 poses on compatible appearance variants.

An appearance realization changes mesh, material, and optionally uniformly
scaled rest translations.  It must not silently revive animation clips that
remain embedded in the source GLB when the admitted M2 motion is a separately
baked NPZ.  This module verifies the complete source-package -> appearance ->
rebase chain and emits the admitted rotations again with only their visual
identity changed.

The result remains a non-qualifying research artifact.  Runtime and visual
review are still required for every derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.m2.actions import (
    ActionBakeError,
    BakedActionSet,
    baked_actions_content_sha256,
    read_baked_actions_npz,
    write_baked_actions_npz,
)
from avengine.m2.contracts import validate_animal_asset_package
from avengine.m2.glb import (
    GlbDocument,
    GlbError,
    extract_actions,
    extract_node_hierarchy,
    extract_skins,
    load_glb,
    parse_glb,
)
from avengine.m2.habitat import (
    HabitatMappingError,
    build_habitat_asset_mapping_from_rebase_report,
)


ACTION_REPORT_SCHEMA = "avengine_m2_action_bake_report_v1"
DERIVATION_SCHEMA = "avengine_m2_appearance_action_rebind_v1"
APPEARANCE_REPORT_SCHEMA = "avengine_animal_appearance_realization_v1"
REBASE_REPORT_SCHEMA = "avengine_m2_skin_root_rebase_v1"
DEFORMATION_REPORT_SCHEMA = "avengine_m2_rebase_deformation_verification_v1"
_APPEARANCE_FLOAT_TOLERANCE = 5.0e-5
_COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_ACCESSOR_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


class ActionRebindError(RuntimeError):
    """The source motion is not proven compatible with the target visual."""


@dataclass(frozen=True)
class ActionRebindResult:
    """In-memory bytes and reports ready for exclusive pair emission."""

    actions: BakedActionSet
    artifact_bytes: bytes
    report: Mapping[str, Any]


def _absolute_without_symlinks(path: str | Path, *, owner: str) -> Path:
    absolute = Path(os.path.abspath(Path(path)))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ActionRebindError(
                f"{owner} path must not contain a symbolic link: {absolute}"
            )
    return absolute


def _regular_file(path: str | Path, *, owner: str, suffix: str) -> Path:
    absolute = _absolute_without_symlinks(path, owner=owner)
    if (
        absolute.suffix.lower() != suffix
        or not absolute.is_file()
        or absolute.stat().st_size <= 0
    ):
        raise ActionRebindError(
            f"{owner} must be a non-empty {suffix} regular file: {absolute}"
        )
    return absolute


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _require_reference(
    value: Any,
    *,
    owner: str,
    path: Path | None = None,
    sha256: str,
    byte_size: int,
    require_byte_size: bool = True,
) -> None:
    if not isinstance(value, Mapping):
        raise ActionRebindError(f"{owner} must be an object")
    if value.get("sha256") != sha256 or (
        require_byte_size and value.get("byte_size") != byte_size
    ):
        raise ActionRebindError(f"{owner} does not bind the expected bytes")
    if path is not None:
        raw_path = value.get("path")
        if not isinstance(raw_path, str):
            raise ActionRebindError(f"{owner}.path must be a string")
        try:
            declared = _absolute_without_symlinks(raw_path, owner=f"{owner}.path")
        except (OSError, ValueError) as exc:
            raise ActionRebindError(f"{owner}.path is unsafe: {exc}") from exc
        if declared != path:
            raise ActionRebindError(f"{owner}.path differs from the expected file")


def _package_role(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    matches = [
        record
        for record in manifest.get("files", [])
        if isinstance(record, Mapping) and record.get("role") == role
    ]
    if len(matches) != 1:
        raise ActionRebindError(f"source package must contain one {role!r} role")
    record = matches[0]
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ActionRebindError(f"source package role {role!r} lacks a path")
    try:
        resolved = resolve_declared_path(raw_path, manifest_dir=manifest_path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise ActionRebindError(
            f"unable to resolve source package role {role!r}: {exc}"
        ) from exc
    path = _regular_file(
        resolved,
        owner=f"source package role {role}",
        suffix=Path(raw_path).suffix.lower(),
    )
    _require_reference(
        record,
        owner=f"source package role {role}",
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
    )
    return path


def _load_source_package(
    manifest_path: str | Path,
) -> tuple[Path, Path, Path, Mapping[str, Any]]:
    path = _regular_file(manifest_path, owner="source package manifest", suffix=".json")
    manifest = load_json(path)
    errors = validate_animal_asset_package(manifest, manifest_path=path)
    if errors:
        raise ActionRebindError(
            "source animal package is invalid: " + "; ".join(errors)
        )
    visual = _package_role(path, manifest, "visual")
    idle = _package_role(path, manifest, "idle_poses")
    walk = _package_role(path, manifest, "walk_poses")
    if _record(idle)["sha256"] != _record(walk)["sha256"]:
        raise ActionRebindError(
            "source idle and walk roles must bind the same canonical action set"
        )
    return path, visual, idle, manifest


def _requested_size_scale(report: Mapping[str, Any]) -> float:
    request = report.get("instance_request")
    operations = (
        request.get("realization_operations") if isinstance(request, Mapping) else None
    )
    if not isinstance(operations, list):
        raise ActionRebindError(
            "appearance report lacks requested size realization parameters"
        )
    matches = [
        operation
        for operation in operations
        if isinstance(operation, Mapping) and operation.get("attribute") == "size"
    ]
    if len(matches) != 1 or matches[0].get("operation_id") != "uniform_actor_scale_v1":
        raise ActionRebindError(
            "appearance report must contain one uniform size request"
        )
    parameters = matches[0].get("parameters")
    scale = parameters.get("scale_ratio") if isinstance(parameters, Mapping) else None
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ActionRebindError("appearance requested size scale is invalid")
    return float(scale)


def _accessor_array(document: GlbDocument, index: Any, *, owner: str) -> np.ndarray:
    root = document.json
    accessors = root.get("accessors")
    views = root.get("bufferViews")
    if (
        not isinstance(accessors, list)
        or not isinstance(views, list)
        or isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(accessors)
    ):
        raise ActionRebindError(f"{owner} accessor is invalid")
    accessor = accessors[index]
    if (
        not isinstance(accessor, Mapping)
        or "sparse" in accessor
        or accessor.get("normalized", False) is not False
    ):
        raise ActionRebindError(f"{owner} accessor must be non-sparse/non-normalized")
    view_index = accessor.get("bufferView")
    if (
        isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(views)
    ):
        raise ActionRebindError(f"{owner} bufferView is invalid")
    view = views[view_index]
    if not isinstance(view, Mapping) or view.get("buffer", 0) != 0:
        raise ActionRebindError(f"{owner} must use embedded buffer 0")
    dtype = _COMPONENT_DTYPES.get(accessor.get("componentType"))
    width = _ACCESSOR_WIDTHS.get(accessor.get("type"))
    count = accessor.get("count")
    if (
        dtype is None
        or width is None
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ActionRebindError(f"{owner} accessor layout is unsupported")
    element_bytes = dtype.itemsize * width
    stride = view.get("byteStride", element_bytes)
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (stride, view_offset, accessor_offset, view_length)
        )
        or stride < element_bytes
    ):
        raise ActionRebindError(f"{owner} accessor offsets/stride are invalid")
    required = accessor_offset + (count - 1) * stride + element_bytes
    if required > view_length or view_offset + required > len(document.binary):
        raise ActionRebindError(f"{owner} accessor escapes the GLB BIN chunk")
    result = np.empty((count, width), dtype=dtype)
    first = view_offset + accessor_offset
    for row in range(count):
        result[row] = np.frombuffer(
            document.binary,
            dtype=dtype,
            count=width,
            offset=first + row * stride,
        )
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(result)):
        raise ActionRebindError(f"{owner} accessor contains non-finite values")
    return result


def _accessor_contract(document: GlbDocument, index: Any, *, owner: str) -> tuple:
    accessors = document.json.get("accessors")
    if (
        not isinstance(accessors, list)
        or isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(accessors)
        or not isinstance(accessors[index], Mapping)
    ):
        raise ActionRebindError(f"{owner} accessor metadata is invalid")
    accessor = accessors[index]
    return (
        accessor.get("componentType"),
        accessor.get("type"),
        accessor.get("count"),
        accessor.get("normalized", False),
        "sparse" in accessor,
    )


def _skinned_primitive(
    document: GlbDocument, *, owner: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    root = document.json
    nodes = root.get("nodes")
    meshes = root.get("meshes")
    if not isinstance(nodes, list) or not isinstance(meshes, list):
        raise ActionRebindError(f"{owner} lacks nodes/meshes")
    candidates = [
        node
        for node in nodes
        if isinstance(node, Mapping) and "skin" in node and "mesh" in node
    ]
    if len(candidates) != 1 or candidates[0].get("skin") != 0:
        raise ActionRebindError(
            f"{owner} must contain exactly one mesh node bound to skin 0"
        )
    mesh_index = candidates[0].get("mesh")
    if (
        isinstance(mesh_index, bool)
        or not isinstance(mesh_index, int)
        or not 0 <= mesh_index < len(meshes)
        or not isinstance(meshes[mesh_index], Mapping)
    ):
        raise ActionRebindError(f"{owner} skinned mesh index is invalid")
    primitives = meshes[mesh_index].get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise ActionRebindError(f"{owner} must contain one skinned primitive")
    primitive = primitives[0]
    attributes = primitive.get("attributes") if isinstance(primitive, Mapping) else None
    required = {"POSITION", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    if (
        not isinstance(primitive, Mapping)
        or "targets" in primitive
        or primitive.get("mode", 4) != 4
        or not isinstance(attributes, Mapping)
        or not required.issubset(attributes)
        or {"JOINTS_1", "WEIGHTS_1"} & set(attributes)
        or not isinstance(primitive.get("indices"), int)
    ):
        raise ActionRebindError(f"{owner} skinned primitive contract is invalid")
    return primitive, attributes


def _maximum_abs(left: np.ndarray, right: np.ndarray, *, owner: str) -> float:
    if left.shape != right.shape:
        raise ActionRebindError(f"{owner} shape changed")
    if not left.size:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _quaternion_error(left: np.ndarray, right: np.ndarray, *, owner: str) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ActionRebindError(f"{owner} quaternion shape changed")
    direct = np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), axis=1)
    negated = np.max(np.abs(left.astype(np.float64) + right.astype(np.float64)), axis=1)
    return float(np.max(np.minimum(direct, negated))) if len(left) else 0.0


def verify_appearance_glb_compatibility(
    source_visual: str | Path,
    output_visual: str | Path,
    *,
    requested_size_scale: float,
    source_payload: bytes | None = None,
    output_payload: bytes | None = None,
) -> dict[str, Any]:
    """Independently measure the immutable mesh, rig and action contracts."""

    if (
        isinstance(requested_size_scale, bool)
        or not isinstance(requested_size_scale, (int, float))
        or not math.isfinite(float(requested_size_scale))
        or float(requested_size_scale) <= 0.0
    ):
        raise ActionRebindError("requested appearance size scale is invalid")
    size_scale = float(requested_size_scale)
    if (source_payload is None) != (output_payload is None):
        raise ActionRebindError(
            "appearance compatibility payloads must be supplied as a pair"
        )
    try:
        if source_payload is None:
            source = load_glb(source_visual)
            output = load_glb(output_visual)
        else:
            source = parse_glb(source_payload)
            output = parse_glb(output_payload)
    except (OSError, GlbError) as exc:
        raise ActionRebindError(
            f"appearance source/output GLB is invalid: {exc}"
        ) from exc

    source_primitive, source_attributes = _skinned_primitive(source, owner="source GLB")
    output_primitive, output_attributes = _skinned_primitive(output, owner="output GLB")
    invariants = ("POSITION", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0")
    for attribute in invariants:
        if _accessor_contract(
            source, source_attributes[attribute], owner=f"source {attribute}"
        ) != _accessor_contract(
            output, output_attributes[attribute], owner=f"output {attribute}"
        ):
            raise ActionRebindError(f"appearance {attribute} accessor contract changed")
    if _accessor_contract(
        source, source_primitive["indices"], owner="source indices"
    ) != _accessor_contract(
        output, output_primitive["indices"], owner="output indices"
    ):
        raise ActionRebindError("appearance topology/index accessor contract changed")
    source_indices = _accessor_array(
        source, source_primitive["indices"], owner="source indices"
    )
    output_indices = _accessor_array(
        output, output_primitive["indices"], owner="output indices"
    )
    if not np.array_equal(source_indices, output_indices):
        raise ActionRebindError("appearance topology/indices changed")
    source_positions = _accessor_array(
        source, source_attributes["POSITION"], owner="source POSITION"
    )
    output_positions = _accessor_array(
        output, output_attributes["POSITION"], owner="output POSITION"
    )
    if source_positions.shape != output_positions.shape:
        raise ActionRebindError("appearance topology vertex count changed")
    uv_error = _maximum_abs(
        _accessor_array(source, source_attributes["TEXCOORD_0"], owner="source UV"),
        _accessor_array(output, output_attributes["TEXCOORD_0"], owner="output UV"),
        owner="TEXCOORD_0",
    )
    if uv_error != 0.0:
        raise ActionRebindError("appearance TEXCOORD_0 changed")
    source_joints = _accessor_array(
        source, source_attributes["JOINTS_0"], owner="source JOINTS_0"
    )
    output_joints = _accessor_array(
        output, output_attributes["JOINTS_0"], owner="output JOINTS_0"
    )
    if not np.array_equal(source_joints, output_joints):
        raise ActionRebindError("appearance JOINTS_0 changed")
    source_weights = _accessor_array(
        source, source_attributes["WEIGHTS_0"], owner="source WEIGHTS_0"
    )
    output_weights = _accessor_array(
        output, output_attributes["WEIGHTS_0"], owner="output WEIGHTS_0"
    )
    if source_weights.dtype != np.dtype("<f4"):
        raise ActionRebindError("appearance WEIGHTS_0 must be FLOAT VEC4")
    weight_sums = np.sum(source_weights, axis=1, keepdims=True, dtype=np.float32)
    if np.any(weight_sums <= 0.0):
        raise ActionRebindError("appearance source WEIGHTS_0 has an unbound vertex")
    # Blender deterministically normalizes each exported binary32 weight row.
    # Compare that canonical result exactly so even a one-ULP output edit fails.
    expected_weights = source_weights / weight_sums
    weight_error = _maximum_abs(
        expected_weights,
        output_weights,
        owner="WEIGHTS_0",
    )
    if not np.array_equal(expected_weights, output_weights):
        raise ActionRebindError("appearance WEIGHTS_0 changed")

    try:
        source_skins = extract_skins(source)
        output_skins = extract_skins(output)
        source_nodes = {
            node.node_index: node for node in extract_node_hierarchy(source)
        }
        output_nodes = {
            node.node_index: node for node in extract_node_hierarchy(output)
        }
    except GlbError as exc:
        raise ActionRebindError(f"appearance skin is invalid: {exc}") from exc
    if len(source_skins) != 1 or len(output_skins) != 1:
        raise ActionRebindError("appearance source/output must contain one skin")
    source_skin = source_skins[0]
    output_skin = output_skins[0]
    source_names = tuple(joint.name for joint in source_skin.joints)
    output_names = tuple(joint.name for joint in output_skin.joints)
    if (
        source_names != output_names
        or any(name is None or not name for name in source_names)
        or len(set(source_names)) != len(source_names)
    ):
        raise ActionRebindError("appearance skin joint order/names changed")

    source_joint_ordinals = {
        joint.node_index: joint.joint_ordinal for joint in source_skin.joints
    }
    output_joint_ordinals = {
        joint.node_index: joint.joint_ordinal for joint in output_skin.joints
    }

    def node_name(nodes: Mapping[int, Any], index: int | None) -> str | None:
        return nodes[index].name if index is not None and index in nodes else None

    if node_name(source_nodes, source_skin.skeleton_node_index) != node_name(
        output_nodes, output_skin.skeleton_node_index
    ):
        raise ActionRebindError("appearance skin skeleton root changed")
    translation_error = 0.0
    rotation_error = 0.0
    scale_error = 0.0
    for source_joint, output_joint in zip(
        source_skin.joints, output_skin.joints, strict=True
    ):
        source_parent_ordinal = source_joint_ordinals.get(
            source_joint.parent_joint_node_index
        )
        output_parent_ordinal = output_joint_ordinals.get(
            output_joint.parent_joint_node_index
        )
        if source_parent_ordinal != output_parent_ordinal or node_name(
            source_nodes, source_joint.parent_node_index
        ) != node_name(output_nodes, output_joint.parent_node_index):
            raise ActionRebindError(
                f"appearance joint parent changed for {source_joint.name}"
            )
        translation_error = max(
            translation_error,
            _maximum_abs(
                np.asarray(source_joint.local_trs.translation)[None, :] * size_scale,
                np.asarray(output_joint.local_trs.translation)[None, :],
                owner=f"joint {source_joint.name} rest translation",
            ),
        )
        rotation_error = max(
            rotation_error,
            _quaternion_error(
                np.asarray(source_joint.local_trs.rotation_xyzw)[None, :],
                np.asarray(output_joint.local_trs.rotation_xyzw)[None, :],
                owner=f"joint {source_joint.name} rest rotation",
            ),
        )
        scale_error = max(
            scale_error,
            _maximum_abs(
                np.asarray(source_joint.local_trs.scale)[None, :],
                np.asarray(output_joint.local_trs.scale)[None, :],
                owner=f"joint {source_joint.name} rest scale",
            ),
        )
    if (
        source_skin.inverse_bind_matrices is None
        or output_skin.inverse_bind_matrices is None
    ):
        raise ActionRebindError("appearance source/output skins require inverse binds")
    expected_ibm = np.asarray(source_skin.inverse_bind_matrices, dtype=np.float64)
    expected_ibm = expected_ibm.copy()
    expected_ibm[:, 12:15] *= size_scale
    ibm_error = _maximum_abs(
        expected_ibm,
        np.asarray(output_skin.inverse_bind_matrices, dtype=np.float64),
        owner="inverse bind matrices",
    )
    skin_maximum = max(translation_error, rotation_error, scale_error, ibm_error)
    if skin_maximum > _APPEARANCE_FLOAT_TOLERANCE:
        raise ActionRebindError(
            "appearance skin rest/IBM differs from requested size scale"
        )

    try:
        source_actions = {action.name: action for action in extract_actions(source)}
        output_actions = {action.name: action for action in extract_actions(output)}
    except GlbError as exc:
        raise ActionRebindError(f"appearance actions are invalid: {exc}") from exc
    if set(source_actions) != {"Idle", "Walking"} or set(output_actions) != {
        "Idle",
        "Walking",
    }:
        raise ActionRebindError(
            "appearance source/output must contain exactly Idle and Walking"
        )
    action_maxima = {
        "timestamps": 0.0,
        "translation": 0.0,
        "rotation": 0.0,
        "scale": 0.0,
    }
    for action_name in ("Idle", "Walking"):

        def channels(action: Any) -> dict[tuple[str, str], Any]:
            result: dict[tuple[str, str], Any] = {}
            for channel in action.channels:
                if not isinstance(channel.target_node_name, str) or not (
                    channel.target_node_name
                ):
                    raise ActionRebindError(
                        f"appearance {action_name} has an unnamed target"
                    )
                key = (channel.target_node_name, channel.target_path)
                if key in result:
                    raise ActionRebindError(
                        f"appearance {action_name} has duplicate target {key}"
                    )
                result[key] = channel
            return result

        source_channels = channels(source_actions[action_name])
        output_channels = channels(output_actions[action_name])
        if set(source_channels) != set(output_channels):
            raise ActionRebindError(
                f"appearance {action_name} channel target/path set changed"
            )
        for key in source_channels:
            source_channel = source_channels[key]
            output_channel = output_channels[key]
            if source_channel.interpolation != output_channel.interpolation:
                raise ActionRebindError(
                    f"appearance {action_name} {key} interpolation changed"
                )
            action_maxima["timestamps"] = max(
                action_maxima["timestamps"],
                _maximum_abs(
                    np.asarray(source_channel.timestamps_seconds)[:, None],
                    np.asarray(output_channel.timestamps_seconds)[:, None],
                    owner=f"{action_name} {key} timestamps",
                ),
            )
            source_values = np.asarray(source_channel.values, dtype=np.float64)
            output_values = np.asarray(output_channel.values, dtype=np.float64)
            if key[1] == "translation":
                error = _maximum_abs(
                    source_values * size_scale,
                    output_values,
                    owner=f"{action_name} {key} translation",
                )
            elif key[1] == "rotation":
                error = _quaternion_error(
                    source_values,
                    output_values,
                    owner=f"{action_name} {key} rotation",
                )
            elif key[1] == "scale":
                error = _maximum_abs(
                    source_values,
                    output_values,
                    owner=f"{action_name} {key} scale",
                )
            else:  # pragma: no cover - extract_actions already constrains paths
                raise ActionRebindError(f"unsupported action path {key[1]}")
            action_maxima[key[1]] = max(action_maxima[key[1]], error)
    if max(action_maxima.values()) > _APPEARANCE_FLOAT_TOLERANCE:
        raise ActionRebindError(
            "appearance action timestamps/values differ from requested size scale"
        )
    return {
        "requested_size_scale": size_scale,
        "mesh": {
            "index_count": int(output_indices.size),
            "vertex_count": int(output_positions.shape[0]),
            "maximum_texcoord_0_error": uv_error,
            "maximum_weights_0_error": weight_error,
        },
        "skin": {
            "joint_count": len(output_names),
            "maximum_scaled_rest_translation_error_m": translation_error,
            "maximum_rest_rotation_error": rotation_error,
            "maximum_rest_scale_error": scale_error,
            "maximum_scaled_inverse_bind_matrix_error": ibm_error,
        },
        "actions": {
            "names": ["Idle", "Walking"],
            "maximum_errors": action_maxima,
        },
        "tolerance": _APPEARANCE_FLOAT_TOLERANCE,
    }


def _strict_appearance_chain(
    *,
    appearance_report_path: str | Path,
    source_visual: Path,
    target_visual: Path,
    rebase_report_path: str | Path,
    deformation_report_path: str | Path,
) -> tuple[Path, Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    appearance_path = _regular_file(
        appearance_report_path, owner="appearance report", suffix=".json"
    )
    rebase_path = _regular_file(
        rebase_report_path, owner="target rebase report", suffix=".json"
    )
    deformation_path = _regular_file(
        deformation_report_path,
        owner="target rebase deformation report",
        suffix=".json",
    )
    appearance = load_json(appearance_path)
    if (
        appearance.get("schema") != APPEARANCE_REPORT_SCHEMA
        or appearance.get("status") != "pass"
        or appearance.get("qualification_claim") is not False
        or appearance.get("formal_dataset_registration_authorized") is not False
    ):
        raise ActionRebindError("appearance report is not a non-qualifying pass")
    _require_reference(
        appearance.get("source"),
        owner="appearance report source",
        path=source_visual,
        sha256=sha256_file(source_visual),
        byte_size=source_visual.stat().st_size,
    )
    output = appearance.get("output")
    appearance_glb = output.get("glb") if isinstance(output, Mapping) else None
    if not isinstance(appearance_glb, Mapping):
        raise ActionRebindError("appearance report lacks its output GLB")
    raw_appearance_visual = appearance_glb.get("path")
    if not isinstance(raw_appearance_visual, str):
        raise ActionRebindError("appearance output path must be a string")
    pre_rebase = _regular_file(
        raw_appearance_visual,
        owner="appearance output GLB",
        suffix=".glb",
    )
    _require_reference(
        appearance_glb,
        owner="appearance output GLB",
        path=pre_rebase,
        sha256=sha256_file(pre_rebase),
        byte_size=pre_rebase.stat().st_size,
    )
    tool = appearance.get("tool_identity")
    if not isinstance(tool, Mapping) or not isinstance(tool.get("path"), str):
        raise ActionRebindError("appearance report lacks tool identity")
    tool_path = _regular_file(tool["path"], owner="appearance tool", suffix=".py")
    if tool.get("sha256") != sha256_file(tool_path):
        raise ActionRebindError("appearance tool bytes changed")
    compatibility = verify_appearance_glb_compatibility(
        source_visual,
        pre_rebase,
        requested_size_scale=_requested_size_scale(appearance),
    )

    rebase = load_json(rebase_path)
    if (
        rebase.get("schema") != REBASE_REPORT_SCHEMA
        or rebase.get("status") != "pass"
        or rebase.get("qualification_claim") is not False
    ):
        raise ActionRebindError("target rebase report is not a research pass")
    _require_reference(
        rebase.get("source"),
        owner="target rebase source",
        path=pre_rebase,
        sha256=sha256_file(pre_rebase),
        byte_size=pre_rebase.stat().st_size,
    )
    _require_reference(
        rebase.get("output"),
        owner="target rebase output",
        path=target_visual,
        sha256=sha256_file(target_visual),
        byte_size=target_visual.stat().st_size,
    )

    deformation = load_json(deformation_path)
    maximum = deformation.get("maximum_vertex_error_m")
    threshold = deformation.get("threshold_maximum_vertex_error_m")
    if (
        deformation.get("schema") != DEFORMATION_REPORT_SCHEMA
        or deformation.get("status") != "pass"
        or deformation.get("qualification_claim") is not False
        or not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(float(maximum))
        or not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or float(maximum) < 0.0
        or float(maximum) > float(threshold)
    ):
        raise ActionRebindError("target rebase deformation verification did not pass")
    _require_reference(
        deformation.get("source"),
        owner="deformation report source",
        path=pre_rebase,
        sha256=sha256_file(pre_rebase),
        byte_size=pre_rebase.stat().st_size,
        require_byte_size=False,
    )
    _require_reference(
        deformation.get("rebased"),
        owner="deformation report target",
        path=target_visual,
        sha256=sha256_file(target_visual),
        byte_size=target_visual.stat().st_size,
        require_byte_size=False,
    )
    rebase_binding = deformation.get("rebase_report")
    _require_reference(
        rebase_binding,
        owner="deformation report rebase binding",
        path=rebase_path,
        sha256=sha256_file(rebase_path),
        byte_size=rebase_path.stat().st_size,
        require_byte_size=False,
    )
    return (
        appearance_path,
        rebase_path,
        deformation_path,
        appearance,
        compatibility,
    )


def _action_records(actions: BakedActionSet) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for clip in actions.actions:
        array = np.asarray(clip.rotations_xyzw, dtype=np.float64)
        norms = np.linalg.norm(array, axis=-1)
        records.append(
            {
                "semantic_action_id": clip.semantic_action_id,
                "source_action_name": clip.source_action_name,
                "clip_start_seconds": clip.clip_start_seconds,
                "clip_end_seconds": clip.clip_end_seconds,
                "sample_count": clip.sample_count,
                "first_sample_tick": clip.sample_ticks[0],
                "last_sample_tick": clip.sample_ticks[-1],
                "loop_duration_ticks": clip.loop_duration_ticks,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.all(np.isfinite(array))),
                "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
            }
        )
    return records


def rebind_compatible_action_set(
    source: BakedActionSet,
    *,
    target_glb_sha256: str,
    target_runtime_joint_order: Sequence[str],
) -> BakedActionSet:
    """Return the exact source samples bound to one identical target rig order."""

    target_order = tuple(target_runtime_joint_order)
    if target_order != source.runtime_joint_order:
        raise ActionRebindError(
            "source action joint order differs from target runtime joint order"
        )
    return BakedActionSet(
        source_glb_sha256=target_glb_sha256,
        runtime_joint_order=source.runtime_joint_order,
        actions=source.actions,
        sample_rate_hz=source.sample_rate_hz,
        time_base_hz=source.time_base_hz,
    )


def build_action_rebind(
    *,
    source_package_manifest: str | Path,
    appearance_report: str | Path,
    target_visual_glb: str | Path,
    target_rebase_report: str | Path,
    target_rebase_deformation_report: str | Path,
) -> ActionRebindResult:
    """Validate the derivative chain and return canonical rebound action bytes."""

    package_path, source_visual, source_actions_path, _manifest = _load_source_package(
        source_package_manifest
    )
    target_visual = _regular_file(
        target_visual_glb, owner="target visual GLB", suffix=".glb"
    )
    appearance_path, rebase_path, deformation_path, appearance, compatibility = (
        _strict_appearance_chain(
            appearance_report_path=appearance_report,
            source_visual=source_visual,
            target_visual=target_visual,
            rebase_report_path=target_rebase_report,
            deformation_report_path=target_rebase_deformation_report,
        )
    )
    try:
        source_actions = read_baked_actions_npz(source_actions_path)
    except (OSError, ActionBakeError) as exc:
        raise ActionRebindError(f"source baked actions are invalid: {exc}") from exc
    if source_actions.source_glb_sha256 != sha256_file(source_visual):
        raise ActionRebindError("source baked actions do not bind source visual GLB")
    if baked_actions_content_sha256(source_actions) != sha256_file(source_actions_path):
        raise ActionRebindError("source baked action content differs from file bytes")

    try:
        target_document = load_glb(target_visual)
        target_mapping = build_habitat_asset_mapping_from_rebase_report(
            target_document,
            load_json(rebase_path),
        )
    except (OSError, GlbError, HabitatMappingError, ValueError) as exc:
        raise ActionRebindError(f"target visual/mapping is invalid: {exc}") from exc
    if target_document.sha256 != sha256_file(target_visual):
        raise ActionRebindError("parsed target visual hash differs from file bytes")
    rebound = rebind_compatible_action_set(
        source_actions,
        target_glb_sha256=target_document.sha256,
        target_runtime_joint_order=target_mapping.runtime_joint_order,
    )
    with tempfile.TemporaryDirectory(prefix="avengine-action-rebind-") as directory:
        temporary = Path(directory) / "actions.npz"
        artifact_sha256 = write_baked_actions_npz(rebound, temporary)
        artifact_bytes = temporary.read_bytes()
    if artifact_sha256 != hashlib.sha256(artifact_bytes).hexdigest():
        raise ActionRebindError("temporary action artifact hash changed")
    if artifact_sha256 != baked_actions_content_sha256(rebound):
        raise ActionRebindError("rebound action canonical hash differs from bytes")

    artifact_record = {
        "path": None,
        "sha256": artifact_sha256,
        "byte_size": len(artifact_bytes),
        "canonical_content_sha256": artifact_sha256,
        "readback_equal": True,
    }
    report: dict[str, Any] = {
        "schema": ACTION_REPORT_SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_glb": _record(target_visual),
        "artifact": artifact_record,
        "clock": {
            "sample_rate_hz": rebound.sample_rate_hz,
            "time_base_hz": rebound.time_base_hz,
        },
        "runtime_joint_order": list(rebound.runtime_joint_order),
        "actions": _action_records(rebound),
        "derivation": {
            "schema": DERIVATION_SCHEMA,
            "method": "reuse_source_package_rotations_on_appearance_compatible_rig",
            "tool_identity": _record(Path(__file__).resolve()),
            "source_package_manifest": _record(package_path),
            "source_visual_glb": _record(source_visual),
            "source_baked_actions": _record(source_actions_path),
            "appearance_report": _record(appearance_path),
            "appearance_request_sha256": appearance.get("instance_request", {}).get(
                "request_sha256"
            ),
            "target_rebase_report": _record(rebase_path),
            "target_rebase_deformation_report": _record(deformation_path),
            "appearance_glb_compatibility": compatibility,
            "compatibility_gates": {
                "source_package_valid": True,
                "appearance_source_is_package_visual": True,
                "appearance_joint_order_unchanged": True,
                "appearance_action_targets_unchanged": True,
                "target_rebase_deformation_pass": True,
                "runtime_joint_order_exact": True,
                "rotation_samples_byte_equivalent_to_source": True,
            },
            "notes": [
                "The embedded GLB clips are not re-baked because the admitted M2 motion is the package NPZ.",
                "Only source_glb_sha256 changes; clip clocks, rotations, and joint order remain exact.",
                "This derivative still requires independent Habitat and human visual review.",
            ],
        },
    }
    return ActionRebindResult(
        actions=rebound,
        artifact_bytes=artifact_bytes,
        report=report,
    )


def _output_path(path: str | Path, *, owner: str, suffix: str) -> Path:
    output = _absolute_without_symlinks(path, owner=owner)
    if output.suffix.lower() != suffix:
        raise ActionRebindError(f"{owner} must use the {suffix} suffix")
    if output.exists() or output.is_symlink():
        raise ActionRebindError(f"refusing to replace {owner}: {output}")
    return output


def write_action_rebind(
    result: ActionRebindResult,
    *,
    output_npz: str | Path,
    report_output: str | Path,
) -> tuple[Path, Path]:
    """Exclusively emit both artifact and report, cleaning partial writes."""

    if not isinstance(result, ActionRebindResult):
        raise ActionRebindError("result must come from build_action_rebind")
    artifact_path = _output_path(
        output_npz, owner="rebound action artifact", suffix=".npz"
    )
    report_path = _output_path(
        report_output, owner="rebound action report", suffix=".json"
    )
    if artifact_path == report_path:
        raise ActionRebindError("artifact and report outputs must differ")
    report = dict(result.report)
    artifact = dict(report["artifact"])
    artifact["path"] = str(artifact_path)
    report["artifact"] = artifact
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    outputs = (
        (artifact_path, result.artifact_bytes),
        (report_path, report_bytes),
    )
    streams: list[tuple[Path, Any]] = []
    try:
        for path, _payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        for path, _payload in outputs:
            streams.append((path, path.open("xb")))
        for (path, stream), (_expected_path, payload) in zip(
            streams, outputs, strict=True
        ):
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            if path.stat().st_size != len(payload):
                raise OSError(f"short write: {path}")
    except OSError as exc:
        for _path, stream in streams:
            stream.close()
        cleanup_errors: list[str] = []
        for path, _stream in streams:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{path}: {cleanup_exc}")
        suffix = f"; cleanup failed: {cleanup_errors}" if cleanup_errors else ""
        raise ActionRebindError(
            f"unable to emit action rebind pair: {exc}{suffix}"
        ) from exc
    finally:
        for _path, stream in streams:
            if not stream.closed:
                stream.close()

    try:
        readback = read_baked_actions_npz(artifact_path)
        if readback != result.actions:
            raise ActionRebindError("rebound action readback differs")
        emitted_report = load_json(report_path)
        if emitted_report != report:
            raise ActionRebindError("rebound action report readback differs")
        if (
            sha256_file(artifact_path) != report["artifact"]["sha256"]
            or artifact_path.stat().st_size != report["artifact"]["byte_size"]
        ):
            raise ActionRebindError("emitted report does not bind action bytes")
    except ActionRebindError:
        artifact_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, ActionBakeError) as exc:
        artifact_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise ActionRebindError(f"action rebind readback failed: {exc}") from exc
    return artifact_path, report_path


__all__ = [
    "ACTION_REPORT_SCHEMA",
    "DERIVATION_SCHEMA",
    "ActionRebindError",
    "ActionRebindResult",
    "build_action_rebind",
    "rebind_compatible_action_set",
    "verify_appearance_glb_compatibility",
    "write_action_rebind",
]
