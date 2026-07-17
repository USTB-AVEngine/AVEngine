"""Fail-closed GLB action selection and unweighted skin-root pruning.

Some otherwise useful animal GLBs export deformation bones and disconnected
IK/controller branches in the same skin.  The controllers are meaningful in
the authoring package, but glTF has no live constraint graph: once deformation
has been baked into joint channels, a controller branch with exactly zero mesh
weight has no runtime deformation effect.  This compiler performs two bounded,
body-plan-neutral operations before the M2 root rebase:

* select actions by exact source name and give them explicit output names;
* when a skin has multiple joint roots, retain the only root branch with a
  non-zero mesh influence and remove only root branches proven to have exactly
  zero weight.

The proof covers every ``JOINTS_0``/``WEIGHTS_0`` pair used by the skin.  A
removed ordinal that is present only in a zero-weight slot is remapped to the
retained root so every emitted ordinal remains valid.  Any second weighted root
fails closed.  Nodes are deliberately not renumbered or erased from the scene
graph; they are removed from the skin and selected animation channels, avoiding
unrelated node-index changes while making the emitted skin a single tree.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m2.glb import (
    AnimationAction,
    AnimationChannel,
    GlbDocument,
    GlbError,
    SkinRecord,
    decode_accessor,
    extract_actions,
    extract_node_hierarchy,
    extract_skins,
    load_glb,
    parse_glb,
)
from avengine.m2.glb_write import build_glb


_FLOAT = 5126
_UNSIGNED_BYTE = 5121
_UNSIGNED_SHORT = 5123
_DEFAULT_CHANNEL_TOLERANCE = 5.0e-5
_WEIGHT_SUM_TOLERANCE = 1.0e-5


class GlbPreprocessError(ValueError):
    """The GLB cannot be transformed within the bounded safety proof."""


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create one output without following/replacing an existing path."""

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


@dataclass(frozen=True)
class _PrimitiveInfluence:
    mesh_index: int
    primitive_index: int
    joint_accessor: int
    weight_accessor: int
    joint_component_type: int
    joints: np.ndarray
    weights: np.ndarray


def _objects(value: Any, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise GlbPreprocessError(f"{owner} must be an array of objects")
    return value


def _integer(value: Any, owner: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GlbPreprocessError(f"{owner} must be an integer >= {minimum}")
    return value


def _action_pairs(
    action_map: Mapping[str, str] | Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    pairs = (
        list(action_map.items())
        if isinstance(action_map, Mapping)
        else list(action_map)
    )
    if not pairs:
        raise GlbPreprocessError("at least one exact action mapping is required")
    sources: set[str] = set()
    targets: set[str] = set()
    result: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise GlbPreprocessError(
                f"action_map[{index}] must be a (source, target) pair"
            )
        source, target = pair
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(target, str)
            or not target.strip()
        ):
            raise GlbPreprocessError(
                "action source and target names must be non-empty strings"
            )
        if source in sources:
            raise GlbPreprocessError(
                f"action source is mapped more than once: {source!r}"
            )
        if target in targets:
            raise GlbPreprocessError(f"action target is ambiguous: {target!r}")
        sources.add(source)
        targets.add(target)
        result.append((source, target))
    return result


def _select_actions(
    source: GlbDocument,
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[dict[str, Any]], tuple[AnimationAction, ...], list[str]]:
    document = source.json
    animations = _objects(document.get("animations", []), "animations")
    by_name: dict[str, dict[str, Any]] = {}
    source_names: list[str] = []
    for index, animation in enumerate(animations):
        name = animation.get("name")
        if not isinstance(name, str) or not name.strip():
            raise GlbPreprocessError(f"animations[{index}] must have a non-empty name")
        if name in by_name:
            raise GlbPreprocessError(f"source animation name is ambiguous: {name!r}")
        by_name[name] = animation
        source_names.append(name)
    missing = [source_name for source_name, _ in pairs if source_name not in by_name]
    if missing:
        raise GlbPreprocessError(
            "exact source animation(s) not found: "
            + ", ".join(repr(name) for name in missing)
        )

    selected_source = [copy.deepcopy(by_name[source_name]) for source_name, _ in pairs]
    inspection_json = copy.deepcopy(document)
    inspection_json["animations"] = copy.deepcopy(selected_source)
    inspection = GlbDocument(
        json=inspection_json,
        binary=source.binary,
        sha256=source.sha256,
        byte_length=source.byte_length,
        source_path=source.source_path,
    )
    try:
        actions = extract_actions(inspection)
    except GlbError as exc:
        raise GlbPreprocessError(f"selected action validation failed: {exc}") from exc
    dropped = [
        name for name in source_names if name not in {source for source, _ in pairs}
    ]
    return selected_source, actions, dropped


def _decode_joint_accessor(
    source: GlbDocument, accessor_index: int
) -> tuple[np.ndarray, int]:
    document = source.json
    accessors = _objects(document.get("accessors", []), "accessors")
    views = _objects(document.get("bufferViews", []), "bufferViews")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise GlbPreprocessError(f"JOINTS_0 accessor is out of range: {accessor_index}")
    accessor = accessors[accessor_index]
    if accessor.get("type") != "VEC4" or accessor.get("normalized", False):
        raise GlbPreprocessError("JOINTS_0 must be a non-normalized VEC4")
    if "sparse" in accessor:
        raise GlbPreprocessError("sparse JOINTS_0 is outside the preprocess boundary")
    component_type = accessor.get("componentType")
    component_format = {_UNSIGNED_BYTE: "B", _UNSIGNED_SHORT: "H"}.get(component_type)
    if component_format is None:
        raise GlbPreprocessError("JOINTS_0 must use UNSIGNED_BYTE or UNSIGNED_SHORT")
    count = _integer(
        accessor.get("count"), f"accessors[{accessor_index}].count", minimum=1
    )
    view_index = _integer(
        accessor.get("bufferView"), f"accessors[{accessor_index}].bufferView"
    )
    if view_index >= len(views):
        raise GlbPreprocessError("JOINTS_0 bufferView is out of range")
    view = views[view_index]
    if view.get("buffer") != 0:
        raise GlbPreprocessError("JOINTS_0 must reference embedded buffer 0")
    view_offset = _integer(
        view.get("byteOffset", 0), f"bufferViews[{view_index}].byteOffset"
    )
    view_length = _integer(
        view.get("byteLength"), f"bufferViews[{view_index}].byteLength", minimum=1
    )
    accessor_offset = _integer(
        accessor.get("byteOffset", 0), f"accessors[{accessor_index}].byteOffset"
    )
    packer = struct.Struct("<" + component_format * 4)
    stride = _integer(
        view.get("byteStride", packer.size),
        f"bufferViews[{view_index}].byteStride",
        minimum=packer.size,
    )
    if stride % (packer.size // 4):
        raise GlbPreprocessError("JOINTS_0 byteStride is not component aligned")
    required = accessor_offset + (count - 1) * stride + packer.size
    if required > view_length or view_offset + required > len(source.binary):
        raise GlbPreprocessError("JOINTS_0 extends beyond its bufferView")
    values = np.empty((count, 4), dtype=np.int64)
    start = view_offset + accessor_offset
    for item_index in range(count):
        values[item_index] = packer.unpack_from(
            source.binary, start + item_index * stride
        )
    return values, int(component_type)


def _skin_primitives(
    source: GlbDocument, skin: SkinRecord
) -> list[_PrimitiveInfluence]:
    document = source.json
    nodes = _objects(document.get("nodes", []), "nodes")
    meshes = _objects(document.get("meshes", []), "meshes")
    mesh_indices: set[int] = set()
    for node_index, node in enumerate(nodes):
        if node.get("skin") != skin.skin_index:
            continue
        mesh_index = _integer(node.get("mesh"), f"nodes[{node_index}].mesh")
        if mesh_index >= len(meshes):
            raise GlbPreprocessError(f"nodes[{node_index}].mesh is out of range")
        mesh_indices.add(mesh_index)
    if not mesh_indices:
        raise GlbPreprocessError("skin is not bound to any mesh node")

    result: list[_PrimitiveInfluence] = []
    for mesh_index in sorted(mesh_indices):
        primitives = _objects(
            meshes[mesh_index].get("primitives", []), f"meshes[{mesh_index}].primitives"
        )
        for primitive_index, primitive in enumerate(primitives):
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict):
                raise GlbPreprocessError("mesh primitive attributes must be an object")
            extra_sets = sorted(
                name
                for name in attributes
                if (name.startswith("JOINTS_") or name.startswith("WEIGHTS_"))
                and name not in {"JOINTS_0", "WEIGHTS_0"}
            )
            if extra_sets:
                raise GlbPreprocessError(
                    "additional skin influence sets are outside the preprocess boundary: "
                    + ", ".join(extra_sets)
                )
            if "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
                raise GlbPreprocessError(
                    "every skinned primitive must contain JOINTS_0 and WEIGHTS_0"
                )
            joint_accessor = _integer(attributes["JOINTS_0"], "JOINTS_0 accessor")
            weight_accessor = _integer(attributes["WEIGHTS_0"], "WEIGHTS_0 accessor")
            joints, component_type = _decode_joint_accessor(source, joint_accessor)
            try:
                decoded_weights = decode_accessor(source, weight_accessor)
            except GlbError as exc:
                raise GlbPreprocessError(f"invalid WEIGHTS_0 accessor: {exc}") from exc
            if decoded_weights.element_type != "VEC4":
                raise GlbPreprocessError("WEIGHTS_0 must be FLOAT VEC4")
            weights = np.asarray(decoded_weights.values, dtype=np.float64)
            if joints.shape != weights.shape:
                raise GlbPreprocessError("JOINTS_0 and WEIGHTS_0 counts must match")
            if int(np.max(joints)) >= len(skin.joints):
                raise GlbPreprocessError(
                    "JOINTS_0 references an ordinal outside the skin"
                )
            minimum_weight = float(np.min(weights))
            maximum_sum_error = float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
            if minimum_weight < 0.0 or maximum_sum_error > _WEIGHT_SUM_TOLERANCE:
                raise GlbPreprocessError(
                    "WEIGHTS_0 must be non-negative and sum to one: "
                    f"minimum={minimum_weight:.9g}, maximum_sum_error={maximum_sum_error:.9g}"
                )
            result.append(
                _PrimitiveInfluence(
                    mesh_index=mesh_index,
                    primitive_index=primitive_index,
                    joint_accessor=joint_accessor,
                    weight_accessor=weight_accessor,
                    joint_component_type=component_type,
                    joints=joints,
                    weights=weights,
                )
            )
    return result


def _joint_weight_proof(
    skin: SkinRecord, primitives: Sequence[_PrimitiveInfluence]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = np.zeros(len(skin.joints), dtype=np.float64)
    nonzero_slots = np.zeros(len(skin.joints), dtype=np.int64)
    zero_slots = np.zeros(len(skin.joints), dtype=np.int64)
    for primitive in primitives:
        for ordinal in range(len(skin.joints)):
            mask = primitive.joints == ordinal
            if not np.any(mask):
                continue
            values = primitive.weights[mask]
            maximum[ordinal] = max(maximum[ordinal], float(np.max(values)))
            nonzero_slots[ordinal] += int(np.count_nonzero(values != 0.0))
            zero_slots[ordinal] += int(np.count_nonzero(values == 0.0))
    return maximum, nonzero_slots, zero_slots


def _root_branches(skin: SkinRecord) -> list[tuple[int, tuple[int, ...]]]:
    ordinal_by_node = {joint.node_index: joint.joint_ordinal for joint in skin.joints}
    children: dict[int, list[int]] = {joint.joint_ordinal: [] for joint in skin.joints}
    roots: list[int] = []
    for joint in skin.joints:
        parent_node = joint.parent_joint_node_index
        if parent_node is None:
            roots.append(joint.joint_ordinal)
        else:
            children[ordinal_by_node[parent_node]].append(joint.joint_ordinal)

    result: list[tuple[int, tuple[int, ...]]] = []
    for root in roots:
        branch: list[int] = []
        stack = [root]
        while stack:
            ordinal = stack.pop()
            branch.append(ordinal)
            stack.extend(reversed(children[ordinal]))
        result.append((root, tuple(sorted(branch))))
    return result


def _validate_removed_branch_payload(
    source: GlbDocument,
    skin: SkinRecord,
    removed_ordinals: set[int],
) -> set[int]:
    """Prove stripped controller branches have no separate scene payload."""

    removed_nodes = {skin.joints[ordinal].node_index for ordinal in removed_ordinals}
    nodes = _objects(source.json.get("nodes", []), "nodes")
    for node_index in sorted(removed_nodes):
        node = nodes[node_index]
        payload_keys = [
            key for key in ("mesh", "camera", "skin", "extensions") if key in node
        ]
        if payload_keys:
            raise GlbPreprocessError(
                "zero-weight joint branch carries separate scene payload and cannot "
                f"be stripped safely: node={node_index}, keys={payload_keys}"
            )
        children = node.get("children", [])
        if not isinstance(children, list):
            raise GlbPreprocessError(f"nodes[{node_index}].children must be an array")
        external_children = [child for child in children if child not in removed_nodes]
        if external_children:
            raise GlbPreprocessError(
                "zero-weight joint branch has non-joint descendants and cannot be "
                f"stripped safely: node={node_index}, children={external_children}"
            )
    return removed_nodes


def _append_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    component_type: int,
    element_type: str,
    values: np.ndarray,
) -> int:
    component_formats = {_UNSIGNED_BYTE: "B", _UNSIGNED_SHORT: "H", _FLOAT: "f"}
    component_format = component_formats.get(component_type)
    if component_format is None:
        raise GlbPreprocessError(
            f"unsupported emitted component type: {component_type}"
        )
    component_counts = {"VEC4": 4, "MAT4": 16}
    component_count = component_counts.get(element_type)
    if component_count is None:
        raise GlbPreprocessError(f"unsupported emitted accessor type: {element_type}")
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != component_count or len(array) == 0:
        raise GlbPreprocessError("emitted accessor values have an invalid shape")
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    packer = struct.Struct("<" + component_format * component_count)
    for row in array:
        packer_values = row.tolist()
        binary.extend(packer.pack(*packer_values))
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    if not isinstance(views, list) or not isinstance(accessors, list):
        raise GlbPreprocessError("bufferViews and accessors must be arrays")
    view_index = len(views)
    views.append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
    )
    accessor_index = len(accessors)
    accessors.append(
        {
            "bufferView": view_index,
            "componentType": component_type,
            "count": len(array),
            "type": element_type,
        }
    )
    return accessor_index


def _default_channel_error(
    channel: AnimationChannel,
    default: Sequence[float],
) -> float:
    if channel.interpolation == "CUBICSPLINE":
        raise GlbPreprocessError(
            "CUBICSPLINE on a non-skin node cannot be proven redundant"
        )
    values = np.asarray(channel.values, dtype=np.float64)
    reference = np.asarray(default, dtype=np.float64)
    if channel.target_path != "rotation":
        return float(np.max(np.abs(values - reference)))
    reference /= np.linalg.norm(reference)
    maximum = 0.0
    for value in values:
        norm = float(np.linalg.norm(value))
        if norm < 1.0e-12:
            return float("inf")
        normalized = value / norm
        maximum = max(
            maximum,
            min(
                float(np.max(np.abs(normalized - reference))),
                float(np.max(np.abs(normalized + reference))),
            ),
        )
    return maximum


def _filter_action_channels(
    selected_animations: Sequence[dict[str, Any]],
    selected_actions: Sequence[AnimationAction],
    pairs: Sequence[tuple[str, str]],
    *,
    retained_joint_nodes: set[int],
    removed_joint_nodes: set[int],
    source: GlbDocument,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hierarchy = {node.node_index: node for node in extract_node_hierarchy(source)}
    output: list[dict[str, Any]] = []
    removed_reports: list[dict[str, Any]] = []
    for raw_animation, action, (source_name, target_name) in zip(
        selected_animations, selected_actions, pairs, strict=True
    ):
        raw_samplers = _objects(
            raw_animation.get("samplers"), f"animation {source_name!r}.samplers"
        )
        raw_channels = _objects(
            raw_animation.get("channels"), f"animation {source_name!r}.channels"
        )
        channel_records = {
            channel.channel_index: channel for channel in action.channels
        }
        retained_channels: list[dict[str, Any]] = []
        sampler_remap: dict[int, int] = {}
        emitted_samplers: list[dict[str, Any]] = []
        for channel_index, raw_channel in enumerate(raw_channels):
            channel = channel_records[channel_index]
            node_index = channel.target_node_index
            reason: str | None = None
            error: float | None = None
            if node_index in removed_joint_nodes:
                reason = "zero_weight_removed_joint_branch"
            elif node_index not in retained_joint_nodes:
                node = hierarchy[node_index]
                defaults = {
                    "translation": node.local_trs.translation,
                    "rotation": node.local_trs.rotation_xyzw,
                    "scale": node.local_trs.scale,
                }
                error = _default_channel_error(channel, defaults[channel.target_path])
                if error > _DEFAULT_CHANNEL_TOLERANCE:
                    raise GlbPreprocessError(
                        "selected action targets a non-skin node with non-redundant motion: "
                        f"{source_name}/{channel.target_node_name}/{channel.target_path} "
                        f"error={error:.9g}"
                    )
                reason = "non_skin_channel_constant_at_node_default"

            if reason is not None:
                report = {
                    "source_action": source_name,
                    "output_action": target_name,
                    "node_index": node_index,
                    "node_name": channel.target_node_name,
                    "path": channel.target_path,
                    "reason": reason,
                }
                if error is not None:
                    report["maximum_default_error"] = error
                removed_reports.append(report)
                continue

            old_sampler = _integer(
                raw_channel.get("sampler"), "animation channel sampler"
            )
            if old_sampler >= len(raw_samplers):
                raise GlbPreprocessError("animation channel sampler is out of range")
            if old_sampler not in sampler_remap:
                sampler_remap[old_sampler] = len(emitted_samplers)
                emitted_samplers.append(copy.deepcopy(raw_samplers[old_sampler]))
            emitted_channel = copy.deepcopy(raw_channel)
            emitted_channel["sampler"] = sampler_remap[old_sampler]
            retained_channels.append(emitted_channel)
        if not retained_channels:
            raise GlbPreprocessError(
                f"selected action {source_name!r} has no retained deformation channels"
            )
        emitted = copy.deepcopy(raw_animation)
        emitted["name"] = target_name
        emitted["samplers"] = emitted_samplers
        emitted["channels"] = retained_channels
        output.append(emitted)
    return output, removed_reports


def preprocess_glb(
    source_path: str | Path,
    output_path: str | Path,
    *,
    action_map: Mapping[str, str] | Sequence[tuple[str, str]],
    prune_zero_weight_leaves: bool = False,
) -> dict[str, Any]:
    """Select actions and prune provably unweighted disconnected skin roots.

    ``action_map`` is exact and ordered.  For example, ``[("Idle", "Idle"),
    ("Walk", "Walking")]`` filters a farm-animal GLB to the two M2 actions
    while giving the locomotion action its canonical name.
    """

    source_resolved = Path(source_path).resolve()
    output_argument = Path(output_path)
    if output_argument.exists() or output_argument.is_symlink():
        raise GlbPreprocessError(f"refusing to replace output: {output_argument}")
    output_resolved = output_argument.resolve()
    if source_resolved == output_resolved:
        raise GlbPreprocessError("output must not overwrite the source GLB")
    pairs = _action_pairs(action_map)
    try:
        source = load_glb(source_resolved)
        skins = extract_skins(source)
    except GlbError as exc:
        raise GlbPreprocessError(f"invalid input GLB: {exc}") from exc
    if len(skins) != 1:
        raise GlbPreprocessError(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    if skin.inverse_bind_matrices is None:
        raise GlbPreprocessError("skin must declare inverseBindMatrices")
    selected_animations, selected_actions, dropped_actions = _select_actions(
        source, pairs
    )
    primitives = _skin_primitives(source, skin)
    maximum_weights, nonzero_slots, zero_slots = _joint_weight_proof(skin, primitives)
    branches = _root_branches(skin)
    if not branches:
        raise GlbPreprocessError("skin has no joint root")
    weighted_branches = [
        (root, ordinals)
        for root, ordinals in branches
        if any(maximum_weights[ordinal] != 0.0 for ordinal in ordinals)
    ]
    if not isinstance(prune_zero_weight_leaves, bool):
        raise GlbPreprocessError("prune_zero_weight_leaves must be a boolean")
    if len(branches) == 1:
        retained_root, retained_branch = branches[0]
        removed_ordinals: set[int] = set()
    else:
        if len(weighted_branches) != 1:
            raise GlbPreprocessError(
                "cannot choose one deformation root without deleting a weighted branch: "
                f"root_count={len(branches)}, weighted_root_count={len(weighted_branches)}"
            )
        retained_root, retained_branch = weighted_branches[0]
        removed_ordinals = {
            ordinal
            for root, ordinals in branches
            if root != retained_root
            for ordinal in ordinals
        }
    leaf_removed_ordinals: set[int] = set()
    if prune_zero_weight_leaves:
        joint_nodes = {joint.node_index for joint in skin.joints}
        nodes = _objects(source.json.get("nodes", []), "nodes")
        for joint in skin.joints:
            children = nodes[joint.node_index].get("children", [])
            if not isinstance(children, list):
                raise GlbPreprocessError("joint node children must be an array")
            if (
                joint.joint_ordinal != retained_root
                and maximum_weights[joint.joint_ordinal] == 0.0
                and not any(child in joint_nodes for child in children)
                and not children
            ):
                leaf_removed_ordinals.add(joint.joint_ordinal)
        removed_ordinals |= leaf_removed_ordinals
    if not any(maximum_weights[ordinal] != 0.0 for ordinal in retained_branch):
        raise GlbPreprocessError(
            "retained skin root branch has no non-zero mesh influence"
        )
    violating = [
        ordinal for ordinal in removed_ordinals if maximum_weights[ordinal] != 0.0
    ]
    if violating:
        raise GlbPreprocessError(
            "refusing to delete joint ordinal(s) with non-zero skin weight: "
            + ", ".join(map(str, violating))
        )
    removed_nodes = _validate_removed_branch_payload(source, skin, removed_ordinals)

    document = copy.deepcopy(source.json)
    buffers = _objects(document.get("buffers", []), "buffers")
    if len(buffers) != 1:
        raise GlbPreprocessError("preprocess requires exactly one embedded buffer")
    declared_length = _integer(
        buffers[0].get("byteLength"), "buffers[0].byteLength", minimum=1
    )
    binary = bytearray(source.binary[:declared_length])
    document_skins = _objects(document.get("skins", []), "skins")
    document_skin = document_skins[skin.skin_index]

    kept_ordinals = [
        ordinal
        for ordinal in range(len(skin.joints))
        if ordinal not in removed_ordinals
    ]
    old_to_new = {ordinal: new for new, ordinal in enumerate(kept_ordinals)}
    fallback_ordinal = old_to_new[retained_root]
    primitive_reports: list[dict[str, Any]] = []
    if removed_ordinals:
        document_skin["joints"] = [
            skin.joints[ordinal].node_index for ordinal in kept_ordinals
        ]
        skeleton = document_skin.get("skeleton")
        if skeleton in removed_nodes:
            raise GlbPreprocessError("skin.skeleton points into a removed joint branch")
        inverse_values = np.asarray(
            [skin.inverse_bind_matrices[ordinal] for ordinal in kept_ordinals],
            dtype=np.float64,
        )
        document_skin["inverseBindMatrices"] = _append_accessor(
            document,
            binary,
            component_type=_FLOAT,
            element_type="MAT4",
            values=inverse_values,
        )

        meshes = _objects(document.get("meshes", []), "meshes")
        for primitive in primitives:
            remapped = np.empty_like(primitive.joints)
            zero_weight_substitutions = 0
            for row in range(primitive.joints.shape[0]):
                for column in range(4):
                    old = int(primitive.joints[row, column])
                    if old in removed_ordinals:
                        weight = float(primitive.weights[row, column])
                        if weight != 0.0:
                            raise GlbPreprocessError(
                                "non-zero weight reached removed-joint remap; fail closed"
                            )
                        remapped[row, column] = fallback_ordinal
                        zero_weight_substitutions += 1
                    else:
                        remapped[row, column] = old_to_new[old]
            new_accessor = _append_accessor(
                document,
                binary,
                component_type=primitive.joint_component_type,
                element_type="VEC4",
                values=remapped,
            )
            target_primitive = meshes[primitive.mesh_index]["primitives"][
                primitive.primitive_index
            ]
            target_primitive["attributes"]["JOINTS_0"] = new_accessor
            primitive_reports.append(
                {
                    "mesh_index": primitive.mesh_index,
                    "primitive_index": primitive.primitive_index,
                    "source_joint_accessor": primitive.joint_accessor,
                    "weight_accessor": primitive.weight_accessor,
                    "output_joint_accessor": new_accessor,
                    "vertex_count": int(primitive.joints.shape[0]),
                    "zero_weight_removed_ordinal_substitutions": zero_weight_substitutions,
                }
            )
    else:
        primitive_reports = [
            {
                "mesh_index": primitive.mesh_index,
                "primitive_index": primitive.primitive_index,
                "source_joint_accessor": primitive.joint_accessor,
                "weight_accessor": primitive.weight_accessor,
                "output_joint_accessor": primitive.joint_accessor,
                "vertex_count": int(primitive.joints.shape[0]),
                "zero_weight_removed_ordinal_substitutions": 0,
            }
            for primitive in primitives
        ]

    retained_nodes = {skin.joints[ordinal].node_index for ordinal in kept_ordinals}
    output_animations, removed_channels = _filter_action_channels(
        selected_animations,
        selected_actions,
        pairs,
        retained_joint_nodes=retained_nodes,
        removed_joint_nodes=removed_nodes,
        source=source,
    )
    document["animations"] = output_animations
    buffers[0]["byteLength"] = len(binary)
    payload = build_glb(document, binary)

    try:
        verified = parse_glb(payload)
        verified_skins = extract_skins(verified)
        verified_actions = extract_actions(verified)
    except GlbError as exc:
        raise GlbPreprocessError(f"output GLB readback failed: {exc}") from exc
    if len(verified_skins) != 1:
        raise GlbPreprocessError("output GLB skin readback count mismatch")
    verified_roots = [
        joint
        for joint in verified_skins[0].joints
        if joint.parent_joint_node_index is None
    ]
    if len(verified_roots) != 1:
        raise GlbPreprocessError("output skin is not one connected joint tree")
    expected_action_names = [target for _, target in pairs]
    if [action.name for action in verified_actions] != expected_action_names:
        raise GlbPreprocessError("output action-name readback mismatch")
    if any(
        channel.target_node_index not in retained_nodes
        for action in verified_actions
        for channel in action.channels
    ):
        raise GlbPreprocessError("output action still targets a non-retained joint")

    try:
        _write_exclusive(output_resolved, payload)
    except OSError as exc:
        raise GlbPreprocessError(
            f"failed to create output exclusively: {output_resolved}: {exc}"
        ) from exc
    output_sha256 = hashlib.sha256(payload).hexdigest()
    branch_reports: list[dict[str, Any]] = []
    for root_ordinal, ordinals in branches:
        root_joint = skin.joints[root_ordinal]
        branch_reports.append(
            {
                "root_ordinal": root_ordinal,
                "root_node_index": root_joint.node_index,
                "root_name": root_joint.name,
                "joint_ordinals": list(ordinals),
                "maximum_weight": max(
                    float(maximum_weights[ordinal]) for ordinal in ordinals
                ),
                "nonzero_weight_slots": sum(
                    int(nonzero_slots[ordinal]) for ordinal in ordinals
                ),
                "disposition": "retained"
                if root_ordinal == retained_root
                else "removed",
            }
        )
    removed_joint_reports = [
        {
            "old_ordinal": ordinal,
            "node_index": skin.joints[ordinal].node_index,
            "name": skin.joints[ordinal].name,
            "maximum_weight": float(maximum_weights[ordinal]),
            "nonzero_weight_slots": int(nonzero_slots[ordinal]),
            "zero_weight_reference_slots": int(zero_slots[ordinal]),
        }
        for ordinal in sorted(removed_ordinals)
    ]
    return {
        "schema": "avengine_m2_glb_preprocess_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_resolved),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output_resolved),
            "sha256": output_sha256,
            "byte_size": len(payload),
        },
        "actions": {
            "selected": [
                {"source_name": source_name, "output_name": target_name}
                for source_name, target_name in pairs
            ],
            "dropped_source_actions": dropped_actions,
            "removed_channels": removed_channels,
        },
        "skin": {
            "skin_index": skin.skin_index,
            "input_joint_count": len(skin.joints),
            "output_joint_count": len(kept_ordinals),
            "input_root_count": len(branches),
            "output_root_count": 1,
            "retained_root": {
                "old_ordinal": retained_root,
                "new_ordinal": fallback_ordinal,
                "node_index": skin.joints[retained_root].node_index,
                "name": skin.joints[retained_root].name,
            },
            "root_branches": branch_reports,
            "removed_joints": removed_joint_reports,
            "zero_weight_leaf_pruning_requested": prune_zero_weight_leaves,
            "zero_weight_leaf_removed_ordinals": sorted(leaf_removed_ordinals),
            "old_to_new_joint_ordinals": {
                str(old): new for old, new in sorted(old_to_new.items())
            },
            "weight_proof": {
                "rule": "removed joint maximum weight must equal exactly 0.0",
                "primitive_count": len(primitives),
                "removed_nonzero_weight_slots": 0,
            },
        },
        "primitives": primitive_reports,
        "notes": [
            "This preprocess step is body-plan neutral and does not identify joints by animal-specific names.",
            "Removed controller nodes remain inert in the scene node table; only skin membership and selected channels are stripped.",
            "Mesh/animation quality, provenance, contacts, Habitat playback, and human review remain separate gates.",
        ],
    }
