"""Bake one uniform skin-ancestor scale into GLB geometry and bind data.

This is a narrow similarity transform compiler, not a relaxation of the M2
rigid rebase gate.  It accepts exactly one positive uniform scale on the skin
root's ancestor path, pushes that scale into descendant translations, skinned
mesh positions, translation animation samples and inverse bind matrices, then
emits an ancestor-scale-free GLB for independent deformation verification.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np

from avengine.assets.glb import (
    GlbError,
    decode_accessor,
    extract_actions,
    extract_skins,
    load_glb,
    parse_glb,
)
from avengine.assets.glb_write import build_glb


_FLOAT = 5126
_SCALE_TOLERANCE = 1.0e-6


class SimilarityBakeError(ValueError):
    """A GLB is outside the bounded uniform-scale bake contract."""


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


def _objects(value: Any, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SimilarityBakeError(f"{owner} must be an array of objects")
    return value


def _parents(nodes: list[dict[str, Any]]) -> list[int | None]:
    result: list[int | None] = [None] * len(nodes)
    for parent, node in enumerate(nodes):
        children = node.get("children", [])
        if not isinstance(children, list):
            raise SimilarityBakeError(f"nodes[{parent}].children must be an array")
        for child in children:
            if (
                isinstance(child, bool)
                or not isinstance(child, int)
                or child < 0
                or child >= len(nodes)
                or result[child] is not None
            ):
                raise SimilarityBakeError(
                    "node hierarchy has an invalid/multiple parent"
                )
            result[child] = parent
    return result


def _ancestors(node: int, parents: list[int | None]) -> list[int]:
    result: list[int] = []
    cursor = parents[node]
    while cursor is not None:
        result.append(cursor)
        cursor = parents[cursor]
    return result


def _descendants(node: int, nodes: list[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    stack = list(nodes[node].get("children", []))
    while stack:
        child = stack.pop()
        if child in result:
            raise SimilarityBakeError("node hierarchy contains a cycle")
        result.add(child)
        stack.extend(nodes[child].get("children", []))
    return result


def _append_float_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    element_type: str,
    values: np.ndarray,
) -> int:
    component_count = {"SCALAR": 1, "VEC3": 3, "MAT4": 16}.get(element_type)
    if component_count is None:
        raise SimilarityBakeError(f"unsupported accessor type: {element_type}")
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != component_count
        or not np.all(np.isfinite(array))
    ):
        raise SimilarityBakeError("replacement accessor values are invalid")
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    packer = struct.Struct("<" + "f" * component_count)
    for row in array:
        binary.extend(packer.pack(*row.tolist()))
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    if not isinstance(views, list) or not isinstance(accessors, list):
        raise SimilarityBakeError("bufferViews/accessors must be arrays")
    view_index = len(views)
    views.append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
    )
    accessor_index = len(accessors)
    accessors.append(
        {
            "bufferView": view_index,
            "componentType": _FLOAT,
            "count": len(array),
            "type": element_type,
        }
    )
    return accessor_index


def bake_uniform_skin_ancestor_scale(
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bake exactly one positive uniform ancestor scale and return evidence."""

    source_resolved = Path(source_path).resolve()
    output_argument = Path(output_path)
    if output_argument.exists() or output_argument.is_symlink():
        raise SimilarityBakeError(f"refusing to replace output: {output_argument}")
    output_resolved = output_argument.resolve()
    if source_resolved == output_resolved:
        raise SimilarityBakeError("output must not overwrite the source GLB")
    try:
        source = load_glb(source_resolved)
        skins = extract_skins(source)
    except GlbError as exc:
        raise SimilarityBakeError(f"invalid input GLB: {exc}") from exc
    if len(skins) != 1:
        raise SimilarityBakeError(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    roots = [joint for joint in skin.joints if joint.parent_joint_node_index is None]
    if len(roots) != 1:
        raise SimilarityBakeError("skin must already contain one joint root")
    if skin.inverse_bind_matrices is None:
        raise SimilarityBakeError("skin must declare inverseBindMatrices")

    document = copy.deepcopy(source.json)
    nodes = _objects(document.get("nodes", []), "nodes")
    parents = _parents(nodes)
    path = [roots[0].node_index, *_ancestors(roots[0].node_index, parents)]
    scaled_nodes: list[tuple[int, float]] = []
    for node_index in path:
        node = nodes[node_index]
        if "matrix" in node:
            raise SimilarityBakeError("matrix-authored ancestor nodes are unsupported")
        raw = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            raise SimilarityBakeError("ancestor scale must contain three finite values")
        error = float(np.max(np.abs(raw - 1.0)))
        if error <= _SCALE_TOLERANCE:
            continue
        uniform_error = float(np.max(np.abs(raw - raw[0])))
        if raw[0] <= 0.0 or uniform_error > _SCALE_TOLERANCE:
            raise SimilarityBakeError(
                f"skin ancestor scale must be positive uniform: node={node_index}, scale={raw.tolist()}"
            )
        scaled_nodes.append((node_index, float(raw[0])))
    if len(scaled_nodes) != 1:
        raise SimilarityBakeError(
            f"expected exactly one non-unit uniform ancestor scale, found {scaled_nodes}"
        )
    scale_node, scale = scaled_nodes[0]
    try:
        actions = extract_actions(source)
    except GlbError as exc:
        raise SimilarityBakeError(f"invalid source actions: {exc}") from exc
    protected_scale_nodes = {scale_node, *_ancestors(scale_node, parents)}
    for action in actions:
        for channel in action.channels:
            if (
                channel.target_path == "scale"
                and channel.target_node_index in protected_scale_nodes
            ):
                raise SimilarityBakeError(
                    "scale animation targets the similarity scale node or a "
                    "relevant ancestor: "
                    f"action={action.name!r}, node={channel.target_node_index}"
                )
    descendants = _descendants(scale_node, nodes)
    joint_nodes = {joint.node_index for joint in skin.joints}
    if not joint_nodes.issubset(descendants):
        raise SimilarityBakeError("not every skin joint is below the scaled ancestor")

    meshes = _objects(document.get("meshes", []), "meshes")
    mesh_instances: dict[int, list[int]] = {}
    for node_index, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mesh_index = node["mesh"]
        if (
            isinstance(mesh_index, bool)
            or not isinstance(mesh_index, int)
            or not 0 <= mesh_index < len(meshes)
        ):
            raise SimilarityBakeError(f"nodes[{node_index}].mesh is invalid")
        mesh_instances.setdefault(mesh_index, []).append(node_index)

    for node_index in sorted(descendants | {scale_node}):
        node = nodes[node_index]
        unsupported = [key for key in ("camera", "extensions") if key in node]
        if unsupported:
            raise SimilarityBakeError(
                "scaled subtree contains unhandled scene payload: "
                f"node={node_index}, keys={unsupported}"
            )
        if "mesh" in node:
            if node_index == scale_node or node.get("skin") != skin.skin_index:
                raise SimilarityBakeError(
                    "scaled subtree contains an unskinned mesh payload that would "
                    f"not be similarity-baked: node={node_index}"
                )
        elif "skin" in node or "weights" in node:
            raise SimilarityBakeError(
                "scaled subtree contains malformed/unhandled skin payload: "
                f"node={node_index}"
            )

    buffers = _objects(document.get("buffers", []), "buffers")
    if len(buffers) != 1:
        raise SimilarityBakeError("similarity bake requires one embedded buffer")
    declared_length = buffers[0].get("byteLength")
    if (
        isinstance(declared_length, bool)
        or not isinstance(declared_length, int)
        or declared_length <= 0
    ):
        raise SimilarityBakeError("buffers[0].byteLength is invalid")
    binary = bytearray(source.binary[:declared_length])

    for node_index in sorted(descendants):
        node = nodes[node_index]
        if "matrix" in node:
            raise SimilarityBakeError(
                "matrix-authored descendant nodes are unsupported"
            )
        translation = np.asarray(
            node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise SimilarityBakeError("descendant translation is invalid")
        node["translation"] = (translation * scale).tolist()
    nodes[scale_node]["scale"] = [1.0, 1.0, 1.0]

    transformed_position_accessors: dict[int, int] = {}
    mesh_records: list[dict[str, Any]] = []
    mesh_nodes = [
        (index, node)
        for index, node in enumerate(nodes)
        if node.get("skin") == skin.skin_index
    ]
    if not mesh_nodes:
        raise SimilarityBakeError("skin is not bound to a mesh node")
    for node_index, node in mesh_nodes:
        if node_index not in descendants:
            raise SimilarityBakeError("skinned mesh is not below the scaled ancestor")
        mesh_index = node.get("mesh")
        if (
            isinstance(mesh_index, bool)
            or not isinstance(mesh_index, int)
            or not 0 <= mesh_index < len(meshes)
        ):
            raise SimilarityBakeError("skinned node has an invalid mesh index")
        instances = mesh_instances[mesh_index]
        if len(instances) != 1:
            raise SimilarityBakeError(
                "shared mesh instancing is outside this similarity bake because "
                f"mutating mesh {mesh_index} would affect nodes {instances}"
            )
        primitives = _objects(
            meshes[mesh_index].get("primitives", []), "mesh.primitives"
        )
        for primitive_index, primitive in enumerate(primitives):
            if "targets" in primitive:
                raise SimilarityBakeError(
                    "morph targets are outside this similarity bake"
                )
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict) or "POSITION" not in attributes:
                raise SimilarityBakeError("mesh primitive must contain POSITION")
            source_accessor = attributes["POSITION"]
            if source_accessor not in transformed_position_accessors:
                try:
                    decoded = decode_accessor(source, source_accessor)
                except GlbError as exc:
                    raise SimilarityBakeError(
                        f"invalid POSITION accessor: {exc}"
                    ) from exc
                if decoded.element_type != "VEC3":
                    raise SimilarityBakeError("POSITION must be FLOAT VEC3")
                values = np.asarray(decoded.values, dtype=np.float64) * scale
                output_accessor = _append_float_accessor(
                    document, binary, element_type="VEC3", values=values
                )
                document["accessors"][output_accessor]["min"] = np.min(
                    values, axis=0
                ).tolist()
                document["accessors"][output_accessor]["max"] = np.max(
                    values, axis=0
                ).tolist()
                transformed_position_accessors[source_accessor] = output_accessor
            attributes["POSITION"] = transformed_position_accessors[source_accessor]
            mesh_records.append(
                {
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "source_position_accessor": source_accessor,
                    "output_position_accessor": attributes["POSITION"],
                }
            )

    animation_records: list[dict[str, Any]] = []
    animations = _objects(document.get("animations", []), "animations")
    scaled_output_accessors: dict[int, int] = {}
    for action in actions:
        raw_animation = animations[action.animation_index]
        samplers = _objects(raw_animation.get("samplers"), "animation.samplers")
        for channel in action.channels:
            if (
                channel.target_path != "translation"
                or channel.target_node_index not in descendants
            ):
                continue
            source_accessor = channel.output_accessor_index
            if source_accessor not in scaled_output_accessors:
                values = np.asarray(channel.values, dtype=np.float64) * scale
                scaled_output_accessors[source_accessor] = _append_float_accessor(
                    document, binary, element_type="VEC3", values=values
                )
            samplers[channel.sampler_index]["output"] = scaled_output_accessors[
                source_accessor
            ]
            animation_records.append(
                {
                    "action": action.name,
                    "node": channel.target_node_name,
                    "source_output_accessor": source_accessor,
                    "output_accessor": scaled_output_accessors[source_accessor],
                }
            )

    similarity = np.diag([scale, scale, scale, 1.0])
    inverse_similarity = np.diag([1.0 / scale, 1.0 / scale, 1.0 / scale, 1.0])
    inverse_values = []
    for raw in skin.inverse_bind_matrices:
        matrix = np.asarray(raw, dtype=np.float64).reshape(4, 4).T
        baked = similarity @ matrix @ inverse_similarity
        inverse_values.append(baked.T.reshape(16))
    inverse_accessor = _append_float_accessor(
        document,
        binary,
        element_type="MAT4",
        values=np.asarray(inverse_values, dtype=np.float64),
    )
    _objects(document.get("skins", []), "skins")[skin.skin_index][
        "inverseBindMatrices"
    ] = inverse_accessor
    buffers[0]["byteLength"] = len(binary)
    payload = build_glb(document, binary)
    try:
        verified = parse_glb(payload)
        verified_skins = extract_skins(verified)
    except GlbError as exc:
        raise SimilarityBakeError(f"output readback failed: {exc}") from exc
    if len(verified_skins) != 1 or len(verified_skins[0].joints) != len(skin.joints):
        raise SimilarityBakeError("output skin readback differs")
    verified_nodes = _objects(verified.json.get("nodes", []), "nodes")
    output_scale = np.asarray(verified_nodes[scale_node].get("scale", [1.0, 1.0, 1.0]))
    if float(np.max(np.abs(output_scale - 1.0))) > _SCALE_TOLERANCE:
        raise SimilarityBakeError("output ancestor scale is not identity")

    try:
        _write_exclusive(output_resolved, payload)
    except OSError as exc:
        raise SimilarityBakeError(
            f"failed to create output exclusively: {output_resolved}: {exc}"
        ) from exc
    return {
        "schema": "avengine_m2_uniform_skin_similarity_bake_v1",
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
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "scale": {
            "node_index": scale_node,
            "node_name": nodes[scale_node].get("name"),
            "uniform_factor": scale,
        },
        "mesh_positions": mesh_records,
        "scaled_translation_channels": animation_records,
        "inverse_bind_matrices": {
            "source_accessor": skin.inverse_bind_matrices_accessor_index,
            "output_accessor": inverse_accessor,
            "count": len(inverse_values),
        },
        "required_followup": [
            "sampled Blender deformation equivalence against the exact source hash",
            "strict M2 root rebase and independent rebase deformation verification",
        ],
    }
