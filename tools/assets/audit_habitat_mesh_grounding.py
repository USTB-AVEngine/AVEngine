#!/usr/bin/env python3
"""Audit exact skinned-vertex grounding and declared source anchors.

This report is a research-candidate measurement.  It evaluates the target
native rebased GLB with the exact baked Idle/Walking local joint rotations,
records the skinned mesh bottom in actor space, and records the declared
muzzle/contact joint positions.  It does not infer physics contact or promote
an asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.assets.actions import baked_actions_content_sha256, read_baked_actions_npz
from avengine.assets.glb import decode_accessor, load_glb
from avengine.contracts.json_io import load_json


class GroundingAuditError(ValueError):
    """Input or measured grounding evidence is invalid."""


_COMPONENT_TYPES: dict[int, tuple[str, int, Any]] = {
    5120: ("b", 1, np.int8),
    5121: ("B", 1, np.uint8),
    5122: ("h", 2, np.int16),
    5123: ("H", 2, np.uint16),
    5125: ("I", 4, np.uint32),
    5126: ("f", 4, np.float32),
}
_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = resolved.read_bytes()
    return {"path": str(resolved), "byte_size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _decode_any(document: Any, accessor_index: int) -> np.ndarray:
    accessors = document.json.get("accessors")
    views = document.json.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise GroundingAuditError("GLB lacks accessors or bufferViews")
    accessor = accessors[accessor_index]
    if "bufferView" not in accessor or "sparse" in accessor:
        raise GroundingAuditError(f"unsupported accessor {accessor_index}")
    try:
        fmt, component_size, dtype = _COMPONENT_TYPES[int(accessor["componentType"])]
        component_count = _COMPONENT_COUNTS[str(accessor["type"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise GroundingAuditError(f"unsupported accessor {accessor_index}") from exc
    view = views[int(accessor["bufferView"])]
    stride = int(view.get("byteStride", component_size * component_count))
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    if stride < component_size * component_count:
        raise GroundingAuditError(f"accessor {accessor_index} stride is too short")
    values = [
        struct.unpack_from(
            "<" + fmt * component_count,
            document.binary,
            offset + index * stride,
        )
        for index in range(count)
    ]
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 2 or result.shape != (count, component_count):
        raise GroundingAuditError(f"accessor {accessor_index} decoded with wrong shape")
    return result


def _qmatrix(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise GroundingAuditError("joint quaternion is not finite xyzw")
    norm = float(np.linalg.norm(q))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        q /= norm
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _node_local(node: Mapping[str, Any]) -> np.ndarray:
    if "matrix" in node:
        matrix = np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
        return matrix
    matrix = _qmatrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)):
        raise GroundingAuditError("node scale is not finite vec3")
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(scale)
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise GroundingAuditError("node translation is not finite vec3")
    matrix[:3, 3] = translation
    return matrix


def _global_nodes(document: Any) -> dict[int, np.ndarray]:
    nodes = document.json.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise GroundingAuditError("GLB has no nodes")
    parents: dict[int, int] = {}
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            if child in parents:
                raise GroundingAuditError(f"node {child} has multiple parents")
            parents[int(child)] = parent
    result: dict[int, np.ndarray] = {}

    def solve(index: int) -> np.ndarray:
        if index in result:
            return result[index]
        parent = parents.get(index)
        result[index] = (
            solve(parent) @ _node_local(nodes[index])
            if parent is not None
            else _node_local(nodes[index])
        )
        return result[index]

    for index in range(len(nodes)):
        solve(index)
    return result


def _geometry(document: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    skins = document.json.get("skins")
    if not isinstance(skins, list) or len(skins) != 1:
        raise GroundingAuditError("visual GLB must contain exactly one skin")
    skin = skins[0]
    joint_indices = [int(index) for index in skin.get("joints", [])]
    if not joint_indices:
        raise GroundingAuditError("skin has no joints")
    nodes = document.json["nodes"]
    names = [nodes[index].get("name") for index in joint_indices]
    if any(not isinstance(name, str) or not name for name in names):
        raise GroundingAuditError("skin joints must have names")
    ibm = _decode_any(document, int(skin["inverseBindMatrices"])).astype(np.float64)
    ibm = ibm.reshape((-1, 4, 4), order="F")
    if ibm.shape != (len(joint_indices), 4, 4):
        raise GroundingAuditError("inverse bind matrix count differs from skin")
    mesh_nodes = [
        (index, node)
        for index, node in enumerate(nodes)
        if "mesh" in node and "skin" in node
    ]
    if len(mesh_nodes) != 1:
        raise GroundingAuditError("visual GLB must have exactly one skinned mesh node")
    mesh_node_index, mesh_node = mesh_nodes[0]
    mesh = document.json["meshes"][int(mesh_node["mesh"])]
    positions: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for primitive in mesh.get("primitives", []):
        attrs = primitive.get("attributes", {})
        required = {"POSITION", "JOINTS_0", "WEIGHTS_0"}
        if not required.issubset(attrs):
            raise GroundingAuditError("skinned primitive lacks position/joints/weights")
        positions.append(np.asarray(decode_accessor(document, int(attrs["POSITION"])).values, dtype=np.float64))
        joints.append(_decode_any(document, int(attrs["JOINTS_0"])).astype(np.int64))
        weights.append(np.asarray(decode_accessor(document, int(attrs["WEIGHTS_0"])).values, dtype=np.float64))
    if not positions:
        raise GroundingAuditError("skinned mesh has no primitives")
    vertex_positions = np.concatenate(positions)
    vertex_joints = np.concatenate(joints)
    vertex_weights = np.concatenate(weights)
    if vertex_positions.ndim != 2 or vertex_positions.shape[1] != 3:
        raise GroundingAuditError("positions are not VEC3")
    if vertex_joints.shape != vertex_weights.shape or vertex_joints.shape[1] != 4:
        raise GroundingAuditError("JOINTS_0 and WEIGHTS_0 are not aligned VEC4")
    if not np.all(np.isfinite(vertex_positions)) or not np.all(np.isfinite(vertex_weights)):
        raise GroundingAuditError("mesh positions or weights are non-finite")
    if np.any(vertex_joints < 0) or np.any(vertex_joints >= len(joint_indices)):
        raise GroundingAuditError("JOINTS_0 contains an out-of-range skin ordinal")
    return (
        vertex_positions,
        vertex_joints,
        vertex_weights,
        ibm,
        _global_nodes(document)[mesh_node_index],
        names,
        joint_indices,
    )


def _matrix_from_mapping(value: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise GroundingAuditError("mapping transform is not finite 4x4")
    return matrix


def _pose_joint_matrices(mapping: Mapping[str, Any], pose: np.ndarray) -> dict[str, np.ndarray]:
    joints = mapping.get("joints")
    runtime = mapping.get("runtime_joint_order")
    if not isinstance(joints, list) or not isinstance(runtime, list):
        raise GroundingAuditError("joint mapping lacks joints/runtime order")
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (len(runtime), 4):
        raise GroundingAuditError(f"pose shape {pose.shape} differs from runtime order {len(runtime)}")
    pose_by_name = {str(name): pose[index] for index, name in enumerate(runtime)}
    records = {str(item["joint_id"]): item for item in joints if isinstance(item, Mapping)}
    if len(records) != len(joints):
        raise GroundingAuditError("joint mapping has duplicate/invalid records")
    solved: dict[str, np.ndarray] = {}

    def solve(name: str) -> np.ndarray:
        if name in solved:
            return solved[name]
        item = records[name]
        parent = item.get("parent_joint_id")
        q = [0.0, 0.0, 0.0, 1.0] if parent is None else pose_by_name[name]
        local = _qmatrix(q)
        local[:3, 3] = np.asarray(item["local_translation_m"], dtype=np.float64)
        if parent is None:
            solved[name] = local
        else:
            solved[name] = solve(str(parent)) @ local
        return solved[name]

    for name in records:
        solve(name)
    return solved


def _skin_actor_vertices(
    *,
    positions: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    inverse_bind: np.ndarray,
    mesh_node_global: np.ndarray,
    actor_from_skin_root: np.ndarray,
    joint_matrices: Mapping[str, np.ndarray],
    skin_joint_names: Sequence[str],
) -> np.ndarray:
    skin_matrices = np.stack([
        joint_matrices[name] for name in skin_joint_names
    ]) @ inverse_bind
    per_vertex = skin_matrices[joints]
    skin_matrix = np.sum(per_vertex * weights[:, :, None, None], axis=1)
    homogeneous = np.concatenate([positions, np.ones((len(positions), 1), dtype=np.float64)], axis=1)
    transformed = np.einsum("vi,vji->vj", homogeneous, skin_matrix)
    actor_matrix = actor_from_skin_root @ mesh_node_global
    actor = transformed @ actor_matrix.T
    if actor.shape != (len(positions), 4) or not np.all(np.isfinite(actor)):
        raise GroundingAuditError("skinned actor vertices are non-finite")
    return actor[:, :3]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(*, visual_glb: Path, actions_npz: Path, joint_mapping: Path, anchors: Path, output: Path, source_glb: Path | None = None, asset_json: Path | None = None, contacts: Path | None = None) -> Path:
    for dest in (output,):
        if dest.exists() or dest.is_symlink():
            raise GroundingAuditError(f"refusing to replace output: {dest}")
    visual = load_glb(visual_glb)
    mapping = load_json(joint_mapping)
    anchor_value = load_json(anchors)
    actions = read_baked_actions_npz(actions_npz)
    if mapping.get("source_glb_sha256") != visual.sha256:
        raise GroundingAuditError("joint mapping does not bind visual GLB")
    if anchor_value.get("source_visual_sha256") != visual.sha256:
        raise GroundingAuditError("anchors do not bind visual GLB")
    if baked_actions_content_sha256(actions) != _hash_file(actions_npz):
        raise GroundingAuditError("actions NPZ canonical hash differs from bytes")
    positions, joints, weights, inverse_bind, mesh_node_global, skin_names, skin_indices = _geometry(visual)
    if list(mapping.get("joint_order", [])) != skin_names:
        raise GroundingAuditError("mapping joint order differs from skin order")
    actor_from_skin_root = _matrix_from_mapping(mapping["actor_from_skin_root"])
    anchor_records = {str(item["anchor_id"]): item for item in anchor_value.get("anchors", []) if isinstance(item, Mapping)}
    contact_value = load_json(contacts) if contacts is not None else {}
    required_contacts = tuple(contact_value.get("contact_order", anchor_value.get("contact_order", [])))
    if required_contacts != ("paw_front_left", "paw_front_right", "paw_hind_left", "paw_hind_right"):
        raise GroundingAuditError("contact order must be the four canonical paw anchors")
    required = ("body", "head", "muzzle", *required_contacts)
    if not all(anchor_id in anchor_records for anchor_id in required):
        raise GroundingAuditError("anchor file lacks required body/head/muzzle/contact records")
    root_name = str(mapping["root_joint_id"])
    results: dict[str, Any] = {}
    for action_id in ("idle", "walk"):
        clip = actions.action(action_id)
        frame_records: list[dict[str, Any]] = []
        minima: list[float] = []
        maxima: list[float] = []
        for frame_index, pose in enumerate(clip.rotations_xyzw):
            matrices = _pose_joint_matrices(mapping, pose)
            actor_matrices = {name: actor_from_skin_root @ matrix for name, matrix in matrices.items()}
            actor_vertices = _skin_actor_vertices(
                positions=positions,
                joints=joints,
                weights=weights,
                inverse_bind=inverse_bind,
                mesh_node_global=mesh_node_global,
                actor_from_skin_root=actor_from_skin_root,
                joint_matrices=matrices,
                skin_joint_names=skin_names,
            )
            minimum = float(np.min(actor_vertices[:, 1]))
            maximum = float(np.max(actor_vertices[:, 1]))
            minima.append(minimum); maxima.append(maximum)
            contact_positions = {
                anchor_id: [float(value) for value in actor_matrices[str(anchor_records[anchor_id]["joint_id"])][:3, 3]]
                for anchor_id in required_contacts
            }
            frame_records.append({
                "frame_index": frame_index,
                "sample_tick": int(clip.sample_ticks[frame_index]),
                "mesh_min_y_m": minimum,
                "mesh_max_y_m": maximum,
                "contact_joint_positions_m": contact_positions,
                "contact_joint_y_minus_mesh_min_y_m": {
                    anchor_id: float(contact_positions[anchor_id][1] - minimum)
                    for anchor_id in required_contacts
                },
                "emitter_position_m": [float(value) for value in actor_matrices[str(anchor_records["muzzle"]["joint_id"])][:3, 3]],
            })
        results[action_id] = {
            "sample_count": len(frame_records),
            "source_action_name": clip.source_action_name,
            "mesh_min_y_m": float(min(minima)),
            "mesh_max_y_m": float(max(maxima)),
            "mesh_min_y_by_frame_m": minima,
            "mesh_max_y_by_frame_m": maxima,
            "frame_records": frame_records,
        }
    # Anchor evidence uses the measured source skin order and actual rest mesh weights.
    anchor_evidence: dict[str, Any] = {}
    matrices_rest = _pose_joint_matrices(mapping, actions.action("idle").rotations_xyzw[0])
    actor_matrices_rest = {name: actor_from_skin_root @ matrix for name, matrix in matrices_rest.items()}
    rest_actor_vertices = _skin_actor_vertices(
        positions=positions, joints=joints, weights=weights,
        inverse_bind=inverse_bind, mesh_node_global=mesh_node_global,
        actor_from_skin_root=actor_from_skin_root,
        joint_matrices=matrices_rest, skin_joint_names=skin_names,
    )
    for anchor_id in required:
        item = anchor_records[anchor_id]; joint_id = str(item["joint_id"])
        if joint_id not in matrices_rest:
            raise GroundingAuditError(f"anchor {anchor_id} references unknown joint {joint_id}")
        ordinal = skin_names.index(joint_id)
        influence = (weights * (joints == ordinal)).sum(axis=1)
        top = joints[np.arange(len(joints)), np.argmax(weights, axis=1)] == ordinal
        anchor_evidence[anchor_id] = {
            "joint_id": joint_id,
            "skin_joint_ordinal": ordinal,
            "joint_from_anchor": item["joint_from_anchor"],
            "joint_world_position_actor_m": [float(value) for value in actor_matrices_rest[joint_id][:3, 3]],
            "weighted_vertex_support_count": int(np.count_nonzero(influence > 0.0)),
            "top_influence_vertex_count": int(np.count_nonzero(top)),
            "top_influence_vertex_centroid_actor_m": [float(value) for value in np.mean(rest_actor_vertices[top], axis=0)] if np.any(top) else None,
        }
    source_anchor_evidence: dict[str, Any] | None = None
    source_mesh_bbox: dict[str, Any] | None = None
    if source_glb is not None:
        source_document = load_glb(source_glb)
        source_positions, source_joints, source_weights, source_inverse_bind, source_mesh_node_global, source_skin_names, source_skin_indices = _geometry(source_document)
        source_nodes_global = _global_nodes(source_document)
        source_joint_matrices = {name: source_nodes_global[int(node_index)] for name, node_index in zip(source_skin_names, source_skin_indices, strict=True)}
        source_rest_vertices = _skin_actor_vertices(
            positions=source_positions, joints=source_joints, weights=source_weights,
            inverse_bind=source_inverse_bind, mesh_node_global=source_mesh_node_global,
            actor_from_skin_root=np.eye(4, dtype=np.float64),
            joint_matrices=source_joint_matrices, skin_joint_names=source_skin_names,
        )
        source_anchor_evidence = {}
        for anchor_id in required:
            item = anchor_records[anchor_id]
            joint_id = str(item["joint_id"])
            if joint_id not in source_skin_names:
                raise GroundingAuditError(f"source GLB lacks anchor joint {joint_id}")
            ordinal = source_skin_names.index(joint_id)
            node_index = int(source_skin_indices[ordinal])
            influence = (source_weights * (source_joints == ordinal)).sum(axis=1)
            top = source_joints[np.arange(len(source_joints)), np.argmax(source_weights, axis=1)] == ordinal
            source_anchor_evidence[anchor_id] = {
                "joint_id": joint_id,
                "source_skin_joint_ordinal": ordinal,
                "source_node_index": node_index,
                "source_joint_global_position_m": [float(v) for v in source_joint_matrices[joint_id][:3, 3]],
                "weighted_vertex_support_count": int(np.count_nonzero(influence > 0.0)),
                "top_influence_vertex_count": int(np.count_nonzero(top)),
                "top_influence_vertex_centroid_m": [float(v) for v in np.mean(source_rest_vertices[top], axis=0)] if np.any(top) else None,
            }
        source_mesh_bbox = {
            "min_m": [float(v) for v in source_rest_vertices.min(axis=0)],
            "max_m": [float(v) for v in source_rest_vertices.max(axis=0)],
            "vertex_count": int(len(source_rest_vertices)),
        }
    value: dict[str, Any] = {
        "schema": "avengine_p12_habitat_mesh_grounding_audit_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "evidence_scope": "exact skinned-vertex geometry and declared joint transforms; no physics support measurement",
        "inputs": {
            "visual_glb": _record(visual_glb),
            "actions_npz": _record(actions_npz),
            "joint_mapping": _record(joint_mapping),
            "anchors": _record(anchors),
            "source_glb": _record(source_glb) if source_glb is not None else None,
            "asset_json": _record(asset_json) if asset_json is not None else None,
            "contacts": _record(contacts) if contacts is not None else None,
        },
        "asset_id": (load_json(asset_json).get("asset_id") if asset_json is not None else anchor_value.get("asset_id")),
        "source_visual_sha256": visual.sha256,
        "mesh": {
            "vertex_count": int(len(positions)),
            "skin_joint_count": len(skin_names),
            "mesh_node_global": mesh_node_global.tolist(),
            "actor_from_skin_root": actor_from_skin_root.tolist(),
            "rest_idle_mesh_min_y_m": results["idle"]["mesh_min_y_m"],
            "rest_idle_mesh_max_y_m": results["idle"]["mesh_max_y_m"],
            "exact_bottom_reference": "actor-root frame +Y; min over all skinned vertices",
        },
        "anchors": anchor_evidence,
        "source_anchor_evidence": source_anchor_evidence,
        "source_mesh_rest_bbox": source_mesh_bbox,
        "actions": results,
        "contact_inference": {
            "source_contact_order": list(required_contacts),
            "contact_phase_source": _record(contacts) if contacts is not None else None,
            "physics_support_measured": False,
            "qualification_claim": False,
        },
        "notes": [
            "Per-frame minima use the exact GLB inverse-bind matrices, exact baked local rotations, and actor_from_skin_root composition.",
            "Declared contact joint origins are anatomical references; they are not soles and do not establish collision support.",
            "Muzzle emitter uses the explicitly declared source joint and joint-local transform; no donor animal anchor or pose value is used.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--joint-mapping", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-glb", type=Path)
    parser.add_argument("--asset-json", type=Path)
    parser.add_argument("--contacts", type=Path)
    args = parser.parse_args()
    result = audit(
        visual_glb=args.visual_glb.resolve(), actions_npz=args.actions_npz.resolve(),
        joint_mapping=args.joint_mapping.resolve(), anchors=args.anchors.resolve(),
        output=args.output.resolve(),
        source_glb=args.source_glb.resolve() if args.source_glb else None,
        asset_json=args.asset_json.resolve() if args.asset_json else None,
        contacts=args.contacts.resolve() if args.contacts else None,
    )
    print(json.dumps({"status": "pass", "output": str(result)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
