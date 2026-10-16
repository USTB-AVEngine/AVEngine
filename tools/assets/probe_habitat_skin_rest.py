#!/usr/bin/env python3
"""Generate and exercise a temporary Habitat skinned-AO rest-pose descriptor.

This is a capability probe, not an asset qualification tool or a trusted
execution attester.  It deliberately writes only to the caller-selected output
directory and hash-binds the local configuration, measured runtime binding and
observation files.  Those bindings establish local artifact integrity only;
runtime execution conclusions belong to an external capture/audit boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np
from PIL import Image


_MIN_QA_SEMANTIC_PIXELS = 256
_MAX_LINK_BIND_ERROR = 5.0e-5


def _read_glb_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError("expected an exact GLB v2 container")
    offset = 12
    json_chunks: list[bytes] = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise ValueError("truncated GLB chunk")
        if chunk_type == 0x4E4F534A:
            json_chunks.append(payload[offset:end])
        offset = end
    if offset != len(payload) or len(json_chunks) != 1:
        raise ValueError("GLB must contain exactly one JSON chunk")
    return json.loads(json_chunks[0].decode("utf-8"))


def _normalise_quaternion(value: list[float]) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid zero/non-finite quaternion")
    return (array / norm).tolist()


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _transform_matrix(
    translation: list[float] | np.ndarray, rotation_xyzw: list[float]
) -> np.ndarray:
    x, y, z, w = _normalise_quaternion(rotation_xyzw)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _gltf_rotation_to_urdf(value: list[float]) -> list[float]:
    # Habitat does not silently rotate URDF coordinates.  This generated URDF
    # intentionally uses the renderer/glTF Y-up basis, so the local rotation
    # is copied in that same basis.  A future Z-up export must declare and test
    # an explicit basis transform instead of relying on an importer guess.
    return _normalise_quaternion(value)


def _gltf_translation_to_urdf(value: list[float], scale: float) -> list[float]:
    x, y, z = value
    return [scale * x, scale * y, scale * z]


def _extract_skin(document: dict[str, Any]) -> dict[str, Any]:
    skins = document.get("skins")
    nodes = document.get("nodes")
    if not isinstance(skins, list) or len(skins) != 1:
        raise ValueError("probe requires exactly one skin")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("GLB has no nodes")
    joint_indices = skins[0].get("joints")
    if not isinstance(joint_indices, list) or not joint_indices:
        raise ValueError("skin has no joints")
    if len(set(joint_indices)) != len(joint_indices):
        raise ValueError("skin has duplicate joint indices")

    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        for child_index in node.get("children", []):
            if child_index in parents:
                raise ValueError(f"node {child_index} has multiple parents")
            parents[child_index] = parent_index

    joint_set = set(joint_indices)
    roots = [index for index in joint_indices if parents.get(index) not in joint_set]
    if len(roots) != 1:
        raise ValueError(f"skin must have one joint root, found {roots}")
    root = roots[0]

    names: dict[int, str] = {}
    for index in joint_indices:
        name = nodes[index].get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"skin joint node {index} lacks a name")
        if name in names.values():
            raise ValueError(f"duplicate skin joint name {name!r}")
        names[index] = name

    ancestor_chain: list[int] = []
    cursor = parents.get(root)
    while cursor is not None:
        ancestor_chain.append(cursor)
        cursor = parents.get(cursor)
    ancestor_chain.reverse()
    external_scale = 1.0
    external_translation = np.zeros(3, dtype=np.float64)
    external_rotation = [0.0, 0.0, 0.0, 1.0]
    for index in ancestor_chain:
        node = nodes[index]
        rotation = _normalise_quaternion(
            list(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        )
        if not np.allclose(rotation, [0.0, 0.0, 0.0, 1.0], atol=1.0e-7):
            raise ValueError("probe does not yet support rotated non-joint ancestors")
        scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if not np.allclose(scale, scale[0], atol=1.0e-7) or scale[0] <= 0:
            raise ValueError("non-joint ancestor scale must be positive and uniform")
        translation = np.asarray(
            node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        external_translation += external_scale * translation
        external_scale *= float(scale[0])

    records: list[dict[str, Any]] = []
    for index in joint_indices:
        node = nodes[index]
        scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        scale_error = float(np.max(np.abs(scale - 1.0)))
        if scale_error > 5.0e-5:
            raise ValueError(
                f"joint {names[index]!r} has unsupported local scale {scale.tolist()}"
            )
        parent_index = parents.get(index)
        if index != root and parent_index not in joint_set:
            raise ValueError(f"joint {names[index]!r} leaves the skin hierarchy")
        translation = list(node.get("translation", [0.0, 0.0, 0.0]))
        rotation = _normalise_quaternion(
            list(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        )
        records.append(
            {
                "node_index": index,
                "name": names[index],
                "parent_name": names.get(parent_index),
                "gltf_local_translation": translation,
                "gltf_local_rotation_xyzw": rotation,
                "gltf_local_scale": scale.tolist(),
                "scale_normalisation_max_abs": scale_error,
                "urdf_origin_xyz": _gltf_translation_to_urdf(
                    translation, external_scale
                ),
                "urdf_joint_rotation_xyzw": _gltf_rotation_to_urdf(rotation),
            }
        )
    if not math.isclose(external_scale, 1.0, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(
            "skin ancestor scale must be baked before Habitat mapping; "
            f"found {external_scale:.17g}"
        )
    return {
        "root_node_index": root,
        "root_name": names[root],
        "external_ancestor_indices": ancestor_chain,
        "external_uniform_scale": external_scale,
        "external_translation_gltf": external_translation.tolist(),
        "external_rotation_gltf_xyzw": external_rotation,
        "joints": records,
    }


def _write_urdf(path: Path, skin: dict[str, Any]) -> None:
    lines = ['<?xml version="1.0"?>', '<robot name="avengine_skin_probe">']
    for joint in skin["joints"]:
        lines.extend(
            [
                f"  <link name={quoteattr(joint['name'])}>",
                "    <inertial>",
                '      <mass value="0.001"/>',
                '      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>',
                "    </inertial>",
                "  </link>",
            ]
        )
    for joint in skin["joints"]:
        parent_name = joint["parent_name"]
        if parent_name is None:
            continue
        xyz = " ".join(f"{value:.17g}" for value in joint["urdf_origin_xyz"])
        lines.extend(
            [
                f'  <joint name={quoteattr("joint_" + joint["name"])} type="spherical">',
                f"    <parent link={quoteattr(parent_name)}/>",
                f"    <child link={quoteattr(joint['name'])}/>",
                f'    <origin xyz="{xyz}" rpy="0 0 0"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_config(path: Path, glb_name: str, *, shader_type: str) -> dict[str, Any]:
    value = {
        "urdf_filepath": "animal.urdf",
        "render_asset": glb_name,
        "uniform_scale": 1.0,
        "mass_scale": 1.0,
        "semantic_id": 200,
        "base_type": "free",
        "inertia_source": "computed",
        "link_order": "tree_traversal",
        "render_mode": "skin",
        "shader_type": shader_type,
        "user_defined": {"avengine_native_gltf_skin_frame": True},
    }
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _file_record(
    path: Path,
    *,
    relative_to: Path | None = None,
    snapshot: Any | None = None,
) -> dict[str, Any]:
    record = {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path.resolve())
        ),
        "byte_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if snapshot is not None:
        record["snapshot"] = snapshot
    return record


def _runtime_binding_data(
    skin: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    by_name = {item["link_name"]: item for item in runtime["link_mapping"]}
    runtime_joint_order = [
        joint["name"] for joint in skin["joints"] if joint["name"] != skin["root_name"]
    ]
    return {
        "runtime_joint_order": runtime_joint_order,
        "joint_position_count": runtime["joint_position_count"],
        "quaternion_order": "xyzw",
        "links": [
            {
                "link_name": name,
                "joint_position_offset": by_name[name]["joint_position_offset"],
                "joint_position_count": by_name[name]["joint_position_count"],
            }
            for name in runtime_joint_order
        ],
    }


def _make_configuration(scene_dataset: Path) -> Any:
    import quaternion  # noqa: F401 -- required before habitat_sim in this build

    import habitat_sim
    import magnum as mn

    sensor_specs: list[Any] = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = mn.Vector2i([480, 640])
        spec.hfov = 60.0
        spec.near = 0.02
        spec.far = 20.0
        spec.gpu2gpu_transfer = False
        # SensorSpec defaults to a 1.5 m eye height.  This probe moves the
        # co-located RGB/depth/semantic optical centre explicitly, so leaving
        # that hidden offset in place would point every QA orbit past the dog.
        spec.position = mn.Vector3(0.0, 0.0, 0.0)
        spec.orientation = mn.Vector3(0.0, 0.0, 0.0)
        if uuid != "rgb":
            spec.channels = 1
        sensor_specs.append(spec)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    # Keep the first skin/rest-pose check independent of room occlusion.  The
    # later M2 formal capture reuses the M1 custom room and its single view0.
    del scene_dataset
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = 0
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def _run_habitat(
    output: Path,
    config_path: Path,
    scene_dataset: Path,
    skin: dict[str, Any],
) -> dict[str, Any]:
    import habitat_sim
    import magnum as mn
    from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs

    configuration = _make_configuration(scene_dataset)
    with habitat_sim.Simulator(configuration) as sim:
        loaded_template_ids = sim.metadata_mediator.ao_template_manager.load_configs(
            str(config_path)
        )
        handles = sim.metadata_mediator.ao_template_manager.get_template_handles(
            config_path.stem.removesuffix(".ao_config")
        )
        if len(loaded_template_ids) != 1 or len(handles) != 1:
            raise RuntimeError(
                "expected one uniquely queryable AO template, got "
                f"ids={loaded_template_ids}, handles={handles}"
            )
        ao = sim.get_articulated_object_manager().add_articulated_object_by_template_handle(
            handles[0]
        )
        if ao is None:
            raise RuntimeError("Habitat failed to instantiate the articulated object")
        ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        by_name = {joint["name"]: joint for joint in skin["joints"]}
        expected_names = set(by_name)
        actual_names = {ao.get_link_name(-1)} | {
            ao.get_link_name(link_id) for link_id in ao.get_link_ids()
        }
        if actual_names != expected_names:
            raise RuntimeError(
                f"Habitat link names differ: missing={expected_names - actual_names}, "
                f"extra={actual_names - expected_names}"
            )

        positions = np.asarray(ao.joint_positions, dtype=np.float64)
        mapping: list[dict[str, Any]] = []
        for link_id in ao.get_link_ids():
            name = ao.get_link_name(link_id)
            offset = int(ao.get_link_joint_pos_offset(link_id))
            count = int(ao.get_link_num_joint_pos(link_id))
            if count != 4 or offset < 0 or offset + count > positions.size:
                raise RuntimeError(
                    f"link {name!r} is not a valid spherical quaternion block"
                )
            positions[offset : offset + 4] = by_name[name]["urdf_joint_rotation_xyzw"]
            mapping.append(
                {
                    "link_id": int(link_id),
                    "link_name": name,
                    "joint_name": ao.get_link_joint_name(link_id),
                    "joint_position_offset": offset,
                    "joint_position_count": count,
                }
            )
        ao.joint_positions = positions

        # Habitat re-anchors skinned rendering to the matched skin-root/base
        # link.  Pelvis is therefore the AO base; its authored global bind
        # transform (and later root trajectory) belongs on the AO, while only
        # the remaining 34 joints occupy quaternion blocks.  A dummy parent
        # would be cancelled by the rig-root inverse.
        root = by_name[skin["root_name"]]
        root_translation = np.asarray(
            skin["external_translation_gltf"], dtype=np.float64
        ) + skin["external_uniform_scale"] * np.asarray(
            root["gltf_local_translation"], dtype=np.float64
        )
        root_q = root["gltf_local_rotation_xyzw"]
        ao.translation = mn.Vector3(root_translation)
        ao.rotation = mn.Quaternion(mn.Vector3(root_q[:3]), root_q[3])

        # The renderer re-anchors the model to the skin-root link.  This
        # candidate's authored Pelvis frame is not the mesh centroid, so the
        # QA-only orbit targets the measured re-anchored visual bounds rather
        # than assuming the original scene origin.
        object_node = ao.root_scene_node
        object_node.compute_cumulative_bb()
        measured_bb = object_node.cumulative_bb
        local_target = 0.5 * (
            np.asarray(measured_bb.min, dtype=np.float64)
            + np.asarray(measured_bb.max, dtype=np.float64)
        )
        state = habitat_sim.AgentState()
        bootstrap_position = local_target + [0.0, -1.5, 0.0]
        bootstrap_rotation = quat_from_two_vectors(
            np.asarray([0.0, 0.0, -1.0]), local_target - bootstrap_position
        )
        state.position = bootstrap_position
        state.rotation = bootstrap_rotation
        agent = sim.initialize_agent(0, state)
        bootstrap = sim.get_sensor_observations()
        bootstrap_rgb = np.asarray(bootstrap["rgb"])
        bootstrap_depth = np.asarray(bootstrap["depth"])
        bootstrap_semantic = np.asarray(bootstrap["semantic"])
        bootstrap_mask = bootstrap_semantic == 200
        Image.fromarray(bootstrap_rgb[..., :3].astype(np.uint8), mode="RGB").save(
            output / "qa_bootstrap_rgb.png"
        )
        bootstrap_pixel_count = int(np.count_nonzero(bootstrap_mask))
        if bootstrap_pixel_count:
            rows, columns = np.nonzero(bootstrap_mask)
            median_depth = float(np.median(bootstrap_depth[bootstrap_mask]))
            focal = 640.0 / (2.0 * math.tan(math.radians(60.0) / 2.0))
            target = np.asarray(
                [
                    bootstrap_position[0]
                    + (float(np.mean(columns)) - 319.5) * median_depth / focal,
                    bootstrap_position[1] + median_depth,
                    bootstrap_position[2]
                    + (239.5 - float(np.mean(rows))) * median_depth / focal,
                ],
                dtype=np.float64,
            )
        else:
            # A failed bootstrap is useful evidence, not grounds to discard the
            # six deterministic views.  Root/bind bugs can make one static
            # drawable bound stale while another direction remains visible.
            # Keeping the measured-bound target lets the report distinguish
            # that failure mode from a fully invisible asset.
            target = local_target.copy()
        camera_positions = [
            ("z_positive", target + [0.0, 0.0, 1.5]),
            ("z_negative", target + [0.0, 0.0, -1.5]),
            ("x_positive", target + [1.5, 0.0, 0.0]),
            ("x_negative", target + [-1.5, 0.0, 0.0]),
            ("y_positive", target + [0.0, 1.5, 0.0]),
            ("y_negative", target + [0.0, -1.5, 0.0]),
        ]
        camera_states = [
            (
                qa_id,
                position,
                quat_from_two_vectors(np.asarray([0.0, 0.0, -1.0]), target - position),
            )
            for qa_id, position in camera_positions
        ]
        candidates: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
        for qa_id, position, rotation in camera_states:
            state.position = position
            state.rotation = rotation
            agent.set_state(state, reset_sensors=True)
            observations = sim.get_sensor_observations()
            candidate_rgb = np.asarray(observations["rgb"])
            candidate_depth = np.asarray(observations["depth"])
            candidate_semantic = np.asarray(observations["semantic"])
            count = int(np.count_nonzero(candidate_semantic == 200))
            candidates.append(
                (count, qa_id, candidate_rgb, candidate_depth, candidate_semantic)
            )
            Image.fromarray(candidate_rgb[..., :3].astype(np.uint8), mode="RGB").save(
                output / f"qa_{qa_id}_rgb.png"
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, selected_qa_id, rgb, depth, semantic = candidates[0]
        Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB").save(
            output / "rest_rgb.png"
        )
        finite_depth = depth[np.isfinite(depth)]
        maximum_depth = float(finite_depth.max()) if finite_depth.size else 1.0
        depth_u16 = np.clip(depth / max(maximum_depth, 1.0e-6), 0.0, 1.0)
        Image.fromarray((depth_u16 * 65535).astype(np.uint16), mode="I;16").save(
            output / "rest_depth.png"
        )
        semantic_u16 = np.asarray(semantic, dtype=np.uint16)
        Image.fromarray(semantic_u16, mode="I;16").save(output / "rest_semantic.png")
        dog_depth = depth[semantic == 200]
        root_node = ao.get_link_scene_node(-1)
        aabb = ao.aabb
        cumulative_bb = measured_bb
        link_transforms = {
            ao.get_link_name(link_id): np.asarray(
                ao.get_link_scene_node(link_id).transformation
            )
            .astype(np.float64)
            .tolist()
            for link_id in [-1, *ao.get_link_ids()]
        }
        expected_link_transforms: dict[str, np.ndarray] = {}
        for joint in skin["joints"]:
            local = _transform_matrix(
                joint["urdf_origin_xyz"], joint["urdf_joint_rotation_xyzw"]
            )
            parent_name = joint["parent_name"]
            if parent_name is None:
                expected = _transform_matrix(root_translation, root_q)
            else:
                expected = expected_link_transforms[parent_name] @ local
            expected_link_transforms[joint["name"]] = expected
        link_transform_errors = {
            name: float(
                np.max(
                    np.abs(
                        np.asarray(link_transforms[name], dtype=np.float64) - expected
                    )
                )
            )
            for name, expected in expected_link_transforms.items()
        }
        return {
            "template_handle": handles[0],
            "object_id": int(ao.object_id),
            "semantic_id": int(object_node.semantic_id),
            "aabb": {"min": list(aabb.min), "max": list(aabb.max)},
            "cumulative_bb": {
                "min": list(cumulative_bb.min),
                "max": list(cumulative_bb.max),
            },
            "root_link_name": ao.get_link_name(-1),
            "root_translation": list(root_node.translation),
            "root_rotation_xyzw": [
                *list(ao.rotation.vector),
                float(ao.rotation.scalar),
            ],
            "joint_position_count": int(positions.size),
            "link_mapping": sorted(mapping, key=lambda item: item["link_id"]),
            "link_transforms": link_transforms,
            "maximum_link_bind_transform_error": max(link_transform_errors.values()),
            "link_bind_transform_errors": link_transform_errors,
            "observation": {
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
                "semantic_shape": list(semantic.shape),
                "dog_semantic_pixel_count": int(np.count_nonzero(semantic == 200)),
                "dog_depth_m": {
                    "minimum": float(np.min(dog_depth)) if dog_depth.size else None,
                    "median": float(np.median(dog_depth)) if dog_depth.size else None,
                    "maximum": float(np.max(dog_depth)) if dog_depth.size else None,
                },
                "selected_qa_camera": selected_qa_id,
                "bootstrap_semantic_pixel_count": bootstrap_pixel_count,
                "measured_bound_target": local_target.tolist(),
                "qa_target": target.tolist(),
                "qa_semantic_pixel_counts": {
                    qa_id: count for count, qa_id, *_ in candidates
                },
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                "semantic_sha256": hashlib.sha256(semantic.tobytes()).hexdigest(),
            },
            "agent_state": {
                "position": np.asarray(agent.state.position).tolist(),
                "rotation_xyzw": list(quat_to_coeffs(agent.state.rotation)),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-dataset", type=Path, required=True)
    parser.add_argument(
        "--shader-type",
        choices=("phong", "pbr"),
        default="phong",
        help="Explicit Habitat AO shader; formal M2 remains phong by default.",
    )
    args = parser.parse_args()

    source = args.input_glb.resolve()
    output = args.output_dir.resolve()
    scene_dataset = args.scene_dataset.resolve()
    if not source.is_file() or not scene_dataset.is_file():
        raise FileNotFoundError("input GLB and scene dataset must exist")
    output.mkdir(parents=True, exist_ok=True)
    document = _read_glb_json(source)
    skin = _extract_skin(document)

    copied_glb = output / "visual.glb"
    shutil.copyfile(source, copied_glb)
    urdf = output / "animal.urdf"
    config = output / "animal.ao_config.json"
    _write_urdf(urdf, skin)
    config_value = _write_config(config, copied_glb.name, shader_type=args.shader_type)
    runtime = _run_habitat(output, config, scene_dataset, skin)
    runtime_binding = _runtime_binding_data(skin, runtime)
    runtime_binding_path = output / "habitat_runtime_binding.json"
    runtime_binding_path.write_text(
        json.dumps(runtime_binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observation_paths = [
        output / relative
        for relative in (
            "qa_bootstrap_rgb.png",
            "qa_x_negative_rgb.png",
            "qa_x_positive_rgb.png",
            "qa_y_negative_rgb.png",
            "qa_y_positive_rgb.png",
            "qa_z_negative_rgb.png",
            "qa_z_positive_rgb.png",
            "rest_depth.png",
            "rest_rgb.png",
            "rest_semantic.png",
        )
    ]
    result = {
        "schema": "avengine_m2_habitat_skin_rest_probe_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "evidence_scope": {
            "local_report_claim": "artifact_integrity_only",
            "trusted_runtime_attestation": False,
            "runtime_execution_conclusion_source": "external_capture_audit_only",
        },
        "producer_source_integrity": _file_record(Path(__file__)),
        "render_configuration_integrity": {
            "configured_shader_type": args.shader_type,
            "ao_config_artifact": _file_record(
                config,
                relative_to=output,
                snapshot=config_value,
            ),
        },
        "runtime_artifact_integrity": {
            "runtime_binding_artifact": _file_record(
                runtime_binding_path,
                relative_to=output,
                snapshot=runtime_binding,
            ),
            "observation_artifacts": [
                _file_record(path, relative_to=output) for path in observation_paths
            ],
        },
        "input": {
            "path": str(source),
            "byte_size": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "normalisation": skin,
        "runtime": runtime,
        "notes": [
            "Local hashes bind the AO config, runtime binding, and observations.",
            "This mutable local report is not a trusted Habitat execution attestation.",
            "Runtime execution conclusions require a separately retained capture/audit.",
        ],
    }
    observation = runtime["observation"]
    qa_counts = observation["qa_semantic_pixel_counts"]
    result["gates"] = {
        "bootstrap_visible": observation["bootstrap_semantic_pixel_count"]
        >= _MIN_QA_SEMANTIC_PIXELS,
        "all_six_orbit_views_visible": len(qa_counts) == 6
        and all(count >= _MIN_QA_SEMANTIC_PIXELS for count in qa_counts.values()),
        "co_located_modalities": observation["rgb_shape"][:2]
        == observation["depth_shape"]
        == observation["semantic_shape"],
        "runtime_joint_mapping_complete": runtime["joint_position_count"]
        == 4 * (len(skin["joints"]) - 1)
        and len(runtime["link_mapping"]) == len(skin["joints"]) - 1,
        "runtime_link_bind_alignment": runtime["maximum_link_bind_transform_error"]
        <= _MAX_LINK_BIND_ERROR,
    }
    result["thresholds"] = {
        "minimum_semantic_pixels_per_view": _MIN_QA_SEMANTIC_PIXELS,
        "maximum_link_bind_transform_error": _MAX_LINK_BIND_ERROR,
    }
    result["status"] = "pass" if all(result["gates"].values()) else "fail"
    (output / "probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
