#!/usr/bin/env python3
"""Prove one bounded local-translation-plus-rotation Habitat AO encoding.

This is a research spike, not a production compiler or an asset admission
gate.  It creates a two-joint glTF skin whose ``child_bone`` animation target
has both a non-axis-aligned local translation and a local rotation.  The
corresponding URDF expands that one transform into three bounded prismatic
dummy links followed by the original, same-named spherical link::

    root_bone -> __tx -> __ty -> __tz -> child_bone

The pinned Habitat runtime is then queried for the real joint-position
offsets.  The spike writes each block by link name and verifies both exact
joint-position readback and the measured root-to-child link matrix against
the float32 glTF target ``T(t) R(q)``.

The generated skin is intentionally tiny.  A before/after semantic render is
used only as supporting evidence that the additional dummy links do not stop
the same-named ``child_bone`` skin link from driving its weighted mesh.  This
does not qualify a general translation-animation compiler.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m2.glb import decode_accessor, load_glb
from avengine.m2.glb_write import build_glb


SCHEMA = "avengine_m2_habitat_local_tr_spike_v1"
ROOT_LINK = "root_bone"
CHILD_LINK = "child_bone"
DUMMY_LINKS = (
    "__avengine_child_bone_tx",
    "__avengine_child_bone_ty",
    "__avengine_child_bone_tz",
)
TARGET_TRANSLATION = (0.23, -0.17, 0.31)
PRISMATIC_LIMIT_LOWER = -0.75
PRISMATIC_LIMIT_UPPER = 0.75
FLOAT32_ATOL = 5.0e-6
SEMANTIC_ID = 200


def target_rotation_xyzw() -> tuple[float, float, float, float]:
    """Return a deterministic rotation about a genuinely oblique axis."""

    axis = np.asarray([1.0, 2.0, -1.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * 0.83
    vector = axis * math.sin(half_angle)
    return (
        float(vector[0]),
        float(vector[1]),
        float(vector[2]),
        float(math.cos(half_angle)),
    )


def _normalise_quaternion(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(array))
    if norm < 1.0e-12:
        raise ValueError("quaternion must be non-zero")
    return array / norm


def transform_matrix(
    translation: Sequence[float], rotation_xyzw: Sequence[float]
) -> np.ndarray:
    """Return a conventional column-vector glTF ``T R`` matrix."""

    translation_array = np.asarray(translation, dtype=np.float64)
    if translation_array.shape != (3,) or not np.all(np.isfinite(translation_array)):
        raise ValueError("translation must contain three finite values")
    x, y, z, w = _normalise_quaternion(rotation_xyzw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    result[:3, 3] = translation_array
    return result


def expanded_chain_matrix(
    translation: Sequence[float], rotation_xyzw: Sequence[float]
) -> np.ndarray:
    """Compose the exact ``Px Py Pz Spherical`` chain used by the URDF."""

    translation_array = np.asarray(translation, dtype=np.float64)
    if translation_array.shape != (3,):
        raise ValueError("translation must contain exactly three values")
    result = np.eye(4, dtype=np.float64)
    for axis, distance in enumerate(translation_array):
        local = np.eye(4, dtype=np.float64)
        local[axis, 3] = float(distance)
        result = result @ local
    return result @ transform_matrix((0.0, 0.0, 0.0), rotation_xyzw)


def _append_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    component_type: int,
    element_type: str,
    values: Sequence[Sequence[int | float]],
    struct_format: str,
    target: int | None = None,
    minimum: Sequence[float] | None = None,
    maximum: Sequence[float] | None = None,
) -> int:
    packer = struct.Struct("<" + struct_format)
    alignment = max(1, min(4, packer.size))
    while len(binary) % alignment:
        binary.append(0)
    offset = len(binary)
    for value in values:
        binary.extend(packer.pack(*value))
    view_index = len(document.setdefault("bufferViews", []))
    view: dict[str, Any] = {
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(binary) - offset,
    }
    if target is not None:
        view["target"] = target
    document["bufferViews"].append(view)
    accessor_index = len(document.setdefault("accessors", []))
    accessor: dict[str, Any] = {
        "bufferView": view_index,
        "componentType": component_type,
        "count": len(values),
        "type": element_type,
    }
    if minimum is not None:
        accessor["min"] = list(minimum)
    if maximum is not None:
        accessor["max"] = list(maximum)
    document["accessors"].append(accessor)
    return accessor_index


def build_probe_glb() -> bytes:
    """Build a deterministic two-bone, one-mesh animated GLB fixture."""

    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "AVEngine bounded Habitat local-TR spike",
        },
        "scene": 0,
        "scenes": [{"nodes": [0, 2]}],
        "nodes": [
            {
                "name": ROOT_LINK,
                "children": [1],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "name": CHILD_LINK,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            },
            {"name": "probe_mesh", "mesh": 0, "skin": 0},
        ],
        "materials": [
            {
                "name": "probe_matte_red",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.05, 0.03, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
    }
    binary = bytearray()

    positions = [
        (-0.25, -0.10, 0.0),
        (0.35, -0.10, 0.0),
        (0.35, 0.10, 0.0),
        (-0.25, 0.10, 0.0),
        (0.62, 0.00, 0.0),
    ]
    position_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=positions,
        struct_format="fff",
        target=34962,
        minimum=(-0.25, -0.10, 0.0),
        maximum=(0.62, 0.10, 0.0),
    )
    normal_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[(0.0, 0.0, 1.0)] * len(positions),
        struct_format="fff",
        target=34962,
    )
    joints_accessor = _append_accessor(
        document,
        binary,
        component_type=5121,
        element_type="VEC4",
        values=[(1, 0, 0, 0)] * len(positions),
        struct_format="BBBB",
        target=34962,
    )
    weights_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC4",
        values=[(1.0, 0.0, 0.0, 0.0)] * len(positions),
        struct_format="ffff",
        target=34962,
    )
    index_accessor = _append_accessor(
        document,
        binary,
        component_type=5123,
        element_type="SCALAR",
        values=[(value,) for value in (0, 1, 2, 0, 2, 3, 1, 4, 2)],
        struct_format="H",
        target=34963,
        minimum=(0,),
        maximum=(4,),
    )
    inverse_bind_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="MAT4",
        values=[
            tuple(np.eye(4, dtype=np.float32).reshape(-1, order="F")),
            tuple(np.eye(4, dtype=np.float32).reshape(-1, order="F")),
        ],
        struct_format="f" * 16,
    )
    time_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="SCALAR",
        values=[(0.0,), (1.0,)],
        struct_format="f",
        minimum=(0.0,),
        maximum=(1.0,),
    )
    translation_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[(0.0, 0.0, 0.0), TARGET_TRANSLATION],
        struct_format="fff",
    )
    rotation_accessor = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC4",
        values=[(0.0, 0.0, 0.0, 1.0), target_rotation_xyzw()],
        struct_format="ffff",
    )

    document["meshes"] = [
        {
            "name": "child_weighted_arrow",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": position_accessor,
                        "NORMAL": normal_accessor,
                        "JOINTS_0": joints_accessor,
                        "WEIGHTS_0": weights_accessor,
                    },
                    "indices": index_accessor,
                    "material": 0,
                    "mode": 4,
                }
            ],
        }
    ]
    document["skins"] = [
        {
            "name": "two_bone_probe_skin",
            "skeleton": 0,
            "joints": [0, 1],
            "inverseBindMatrices": inverse_bind_accessor,
        }
    ]
    document["animations"] = [
        {
            "name": "LocalTRTarget",
            "samplers": [
                {"input": time_accessor, "output": translation_accessor},
                {"input": time_accessor, "output": rotation_accessor},
            ],
            "channels": [
                {"sampler": 0, "target": {"node": 1, "path": "translation"}},
                {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
            ],
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    return build_glb(document, binary)


def render_probe_urdf() -> str:
    """Return the bounded Px/Py/Pz/spherical link expansion."""

    link_names = (ROOT_LINK, *DUMMY_LINKS, CHILD_LINK)
    lines = ['<?xml version="1.0"?>', '<robot name="avengine_local_tr_spike">']
    for name in link_names:
        lines.extend(
            [
                f'  <link name="{name}">',
                "    <inertial>",
                '      <mass value="0.001"/>',
                '      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>',
                "    </inertial>",
                "  </link>",
            ]
        )
    parents = (ROOT_LINK, DUMMY_LINKS[0], DUMMY_LINKS[1])
    children = DUMMY_LINKS
    axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    axis_names = ("tx", "ty", "tz")
    for parent, child, axis, axis_name in zip(
        parents, children, axes, axis_names, strict=True
    ):
        lines.extend(
            [
                f'  <joint name="joint_child_bone_{axis_name}" type="prismatic">',
                f'    <parent link="{parent}"/>',
                f'    <child link="{child}"/>',
                '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>',
                (
                    f'    <limit lower="{PRISMATIC_LIMIT_LOWER}" '
                    f'upper="{PRISMATIC_LIMIT_UPPER}" effort="10" velocity="10"/>'
                ),
                "  </joint>",
            ]
        )
    lines.extend(
        [
            '  <joint name="joint_child_bone_rotation" type="spherical">',
            f'    <parent link="{DUMMY_LINKS[-1]}"/>',
            f'    <child link="{CHILD_LINK}"/>',
            '    <origin xyz="0 0 0" rpy="0 0 0"/>',
            "  </joint>",
            "</robot>",
        ]
    )
    return "\n".join(lines) + "\n"


def probe_ao_config() -> dict[str, Any]:
    return {
        "urdf_filepath": "local_tr_probe.urdf",
        "render_asset": "local_tr_probe.glb",
        "uniform_scale": 1.0,
        "mass_scale": 1.0,
        "semantic_id": SEMANTIC_ID,
        "base_type": "free",
        "inertia_source": "computed",
        "link_order": "tree_traversal",
        "render_mode": "skin",
        "shader_type": "flat",
        "user_defined": {"avengine_native_gltf_skin_frame": True},
    }


def _encoded_gltf_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document = load_glb(path)
    animation = document.json["animations"][0]
    outputs: dict[str, int] = {}
    for channel in animation["channels"]:
        target = channel["target"]
        if target["node"] != 1:
            raise RuntimeError("probe animation target node changed")
        outputs[target["path"]] = animation["samplers"][channel["sampler"]]["output"]
    if set(outputs) != {"translation", "rotation"}:
        raise RuntimeError("probe GLB must encode translation and rotation channels")
    translation = np.asarray(
        decode_accessor(document, outputs["translation"]).values[-1],
        dtype=np.float64,
    )
    rotation = _normalise_quaternion(
        decode_accessor(document, outputs["rotation"]).values[-1]
    )
    return translation, rotation


def _make_configuration(gpu_device_id: int) -> Any:
    import quaternion  # noqa: F401 -- required before habitat_sim in this build

    import habitat_sim
    import magnum as mn

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "semantic"
    sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = mn.Vector2i([256, 256])
    sensor.hfov = 60.0
    sensor.near = 0.02
    sensor.far = 10.0
    sensor.channels = 1
    sensor.gpu2gpu_transfer = False
    sensor.position = mn.Vector3(0.0, 0.0, 0.0)
    sensor.orientation = mn.Vector3(0.0, 0.0, 0.0)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = gpu_device_id
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.action_space = {}
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def _matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise RuntimeError("Habitat returned an invalid 4x4 link matrix")
    return array


def _mask_record(mask: np.ndarray) -> dict[str, Any]:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return {"pixel_count": 0, "centroid_rc": None, "bbox_rc": None}
    return {
        "pixel_count": int(rows.size),
        "centroid_rc": [float(np.mean(rows)), float(np.mean(columns))],
        "bbox_rc": [
            int(rows.min()),
            int(columns.min()),
            int(rows.max()),
            int(columns.max()),
        ],
    }


def _run_runtime(
    output: Path,
    config_path: Path,
    gltf_translation: np.ndarray,
    gltf_rotation: np.ndarray,
    *,
    gpu_device_id: int,
) -> dict[str, Any]:
    import habitat_sim

    configuration = _make_configuration(gpu_device_id)
    with habitat_sim.Simulator(configuration) as sim:
        manager = sim.metadata_mediator.ao_template_manager
        loaded = manager.load_configs(str(config_path))
        handles = manager.get_template_handles("local_tr_probe")
        if len(loaded) != 1 or len(handles) != 1:
            raise RuntimeError(
                f"expected one AO template, got ids={loaded}, handles={handles}"
            )
        ao = sim.get_articulated_object_manager().add_articulated_object_by_template_handle(
            handles[0]
        )
        if ao is None:
            raise RuntimeError("Habitat did not instantiate the local-TR AO")
        ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        link_ids = {ao.get_link_name(-1): -1}
        link_ids.update(
            {ao.get_link_name(link_id): int(link_id) for link_id in ao.get_link_ids()}
        )
        expected_links = {ROOT_LINK, CHILD_LINK, *DUMMY_LINKS}
        if set(link_ids) != expected_links:
            raise RuntimeError(
                "Habitat link names changed: "
                f"missing={sorted(expected_links - set(link_ids))}, "
                f"extra={sorted(set(link_ids) - expected_links)}"
            )

        layout: list[dict[str, Any]] = []
        for name in (*DUMMY_LINKS, CHILD_LINK):
            link_id = link_ids[name]
            layout.append(
                {
                    "link_name": name,
                    "link_id": link_id,
                    "joint_name": ao.get_link_joint_name(link_id),
                    "joint_type": str(ao.get_link_joint_type(link_id)),
                    "joint_position_offset": int(ao.get_link_joint_pos_offset(link_id)),
                    "joint_position_count": int(ao.get_link_num_joint_pos(link_id)),
                }
            )
        by_name = {item["link_name"]: item for item in layout}
        for name in DUMMY_LINKS:
            if by_name[name]["joint_position_count"] != 1:
                raise RuntimeError(f"{name} is not one prismatic position block")
        if by_name[CHILD_LINK]["joint_position_count"] != 4:
            raise RuntimeError("child_bone is not one spherical quaternion block")

        positions_before = np.asarray(ao.joint_positions, dtype=np.float64)
        requested = positions_before.copy()
        for name, value in zip(DUMMY_LINKS, gltf_translation, strict=True):
            offset = by_name[name]["joint_position_offset"]
            requested[offset] = value
        child_offset = by_name[CHILD_LINK]["joint_position_offset"]
        requested[child_offset : child_offset + 4] = gltf_rotation

        lower, upper = (
            np.asarray(value, dtype=np.float64) for value in ao.joint_position_limits
        )
        limit_evidence: list[dict[str, Any]] = []
        for name, target in zip(DUMMY_LINKS, gltf_translation, strict=True):
            offset = by_name[name]["joint_position_offset"]
            if not (
                math.isclose(lower[offset], PRISMATIC_LIMIT_LOWER, abs_tol=1.0e-7)
                and math.isclose(upper[offset], PRISMATIC_LIMIT_UPPER, abs_tol=1.0e-7)
                and lower[offset] <= target <= upper[offset]
            ):
                raise RuntimeError(f"runtime limits for {name} differ from the URDF")
            limit_evidence.append(
                {
                    "link_name": name,
                    "joint_position_offset": offset,
                    "lower": float(lower[offset]),
                    "upper": float(upper[offset]),
                    "target": float(target),
                    "target_within_limits": True,
                }
            )

        state = habitat_sim.AgentState()
        state.position = [0.0, 0.0, 2.5]
        state.rotation = [0.0, 0.0, 0.0, 1.0]
        agent = sim.initialize_agent(0, state)
        del agent
        semantic_before = np.asarray(sim.get_sensor_observations()["semantic"])
        mask_before = semantic_before == SEMANTIC_ID

        ao.joint_positions = requested
        readback = np.asarray(ao.joint_positions, dtype=np.float64)
        readback_max_abs_error = float(np.max(np.abs(readback - requested)))
        if readback_max_abs_error > FLOAT32_ATOL:
            raise RuntimeError(
                "Habitat joint-position readback exceeded float32 tolerance: "
                f"{readback_max_abs_error:.9g} > {FLOAT32_ATOL:.9g}"
            )

        root_absolute = _matrix(ao.get_link_scene_node(-1).absolute_transformation())
        child_absolute = _matrix(
            ao.get_link_scene_node(link_ids[CHILD_LINK]).absolute_transformation()
        )
        root_from_child = np.linalg.inv(root_absolute) @ child_absolute
        expected = transform_matrix(gltf_translation, gltf_rotation)
        expanded = expanded_chain_matrix(gltf_translation, gltf_rotation)
        expansion_error = float(np.max(np.abs(expanded - expected)))
        matrix_error = float(np.max(np.abs(root_from_child - expected)))
        if expansion_error > 1.0e-12:
            raise RuntimeError("pure URDF chain algebra differs from glTF T R")
        if matrix_error > FLOAT32_ATOL:
            raise RuntimeError(
                "Habitat child link matrix exceeded float32 tolerance: "
                f"{matrix_error:.9g} > {FLOAT32_ATOL:.9g}"
            )

        semantic_target = np.asarray(sim.get_sensor_observations()["semantic"])
        mask_target = semantic_target == SEMANTIC_ID
        before_record = _mask_record(mask_before)
        target_record = _mask_record(mask_target)
        if (
            before_record["pixel_count"] < 32
            or target_record["pixel_count"] < 32
            or np.array_equal(mask_before, mask_target)
        ):
            raise RuntimeError(
                "same-name skin render did not produce two distinct visible masks"
            )
        Image.fromarray((mask_before.astype(np.uint8) * 255), mode="L").save(
            output / "semantic_rest_mask.png"
        )
        Image.fromarray((mask_target.astype(np.uint8) * 255), mode="L").save(
            output / "semantic_target_mask.png"
        )

        import habitat_sim._ext.habitat_sim_bindings as native_bindings

        native_path = Path(native_bindings.__file__).resolve()
        return {
            "habitat_version": getattr(habitat_sim, "__version__", None),
            "habitat_python_module": str(Path(habitat_sim.__file__).resolve()),
            "native_binding": {
                "path": str(native_path),
                "byte_size": native_path.stat().st_size,
                "sha256": sha256_file(native_path),
            },
            "template_handle": handles[0],
            "object_id": int(ao.object_id),
            "root_link_name": ao.get_link_name(-1),
            "joint_position_count": int(readback.size),
            "link_layout": layout,
            "prismatic_limits": limit_evidence,
            "joint_positions_before": positions_before.tolist(),
            "joint_positions_requested": requested.tolist(),
            "joint_positions_readback": readback.tolist(),
            "joint_positions_readback_max_abs_error": readback_max_abs_error,
            "root_absolute_matrix": root_absolute.tolist(),
            "child_absolute_matrix": child_absolute.tolist(),
            "root_from_child_measured_matrix": root_from_child.tolist(),
            "gltf_target_matrix": expected.tolist(),
            "expanded_chain_matrix": expanded.tolist(),
            "expanded_chain_vs_gltf_max_abs_error": expansion_error,
            "habitat_matrix_vs_gltf_max_abs_error": matrix_error,
            "float32_absolute_tolerance": FLOAT32_ATOL,
            "same_name_skin_render": {
                "gltf_skin_joint_names": [ROOT_LINK, CHILD_LINK],
                "urdf_link_names": [ROOT_LINK, *DUMMY_LINKS, CHILD_LINK],
                "dummy_links_absent_from_gltf_skin": True,
                "final_skin_link_name": CHILD_LINK,
                "render_mode": "skin",
                "rest_mask": before_record,
                "target_mask": target_record,
                "masks_are_distinct": True,
                "same_named_weighted_mesh_follow_observed": True,
            },
        }


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(output: Path, *, gpu_device_id: int = 0) -> Path:
    """Generate the fixture, run Habitat and publish one bound evidence file."""

    output = output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            f"refusing to replace existing output directory: {output}"
        ) from exc

    glb_path = output / "local_tr_probe.glb"
    urdf_path = output / "local_tr_probe.urdf"
    config_path = output / "local_tr_probe.ao_config.json"
    glb_path.write_bytes(build_probe_glb())
    urdf_path.write_text(render_probe_urdf(), encoding="utf-8")
    config_path.write_text(
        json.dumps(probe_ao_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gltf_translation, gltf_rotation = _encoded_gltf_target(glb_path)
    if not np.all(np.abs(gltf_translation) > 1.0e-6):
        raise RuntimeError("probe translation must have three non-zero components")
    if not np.all(np.abs(gltf_rotation[:3]) > 1.0e-6):
        raise RuntimeError("probe rotation axis must be oblique")
    runtime = _run_runtime(
        output,
        config_path,
        gltf_translation,
        gltf_rotation,
        gpu_device_id=gpu_device_id,
    )

    artifact_names = [
        "local_tr_probe.glb",
        "local_tr_probe.urdf",
        "local_tr_probe.ao_config.json",
        "semantic_rest_mask.png",
        "semantic_target_mask.png",
    ]
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "scope": "bounded two-bone Habitat local T+R capability spike",
        "python_executable": str(Path(sys.executable).resolve()),
        "tool": {
            "path": str(Path(__file__).resolve()),
            "byte_size": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "target": {
            "animation": "LocalTRTarget",
            "sample_time_seconds": 1.0,
            "child_link": CHILD_LINK,
            "translation_float32_readback": gltf_translation.tolist(),
            "rotation_xyzw_float32_readback": gltf_rotation.tolist(),
            "composition": "T(translation) @ R(rotation_xyzw)",
        },
        "encoding": {
            "chain": [ROOT_LINK, *DUMMY_LINKS, CHILD_LINK],
            "joint_types": ["prismatic_x", "prismatic_y", "prismatic_z", "spherical"],
            "prismatic_limit_lower": PRISMATIC_LIMIT_LOWER,
            "prismatic_limit_upper": PRISMATIC_LIMIT_UPPER,
            "spherical_quaternion_order": "xyzw",
            "offset_assignment": "measured Habitat link name -> joint-position offset",
        },
        "runtime": runtime,
        "artifacts": [
            _file_record(output / name, relative_to=output) for name in artifact_names
        ],
        "claims_proven": [
            "three bounded prismatic joint positions plus one spherical xyzw block reproduce one non-axis-aligned glTF child local T(t)R(q)",
            "Habitat joint-position readback equals the name-addressed requested vector within the declared float32 tolerance",
            "the measured Habitat root-to-child link matrix equals the float32 glTF T(t)R(q) matrix within the declared tolerance",
            "three extra dummy links coexist with root_bone and child_bone in skin render mode, and the same-named weighted child mesh remains visible and changes with the target pose",
        ],
        "claims_not_proven": [
            "a production compiler for arbitrary glTF translation channels",
            "automatic dummy-link naming and collision handling across arbitrary animal skeletons",
            "full vertex-by-vertex deformation equivalence or Habitat playback of the glTF animation clock",
            "stability under dynamic physics stepping, contacts, or motors",
            "asset admission or species-general motion quality",
        ],
    }
    evidence["content_sha256"] = canonical_json_sha256(evidence)
    evidence_path = output / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="New directory for generated fixture files and evidence.json",
    )
    parser.add_argument("--gpu-device-id", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = run(Path(args.output), gpu_device_id=args.gpu_device_id)
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
