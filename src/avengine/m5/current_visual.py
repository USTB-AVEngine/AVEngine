"""Current installed-prefix, visual-only M5 research capture.

This module intentionally does not reuse the historical M5 canary writer.
That writer owns the retained two-Beagle counterfactual, RLR audio, and its
v1 evidence reader.  The entrypoint here has a smaller purpose: render a
fresh MP3D visual preview only after a 75-state articulated request has been
authored for the same MP3D room.

It accepts an installed Habitat prefix plus separately supplied MP3D and
Magnum inputs.  There is no checkout-oriented compatibility branch.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import resolve_declared_path, write_json
from avengine.contracts.transforms import normalized_quaternion_xyzw, transform_error
from avengine.m1.contracts import (
    ValidatedM1Inputs,
    load_and_validate_inputs as load_m1_inputs,
    validate_loaded_scene_asset_graph,
    validate_scene_asset_graph,
)
from avengine.m1.habitat_capture import (
    InstalledHabitatRuntime,
    VISUAL_SENSOR_TYPES,
    _logical_listener_pose,
    _state_snapshot,
    prepare_installed_habitat_runtime,
)
from avengine.m2.contracts import (
    FORMAL_MODALITIES,
    ValidatedM2Inputs,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.m2.habitat import HabitatLinkJointBlock, bind_habitat_link_layout
from avengine.m2.habitat_capture import (
    HabitatCaptureError,
    _apply_root_with_habitat,
    compile_frame_applications,
    load_runtime_asset_bundle,
    validate_capture_context,
)


CURRENT_ACTOR_IDS = ("actor0", "actor1")
CURRENT_SOURCE_IDS = ("source0", "source1")
CURRENT_SEMANTIC_IDS = (210, 211)
CURRENT_OBJECT_ID_MODALITY = "object_id"
CURRENT_OBJECT_ID_SENSOR_UUID = "rig_object_id"
CURRENT_ACTOR_OFFSETS_M = ((0.0, 0.0, 0.7), (0.25, 0.0, -0.7))
CURRENT_EMITTER_LINK_NAMES = ("beagle Xtra Mouth", "beagle Xtra Mouth")
CURRENT_POSE_COMPONENT_ORDER = (
    "translation_x_m",
    "translation_y_m",
    "translation_z_m",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
)
_ROOT_READBACK_ATOL = 2.0e-6
_JOINT_READBACK_ATOL = 2.0e-6
_WORLD_TIME_READBACK_ATOL = 1.0e-12


class CurrentVisualError(RuntimeError):
    """The current visual-only M5 route cannot safely run."""


def _pose_vector(pose: Mapping[str, Any], *, owner: str) -> np.ndarray:
    """Pack one already-read pose without replacing it with a planned pose."""

    if not isinstance(pose, Mapping):
        raise CurrentVisualError(f"{owner} must be a pose object")
    translation = np.asarray(pose.get("translation_m"), dtype=np.float64)
    quaternion = np.asarray(pose.get("rotation_xyzw"), dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise CurrentVisualError(
            f"{owner} must contain translation[3] and quaternion_xyzw[4]"
        )
    value = np.concatenate((translation, quaternion))
    if not np.all(np.isfinite(value)):
        raise CurrentVisualError(f"{owner} is not finite")
    return value


def _m1_capture_identity(room_inputs: ValidatedM1Inputs) -> dict[str, Any]:
    """Snapshot value authority that must survive later M1 path rewrites."""

    request = room_inputs.request
    source_axis_order = [source["source_id"] for source in request["sources"]]
    if source_axis_order != list(CURRENT_SOURCE_IDS):
        raise CurrentVisualError(
            "current visual source axes differ from the validated M1 request"
        )
    return deepcopy(
        {
            "request_id": request["request_id"],
            "room_id": request["room_id"],
            "primary_camera_rig": request["primary_camera_rig"],
            "listener": request["listener"],
            "source_axis_order": source_axis_order,
            "qa_views": request.get("qa_views", []),
        }
    )


def _snapshot_world_time(snapshot: Mapping[str, Any], *, owner: str) -> float:
    if not isinstance(snapshot, Mapping):
        raise CurrentVisualError(f"{owner} must be a state snapshot")
    try:
        value = float(snapshot["world_time_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise CurrentVisualError(f"{owner} has no numeric world time") from error
    if not np.isfinite(value):
        raise CurrentVisualError(f"{owner} world time is not finite")
    return value


def _require_render_snapshot_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    sensor_order: Sequence[str],
    frame_index: int,
) -> tuple[float, float]:
    """Prove that a visual render did not move the camera rig or its sensors."""

    before_time = _snapshot_world_time(
        before, owner=f"frame {frame_index} pre-render snapshot"
    )
    after_time = _snapshot_world_time(
        after, owner=f"frame {frame_index} post-render snapshot"
    )
    if abs(after_time - before_time) > _WORLD_TIME_READBACK_ATOL:
        raise CurrentVisualError(
            f"frame {frame_index} render changed Habitat world time"
        )
    if transform_error(before["agent"], after["agent"]) > _ROOT_READBACK_ATOL:
        raise CurrentVisualError(
            f"frame {frame_index} render changed actual agent/camera pose"
        )
    for sensor_uuid in sensor_order:
        if (
            transform_error(
                before["sensors"][sensor_uuid],
                after["sensors"][sensor_uuid],
            )
            > _ROOT_READBACK_ATOL
        ):
            raise CurrentVisualError(
                f"frame {frame_index} render changed actual sensor {sensor_uuid!r} pose"
            )
    return before_time, after_time


def _require_explicit_runtime_inputs(
    *,
    runtime_prefix: str | Path | None,
    mp3d_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> None:
    missing = [
        option
        for option, value in (
            ("--runtime-prefix", runtime_prefix),
            ("--mp3d-root", mp3d_root),
            ("--magnum-python-site", magnum_python_site),
        )
        if value is None or not str(value).strip()
    ]
    if missing:
        raise CurrentVisualError(
            "capture-current-visual requires explicit " + ", ".join(missing)
        )


def _current_mp3d_room_error(room_inputs: ValidatedM1Inputs) -> str | None:
    room = room_inputs.room
    if room.get("room_kind") != "habitat_native":
        return "capture-current-visual supports only habitat_native MP3D rooms"
    scene = room.get("scene")
    if not isinstance(scene, Mapping) or scene.get("scene_id_kind") != "path":
        return "current MP3D visual capture requires a path-backed scene"
    raw_paths = [
        scene.get("scene_id"),
        scene.get("dataset_config_path"),
        scene.get("navmesh_path"),
    ]
    assets = room.get("assets")
    if not isinstance(assets, list):
        return "current MP3D visual capture requires declared room assets"
    raw_paths.extend(
        asset.get("path") for asset in assets if isinstance(asset, Mapping)
    )
    if not raw_paths or any(
        not isinstance(path, str) or "${AVENGINE_MP3D_ROOT}" not in path
        for path in raw_paths
    ):
        return (
            "current MP3D visual capture requires every selected scene path to "
            "use ${AVENGINE_MP3D_ROOT}"
        )
    return None


def _same_room_reason(
    m2_inputs: ValidatedM2Inputs, room_inputs: ValidatedM1Inputs
) -> str | None:
    m2_room_id = m2_inputs.request.get("room_id")
    selected_room_id = room_inputs.room.get("room_id")
    if not isinstance(m2_room_id, str) or not m2_room_id:
        return "M2 request has no room_id for current MP3D coordinate binding"
    if not isinstance(selected_room_id, str) or not selected_room_id:
        return "M1 room has no room_id for current MP3D coordinate binding"
    if m2_room_id != selected_room_id:
        return (
            "M2 request room_id "
            f"{m2_room_id!r} differs from selected MP3D room {selected_room_id!r}; "
            "its actor coordinates must not be rendered in that room"
        )
    return None


def _fresh_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    if os.path.lexists(output):
        raise CurrentVisualError(f"refusing to replace current visual output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    return output


def _receipt_inputs(
    *, m2_inputs: ValidatedM2Inputs, room_inputs: ValidatedM1Inputs
) -> dict[str, str]:
    return {
        "animal_manifest": str(m2_inputs.asset_path),
        "m2_request": str(m2_inputs.request_path),
        "room_manifest": str(room_inputs.room_path),
        "m1_request": str(room_inputs.request_path),
    }


def _write_not_run_receipt(
    output: Path,
    *,
    reason: str,
    m2_inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "status": "not_run",
        "research_only": True,
        "episode_counted": False,
        "reason": reason,
        "inputs": _receipt_inputs(m2_inputs=m2_inputs, room_inputs=room_inputs),
        "selected_room": {
            "room_id": room_inputs.room["room_id"],
            "room_kind": room_inputs.room["room_kind"],
        },
        "next_requirement": (
            "provide a valid 75-state M2 request authored for this exact MP3D room"
        ),
    }
    write_json(output / "research_receipt.json", receipt)
    return receipt


def _checkout_ancestor(path: Path) -> Path | None:
    candidate = path.resolve()
    while True:
        if os.path.lexists(candidate / ".git"):
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def _resolve_external_scene(
    room_inputs: ValidatedM1Inputs, installed_runtime: InstalledHabitatRuntime
) -> dict[str, Any]:
    mp3d_root = installed_runtime.mp3d_root
    if mp3d_root is None:
        raise CurrentVisualError("current MP3D visual capture requires --mp3d-root")
    root = Path(mp3d_root).resolve()
    checkout = _checkout_ancestor(root)
    if checkout is not None:
        raise CurrentVisualError(
            "current MP3D visual capture rejects a data root inside a Git checkout: "
            f"{root} (found .git at {checkout})"
        )
    scene = room_inputs.room["scene"]
    environment = {"AVENGINE_MP3D_ROOT": str(root)}

    def resolve(raw_path: Any, *, owner: str) -> Path:
        if not isinstance(raw_path, str):
            raise CurrentVisualError(f"{owner} must be a declared path")
        try:
            resolved = resolve_declared_path(
                raw_path,
                manifest_dir=room_inputs.room_path.parent,
                environment=environment,
            )
            resolved.relative_to(root)
        except (OSError, TypeError, ValueError) as error:
            raise CurrentVisualError(
                f"{owner} must resolve under --mp3d-root: {error}"
            ) from error
        if not resolved.is_file():
            raise CurrentVisualError(f"{owner} is missing: {resolved}")
        return resolved

    resolved = {
        "scene_id": resolve(scene["scene_id"], owner="scene_id"),
        "dataset_config": resolve(
            scene["dataset_config_path"], owner="dataset_config_path"
        ),
        "navmesh": resolve(scene["navmesh_path"], owner="navmesh_path"),
        "load_semantic_mesh": bool(scene.get("load_semantic_mesh", False)),
    }
    for asset in room_inputs.room["assets"]:
        resolve(asset["path"], owner=f"room asset {asset['role']}")
    return resolved


def _semantic_sensor_target(name: str) -> Any:
    """Import the binding only after the installed-prefix import has succeeded."""

    from habitat_sim._ext import habitat_sim_bindings

    try:
        return getattr(habitat_sim_bindings.SemanticSensorTarget, name)
    except AttributeError as exc:
        raise CurrentVisualError(
            f"installed Habitat runtime lacks SemanticSensorTarget.{name}"
        ) from exc


def _make_current_configuration(
    *,
    room_inputs: ValidatedM1Inputs,
    installed_runtime: InstalledHabitatRuntime,
    scene: Mapping[str, Any],
    include_audio_sensor: bool = False,
    gpu_device_id: int = 0,
) -> tuple[Any, dict[str, str]]:
    """Build only the co-located visual sensors for the installed runtime."""

    if include_audio_sensor:
        raise CurrentVisualError("current visual capture never enables an audio sensor")
    if (
        isinstance(gpu_device_id, bool)
        or not isinstance(gpu_device_id, int)
        or gpu_device_id < 0
    ):
        raise CurrentVisualError("gpu_device_id must be a nonnegative integer")
    habitat_sim = installed_runtime.habitat_sim
    mn = installed_runtime.magnum
    rig = room_inputs.request["primary_camera_rig"]
    calibration = rig["shared_calibration"]
    height, width = calibration["resolution_hw"]
    local = calibration["rig_from_sensor"]
    modality_to_uuid = {
        item["modality"]: item["sensor_uuid"] for item in rig["modalities"]
    }
    if list(modality_to_uuid) != list(FORMAL_MODALITIES):
        raise CurrentVisualError(
            "M1 visual modality order changed from rgb/depth/semantic"
        )

    sensor_specs: list[Any] = []
    local_position = mn.Vector3(local["translation_m"])
    local_orientation = mn.Vector3(0.0, 0.0, 0.0)
    for modality in FORMAL_MODALITIES:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = modality_to_uuid[modality]
        spec.sensor_type = getattr(
            habitat_sim.SensorType, VISUAL_SENSOR_TYPES[modality]
        )
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = mn.Vector2i([height, width])
        spec.position = local_position
        spec.orientation = local_orientation
        spec.hfov = float(calibration["hfov_degrees"])
        spec.near = float(calibration["near_m"])
        spec.far = float(calibration["far_m"])
        spec.gpu2gpu_transfer = False
        spec.noise_model = "None"
        if modality != "rgb":
            spec.channels = 1
        if modality == "semantic" and hasattr(spec, "semantic_target"):
            spec.semantic_target = _semantic_sensor_target("SEMANTIC_ID")
        sensor_specs.append(spec)

    object_id_spec = habitat_sim.CameraSensorSpec()
    object_id_spec.uuid = CURRENT_OBJECT_ID_SENSOR_UUID
    object_id_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    object_id_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    object_id_spec.resolution = mn.Vector2i([height, width])
    object_id_spec.position = local_position
    object_id_spec.orientation = local_orientation
    object_id_spec.hfov = float(calibration["hfov_degrees"])
    object_id_spec.near = float(calibration["near_m"])
    object_id_spec.far = float(calibration["far_m"])
    object_id_spec.gpu2gpu_transfer = False
    object_id_spec.noise_model = "None"
    object_id_spec.channels = 1
    if not hasattr(object_id_spec, "semantic_target"):
        raise CurrentVisualError(
            "installed Habitat CameraSensorSpec lacks semantic_target"
        )
    object_id_spec.semantic_target = _semantic_sensor_target("OBJECT_ID")
    sensor_specs.append(object_id_spec)
    modality_to_uuid[CURRENT_OBJECT_ID_MODALITY] = CURRENT_OBJECT_ID_SENSOR_UUID

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene["scene_id"])
    sim_cfg.scene_dataset_config_file = str(scene["dataset_config"])
    sim_cfg.load_semantic_mesh = bool(scene["load_semantic_mesh"])
    # Bullet is needed to create a kinematic AO. The capture never steps it.
    sim_cfg.enable_physics = True
    sim_cfg.physics_config_file = str(installed_runtime.physics_config_path)
    sim_cfg.random_seed = int(room_inputs.request["seed"])
    sim_cfg.gpu_device_id = gpu_device_id

    agent_cfg = habitat_sim.AgentConfiguration()
    navigation = room_inputs.room.get("navigation", {})
    agent_cfg.height = float(navigation.get("agent_height_m", 1.5))
    agent_cfg.radius = float(navigation.get("agent_radius_m", 0.2))
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}

    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_height = agent_cfg.height
    nav_settings.agent_radius = agent_cfg.radius
    nav_settings.include_static_objects = bool(
        navigation.get("include_static_objects", False)
    )
    sim_cfg.navmesh_settings = nav_settings
    return habitat_sim.Configuration(sim_cfg, [agent_cfg]), modality_to_uuid


def _instantiate_semantic_actor(
    simulator: Any,
    *,
    bundle: Any,
    habitat_sim: Any,
    base_handle: str,
    semantic_id: int,
    actor_index: int,
) -> tuple[Any, Any]:
    manager = simulator.metadata_mediator.ao_template_manager
    attributes = manager.get_template_by_handle(base_handle)
    if attributes is None:
        raise CurrentVisualError("cannot retrieve the loaded articulated template")
    attributes.semantic_id = int(semantic_id)
    handle = f"{base_handle}.m5_current_actor{actor_index}_semantic{semantic_id}"
    if int(manager.register_template(attributes, handle)) < 0:
        raise CurrentVisualError("failed to register a current visual actor template")
    articulated_object = simulator.get_articulated_object_manager().add_articulated_object_by_template_handle(
        handle
    )
    if articulated_object is None:
        raise CurrentVisualError("Habitat failed to instantiate a current visual actor")
    articulated_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    link_ids = list(articulated_object.get_link_ids())
    expected_names = set(bundle.joint_mapping["joint_order"])
    actual_names = {articulated_object.get_link_name(-1)} | {
        articulated_object.get_link_name(link_id) for link_id in link_ids
    }
    if actual_names != expected_names:
        raise CurrentVisualError("articulated link names differ from the M2 mapping")
    blocks = [
        HabitatLinkJointBlock(
            link_name=articulated_object.get_link_name(link_id),
            joint_position_offset=int(
                articulated_object.get_link_joint_pos_offset(link_id)
            ),
            joint_position_count=int(
                articulated_object.get_link_num_joint_pos(link_id)
            ),
        )
        for link_id in link_ids
    ]
    binding = bind_habitat_link_layout(
        bundle.joint_mapping["runtime_joint_order"],
        blocks,
        joint_position_count=len(articulated_object.joint_positions),
    )
    if int(articulated_object.creation_attributes.semantic_id) != semantic_id:
        raise CurrentVisualError("actor creation semantic ID differs from its template")
    nodes = [articulated_object.root_scene_node]
    nodes.extend(
        articulated_object.get_link_scene_node(link_id) for link_id in link_ids
    )
    for node in nodes:
        if not hasattr(node, "semantic_id"):
            raise CurrentVisualError("Habitat SceneNode lacks semantic_id")
        node.semantic_id = semantic_id
        if int(node.semantic_id) != semantic_id:
            raise CurrentVisualError("Habitat semantic ID writeback differs")
    return articulated_object, binding


def _link_id_by_name(articulated_object: Any, name: str) -> int:
    matches = [
        int(link_id)
        for link_id in articulated_object.get_link_ids()
        if articulated_object.get_link_name(link_id) == name
    ]
    if len(matches) != 1:
        raise CurrentVisualError(f"expected one articulated link named {name!r}")
    return matches[0]


def _node_world_position(articulated_object: Any, link_id: int) -> np.ndarray:
    matrix = np.asarray(
        articulated_object.get_link_scene_node(link_id).absolute_transformation(),
        dtype=np.float64,
    )
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise CurrentVisualError("emitter SceneNode transform is not a finite 4x4")
    return matrix[:3, 3].copy()


def _read_actor_state(articulated_object: Any) -> tuple[np.ndarray, np.ndarray]:
    root = np.asarray(
        articulated_object.root_scene_node.absolute_transformation(), dtype=np.float64
    )
    joints = np.asarray(articulated_object.joint_positions, dtype=np.float64).reshape(
        -1
    )
    if root.shape != (4, 4) or not np.all(np.isfinite(root)):
        raise CurrentVisualError("articulated root readback is not a finite 4x4")
    if not np.all(np.isfinite(joints)):
        raise CurrentVisualError("articulated joint readback is not finite")
    return root, joints


def _quaternion_block_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape or actual.ndim != 1 or actual.size % 4:
        return float("inf")
    maximum = 0.0
    for offset in range(0, actual.size, 4):
        left = actual[offset : offset + 4]
        right = expected[offset : offset + 4]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1.0e-15 or right_norm <= 1.0e-15:
            return float("inf")
        left = left / left_norm
        right = right / right_norm
        maximum = max(
            maximum,
            min(
                float(np.max(np.abs(left - right))),
                float(np.max(np.abs(left + right))),
            ),
        )
    return maximum


def _observation_arrays(
    observation: Mapping[str, Any], modality_to_uuid: Mapping[str, str]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for modality in (*FORMAL_MODALITIES, CURRENT_OBJECT_ID_MODALITY):
        uuid = modality_to_uuid[modality]
        if uuid not in observation:
            raise CurrentVisualError(f"Habitat observation lacks {modality} sensor")
        arrays[modality] = np.ascontiguousarray(np.asarray(observation[uuid])).copy()
    rgb = arrays["rgb"]
    depth = arrays["depth"]
    semantic = arrays["semantic"]
    object_id = arrays[CURRENT_OBJECT_ID_MODALITY]
    if rgb.ndim != 3 or rgb.shape[-1] not in {3, 4}:
        raise CurrentVisualError(f"RGB frame must be HxWx3/4, got {rgb.shape}")
    if depth.ndim != 2 or semantic.ndim != 2 or object_id.ndim != 2:
        raise CurrentVisualError("depth, semantic and object-ID frames must be HxW")
    if (
        rgb.shape[:2] != depth.shape
        or depth.shape != semantic.shape
        or semantic.shape != object_id.shape
    ):
        raise CurrentVisualError(
            "RGB, depth, semantic and object-ID frames are not co-registered"
        )
    if not np.issubdtype(object_id.dtype, np.integer):
        raise CurrentVisualError("object-ID frame must use an integer dtype")
    return arrays


def _object_id_pixel_record(image: Any, runtime_object_id: int) -> dict[str, Any]:
    array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise CurrentVisualError("object-ID observation must be an integer HxW array")
    rows, columns = np.nonzero(array == runtime_object_id)
    count = int(rows.size)
    bbox = None
    if count:
        x_min = int(columns.min())
        x_max = int(columns.max())
        y_min = int(rows.min())
        y_max = int(rows.max())
        bbox = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width_pixels": x_max - x_min + 1,
            "height_pixels": y_max - y_min + 1,
        }
    return {
        "runtime_object_id": runtime_object_id,
        "pixel_count": count,
        "bbox_xyxy": bbox,
    }


def _require_no_actor_semantic_collision(semantic: Any) -> None:
    """Reject a scene that already uses either per-run actor semantic ID."""

    image = np.asarray(semantic)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise CurrentVisualError(
            "no-actor semantic preflight is not an integer HxW observation"
        )
    collisions = {
        semantic_id: int(np.count_nonzero(image == semantic_id))
        for semantic_id in CURRENT_SEMANTIC_IDS
    }
    if any(collisions.values()):
        raise CurrentVisualError(
            "selected MP3D scene already uses a current actor semantic ID: "
            f"{collisions}"
        )


def _require_bullet(installed_runtime: InstalledHabitatRuntime) -> None:
    if not bool(getattr(installed_runtime.habitat_sim, "built_with_bullet", False)):
        raise CurrentVisualError(
            "current MP3D visual capture requires a Bullet-enabled Habitat runtime"
        )


def _capture_current_visual(
    *,
    m2_inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    installed_runtime: InstalledHabitatRuntime,
    output: Path,
    bundle: Any,
    frames: Sequence[Any],
) -> dict[str, Any]:
    m1_capture_identity = _m1_capture_identity(room_inputs)
    scene = _resolve_external_scene(room_inputs, installed_runtime)
    configuration, modality_to_uuid = _make_current_configuration(
        room_inputs=room_inputs,
        installed_runtime=installed_runtime,
        scene=scene,
        include_audio_sensor=False,
    )

    habitat_sim = installed_runtime.habitat_sim
    qt = installed_runtime.quaternion
    mn = installed_runtime.magnum
    rig = room_inputs.request["primary_camera_rig"]
    rig_transform = rig["world_from_rig"]
    camera_position = np.asarray(rig_transform["translation_m"], dtype=np.float64)
    qx, qy, qz, qw = normalized_quaternion_xyzw(rig_transform["rotation_xyzw"])
    offsets = np.asarray(CURRENT_ACTOR_OFFSETS_M, dtype=np.float64)
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    semantic_frames: list[np.ndarray] = []
    object_id_frames: list[np.ndarray] = []
    actor_matrices: list[np.ndarray] = []
    planned_skin_root_matrices: list[np.ndarray] = []
    actual_skin_root_matrices: list[np.ndarray] = []
    actual_agent_camera_poses: list[np.ndarray] = []
    actual_sensor_poses: list[np.ndarray] = []
    actual_logical_listener_poses: list[np.ndarray] = []
    source_positions: list[np.ndarray] = []
    visibility_records: list[tuple[int, int]] = []
    object_id_records: list[list[dict[str, Any]]] = []
    frame_records: list[dict[str, Any]] = []

    with habitat_sim.Simulator(configuration) as simulator:
        navmesh_loaded = bool(simulator.pathfinder.load_nav_mesh(str(scene["navmesh"])))
        if not navmesh_loaded or not bool(simulator.pathfinder.is_loaded):
            raise CurrentVisualError("Habitat could not load the declared MP3D navmesh")
        try:
            loaded_scene_errors, _ = validate_loaded_scene_asset_graph(
                room_inputs,
                None,
                simulator,
                declared_navmesh_loaded=navmesh_loaded,
                mp3d_root=installed_runtime.mp3d_root,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise CurrentVisualError(
                f"unable to validate loaded MP3D scene graph: {error}"
            ) from error
        if loaded_scene_errors:
            raise CurrentVisualError(
                "loaded MP3D scene graph differs from the declared room: "
                + "; ".join(loaded_scene_errors)
            )
        simulator.seed(int(m2_inputs.request["seed"]))
        camera_state = habitat_sim.AgentState()
        camera_state.position = camera_position
        camera_state.rotation = qt.quaternion(qw, qx, qy, qz)
        agent = simulator.initialize_agent(0, camera_state)

        captured_modalities = (*FORMAL_MODALITIES, CURRENT_OBJECT_ID_MODALITY)
        sensors = [
            simulator.sensors[modality_to_uuid[modality]]
            for modality in captured_modalities
        ]
        initial_world_time = float(simulator.get_world_time())
        sensor_uuids = [modality_to_uuid[modality] for modality in captured_modalities]
        sensor_order = sorted(sensor_uuids)
        modality_by_sensor_uuid = {
            sensor_uuid: modality for modality, sensor_uuid in modality_to_uuid.items()
        }
        listener = room_inputs.request["listener"]
        preflight_observation = simulator.render_sensors(sensors)
        preflight_arrays = _observation_arrays(preflight_observation, modality_to_uuid)
        _require_no_actor_semantic_collision(preflight_arrays["semantic"])
        if float(simulator.get_world_time()) != initial_world_time:
            raise CurrentVisualError(
                "no-actor semantic preflight advanced Habitat world time"
            )

        template_manager = simulator.metadata_mediator.ao_template_manager
        config_path = bundle.paths_by_role["habitat_ao_config"]
        loaded_ids = template_manager.load_configs(str(config_path))
        handle_prefix = config_path.stem.removesuffix(".ao_config")
        base_handles = template_manager.get_template_handles(handle_prefix)
        if len(loaded_ids) != 1 or len(base_handles) != 1:
            raise CurrentVisualError(
                "current visual capture expected one source AO template"
            )
        actors: list[Any] = []
        bindings: list[Any] = []
        emitter_link_ids: list[int] = []
        runtime_object_ids: list[int] = []
        for actor_index, semantic_id in enumerate(CURRENT_SEMANTIC_IDS):
            actor, binding = _instantiate_semantic_actor(
                simulator,
                bundle=bundle,
                habitat_sim=habitat_sim,
                base_handle=base_handles[0],
                semantic_id=semantic_id,
                actor_index=actor_index,
            )
            actors.append(actor)
            bindings.append(binding)
            runtime_object_id = getattr(actor, "object_id", None)
            if (
                isinstance(runtime_object_id, bool)
                or not isinstance(runtime_object_id, int)
                or runtime_object_id < 0
                or runtime_object_id in runtime_object_ids
            ):
                raise CurrentVisualError(
                    "Habitat actor runtime object IDs must be unique nonnegative integers"
                )
            runtime_object_ids.append(runtime_object_id)
            emitter_link_ids.append(
                _link_id_by_name(actor, CURRENT_EMITTER_LINK_NAMES[actor_index])
            )
        for frame in frames:
            frame_camera = habitat_sim.AgentState()
            frame_camera.position = camera_position
            frame_camera.rotation = qt.quaternion(qw, qx, qy, qz)
            agent.set_state(frame_camera, reset_sensors=False, infer_sensor_states=True)
            pre_render_snapshot = _state_snapshot(
                simulator, agent, sensor_uuids, installed_runtime.quat_to_coeffs
            )
            pose_errors = [
                transform_error(rig_transform, pre_render_snapshot["agent"]),
                *[
                    transform_error(rig_transform, sensor_pose)
                    for sensor_pose in pre_render_snapshot["sensors"].values()
                ],
            ]
            if max(pose_errors) > _ROOT_READBACK_ATOL:
                raise CurrentVisualError(
                    f"frame {frame.frame_index} camera/readback differs from M1 rig"
                )

            actor_world: list[np.ndarray] = []
            planned_skin_roots: list[np.ndarray] = []
            before_states: list[tuple[np.ndarray, np.ndarray]] = []
            for actor_index, (actor, binding) in enumerate(
                zip(actors, bindings, strict=True)
            ):
                skin_root = np.asarray(
                    frame.world_from_skin_root, dtype=np.float64
                ).copy()
                skin_root[:3, 3] += offsets[actor_index]
                joints = np.asarray(
                    binding.map_pose(frame.joint_rotations_xyzw), dtype=np.float64
                )
                _apply_root_with_habitat(actor, skin_root, qt=qt, mn=mn)
                actor.joint_positions = joints.copy()
                root_readback, joint_readback = _read_actor_state(actor)
                root_error = float(np.max(np.abs(root_readback - skin_root)))
                joint_error = _quaternion_block_error(joint_readback, joints)
                if (
                    root_error > _ROOT_READBACK_ATOL
                    or joint_error > _JOINT_READBACK_ATOL
                ):
                    raise CurrentVisualError(
                        f"frame {frame.frame_index} actor {actor_index} readback failed"
                    )
                planned_skin_roots.append(skin_root)
                before_states.append((root_readback, joint_readback))
                matrix = np.asarray(frame.world_from_actor, dtype=np.float64).copy()
                matrix[:3, 3] += offsets[actor_index]
                actor_world.append(matrix)

            observation = simulator.render_sensors(sensors)
            arrays = _observation_arrays(observation, modality_to_uuid)
            frame_visibility = tuple(
                int(np.count_nonzero(arrays["semantic"] == semantic_id))
                for semantic_id in CURRENT_SEMANTIC_IDS
            )
            frame_object_ids = [
                _object_id_pixel_record(arrays[CURRENT_OBJECT_ID_MODALITY], object_id)
                for object_id in runtime_object_ids
            ]
            after_states = [_read_actor_state(actor) for actor in actors]
            for before, after in zip(before_states, after_states, strict=True):
                if not np.allclose(
                    before[0], after[0], rtol=0.0, atol=1.0e-12
                ) or not np.allclose(before[1], after[1], rtol=0.0, atol=1.0e-12):
                    raise CurrentVisualError(
                        f"frame {frame.frame_index} render changed an articulated state"
                    )
            post_render_snapshot = _state_snapshot(
                simulator, agent, sensor_uuids, installed_runtime.quat_to_coeffs
            )
            pre_render_world_time, post_render_world_time = (
                _require_render_snapshot_unchanged(
                    pre_render_snapshot,
                    post_render_snapshot,
                    sensor_order=sensor_order,
                    frame_index=int(frame.frame_index),
                )
            )
            if (
                abs(post_render_world_time - initial_world_time)
                > _WORLD_TIME_READBACK_ATOL
            ):
                raise CurrentVisualError(
                    f"frame {frame.frame_index} advanced Habitat world time"
                )
            frame_agent_camera_pose = _pose_vector(
                post_render_snapshot["agent"],
                owner=f"frame {frame.frame_index} post-render agent/camera readback",
            )
            frame_sensor_poses = np.stack(
                [
                    _pose_vector(
                        post_render_snapshot["sensors"][sensor_uuid],
                        owner=(
                            f"frame {frame.frame_index} post-render sensor "
                            f"{sensor_uuid!r} readback"
                        ),
                    )
                    for sensor_uuid in sensor_order
                ],
                axis=0,
            )
            frame_listener_pose = _pose_vector(
                _logical_listener_pose(post_render_snapshot, listener),
                owner=f"frame {frame.frame_index} post-render logical listener pose",
            )

            frame_sources = np.stack(
                [
                    _node_world_position(actor, emitter_link_ids[index])
                    for index, actor in enumerate(actors)
                ],
                axis=0,
            )
            rgb_frames.append(
                np.asarray(arrays["rgb"])[..., :3].astype(np.uint8, copy=True)
            )
            depth_frames.append(np.asarray(arrays["depth"]).copy())
            semantic_frames.append(np.asarray(arrays["semantic"]).copy())
            object_id_frames.append(
                np.asarray(arrays[CURRENT_OBJECT_ID_MODALITY]).copy()
            )
            actor_matrices.append(np.stack(actor_world, axis=0))
            planned_skin_root_matrices.append(np.stack(planned_skin_roots, axis=0))
            actual_skin_root_matrices.append(
                np.stack([state[0] for state in after_states], axis=0)
            )
            actual_agent_camera_poses.append(frame_agent_camera_pose)
            actual_sensor_poses.append(frame_sensor_poses)
            actual_logical_listener_poses.append(frame_listener_pose)
            source_positions.append(frame_sources)
            visibility_records.append(frame_visibility)
            object_id_records.append(frame_object_ids)
            frame_records.append(
                {
                    "frame_index": int(frame.frame_index),
                    "pts_ticks": int(frame.pts_ticks),
                    "action_id": str(frame.action_id),
                    "action_sample_index": int(frame.action_sample_index),
                    "actor_world_positions_m": [
                        matrix[:3, 3].tolist() for matrix in actor_world
                    ],
                    "source_positions_m": frame_sources.tolist(),
                    "semantic_visibility_pixels": list(frame_visibility),
                    "object_id_pixel_counts": [
                        record["pixel_count"] for record in frame_object_ids
                    ],
                    "object_id_bboxes_xyxy": [
                        record["bbox_xyxy"] for record in frame_object_ids
                    ],
                    "snapshot_world_time_seconds": post_render_world_time,
                    "pre_render_snapshot_world_time_seconds": pre_render_world_time,
                    "post_render_snapshot_world_time_seconds": post_render_world_time,
                    "pose_alignment": {
                        "array_frame_index": int(frame.frame_index),
                        "actor_axis_order": list(CURRENT_ACTOR_IDS),
                        "source_axis_order": list(CURRENT_SOURCE_IDS),
                        "sensor_axis_order": list(sensor_order),
                        "pose_component_order": list(CURRENT_POSE_COMPONENT_ORDER),
                    },
                    "observation_calls": 1,
                }
            )

    if len(rgb_frames) != 75 or len(object_id_frames) != 75:
        raise CurrentVisualError("current visual capture must retain exactly 75 frames")
    arrays_root = output / "arrays"
    arrays_root.mkdir()
    artifacts = {
        "rgb": "arrays/rgb.npy",
        "depth": "arrays/depth.npy",
        "semantic": "arrays/semantic.npy",
        "object_id": "arrays/object_id.npy",
        "actor_world_matrices": "arrays/actor_world_matrices.npy",
        "planned_applied_skin_root_matrices": (
            "arrays/planned_applied_skin_root_matrices.npy"
        ),
        "actual_skin_root_readback_matrices": (
            "arrays/actual_skin_root_readback_matrices.npy"
        ),
        "actual_agent_camera_poses": (
            "arrays/actual_agent_camera_poses_txyz_qxyzw.npy"
        ),
        "actual_sensor_poses": "arrays/actual_sensor_poses_txyz_qxyzw.npy",
        "actual_logical_listener_poses": (
            "arrays/actual_logical_listener_poses_txyz_qxyzw.npy"
        ),
        "source_positions_m": "arrays/source_positions_m.npy",
        "semantic_visibility_pixels": "arrays/semantic_visibility_pixels.npy",
        "frame_records": "frame_records.json",
    }
    np.save(arrays_root / "rgb.npy", np.ascontiguousarray(np.stack(rgb_frames)))
    np.save(arrays_root / "depth.npy", np.ascontiguousarray(np.stack(depth_frames)))
    np.save(
        arrays_root / "semantic.npy", np.ascontiguousarray(np.stack(semantic_frames))
    )
    np.save(
        arrays_root / "object_id.npy",
        np.ascontiguousarray(np.stack(object_id_frames)),
    )
    np.save(
        arrays_root / "actor_world_matrices.npy",
        np.ascontiguousarray(np.stack(actor_matrices)),
    )
    np.save(
        arrays_root / "planned_applied_skin_root_matrices.npy",
        np.ascontiguousarray(np.stack(planned_skin_root_matrices)),
    )
    np.save(
        arrays_root / "actual_skin_root_readback_matrices.npy",
        np.ascontiguousarray(np.stack(actual_skin_root_matrices)),
    )
    np.save(
        arrays_root / "actual_agent_camera_poses_txyz_qxyzw.npy",
        np.ascontiguousarray(np.stack(actual_agent_camera_poses)),
    )
    np.save(
        arrays_root / "actual_sensor_poses_txyz_qxyzw.npy",
        np.ascontiguousarray(np.stack(actual_sensor_poses)),
    )
    np.save(
        arrays_root / "actual_logical_listener_poses_txyz_qxyzw.npy",
        np.ascontiguousarray(np.stack(actual_logical_listener_poses)),
    )
    np.save(
        arrays_root / "source_positions_m.npy",
        np.ascontiguousarray(np.stack(source_positions)),
    )
    visibility_array = np.asarray(visibility_records, dtype=np.int64)
    np.save(
        arrays_root / "semantic_visibility_pixels.npy",
        visibility_array,
    )
    write_json(output / "frame_records.json", {"frames": frame_records})
    visibility_by_actor = {}
    for actor_index, actor_id in enumerate(CURRENT_ACTOR_IDS):
        pixels = visibility_array[:, actor_index]
        visible = pixels > 0
        visibility_by_actor[actor_id] = {
            "visible_frame_count": int(np.count_nonzero(visible)),
            "zero_visible_frame_indices": np.flatnonzero(~visible).astype(int).tolist(),
            "minimum_visible_pixels": int(np.min(pixels)),
            "maximum_visible_pixels": int(np.max(pixels)),
            "per_frame_visible_pixels": pixels.tolist(),
        }
    object_id_by_actor = {}
    for actor_index, actor_id in enumerate(CURRENT_ACTOR_IDS):
        records = [frame[actor_index] for frame in object_id_records]
        counts = [int(record["pixel_count"]) for record in records]
        object_id_by_actor[actor_id] = {
            "runtime_object_id": runtime_object_ids[actor_index],
            "pixel_counts_by_frame": counts,
            "bboxes_xyxy_by_frame": [record["bbox_xyxy"] for record in records],
            "positive_pixel_frame_count": int(np.count_nonzero(counts)),
            "zero_pixel_frame_indices": [
                index for index, count in enumerate(counts) if count == 0
            ],
        }
    habitat_module = getattr(installed_runtime.habitat_sim, "__file__", None)
    if not isinstance(habitat_module, str) or not habitat_module:
        raise CurrentVisualError("installed Habitat runtime has no module path")
    receipt: dict[str, Any] = {
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "runtime": {
            "mode": "installed_prefix",
            "prefix": str(installed_runtime.prefix),
            "mp3d_root": str(installed_runtime.mp3d_root),
            "magnum_python_site": str(installed_runtime.magnum_python_site),
            "habitat_sim_module": str(Path(habitat_module).resolve()),
        },
        "inputs": _receipt_inputs(m2_inputs=m2_inputs, room_inputs=room_inputs),
        "selected_room": {
            "room_id": room_inputs.room["room_id"],
            "room_kind": room_inputs.room["room_kind"],
        },
        "m1_capture_identity": m1_capture_identity,
        "capture": {
            "native_habitat_started": True,
            "m1_request_id": m1_capture_identity["request_id"],
            "room_id": m1_capture_identity["room_id"],
            "frame_count": len(frame_records),
            "modalities": [*FORMAL_MODALITIES, CURRENT_OBJECT_ID_MODALITY],
            "observation_calls_per_frame": 1,
            "state_snapshot_calls_per_frame": 2,
            "persisted_pose_snapshot_phase": "post_render",
            "physics_steps": 0,
            "audio_sensor_included": False,
        },
        "actors": {
            "actor_ids": list(CURRENT_ACTOR_IDS),
            "source_ids": list(CURRENT_SOURCE_IDS),
            "semantic_ids": list(CURRENT_SEMANTIC_IDS),
            "runtime_object_ids": list(runtime_object_ids),
            "instance_offsets_m": offsets.tolist(),
            "emitter_link_names": list(CURRENT_EMITTER_LINK_NAMES),
            "axis_bindings": [
                {
                    "axis_index": axis_index,
                    "actor_id": CURRENT_ACTOR_IDS[axis_index],
                    "source_id": CURRENT_SOURCE_IDS[axis_index],
                    "semantic_id": CURRENT_SEMANTIC_IDS[axis_index],
                    "runtime_object_id": runtime_object_ids[axis_index],
                    "emitter_link_name": CURRENT_EMITTER_LINK_NAMES[axis_index],
                    "runtime_emitter_link_id": emitter_link_ids[axis_index],
                    "audio_endpoint_status": "pending",
                    "audio_endpoint_id": None,
                }
                for axis_index in range(len(CURRENT_ACTOR_IDS))
            ],
        },
        "pose_readback": {
            "status": "native_habitat_pose_readback_recorded",
            "frame_count": len(frame_records),
            "pose_component_order": list(CURRENT_POSE_COMPONENT_ORDER),
            "coordinate_convention": {
                "frame": "habitat_world",
                "axis_order": ["X", "Y", "Z"],
                "up_axis": "+Y",
            },
            "matrix_convention": {
                "transform_direction": "world_from_local",
                "translation_location": "last_column",
                "homogeneous_last_row": [0.0, 0.0, 0.0, 1.0],
            },
            "actor_axis_order": list(CURRENT_ACTOR_IDS),
            "source_axis_order": list(CURRENT_SOURCE_IDS),
            "sensor_order": [
                {
                    "axis_index": axis_index,
                    "sensor_uuid": sensor_uuid,
                    "modality": modality_by_sensor_uuid[sensor_uuid],
                }
                for axis_index, sensor_uuid in enumerate(sensor_order)
            ],
            "planned_actor_world_matrices": {
                "artifact_key": "actor_world_matrices",
                "pose_kind": "planned_actor_transform_with_instance_offset",
                "actual_readback": False,
            },
            "planned_applied_skin_root_matrices": {
                "artifact_key": "planned_applied_skin_root_matrices",
                "pose_kind": "planned_transform_applied_to_skin_root",
                "actual_readback": False,
            },
            "actual_skin_root_readback_matrices": {
                "artifact_key": "actual_skin_root_readback_matrices",
                "pose_kind": "articulated_root_scene_node_after_render",
                "actual_readback": True,
            },
            "actual_agent_camera_poses": {
                "artifact_key": "actual_agent_camera_poses",
                "pose_kind": "post_render_agent_camera_from_state_snapshot",
                "actual_readback": True,
            },
            "actual_sensor_poses": {
                "artifact_key": "actual_sensor_poses",
                "pose_kind": "post_render_sensor_states_from_state_snapshot",
                "actual_readback": True,
            },
            "actual_logical_listener_poses": {
                "artifact_key": "actual_logical_listener_poses",
                "pose_kind": (
                    "logical_listener_composed_from_post_render_agent_readback"
                ),
                "actual_readback": False,
                "derived_from_actual_agent_readback": True,
                "native_listener_sensor_readback": False,
            },
        },
        "visibility": {
            "status": "native_semantic_id_raster_recorded_coverage_incomplete",
            "authority": "same_habitat_renderer_semantic_id_raster_only",
            "semantic_id_coverage_completeness_status": "not_certified",
            "frame_count": len(frame_records),
            "all_frames_visibility_required": False,
            "candidate_filtering": "disabled",
            "candidate_legality_effect": "descriptor_only",
            "per_actor": visibility_by_actor,
        },
        "object_id_visibility": {
            "status": "native_object_id_raster_recorded",
            "sensor_uuid": CURRENT_OBJECT_ID_SENSOR_UUID,
            "semantic_sensor_target": "OBJECT_ID",
            "frame_count": len(frame_records),
            "runtime_id_scope": ("one_simulator_invocation_not_stable_across_runs"),
            "runtime_id_mapping_authority": "articulated_object.object_id_readback",
            "candidate_filtering": "disabled",
            "candidate_legality_effect": "descriptor_only",
            "zero_pixels_affect_legality": False,
            "actor_modal_visibility_authority_status": (
                "pending_fresh_runtime_a_b_and_visual_review"
            ),
            "per_actor": object_id_by_actor,
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "visual-only research capture; OBJECT_ID is retained as a candidate modal "
            "raster but needs fresh-runtime A/B and visual review before visibility "
            "authority. It does not count an episode or establish an audio, room, "
            "asset, P1/amodal, occlusion, off-screen, or dataset claim"
        ),
    }
    write_json(output / "research_receipt.json", receipt)
    return receipt


def capture_current_visual(
    *,
    animal_manifest_path: str | Path,
    m2_request_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    runtime_prefix: str | Path | None,
    mp3d_root: str | Path | None,
    magnum_python_site: str | Path | None,
    output_directory: str | Path,
    rlr_sdk_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture M5 visuals only on an explicit installed MP3D runtime.

    A request authored for another room returns an ordinary ``not_run`` receipt
    before any Habitat import or GPU work.  The current retained Beagle request
    follows that path because it contains Blender-room coordinates.
    """

    _require_explicit_runtime_inputs(
        runtime_prefix=runtime_prefix,
        mp3d_root=mp3d_root,
        magnum_python_site=magnum_python_site,
    )
    try:
        m2_inputs = load_m2_inputs(animal_manifest_path, m2_request_path)
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    except (HabitatCaptureError, OSError, ValueError) as error:
        raise CurrentVisualError(str(error)) from error
    room_error = _current_mp3d_room_error(room_inputs)
    if room_error is not None:
        raise CurrentVisualError(room_error)
    room_reason = _same_room_reason(m2_inputs, room_inputs)
    if room_reason is not None:
        output = _fresh_output_directory(output_directory)
        return _write_not_run_receipt(
            output,
            reason=room_reason,
            m2_inputs=m2_inputs,
            room_inputs=room_inputs,
        )
    context_errors = validate_capture_context(m2_inputs, room_inputs)
    if context_errors:
        raise CurrentVisualError(
            "current MP3D visual input context is invalid: " + "; ".join(context_errors)
        )
    try:
        static_scene_errors = validate_scene_asset_graph(
            room_inputs,
            None,
            mp3d_root=mp3d_root,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CurrentVisualError(
            f"unable to validate MP3D scene graph before runtime: {error}"
        ) from error
    if static_scene_errors:
        raise CurrentVisualError(
            "MP3D scene graph differs from the declared room: "
            + "; ".join(static_scene_errors)
        )
    try:
        bundle = load_runtime_asset_bundle(m2_inputs)
        frames = compile_frame_applications(m2_inputs, bundle)
    except (HabitatCaptureError, OSError, ValueError) as error:
        raise CurrentVisualError(str(error)) from error
    if len(frames) != 75:
        raise CurrentVisualError(
            "current visual capture requires a 75-state M2 request"
        )
    try:
        installed_runtime = prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_python_site,
            rlr_sdk_root=rlr_sdk_root,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CurrentVisualError(
            f"installed Habitat runtime is unavailable: {error}"
        ) from error
    _require_bullet(installed_runtime)
    output = _fresh_output_directory(output_directory)
    try:
        return _capture_current_visual(
            m2_inputs=m2_inputs,
            room_inputs=room_inputs,
            installed_runtime=installed_runtime,
            output=output,
            bundle=bundle,
            frames=frames,
        )
    except (HabitatCaptureError, OSError, ValueError) as error:
        if isinstance(error, CurrentVisualError):
            raise
        raise CurrentVisualError(str(error)) from error


__all__ = ["CurrentVisualError", "capture_current_visual"]
