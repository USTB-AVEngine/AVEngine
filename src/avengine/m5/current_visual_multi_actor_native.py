"""Native Habitat runner for one research-only two-actor MP3D case.

The CPU loader and apply-plan are the authority for input validation.  This
module adds exactly one native Simulator invocation: two independently loaded
actor bundles share the validated M1 camera/sensor rig, while every actor
root/joint application and readback remains on its own axis.  It is visual
research evidence only; it does not run audio or turn visibility into a gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.contracts.transforms import normalized_quaternion_xyzw, transform_error
from avengine.m1.contracts import (
    load_and_validate_inputs as load_m1_inputs,
    validate_loaded_scene_asset_graph,
    validate_scene_asset_graph,
)
from avengine.m1.habitat_capture import (
    _logical_listener_pose,
    _state_snapshot,
    prepare_installed_habitat_runtime,
)
from avengine.m2.contracts import FORMAL_MODALITIES
from avengine.m2.habitat_capture import (
    _apply_root_with_habitat,
)
from avengine.m5.current_visual import (
    CURRENT_OBJECT_ID_MODALITY,
    _m1_capture_identity,
    _instantiate_semantic_actor,
    _link_id_by_name,
    _make_current_configuration,
    _node_world_position,
    _observation_arrays,
    _object_id_pixel_record,
    _pose_vector,
    _quaternion_block_error,
    _read_actor_state,
    _require_bullet,
    _resolve_external_scene,
    _require_render_snapshot_unchanged,
)
from avengine.m5.current_visual_multi_actor import (
    FRAME_COUNT,
    build_apply_plan,
    classify_collision_contacts,
    load_multi_actor_case,
)
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth


ROOT_READBACK_ATOL = 2.0e-6
JOINT_READBACK_ATOL = 2.0e-6
WORLD_TIME_ATOL = 1.0e-12
ARRAY_ARTIFACTS = (
    "rgb.npy",
    "depth.npy",
    "semantic.npy",
    "object_id.npy",
    "planned_actor_world_matrices.npy",
    "planned_applied_skin_root_matrices.npy",
    "actual_skin_root_readback_matrices.npy",
    "target_only_semantic_actor0.npy",
    "target_only_semantic_actor1.npy",
    "target_only_object_id_actor0.npy",
    "target_only_object_id_actor1.npy",
    "target_only_skin_root_readback_actor0.npy",
    "target_only_skin_root_readback_actor1.npy",
    "target_only_joint_readback_actor0.npy",
    "target_only_joint_readback_actor1.npy",
    "target_only_camera_poses_actor0.npy",
    "target_only_camera_poses_actor1.npy",
    "target_only_sensor_poses_actor0.npy",
    "target_only_sensor_poses_actor1.npy",
    "actual_agent_camera_poses.npy",
    "actual_sensor_poses.npy",
    "actual_logical_listener_poses.npy",
    "emitter_positions_m.npy",
    "actual_joint_readback_actor0.npy",
    "actual_joint_readback_actor1.npy",
)
JSON_ARTIFACTS = (
    "frame_records.json",
    "target_only_frame_records.json",
    "pixel_visibility_truth.json",
)


class NativeMultiActorVisualError(RuntimeError):
    """Native visual evidence cannot be trusted."""


def _fail(message: str) -> NativeMultiActorVisualError:
    return NativeMultiActorVisualError(message)


def _fresh_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to replace existing capture output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    return output


def _save_array(output: Path, name: str, values: Sequence[np.ndarray]) -> None:
    np.save(output / name, np.ascontiguousarray(np.stack(values)))


def _runtime_contact_api(simulator: Any) -> tuple[Any, Any]:
    detector = getattr(simulator, "perform_discrete_collision_detection", None)
    reader = getattr(simulator, "get_physics_contact_points", None)
    if not callable(detector) or not callable(reader):
        raise _fail(
            "installed Habitat runtime lacks perform_discrete_collision_detection "
            "or get_physics_contact_points"
        )
    return detector, reader


def _state_equal(
    before: tuple[np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray],
    *,
    frame_index: int,
    actor_id: str,
) -> None:
    if not np.allclose(before[0], after[0], rtol=0.0, atol=1.0e-12):
        raise _fail(
            f"frame {frame_index} actor {actor_id} root changed during readback"
        )
    if not np.allclose(before[1], after[1], rtol=0.0, atol=1.0e-12):
        raise _fail(
            f"frame {frame_index} actor {actor_id} joints changed during readback"
        )


def _load_actor_templates(simulator: Any, tracks: Sequence[Any]) -> list[str]:
    handles: list[str] = []
    manager = simulator.metadata_mediator.ao_template_manager
    for track in tracks:
        config_path = track.bundle.paths_by_role["habitat_ao_config"]
        loaded = manager.load_configs(str(config_path))
        prefix = config_path.stem.removesuffix(".ao_config")
        candidates = manager.get_template_handles(prefix)
        if len(loaded) != 1 or len(candidates) != 1:
            raise _fail(f"actor {track.actor_id} does not have one AO template")
        handles.append(candidates[0])
    return handles


def _camera_pose(room_inputs: Any, installed_runtime: Any) -> tuple[np.ndarray, Any]:
    rig_transform = room_inputs.request["primary_camera_rig"]["world_from_rig"]
    position = np.asarray(rig_transform["translation_m"], dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise _fail("M1 camera rig position is not finite")
    x, y, z, w = normalized_quaternion_xyzw(rig_transform["rotation_xyzw"])
    return position, installed_runtime.quaternion.quaternion(w, x, y, z)


def _require_camera_and_sensor_pose_match(
    rig_transform: Any,
    snapshot: Any,
    *,
    frame_index: int,
) -> None:
    errors = [transform_error(rig_transform, snapshot["agent"])]
    errors.extend(
        transform_error(rig_transform, sensor_pose)
        for sensor_pose in snapshot["sensors"].values()
    )
    if not errors or max(errors) > ROOT_READBACK_ATOL:
        raise _fail(f"frame {frame_index} M1 camera/sensor readback differs")


def _require_no_case_semantic_collision(
    semantic: Any, semantic_ids: Sequence[int]
) -> None:
    image = np.asarray(semantic)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise _fail("no-actor semantic preflight is not an integer HxW observation")
    collisions = {
        int(semantic_id): int(np.count_nonzero(image == semantic_id))
        for semantic_id in semantic_ids
    }
    if any(collisions.values()):
        raise _fail(
            f"selected MP3D scene already uses a case actor semantic ID: {collisions}"
        )


def _receipt_actor_axis_bindings(
    case: Any,
    runtime_object_ids: Sequence[int],
    runtime_emitter_link_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if not (
        len(case.actor_tracks)
        == len(runtime_object_ids)
        == len(runtime_emitter_link_ids)
        == 2
    ):
        raise _fail("audio-ready visual receipt requires exactly two actor axes")
    return [
        {
            "axis_index": axis_index,
            "actor_id": track.actor_id,
            "source_id": track.source_id,
            "semantic_id": track.semantic_id,
            "runtime_object_id": int(runtime_object_ids[axis_index]),
            "emitter_link_name": track.emitter_link_name,
            "runtime_emitter_link_id": int(runtime_emitter_link_ids[axis_index]),
            "audio_endpoint_status": "pending",
            "audio_endpoint_id": None,
        }
        for axis_index, track in enumerate(case.actor_tracks)
    ]


def _target_only_semantic_frame(
    semantic: Any,
    object_id: Any,
    *,
    semantic_id: int,
    runtime_object_id: int,
) -> np.ndarray:
    """Retain one target mask only after semantic/object-ID pixel agreement."""

    semantic_array = np.asarray(semantic)
    object_id_array = np.asarray(object_id)
    if (
        semantic_array.ndim != 2
        or object_id_array.shape != semantic_array.shape
        or not np.issubdtype(semantic_array.dtype, np.integer)
        or not np.issubdtype(object_id_array.dtype, np.integer)
    ):
        raise _fail("target-only semantic/object-ID observations are invalid")
    semantic_mask = semantic_array == semantic_id
    object_mask = object_id_array == runtime_object_id
    if not np.array_equal(semantic_mask, object_mask):
        raise _fail("target-only semantic and runtime object-ID pixels differ")
    unexpected = np.unique(semantic_array[(semantic_array != 0) & ~semantic_mask])
    if unexpected.size:
        raise _fail(
            "target-only blank-scene semantic pass contains another semantic ID: "
            f"{unexpected.tolist()}"
        )
    return np.where(semantic_mask, semantic_id, 0).astype(np.int32)


def _require_target_only_replay_match(
    *,
    normal_root: np.ndarray,
    normal_joints: np.ndarray,
    target_root: np.ndarray,
    target_joints: np.ndarray,
    frame_index: int,
    actor_id: str,
) -> None:
    if (
        normal_root.shape != target_root.shape
        or float(np.max(np.abs(normal_root - target_root))) > ROOT_READBACK_ATOL
    ):
        raise _fail(
            f"frame {frame_index} actor {actor_id} target-only root differs "
            "from the normal pass"
        )
    if _quaternion_block_error(normal_joints, target_joints) > JOINT_READBACK_ATOL:
        raise _fail(
            f"frame {frame_index} actor {actor_id} target-only joints differ "
            "from the normal pass"
        )


def _target_only_configuration(
    *,
    room_inputs: Any,
    installed_runtime: Any,
    scene: Mapping[str, Any],
    gpu_device_id: int,
) -> tuple[Any, dict[str, str]]:
    configuration, modality_to_uuid = _make_current_configuration(
        room_inputs=room_inputs,
        installed_runtime=installed_runtime,
        scene=scene,
        include_audio_sensor=False,
        gpu_device_id=gpu_device_id,
    )
    configuration.sim_cfg.scene_id = "NONE"
    configuration.sim_cfg.load_semantic_mesh = False
    return configuration, modality_to_uuid


def _capture_target_only_semantic_passes(
    *,
    case: Any,
    plan: Any,
    room_inputs: Any,
    installed_runtime: Any,
    scene: Mapping[str, Any],
    gpu_device_id: int,
    camera_position: np.ndarray,
    camera_rotation: Any,
    rig_transform: Mapping[str, Any],
    normal_sensor_order: Sequence[str],
    normal_camera_poses: Sequence[np.ndarray],
    normal_sensor_poses: Sequence[np.ndarray],
    normal_skin_matrices: Sequence[np.ndarray],
    normal_joint_arrays: Sequence[Sequence[np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """Replay each actor alone in Habitat's blank stage with exact readback."""

    habitat_sim = installed_runtime.habitat_sim
    mn = installed_runtime.magnum
    qt = installed_runtime.quaternion
    results: dict[str, dict[str, Any]] = {}
    for actor_index, (track, track_plan) in enumerate(
        zip(case.actor_tracks, plan.actor_tracks, strict=True)
    ):
        configuration, modality_to_uuid = _target_only_configuration(
            room_inputs=room_inputs,
            installed_runtime=installed_runtime,
            scene=scene,
            gpu_device_id=gpu_device_id,
        )
        sensor_modalities = (*FORMAL_MODALITIES, CURRENT_OBJECT_ID_MODALITY)
        sensor_uuids = [modality_to_uuid[name] for name in sensor_modalities]
        semantic_frames: list[np.ndarray] = []
        object_id_frames: list[np.ndarray] = []
        root_readbacks: list[np.ndarray] = []
        joint_readbacks: list[np.ndarray] = []
        camera_readbacks: list[np.ndarray] = []
        sensor_readbacks: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        with habitat_sim.Simulator(configuration) as simulator:
            simulator.seed(int(room_inputs.request["seed"]))
            camera_state = habitat_sim.AgentState()
            camera_state.position = camera_position
            camera_state.rotation = camera_rotation
            agent = simulator.initialize_agent(0, camera_state)
            sensors = [simulator.sensors[uuid] for uuid in sensor_uuids]
            sensor_order = sorted(sensor_uuids)
            if list(sensor_order) != list(normal_sensor_order):
                raise _fail("target-only sensor axis differs from the normal pass")
            blank = _observation_arrays(
                simulator.render_sensors(sensors), modality_to_uuid
            )
            if np.any(np.asarray(blank["semantic"]) != 0):
                raise _fail("Habitat blank stage is not semantically empty")
            handle = _load_actor_templates(simulator, [track])[0]
            actor, binding = _instantiate_semantic_actor(
                simulator,
                bundle=track.bundle,
                habitat_sim=habitat_sim,
                base_handle=handle,
                semantic_id=track.semantic_id,
                actor_index=actor_index,
            )
            runtime_object_id = getattr(actor, "object_id", None)
            if (
                isinstance(runtime_object_id, bool)
                or not isinstance(runtime_object_id, int)
                or runtime_object_id < 0
            ):
                raise _fail(
                    f"target-only actor {track.actor_id} runtime object_id is invalid"
                )
            initial_world_time = float(simulator.get_world_time())
            for frame_index in range(FRAME_COUNT):
                frame_camera = habitat_sim.AgentState()
                frame_camera.position = camera_position
                frame_camera.rotation = camera_rotation
                agent.set_state(
                    frame_camera, reset_sensors=False, infer_sensor_states=True
                )
                pre_render_snapshot = _state_snapshot(
                    simulator,
                    agent,
                    sensor_uuids,
                    installed_runtime.quat_to_coeffs,
                )
                _require_camera_and_sensor_pose_match(
                    rig_transform,
                    pre_render_snapshot,
                    frame_index=frame_index,
                )
                actor_frame = track_plan.frames[frame_index]
                planned_skin = np.asarray(actor_frame.derived_world_from_skin_root)
                joints = np.asarray(
                    binding.map_pose(
                        np.asarray(
                            [rotation for _, rotation in actor_frame.joint_states]
                        )
                    ),
                    dtype=np.float64,
                )
                _apply_root_with_habitat(actor, planned_skin, qt=qt, mn=mn)
                actor.joint_positions = joints.copy()
                target_root, target_joints = _read_actor_state(actor)
                _require_target_only_replay_match(
                    normal_root=np.asarray(normal_skin_matrices[frame_index])[
                        actor_index
                    ],
                    normal_joints=np.asarray(
                        normal_joint_arrays[actor_index][frame_index]
                    ),
                    target_root=target_root,
                    target_joints=target_joints,
                    frame_index=frame_index,
                    actor_id=track.actor_id,
                )
                arrays = _observation_arrays(
                    simulator.render_sensors(sensors), modality_to_uuid
                )
                after_root, after_joints = _read_actor_state(actor)
                _state_equal(
                    (target_root, target_joints),
                    (after_root, after_joints),
                    frame_index=frame_index,
                    actor_id=track.actor_id,
                )
                post_render_snapshot = _state_snapshot(
                    simulator,
                    agent,
                    sensor_uuids,
                    installed_runtime.quat_to_coeffs,
                )
                _require_render_snapshot_unchanged(
                    pre_render_snapshot,
                    post_render_snapshot,
                    sensor_order=sensor_order,
                    frame_index=frame_index,
                )
                if (
                    abs(float(simulator.get_world_time()) - initial_world_time)
                    > WORLD_TIME_ATOL
                ):
                    raise _fail(
                        f"frame {frame_index} target-only render advanced world time"
                    )
                camera_pose = _pose_vector(
                    post_render_snapshot["agent"], owner="target-only camera"
                )
                sensor_pose = np.stack(
                    [
                        _pose_vector(post_render_snapshot["sensors"][uuid], owner=uuid)
                        for uuid in sensor_order
                    ]
                )
                if not np.allclose(
                    camera_pose,
                    normal_camera_poses[frame_index],
                    rtol=0.0,
                    atol=ROOT_READBACK_ATOL,
                ) or not np.allclose(
                    sensor_pose,
                    normal_sensor_poses[frame_index],
                    rtol=0.0,
                    atol=ROOT_READBACK_ATOL,
                ):
                    raise _fail(
                        f"frame {frame_index} target-only camera readback differs"
                    )
                semantic = _target_only_semantic_frame(
                    arrays["semantic"],
                    arrays[CURRENT_OBJECT_ID_MODALITY],
                    semantic_id=track.semantic_id,
                    runtime_object_id=runtime_object_id,
                )
                object_record = _object_id_pixel_record(
                    arrays[CURRENT_OBJECT_ID_MODALITY], runtime_object_id
                )
                semantic_frames.append(semantic)
                object_id_frames.append(
                    np.asarray(arrays[CURRENT_OBJECT_ID_MODALITY]).copy()
                )
                root_readbacks.append(target_root.copy())
                joint_readbacks.append(target_joints.copy())
                camera_readbacks.append(camera_pose)
                sensor_readbacks.append(sensor_pose)
                records.append(
                    {
                        "frame_index": frame_index,
                        "actor_id": track.actor_id,
                        "source_id": track.source_id,
                        "semantic_id": track.semantic_id,
                        "target_only_runtime_object_id": runtime_object_id,
                        "target_only_pixel_count": object_record["pixel_count"],
                        "target_only_bbox_xyxy": object_record["bbox_xyxy"],
                        "normal_root_readback_match": True,
                        "normal_joint_readback_match": True,
                        "normal_camera_sensor_readback_match": True,
                        "snapshot_world_time_seconds": float(
                            post_render_snapshot["world_time_seconds"]
                        ),
                    }
                )
        results[track.actor_id] = {
            "actor_index": actor_index,
            "semantic_id": track.semantic_id,
            "runtime_object_id": runtime_object_id,
            "semantic_frames": semantic_frames,
            "object_id_frames": object_id_frames,
            "root_readbacks": root_readbacks,
            "joint_readbacks": joint_readbacks,
            "camera_readbacks": camera_readbacks,
            "sensor_readbacks": sensor_readbacks,
            "frame_records": records,
        }
    return results


def _compile_target_only_visibility(
    *,
    case: Any,
    m1_capture_identity: Mapping[str, Any],
    normal_semantic_frames: Sequence[np.ndarray],
    target_only_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    frame_count = len(normal_semantic_frames)
    if frame_count != FRAME_COUNT:
        raise _fail("same-renderer visibility requires every normal frame")
    resolution = list(np.asarray(normal_semantic_frames[0]).shape)
    rig = m1_capture_identity["primary_camera_rig"]
    camera_contract_id = (
        f"{m1_capture_identity['request_id']}:{rig['rig_id']}:{rig['view_id']}"
    )
    camera_pose_ids = [
        f"{camera_contract_id}:native_readback_frame_{index:03d}"
        for index in range(frame_count)
    ]
    common_context = {
        "renderer_backend": "habitat_sim_native_mp3d",
        "rgb_renderer_backend": "habitat_sim_native_mp3d",
        "camera_contract_id": camera_contract_id,
        "semantic_id_namespace": "m5_declared_actor_semantic_ids",
        "resolution_hw": resolution,
        "frame_indices": list(range(frame_count)),
        "camera_pose_ids": camera_pose_ids,
    }
    semantic_ids = {track.actor_id: track.semantic_id for track in case.actor_tracks}
    try:
        return compile_pixel_visibility_truth(
            normal_semantic_masks=normal_semantic_frames,
            target_only_semantic_masks_by_instance={
                track.actor_id: target_only_results[track.actor_id]["semantic_frames"]
                for track in case.actor_tracks
            },
            semantic_ids_by_instance=semantic_ids,
            normal_context={"pass_kind": "modal_scene", **common_context},
            target_only_contexts_by_instance={
                track.actor_id: {
                    "pass_kind": "target_only",
                    "target_instance_id": track.actor_id,
                    **common_context,
                }
                for track in case.actor_tracks
            },
        )
    except ValueError as exc:
        raise _fail(f"same-renderer target-only visibility is invalid: {exc}") from exc


def _capture_m1_request_path(case: Any, override: str | Path | None) -> Path:
    if override is None:
        return case.camera_m1_request_path
    candidate = Path(override)
    if candidate.is_symlink():
        raise _fail(f"capture M1 request must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"capture M1 request is missing: {candidate}") from exc
    if not resolved.is_file():
        raise _fail(f"capture M1 request is not a regular file: {resolved}")
    return resolved


def _capture_native(
    *,
    case: Any,
    plan: Any,
    room_inputs: Any,
    installed_runtime: Any,
    output: Path,
    gpu_device_id: int,
    capture_m1_request_path: Path,
    camera_binding_source: str,
) -> dict[str, Any]:
    habitat_sim = installed_runtime.habitat_sim
    mn = installed_runtime.magnum
    qt = installed_runtime.quaternion
    scene = _resolve_external_scene(room_inputs, installed_runtime)
    m1_capture_identity = _m1_capture_identity(room_inputs)
    configuration, modality_to_uuid = _make_current_configuration(
        room_inputs=room_inputs,
        installed_runtime=installed_runtime,
        scene=scene,
        include_audio_sensor=False,
        gpu_device_id=gpu_device_id,
    )
    camera_position, camera_rotation = _camera_pose(room_inputs, installed_runtime)
    rig_transform = room_inputs.request["primary_camera_rig"]["world_from_rig"]
    sensor_modalities = (*FORMAL_MODALITIES, CURRENT_OBJECT_ID_MODALITY)
    sensor_uuids = [modality_to_uuid[name] for name in sensor_modalities]
    sensors: list[Any]
    actors: list[Any] = []
    bindings: list[Any] = []
    emitter_link_ids: list[int] = []
    runtime_object_ids: list[int] = []
    detector: Any
    reader: Any
    with habitat_sim.Simulator(configuration) as simulator:
        navmesh_loaded = bool(simulator.pathfinder.load_nav_mesh(str(scene["navmesh"])))
        if not navmesh_loaded or not bool(simulator.pathfinder.is_loaded):
            raise _fail("Habitat could not load the declared MP3D navmesh")
        loaded_errors, _ = validate_loaded_scene_asset_graph(
            room_inputs,
            None,
            simulator,
            declared_navmesh_loaded=navmesh_loaded,
            mp3d_root=installed_runtime.mp3d_root,
        )
        if loaded_errors:
            raise _fail("loaded MP3D scene graph differs: " + "; ".join(loaded_errors))
        simulator.seed(int(room_inputs.request["seed"]))
        camera_state = habitat_sim.AgentState()
        camera_state.position = camera_position
        camera_state.rotation = camera_rotation
        agent = simulator.initialize_agent(0, camera_state)
        sensors = [simulator.sensors[uuid] for uuid in sensor_uuids]
        sensor_order = sorted(sensor_uuids)
        initial_world_time = float(simulator.get_world_time())
        preflight = _observation_arrays(
            simulator.render_sensors(sensors), modality_to_uuid
        )
        _require_no_case_semantic_collision(
            preflight["semantic"],
            [track.semantic_id for track in case.actor_tracks],
        )
        if (
            abs(float(simulator.get_world_time()) - initial_world_time)
            > WORLD_TIME_ATOL
        ):
            raise _fail("native visual preflight advanced world time")
        detector, reader = _runtime_contact_api(simulator)
        handles = _load_actor_templates(simulator, case.actor_tracks)
        for actor_index, track in enumerate(case.actor_tracks):
            actor, binding = _instantiate_semantic_actor(
                simulator,
                bundle=track.bundle,
                habitat_sim=habitat_sim,
                base_handle=handles[actor_index],
                semantic_id=track.semantic_id,
                actor_index=actor_index,
            )
            object_id = getattr(actor, "object_id", None)
            if (
                isinstance(object_id, bool)
                or not isinstance(object_id, int)
                or object_id < 0
            ):
                raise _fail(f"actor {track.actor_id} runtime object_id is invalid")
            if object_id in runtime_object_ids:
                raise _fail("native actor runtime object IDs are not unique")
            actors.append(actor)
            bindings.append(binding)
            runtime_object_ids.append(object_id)
            emitter_link_ids.append(_link_id_by_name(actor, track.emitter_link_name))

        rgb_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        semantic_frames: list[np.ndarray] = []
        object_id_frames: list[np.ndarray] = []
        planned_actor_matrices: list[np.ndarray] = []
        planned_skin_matrices: list[np.ndarray] = []
        actual_skin_matrices: list[np.ndarray] = []
        actual_joint_arrays: list[list[np.ndarray]] = [[], []]
        camera_poses: list[np.ndarray] = []
        sensor_poses: list[np.ndarray] = []
        listener_poses: list[np.ndarray] = []
        emitter_positions: list[np.ndarray] = []
        frame_records: list[dict[str, Any]] = []
        pair_collision_frames: list[int] = []

        for frame_index in range(FRAME_COUNT):
            frame_camera = habitat_sim.AgentState()
            frame_camera.position = camera_position
            frame_camera.rotation = camera_rotation
            agent.set_state(frame_camera, reset_sensors=False, infer_sensor_states=True)
            pre_render_snapshot = _state_snapshot(
                simulator, agent, sensor_uuids, installed_runtime.quat_to_coeffs
            )
            _require_camera_and_sensor_pose_match(
                rig_transform,
                pre_render_snapshot,
                frame_index=frame_index,
            )
            before_states: list[tuple[np.ndarray, np.ndarray]] = []
            actor_world: list[np.ndarray] = []
            skin_world: list[np.ndarray] = []
            for actor_index, (actor, binding, track_plan) in enumerate(
                zip(actors, bindings, plan.actor_tracks, strict=True)
            ):
                actor_frame = track_plan.frames[frame_index]
                planned_skin = np.asarray(actor_frame.derived_world_from_skin_root)
                planned_actor = np.asarray(actor_frame.planned_world_from_actor)
                joints = np.asarray(
                    binding.map_pose(
                        np.asarray(
                            [rotation for _, rotation in actor_frame.joint_states]
                        )
                    ),
                    dtype=np.float64,
                )
                _apply_root_with_habitat(actor, planned_skin, qt=qt, mn=mn)
                actor.joint_positions = joints.copy()
                actual_root, actual_joints = _read_actor_state(actor)
                if (
                    float(np.max(np.abs(actual_root - planned_skin)))
                    > ROOT_READBACK_ATOL
                ):
                    raise _fail(
                        f"frame {frame_index} actor {actor_index} root apply failed"
                    )
                if _quaternion_block_error(actual_joints, joints) > JOINT_READBACK_ATOL:
                    raise _fail(
                        f"frame {frame_index} actor {actor_index} joint apply failed"
                    )
                before_states.append((actual_root.copy(), actual_joints.copy()))
                actor_world.append(planned_actor.copy())
                skin_world.append(planned_skin.copy())

            world_before_collision = float(simulator.get_world_time())
            detector()
            world_after_collision = float(simulator.get_world_time())
            if abs(world_after_collision - world_before_collision) > WORLD_TIME_ATOL:
                raise _fail(
                    f"frame {frame_index} collision detection advanced world time"
                )
            contacts = reader()
            if contacts is None:
                raise _fail(f"frame {frame_index} physics contact readback is missing")
            collision = classify_collision_contacts(
                list(contacts), runtime_actor_object_ids=runtime_object_ids
            )
            collision_states = [_read_actor_state(actor) for actor in actors]
            for actor_index, (before, after) in enumerate(
                zip(before_states, collision_states, strict=True)
            ):
                _state_equal(
                    before,
                    after,
                    frame_index=frame_index,
                    actor_id=case.actor_tracks[actor_index].actor_id,
                )
            if not collision.actor_pair_collision_free:
                pair_collision_frames.append(frame_index)

            arrays = _observation_arrays(
                simulator.render_sensors(sensors), modality_to_uuid
            )
            after_states = [_read_actor_state(actor) for actor in actors]
            for actor_index, (before, after) in enumerate(
                zip(collision_states, after_states, strict=True)
            ):
                _state_equal(
                    before,
                    after,
                    frame_index=frame_index,
                    actor_id=case.actor_tracks[actor_index].actor_id,
                )
            post_render_snapshot = _state_snapshot(
                simulator, agent, sensor_uuids, installed_runtime.quat_to_coeffs
            )
            pre_time, post_time = _require_render_snapshot_unchanged(
                pre_render_snapshot,
                post_render_snapshot,
                sensor_order=sensor_order,
                frame_index=frame_index,
            )
            if abs(post_time - initial_world_time) > WORLD_TIME_ATOL:
                raise _fail(f"frame {frame_index} render advanced world time")
            frame_sources = np.stack(
                [
                    _node_world_position(actor, link_id)
                    for actor, link_id in zip(actors, emitter_link_ids, strict=True)
                ]
            )
            object_records = [
                _object_id_pixel_record(arrays[CURRENT_OBJECT_ID_MODALITY], object_id)
                for object_id in runtime_object_ids
            ]
            rgb_frames.append(
                np.asarray(arrays["rgb"])[..., :3].astype(np.uint8, copy=True)
            )
            depth_frames.append(np.asarray(arrays["depth"]).copy())
            semantic_frames.append(np.asarray(arrays["semantic"]).copy())
            object_id_frames.append(
                np.asarray(arrays[CURRENT_OBJECT_ID_MODALITY]).copy()
            )
            planned_actor_matrices.append(np.stack(actor_world))
            planned_skin_matrices.append(np.stack(skin_world))
            actual_skin_matrices.append(np.stack([state[0] for state in after_states]))
            for actor_index, (_, joints) in enumerate(after_states):
                actual_joint_arrays[actor_index].append(joints.copy())
            camera_poses.append(
                _pose_vector(post_render_snapshot["agent"], owner="camera")
            )
            sensor_poses.append(
                np.stack(
                    [
                        _pose_vector(post_render_snapshot["sensors"][uuid], owner=uuid)
                        for uuid in sensor_order
                    ]
                )
            )
            listener_poses.append(
                _pose_vector(
                    _logical_listener_pose(
                        post_render_snapshot, room_inputs.request["listener"]
                    ),
                    owner="logical listener",
                )
            )
            emitter_positions.append(frame_sources)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "pts_ticks": frame_index * 3200,
                    "source_center_plan_m": [
                        list(
                            plan.actor_tracks[actor_index]
                            .frames[frame_index]
                            .route_source_center_m
                        )
                        for actor_index in range(2)
                    ],
                    "planned_world_from_actor": [
                        matrix.tolist() for matrix in actor_world
                    ],
                    "planned_world_from_skin_root": [
                        matrix.tolist() for matrix in skin_world
                    ],
                    "actual_skin_root_readback": [
                        state[0].tolist() for state in after_states
                    ],
                    "actual_joint_readback": [
                        state[1].tolist() for state in after_states
                    ],
                    "actual_emitter_position_m": frame_sources.tolist(),
                    "source_positions_m": frame_sources.tolist(),
                    "runtime_object_ids": list(runtime_object_ids),
                    "object_id_modal": object_records,
                    "collision": {
                        "actor_actor_contacts": list(collision.actor_actor_contacts),
                        "actor_stage_contacts": list(collision.actor_stage_contacts),
                        "actor_other_contacts": list(collision.actor_other_contacts),
                        "unrelated_contacts": list(collision.unrelated_contacts),
                        "actor_pair_collision_free": collision.actor_pair_collision_free,
                    },
                    "snapshot_world_time_seconds": post_time,
                    "pre_render_snapshot_world_time_seconds": pre_time,
                    "physics_step_count": 0,
                    "pose_alignment": {
                        "array_frame_index": frame_index,
                        "actor_axis_order": [
                            track.actor_id for track in case.actor_tracks
                        ],
                        "source_axis_order": [
                            track.source_id for track in case.actor_tracks
                        ],
                        "sensor_axis_order": list(sensor_order),
                        "pose_component_order": [
                            "translation_x_m",
                            "translation_y_m",
                            "translation_z_m",
                            "quaternion_x",
                            "quaternion_y",
                            "quaternion_z",
                            "quaternion_w",
                        ],
                    },
                    "observation_calls": 1,
                }
            )

    if len(frame_records) != FRAME_COUNT:
        raise _fail("native multi-actor capture must retain exactly 75 frames")
    arrays_root = output / "arrays"
    arrays_root.mkdir()
    target_only_results = _capture_target_only_semantic_passes(
        case=case,
        plan=plan,
        room_inputs=room_inputs,
        installed_runtime=installed_runtime,
        scene=scene,
        gpu_device_id=gpu_device_id,
        camera_position=camera_position,
        camera_rotation=camera_rotation,
        rig_transform=rig_transform,
        normal_sensor_order=sensor_order,
        normal_camera_poses=camera_poses,
        normal_sensor_poses=sensor_poses,
        normal_skin_matrices=actual_skin_matrices,
        normal_joint_arrays=actual_joint_arrays,
    )
    visibility_truth = _compile_target_only_visibility(
        case=case,
        m1_capture_identity=m1_capture_identity,
        normal_semantic_frames=semantic_frames,
        target_only_results=target_only_results,
    )
    target_only_records: dict[str, Any] = {}
    for actor_index, track in enumerate(case.actor_tracks):
        result = target_only_results[track.actor_id]
        _save_array(
            arrays_root,
            f"target_only_semantic_actor{actor_index}.npy",
            result["semantic_frames"],
        )
        _save_array(
            arrays_root,
            f"target_only_object_id_actor{actor_index}.npy",
            result["object_id_frames"],
        )
        _save_array(
            arrays_root,
            f"target_only_skin_root_readback_actor{actor_index}.npy",
            result["root_readbacks"],
        )
        _save_array(
            arrays_root,
            f"target_only_joint_readback_actor{actor_index}.npy",
            result["joint_readbacks"],
        )
        _save_array(
            arrays_root,
            f"target_only_camera_poses_actor{actor_index}.npy",
            result["camera_readbacks"],
        )
        _save_array(
            arrays_root,
            f"target_only_sensor_poses_actor{actor_index}.npy",
            result["sensor_readbacks"],
        )
        target_only_records[track.actor_id] = {
            "actor_index": actor_index,
            "semantic_id": track.semantic_id,
            "target_only_runtime_object_id": result["runtime_object_id"],
            "frames": result["frame_records"],
        }
    write_json(
        output / "target_only_frame_records.json",
        {"per_actor": target_only_records},
    )
    write_json(output / "pixel_visibility_truth.json", visibility_truth)
    _save_array(arrays_root, "rgb.npy", rgb_frames)
    _save_array(arrays_root, "depth.npy", depth_frames)
    _save_array(arrays_root, "semantic.npy", semantic_frames)
    _save_array(arrays_root, "object_id.npy", object_id_frames)
    _save_array(arrays_root, "planned_actor_world_matrices.npy", planned_actor_matrices)
    _save_array(
        arrays_root, "planned_applied_skin_root_matrices.npy", planned_skin_matrices
    )
    _save_array(
        arrays_root, "actual_skin_root_readback_matrices.npy", actual_skin_matrices
    )
    _save_array(arrays_root, "actual_agent_camera_poses.npy", camera_poses)
    _save_array(arrays_root, "actual_sensor_poses.npy", sensor_poses)
    _save_array(arrays_root, "actual_logical_listener_poses.npy", listener_poses)
    _save_array(arrays_root, "emitter_positions_m.npy", emitter_positions)
    for actor_index, values in enumerate(actual_joint_arrays):
        _save_array(
            arrays_root, f"actual_joint_readback_actor{actor_index}.npy", values
        )
    write_json(output / "frame_records.json", {"frames": frame_records})
    receipt = {
        "status": "rejected_actor_pair_collision"
        if pair_collision_frames
        else "research_only",
        "research_only": True,
        "formal_m2_request": False,
        "episode_counted": False,
        "inputs": {
            "m1_request": {"path": str(capture_m1_request_path)},
            "room_manifest": {"path": str(room_inputs.room_path)},
            "case_manifest": {"path": str(case.manifest_path)},
        },
        "selected_room": {
            "room_id": room_inputs.room["room_id"],
            "room_kind": room_inputs.room["room_kind"],
        },
        "m1_capture_identity": m1_capture_identity,
        "capture": {
            "native_habitat_started": True,
            "m1_request_id": m1_capture_identity["request_id"],
            "room_id": m1_capture_identity["room_id"],
            "frame_count": FRAME_COUNT,
            "fps": 15,
            "pts_ticks": "i*3200",
            "physics_steps": 0,
            "gpu_device_id": gpu_device_id,
            "collision_detection": "perform_discrete_collision_detection",
            "pair_collision_frames": pair_collision_frames,
            "actor_pair_collision_free": not pair_collision_frames,
            "support_contact_status": "raw stage contacts only; support pending",
            "modalities": [*FORMAL_MODALITIES, CURRENT_OBJECT_ID_MODALITY],
            "audio_sensor_included": False,
            "target_only_passes": {
                "renderer_backend": "habitat_sim_native_mp3d",
                "pass_count": len(case.actor_tracks),
                "modalities": ["semantic", "object_id"],
            },
        },
        "case": {
            "case_id": plan.case_id,
            "region_instance_id": plan.region_instance_id,
            "route_family_id": plan.route_family_id,
            "motion_case": plan.motion_case,
            "camera_m1_request": str(capture_m1_request_path),
            "declared_camera_m1_request": str(case.camera_m1_request_path),
            "camera_binding_source": camera_binding_source,
            "camera_participation_in_route_generation": False,
        },
        "actors": {
            "actor_ids": [track.actor_id for track in case.actor_tracks],
            "source_ids": [track.source_id for track in case.actor_tracks],
            "semantic_ids": [track.semantic_id for track in case.actor_tracks],
            "runtime_object_ids": runtime_object_ids,
            "emitter_link_names": [
                track.emitter_link_name for track in case.actor_tracks
            ],
            "target_only_runtime_object_ids": [
                target_only_results[track.actor_id]["runtime_object_id"]
                for track in case.actor_tracks
            ],
            "source_center_plan_authority": "r4 route source-center points",
            "actual_emitter_authority": "native link SceneNode readback",
            "axis_bindings": _receipt_actor_axis_bindings(
                case, runtime_object_ids, emitter_link_ids
            ),
        },
        "visibility": {
            "object_id_zero_pixels_affect_legality": False,
            "semantic_visibility_affect_legality": False,
            "candidate_filtering": "disabled",
            "authority": visibility_truth["authority"],
            "formal_modal_visibility": visibility_truth,
            "target_only_authority": {
                "renderer_backend": "habitat_sim_native_mp3d",
                "stage_policy": "Habitat NONE blank stage",
                "actor_pose_policy": "exact normal-pass root/joint readback replay",
                "camera_policy": "exact normal-pass agent/sensor readback",
                "semantic_object_id_pixel_agreement_required": True,
            },
        },
        "artifacts": {
            "arrays": [f"arrays/{name}" for name in ARRAY_ARTIFACTS],
            "frame_records": "frame_records.json",
            "target_only_frame_records": "target_only_frame_records.json",
            "pixel_visibility_truth": "pixel_visibility_truth.json",
            "source_positions_m": "arrays/emitter_positions_m.npy",
            "actual_agent_camera_poses": "arrays/actual_agent_camera_poses.npy",
            "actual_sensor_poses": "arrays/actual_sensor_poses.npy",
            "actual_logical_listener_poses": (
                "arrays/actual_logical_listener_poses.npy"
            ),
        },
        "native_pending": {"rlr": None, "formal_admission": None},
    }
    write_json(output / "research_receipt.json", receipt)
    return receipt


def capture_mp3d_region_case(
    *,
    case_manifest_path: str | Path,
    room_manifest_path: str | Path,
    runtime_prefix: str | Path,
    mp3d_root: str | Path,
    magnum_python_site: str | Path,
    output_directory: str | Path,
    rlr_sdk_root: str | Path | None = None,
    gpu_device_id: int = 0,
    m1_request_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and natively capture one case with no output replacement."""

    case = load_multi_actor_case(case_manifest_path)
    plan = build_apply_plan(case)
    capture_m1_request = _capture_m1_request_path(case, m1_request_path)
    camera_binding_source = (
        "case_manifest" if m1_request_path is None else "explicit_posthoc_rebind"
    )
    try:
        room_inputs = load_m1_inputs(room_manifest_path, capture_m1_request)
        static_errors = validate_scene_asset_graph(
            room_inputs, None, mp3d_root=mp3d_root
        )
    except Exception as exc:
        raise _fail(f"M1/MP3D room validation failed: {exc}") from exc
    if static_errors:
        raise _fail(
            "MP3D scene graph differs from declared room: " + "; ".join(static_errors)
        )
    try:
        installed = prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_python_site,
            rlr_sdk_root=rlr_sdk_root,
        )
        _require_bullet(installed)
    except Exception as exc:
        raise _fail(f"installed Habitat runtime unavailable: {exc}") from exc
    output = _fresh_output_directory(output_directory)
    return _capture_native(
        case=case,
        plan=plan,
        room_inputs=room_inputs,
        installed_runtime=installed,
        output=output,
        gpu_device_id=gpu_device_id,
        capture_m1_request_path=capture_m1_request,
        camera_binding_source=camera_binding_source,
    )


def compare_capture_outputs(first: str | Path, second: str | Path) -> dict[str, Any]:
    """Compare repeated fresh captures; any byte-level evidence drift fails."""

    first_root = Path(first).resolve()
    second_root = Path(second).resolve()
    differing: list[str] = []
    for name in ARRAY_ARTIFACTS:
        left = np.load(first_root / "arrays" / name, allow_pickle=False)
        right = np.load(second_root / "arrays" / name, allow_pickle=False)
        if left.shape != right.shape or not np.array_equal(left, right):
            differing.append(f"arrays/{name}")
    for name in JSON_ARTIFACTS:
        left_records = load_json(first_root / name)
        right_records = load_json(second_root / name)
        if left_records != right_records:
            differing.append(name)
    return {
        "status": "pass" if not differing else "fail",
        "differing_artifacts": differing,
    }


__all__ = [
    "NativeMultiActorVisualError",
    "capture_mp3d_region_case",
    "compare_capture_outputs",
]
