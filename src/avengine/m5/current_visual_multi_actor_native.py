"""Native Habitat runner for one research-only two-actor MP3D case.

The CPU loader and apply-plan are the authority for input validation.  This
module adds exactly one native Simulator invocation: two independently loaded
actor bundles share the validated M1 camera/sensor rig, while every actor
root/joint application and readback remains on its own axis.  It is visual
research evidence only; it does not run audio or turn visibility into a gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
    "actual_agent_camera_poses.npy",
    "actual_sensor_poses.npy",
    "actual_logical_listener_poses.npy",
    "emitter_positions_m.npy",
    "actual_joint_readback_actor0.npy",
    "actual_joint_readback_actor1.npy",
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
                }
            )

    if len(frame_records) != FRAME_COUNT:
        raise _fail("native multi-actor capture must retain exactly 75 frames")
    arrays_root = output / "arrays"
    arrays_root.mkdir()
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
        "capture": {
            "native_habitat_started": True,
            "frame_count": FRAME_COUNT,
            "fps": 15,
            "pts_ticks": "i*3200",
            "physics_steps": 0,
            "gpu_device_id": gpu_device_id,
            "collision_detection": "perform_discrete_collision_detection",
            "pair_collision_frames": pair_collision_frames,
            "actor_pair_collision_free": not pair_collision_frames,
            "support_contact_status": "raw stage contacts only; support pending",
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
            "source_center_plan_authority": "r4 route source-center points",
            "actual_emitter_authority": "native link SceneNode readback",
        },
        "visibility": {
            "object_id_zero_pixels_affect_legality": False,
            "semantic_visibility_affect_legality": False,
            "authority": (
                "installed-runtime OBJECT_ID raster; active-RGB depth equivalence "
                "and formal modal visibility remain pending"
            ),
            "formal_modal_visibility": None,
        },
        "artifacts": {
            "arrays": [f"arrays/{name}" for name in ARRAY_ARTIFACTS],
            "frame_records": "frame_records.json",
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
    left_records = load_json(first_root / "frame_records.json")
    right_records = load_json(second_root / "frame_records.json")
    if left_records != right_records:
        differing.append("frame_records.json")
    return {
        "status": "pass" if not differing else "fail",
        "differing_artifacts": differing,
    }


__all__ = [
    "NativeMultiActorVisualError",
    "capture_mp3d_region_case",
    "compare_capture_outputs",
]
