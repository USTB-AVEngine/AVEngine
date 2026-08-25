"""Habitat-only internal QA capture for an asset-bound two-animal M7 route.

The M7 throughput batch uses generic source slots only until an asset pair is
chosen.  This module closes a review render over that final binding: it applies
the exact root paths that generated the cache, keeps an asset's declared
anatomical forward axis, and derives the acoustic source centre from the same
constant asset-local emitter offset.  It is deliberately visual-only; it does
not call RLR, change a cache, or promote a research asset to dataset admission.

This is not the SPEAR/UE Apartment presentation backend.  It exists solely to
check that a chosen asset pair, its root paths, and its acoustic source centres
remain mutually consistent before an optional UE visual render is requested.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, file_record, sha256_file
from avengine.contracts.transforms import transform_error
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import (
    _git_checkout_ancestor,
    _make_configuration,
    _resolved_assets,
    _state_snapshot,
)
from avengine.assets.contracts import FORMAL_MODALITIES, load_and_validate_inputs as load_m2_inputs
from avengine.assets.habitat_capture import (
    HabitatCaptureError,
    _apply_root_with_habitat,
    _quaternion_block_error,
    _runtime_snapshot,
    _validate_observation_arrays,
    load_research_review_inputs,
    load_runtime_asset_bundle,
)
from avengine.m5.visual import _instantiate_actor_with_semantic_template
from avengine.m5_1.mixed_capture import (
    LOCOMOTION_POLICY_ID,
    locomotion_schedule_from_root_trajectory,
    trajectory_world_matrices,
)
from avengine.m5_1.orientation import habitat_yaw_degrees_from_xyzw
from avengine.m7.sensor_rig import (
    resolve_m7_sensor_rig_trajectory,
)


SCHEMA = "avengine_m7_asset_bound_two_animal_visual_capture_v1"
FRAME_RATE_HZ = 15
_ROOT_READBACK_ATOL = 2.0e-6
_JOINT_READBACK_ATOL = 2.0e-6
_RIG_READBACK_ATOL = 2.0e-6


class AssetBoundVisualReviewError(RuntimeError):
    """An asset-bound review request cannot be captured faithfully."""


def _require_explicit_runtime_root(runtime_root: str | Path | None) -> Path:
    """Resolve an explicit non-checkout runtime root for the direct writer."""

    if runtime_root is None:
        raise AssetBoundVisualReviewError(
            "direct M7 visual review requires an explicit runtime_root; "
            "ambient AVENGINE_HABITAT_RUNTIME_ROOT and sibling checkout "
            "discovery are retired"
        )
    runtime = Path(runtime_root).resolve()
    if not runtime.is_dir():
        raise AssetBoundVisualReviewError(
            f"direct M7 visual review runtime root is missing: {runtime}"
        )
    checkout_root = _git_checkout_ancestor(runtime)
    if checkout_root is not None:
        raise AssetBoundVisualReviewError(
            "direct M7 visual review runtime root must not be inside a Git "
            f"checkout: {runtime} (found .git at {checkout_root})"
        )
    return runtime


@dataclass(frozen=True)
class TwoAnimalVisualCapture:
    """Same-view RGB/semantic observations and their asset-bound closure."""

    output_dir: Path
    rgb: np.ndarray
    semantic: np.ndarray
    actor_world_matrices: np.ndarray
    acoustic_source_centers_m: np.ndarray
    semantic_visibility_pixels: np.ndarray
    listener_positions_m: np.ndarray
    listener_rotations_xyzw: np.ndarray
    sensor_rig_trajectory: Mapping[str, Any]
    evidence: Mapping[str, Any]


def _paths(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssetBoundVisualReviewError(f"{owner} must be finite [75,3]") from exc
    if result.shape != (75, 3) or not np.all(np.isfinite(result)):
        raise AssetBoundVisualReviewError(f"{owner} must be finite [75,3]")
    return np.ascontiguousarray(result)


def _vector(value: Any, *, owner: str, horizontal: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssetBoundVisualReviewError(f"{owner} must be a finite vec3") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise AssetBoundVisualReviewError(f"{owner} must be a finite vec3")
    if horizontal and (abs(float(result[1])) > 1.0e-12 or np.linalg.norm(result) <= 1.0e-12):
        raise AssetBoundVisualReviewError(f"{owner} must be a nonzero horizontal vec3")
    return np.ascontiguousarray(result)


def _save_array(output: Path, name: str, value: np.ndarray) -> Mapping[str, Any]:
    path = output / "arrays" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    reloaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if reloaded.dtype != array.dtype or reloaded.shape != array.shape or not np.array_equal(reloaded, array):
        raise AssetBoundVisualReviewError(f"{name} array readback differs")
    return {**file_record(path, relative_to=output), "dtype": array.dtype.str, "shape": list(array.shape), "readback_verified": True}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _sensor_rig_readback_errors(
    *,
    expected_world_from_rig: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    rgb_sensor_uuid: str,
    listener_uuid: str,
) -> dict[str, float]:
    """Measure the agent, formal camera and listener against one rig pose."""

    sensors = snapshot.get("sensors")
    if (
        not isinstance(sensors, Mapping)
        or rgb_sensor_uuid not in sensors
        or listener_uuid not in sensors
        or not isinstance(snapshot.get("agent"), Mapping)
    ):
        raise AssetBoundVisualReviewError(
            "camera/listener sensor-rig readback is incomplete"
        )
    try:
        sensor_errors = {
            str(sensor_uuid): transform_error(
                dict(expected_world_from_rig), dict(sensor_pose)
            )
            for sensor_uuid, sensor_pose in sensors.items()
        }
        return {
            "agent": transform_error(
                dict(expected_world_from_rig), dict(snapshot["agent"])
            ),
            "camera": sensor_errors[rgb_sensor_uuid],
            "listener": sensor_errors[listener_uuid],
            "all_sensors": max(sensor_errors.values()),
        }
    except (TypeError, ValueError, KeyError) as error:
        raise AssetBoundVisualReviewError(
            "camera/listener sensor-rig readback is invalid"
        ) from error


def _load_animal_inputs(manifest_path: str | Path, request_path: str | Path, *, research_candidate: bool) -> Any:
    return (
        load_research_review_inputs(manifest_path, request_path)
        if research_candidate
        else load_m2_inputs(manifest_path, request_path)
    )


def _world_points(matrices: np.ndarray, local_offsets: Sequence[np.ndarray]) -> np.ndarray:
    output = np.empty((75, 2, 3), dtype=np.float64)
    for index, offset in enumerate(local_offsets):
        homogeneous = np.concatenate((offset, np.asarray([1.0])))
        output[:, index] = np.einsum("nij,j->ni", matrices[:, index], homogeneous)[:, :3]
    return np.ascontiguousarray(output)


def capture_two_m2_animal_paths(
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    animal_manifest_paths: Sequence[str | Path],
    m2_request_paths: Sequence[str | Path],
    research_candidates: Sequence[bool],
    root_paths_m: Sequence[Any],
    local_forward_axes: Sequence[Any],
    emitter_offsets_m: Sequence[Any],
    actor_ids: Sequence[str],
    actor_classes: Sequence[str],
    semantic_ids: Sequence[int],
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
    route_provenance: Mapping[str, Any] | None = None,
    sensor_rig_trajectory: Mapping[str, Any] | None = None,
) -> TwoAnimalVisualCapture:
    """Capture two M2 animal assets over their selected asset-bound paths.

    The function accepts different M2 packages for the two actors.  Root paths
    are authoritative visual placements; ``emitter_offsets_m`` produce the
    acoustic centres independently of animated muzzle-bone jitter, matching
    the M7 cache contract.  The optional SensorRigTrajectory drives the formal
    camera and co-located listener every frame; omission materializes a
    complete HOLD trajectory from the historical M1 camera pose.
    """

    two = (animal_manifest_paths, m2_request_paths, research_candidates, root_paths_m, local_forward_axes, emitter_offsets_m, actor_ids, actor_classes, semantic_ids)
    if any(len(value) != 2 for value in two):
        raise AssetBoundVisualReviewError("two-actor capture requires exactly two values per field")
    if len(set(actor_ids)) != 2 or any(not isinstance(value, str) or not value for value in actor_ids):
        raise AssetBoundVisualReviewError("actor_ids must be two distinct non-empty strings")
    if len(set(semantic_ids)) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in semantic_ids):
        raise AssetBoundVisualReviewError("semantic_ids must be two distinct non-negative integers")
    if any(not isinstance(value, str) or not value for value in actor_classes):
        raise AssetBoundVisualReviewError("actor_classes must be non-empty strings")
    if any(not isinstance(value, bool) for value in research_candidates):
        raise AssetBoundVisualReviewError("research_candidates must contain booleans")

    paths = tuple(_paths(value, owner=f"actor {index} root path") for index, value in enumerate(root_paths_m))
    axes = tuple(_vector(value, owner=f"actor {index} local forward axis", horizontal=True) for index, value in enumerate(local_forward_axes))
    offsets = tuple(_vector(value, owner=f"actor {index} emitter offset") for index, value in enumerate(emitter_offsets_m))
    worlds = tuple(trajectory_world_matrices(paths[index], local_forward_axis=axes[index]) for index in range(2))
    actor_world_matrices = np.ascontiguousarray(np.stack(worlds, axis=1))
    expected_centers = _world_points(actor_world_matrices, offsets)

    output = Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise AssetBoundVisualReviewError(f"refusing to replace capture output: {output}")
    output.mkdir(parents=True)
    try:
        inputs = tuple(
            _load_animal_inputs(animal_manifest_paths[index], m2_request_paths[index], research_candidate=research_candidates[index])
            for index in range(2)
        )
        bundles = tuple(load_runtime_asset_bundle(value) for value in inputs)
        actions = tuple(
            {
                action_id: bundle.action_sets_by_role[bundle.action_roles_by_id[action_id]].action(action_id)
                for action_id in ("idle", "walk")
            }
            for bundle in bundles
        )
        schedules = tuple(
            locomotion_schedule_from_root_trajectory(paths[index], action_sample_counts={action_id: action.sample_count for action_id, action in actions[index].items()})
            for index in range(2)
        )
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
        fixed_world_from_rig = room_inputs.request["primary_camera_rig"][
            "world_from_rig"
        ]
        retained_rig_trajectory = resolve_m7_sensor_rig_trajectory(
            sensor_rig_trajectory=sensor_rig_trajectory,
            listener_position_m=fixed_world_from_rig["translation_m"],
            listener_yaw_deg=habitat_yaw_degrees_from_xyzw(
                fixed_world_from_rig["rotation_xyzw"]
            ),
        )
        rig_frames = tuple(retained_rig_trajectory["frames"])
        runtime = _require_explicit_runtime_root(runtime_root)
        if any(not record["exists"] for record in _resolved_assets(room_inputs, runtime)):
            raise AssetBoundVisualReviewError("validated room has a missing runtime asset")

        # The pinned native extension must follow numpy-quaternion in import order.
        import quaternion as qt
        import habitat_sim
        import magnum as mn
        from habitat_sim.utils.common import quat_to_coeffs

        configuration, modality_to_uuid, _listener_uuid, resolved_scene = _make_configuration(room_inputs, runtime, output / "scene_scratch")
        configuration.sim_cfg.enable_hbao = True
        if not bool(resolved_scene.get("enable_physics", False)):
            configuration.sim_cfg.enable_physics = True
        rgb_frames: list[np.ndarray] = []
        semantic_frames: list[np.ndarray] = []
        visibility_frames: list[tuple[int, int]] = []
        records: list[Mapping[str, Any]] = []
        maximum_root_error = [0.0, 0.0]
        maximum_joint_error = [0.0, 0.0]
        pose_hashes = [{"idle": set(), "walk": set()} for _ in range(2)]
        listener_positions: list[np.ndarray] = []
        listener_rotations: list[np.ndarray] = []
        maximum_rig_error = {
            "agent": 0.0,
            "camera": 0.0,
            "listener": 0.0,
            "all_sensors": 0.0,
        }

        with habitat_sim.Simulator(configuration) as simulator:
            navmesh_path = resolved_scene.get("navmesh")
            if navmesh_path is not None and Path(navmesh_path).is_file():
                simulator.pathfinder.load_nav_mesh(str(navmesh_path))
            simulator.seed(int(room_inputs.request["seed"]))
            first_world_from_rig = rig_frames[0]["world_from_rig"]
            camera_state = habitat_sim.AgentState()
            camera_state.position = np.asarray(
                first_world_from_rig["translation_m"], dtype=np.float64
            )
            x, y, z, w = first_world_from_rig["rotation_xyzw"]
            camera_state.rotation = qt.quaternion(w, x, y, z)
            agent = simulator.initialize_agent(0, camera_state)

            manager = simulator.metadata_mediator.ao_template_manager
            actors: list[Any] = []
            bindings: list[Any] = []
            for index, bundle in enumerate(bundles):
                config_path = bundle.paths_by_role["habitat_ao_config"]
                loaded = manager.load_configs(str(config_path))
                handles = manager.get_template_handles()
                base_handle = str(config_path)
                if len(loaded) != 1 or base_handle not in handles:
                    raise AssetBoundVisualReviewError(
                        f"actor {index} AO template did not load uniquely: {handles}"
                    )
                actor, binding = _instantiate_actor_with_semantic_template(
                    simulator,
                    bundle=bundle,
                    habitat_sim=habitat_sim,
                    base_handle=base_handle,
                    semantic_id=int(semantic_ids[index]),
                    actor_index=index,
                    shader_type="pbr",
                )
                actors.append(actor)
                bindings.append(binding)
            sensors = [simulator.sensors[modality_to_uuid[modality]] for modality in FORMAL_MODALITIES]
            all_sensor_uuids = sorted(
                {*modality_to_uuid.values(), _listener_uuid}
            )
            initial_world_time = float(simulator.get_world_time())

            for frame_index in range(75):
                rig_frame = rig_frames[frame_index]
                if (
                    rig_frame["frame_index"] != frame_index
                    or rig_frame["pts_ticks"] != frame_index * 3_200
                ):
                    raise AssetBoundVisualReviewError(
                        "sensor-rig trajectory differs from the visual frame clock"
                    )
                world_from_rig = rig_frame["world_from_rig"]
                frame_position = np.asarray(
                    world_from_rig["translation_m"], dtype=np.float64
                )
                qx, qy, qz, qw = world_from_rig["rotation_xyzw"]
                camera_state = habitat_sim.AgentState()
                camera_state.position = frame_position
                camera_state.rotation = qt.quaternion(qw, qx, qy, qz)
                agent.set_state(
                    camera_state,
                    reset_sensors=False,
                    infer_sensor_states=True,
                )
                rig_snapshot = _state_snapshot(
                    simulator,
                    agent,
                    all_sensor_uuids,
                    quat_to_coeffs,
                )
                rig_errors = _sensor_rig_readback_errors(
                    expected_world_from_rig=world_from_rig,
                    snapshot=rig_snapshot,
                    rgb_sensor_uuid=modality_to_uuid["rgb"],
                    listener_uuid=_listener_uuid,
                )
                for role, error in rig_errors.items():
                    maximum_rig_error[role] = max(
                        maximum_rig_error[role], error
                    )
                if max(rig_errors.values()) > _RIG_READBACK_ATOL:
                    raise AssetBoundVisualReviewError(
                        f"frame {frame_index} camera/listener readback differs"
                    )
                listener_pose = rig_snapshot["sensors"][_listener_uuid]
                listener_positions.append(
                    np.asarray(
                        listener_pose["translation_m"], dtype=np.float64
                    )
                )
                listener_rotations.append(
                    np.asarray(
                        listener_pose["rotation_xyzw"], dtype=np.float64
                    )
                )
                before: list[Mapping[str, Any]] = []
                action_rows: list[Mapping[str, Any]] = []
                for index, actor in enumerate(actors):
                    state = schedules[index][frame_index]
                    action = actions[index][state.action_id]
                    sample_index = state.action_sample_index
                    skin = worlds[index][frame_index] @ np.asarray(bundles[index].actor_from_skin_root, dtype=np.float64)
                    joints = np.asarray(bindings[index].map_pose(action.rotations_xyzw[sample_index]), dtype=np.float64)
                    _apply_root_with_habitat(actor, skin, qt=qt, mn=mn)
                    actor.joint_positions = joints.copy()
                    snapshot = _runtime_snapshot(simulator, actor)
                    root_error = float(np.max(np.abs(np.asarray(snapshot["world_from_skin_root"], dtype=np.float64) - skin)))
                    joint_error = _quaternion_block_error(np.asarray(snapshot["joint_positions_xyzw"], dtype=np.float64), joints)
                    maximum_root_error[index] = max(maximum_root_error[index], root_error)
                    maximum_joint_error[index] = max(maximum_joint_error[index], joint_error)
                    if root_error > _ROOT_READBACK_ATOL or joint_error > _JOINT_READBACK_ATOL:
                        raise AssetBoundVisualReviewError(f"frame {frame_index} actor {actor_ids[index]} state readback differs")
                    before.append(snapshot)
                    pose_hash = str(snapshot["sha256"])
                    pose_hashes[index][state.action_id].add(pose_hash)
                    action_rows.append({
                        "actor_id": actor_ids[index], "action_id": state.action_id,
                        "action_sample_index": sample_index, "action_phase": state.action_phase,
                        "root_horizontal_speed_m_s": state.horizontal_speed_m_s,
                        "root_transform_m": paths[index][frame_index].tolist(),
                        "world_from_skin_root": skin.tolist(), "state_sha256": pose_hash,
                    })
                observation = simulator.render_sensors(sensors)
                arrays = _validate_observation_arrays(observation, modality_to_uuid)
                visible = tuple(int(np.count_nonzero(arrays["semantic"] == semantic_ids[index])) for index in range(2))
                after = [_runtime_snapshot(simulator, actor) for actor in actors]
                if any(left["sha256"] != right["sha256"] for left, right in zip(before, after, strict=True)):
                    raise AssetBoundVisualReviewError(f"frame {frame_index} render mutated an articulated state")
                if not math.isclose(float(simulator.get_world_time()), initial_world_time, rel_tol=0.0, abs_tol=0.0):
                    raise AssetBoundVisualReviewError("capture unexpectedly advanced Habitat world time")
                rgb_frames.append(np.asarray(arrays["rgb"])[..., :3].astype(np.uint8, copy=True))
                semantic_frames.append(np.asarray(arrays["semantic"]).copy())
                visibility_frames.append(visible)
                records.append(
                    {
                        "frame_index": frame_index,
                        "pts_ticks": frame_index * 3200,
                        "actors": action_rows,
                        "semantic_visibility_pixels": list(visible),
                        "acoustic_source_centers_m": expected_centers[
                            frame_index
                        ].tolist(),
                        "sensor_rig": {
                            "trajectory_id": retained_rig_trajectory[
                                "trajectory_id"
                            ],
                            "view_pose_hash": rig_frame["pose_hash"],
                            "expected_world_from_rig": deepcopy(
                                world_from_rig
                            ),
                            "agent_readback": rig_snapshot["agent"],
                            "camera_readback": rig_snapshot["sensors"][
                                modality_to_uuid["rgb"]
                            ],
                            "listener_readback": listener_pose,
                            "sensor_readbacks": rig_snapshot["sensors"],
                            "transform_errors": rig_errors,
                        },
                        "observation_calls": 1,
                    }
                )

        rgb = np.ascontiguousarray(np.stack(rgb_frames))
        semantic = np.ascontiguousarray(np.stack(semantic_frames))
        visibility = np.asarray(visibility_frames, dtype=np.int64)
        listener_position_array = np.ascontiguousarray(
            np.stack(listener_positions)
        )
        listener_rotation_array = np.ascontiguousarray(
            np.stack(listener_rotations)
        )
        if np.any(np.max(visibility, axis=0) <= 0):
            raise AssetBoundVisualReviewError("the camera never observed one requested actor")
        artifacts = {
            "rgb": _save_array(output, "rgb", rgb),
            "semantic": _save_array(output, "semantic", semantic),
            "actor_world_matrices": _save_array(output, "actor_world_matrices", actor_world_matrices),
            "acoustic_source_centers_m": _save_array(output, "acoustic_source_centers_m", expected_centers),
            "semantic_visibility_pixels": _save_array(output, "semantic_visibility_pixels", visibility),
            "listener_positions_m": _save_array(
                output, "listener_positions_m", listener_position_array
            ),
            "listener_rotations_xyzw": _save_array(
                output,
                "listener_rotations_xyzw",
                listener_rotation_array,
            ),
        }
        trajectory_path = output / "sensor_rig_trajectory.json"
        _write_json(trajectory_path, retained_rig_trajectory)
        records_path = output / "frame_readback.json"
        _write_json(records_path, records)
        evidence: dict[str, Any] = {
            "schema": SCHEMA, "status": "pass", "research_only": True, "qualification_claim": False,
            "claim_boundary": "visual review closure only; source-center placement only; no new RLR call or dataset-admission claim",
            "frame_count": 75, "frame_rate_hz": FRAME_RATE_HZ, "physics_steps": 0, "observation_calls_per_frame": 1,
            "muzzle_animation_policy": "not used for acoustic position; constant final_scaled_asset_root emitter_offset_m is authoritative",
            "actors": [
                {"actor_id": actor_ids[index], "actor_class": actor_classes[index], "asset_id": inputs[index].asset["asset_id"], "asset_admission_state": inputs[index].asset["admission_state"], "semantic_id": semantic_ids[index], "local_anatomical_forward_axis": axes[index].tolist(), "emitter_offset_m": offsets[index].tolist(), "action_selection": LOCOMOTION_POLICY_ID, "actions": ["idle", "walk"]}
                for index in range(2)
            ],
            "input_paths": {"room_manifest": {"path": str(Path(room_manifest_path).resolve()), "sha256": sha256_file(room_manifest_path)}, "m1_request": {"path": str(Path(m1_request_path).resolve()), "sha256": sha256_file(m1_request_path)}, "animals": [{"asset_manifest": {"path": str(inputs[index].asset_path), "sha256": sha256_file(inputs[index].asset_path)}, "m2_request": {"path": str(inputs[index].request_path), "sha256": sha256_file(inputs[index].request_path)}} for index in range(2)], "route_provenance": dict(route_provenance or {})},
            "sensor_rig_trajectory": deepcopy(retained_rig_trajectory),
            "sensor_rig_binding": {
                "trajectory_id": retained_rig_trajectory["trajectory_id"],
                "content_sha256": canonical_json_sha256(
                    retained_rig_trajectory
                ),
                "artifact": file_record(
                    trajectory_path, relative_to=output
                ),
            },
            "readback": {"maximum_root_error_m": dict(zip(actor_ids, maximum_root_error, strict=True)), "maximum_joint_quaternion_error": dict(zip(actor_ids, maximum_joint_error, strict=True)), "maximum_sensor_rig_transform_error": maximum_rig_error, "semantic_visible_frame_count": dict(zip(actor_ids, [int(np.count_nonzero(visibility[:, index] > 0)) for index in range(2)], strict=True)), "distinct_rendered_state_count_by_action": {actor_ids[index]: {action_id: len(values) for action_id, values in pose_hashes[index].items()} for index in range(2)}, "maximum_acoustic_center_reconstruction_error_m": 0.0, "frame_records": file_record(records_path, relative_to=output)},
            "array_artifacts": artifacts,
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        _write_json(output / "evidence.json", evidence)
        return TwoAnimalVisualCapture(
            output,
            rgb,
            semantic,
            actor_world_matrices,
            expected_centers,
            visibility,
            listener_position_array,
            listener_rotation_array,
            retained_rig_trajectory,
            evidence,
        )
    except (HabitatCaptureError, OSError, ValueError) as exc:
        raise AssetBoundVisualReviewError(str(exc)) from exc


__all__ = ["AssetBoundVisualReviewError", "TwoAnimalVisualCapture", "capture_two_m2_animal_paths"]
