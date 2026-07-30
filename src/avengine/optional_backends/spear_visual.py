"""Compile authoritative AVEngine records into a SPEAR visual-only plan.

This module is deliberately a pure-Python boundary: importing or compiling a
plan never imports SPEAR or Unreal bindings and never starts a native runtime.
Timeline v2 owns every actor state and binds each SensorRig pose hash, the
SensorRigTrajectory owns exact camera/listener state, the RoomCapsule owns the
room identity and layout, and native room qualification owns the source-center
gate. A SPEAR consumer may render the resulting plan only as
``comparison_visual``; it is not a second navigation, source-logic, audio, or
admission authority.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m5_1.orientation import habitat_yaw_degrees_from_xyzw


PLAN_SCHEMA = "avengine_optional_spear_visual_plan_v1"
BACKEND_ROLE = "comparison_visual"
TIMELINE_SCHEMA = "avengine_authoritative_timeline_v2"
ROOM_CAPSULE_SCHEMA = "avengine_m6x_room_capsule_v1"
FRAME_COUNT = 75

# These axes are declared by the current Habitat runtime packages, not inferred
# from species names.  Callers can extend/override the mapping for another
# versioned body plan without adding a cat/bird/etc. branch to this compiler.
DEFAULT_BODY_PLAN_FORWARD_AXES: Mapping[str, tuple[float, float, float]] = {
    "biped_human": (0.0, 0.0, 1.0),
    "quadruped_canine": (1.0, 0.0, 0.0),
    "quadruped_mammal_felid_v1": (1.0, 0.0, 0.0),
    "rigid_object": (1.0, 0.0, 0.0),
    "environmental_source": (1.0, 0.0, 0.0),
}


class SpearVisualPlanError(ValueError):
    """An input cannot produce a closed comparison-visual plan."""


@dataclass(frozen=True)
class SpearActorBinding:
    """One asset's already-imported UE visual and animation binding.

    ``ue_anatomical_forward_yaw_deg`` is the yaw of the asset's anatomical
    forward axis in its own UE actor frame: ``+X`` is 0 degrees and ``+Y`` is
    90 degrees.  Subtracting it from the desired UE world-forward yaw yields
    the UE actor rotation for any compatible body plan.
    """

    blueprint_class_path: str
    idle_animation: str
    walking_animation: str
    ue_anatomical_forward_yaw_deg: float

    @classmethod
    def from_value(cls, value: Any, *, asset_id: str) -> "SpearActorBinding":
        if isinstance(value, cls):
            binding = value
        elif isinstance(value, Mapping):
            required = {
                "blueprint_class_path",
                "idle_animation",
                "walking_animation",
                "ue_anatomical_forward_yaw_deg",
            }
            missing = sorted(required - set(value))
            if missing:
                raise SpearVisualPlanError(
                    f"actor binding {asset_id!r} lacks {', '.join(missing)}"
                )
            binding = cls(
                blueprint_class_path=value["blueprint_class_path"],
                idle_animation=value["idle_animation"],
                walking_animation=value["walking_animation"],
                ue_anatomical_forward_yaw_deg=value[
                    "ue_anatomical_forward_yaw_deg"
                ],
            )
        else:
            raise SpearVisualPlanError(
                f"actor binding {asset_id!r} must be a mapping or SpearActorBinding"
            )
        for field in (
            "blueprint_class_path",
            "idle_animation",
            "walking_animation",
        ):
            item = getattr(binding, field)
            if not isinstance(item, str) or not item.strip():
                raise SpearVisualPlanError(
                    f"actor binding {asset_id!r}.{field} must be a non-empty string"
                )
        yaw = binding.ue_anatomical_forward_yaw_deg
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
            raise SpearVisualPlanError(
                f"actor binding {asset_id!r} UE forward yaw must be finite"
            )
        if not math.isfinite(float(yaw)):
            raise SpearVisualPlanError(
                f"actor binding {asset_id!r} UE forward yaw must be finite"
            )
        return cls(
            blueprint_class_path=binding.blueprint_class_path,
            idle_animation=binding.idle_animation,
            walking_animation=binding.walking_animation,
            ue_anatomical_forward_yaw_deg=float(yaw),
        )


def _number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpearVisualPlanError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SpearVisualPlanError(f"{owner} must be a finite number")
    return result


def _finite_vector(value: Any, length: int, *, owner: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SpearVisualPlanError(
            f"{owner} must contain exactly {length} finite numbers"
        )
    if len(value) != length:
        raise SpearVisualPlanError(
            f"{owner} must contain exactly {length} finite numbers"
        )
    return tuple(_number(item, owner=f"{owner}[{index}]") for index, item in enumerate(value))


def _assert_finite_tree(value: Any, *, owner: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _number(value, owner=owner)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, owner=f"{owner}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, owner=f"{owner}[{index}]")
        return
    raise SpearVisualPlanError(f"{owner} contains a non-JSON value")


def _nonempty_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpearVisualPlanError(f"{owner} must be a non-empty string")
    return value


def _wrap_yaw_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0.0 else wrapped


def habitat_point_to_apartment_ue_cm(
    point_habitat_m: Sequence[float],
) -> tuple[float, float, float]:
    """Apply the legacy glTF-import transform ``U=100*(H.x,H.z,H.y)``."""

    x, y, z = _finite_vector(
        point_habitat_m, 3, owner="Habitat point"
    )
    return (100.0 * x, 100.0 * z, 100.0 * y)


def camera_ue_yaw_degrees(habitat_yaw_degrees: float) -> float:
    """Map Habitat's ``-Z`` camera forward to UE using ``-90-H_yaw``."""

    yaw = _number(habitat_yaw_degrees, owner="Habitat camera yaw")
    return _wrap_yaw_degrees(-90.0 - yaw)


def _unit_forward_axis(value: Any, *, owner: str) -> tuple[float, float, float]:
    axis = _finite_vector(value, 3, owner=owner)
    norm = math.sqrt(sum(item * item for item in axis))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise SpearVisualPlanError(f"{owner} must be a unit vector")
    if math.hypot(axis[0], axis[2]) <= 1.0e-12:
        raise SpearVisualPlanError(f"{owner} must have a horizontal component")
    return axis


def _rotate_vector_xyzw(
    rotation_xyzw: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    x, y, z, w = _finite_vector(
        rotation_xyzw, 4, owner="Timeline rotation_xyzw"
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise SpearVisualPlanError("Timeline rotation_xyzw must be a unit quaternion")
    vx, vy, vz = vector
    # Unit-quaternion vector rotation: v' = v + 2*w*(q_xyz x v)
    #                                      + 2*(q_xyz x (q_xyz x v)).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _actor_forward_and_yaw(
    rotation_xyzw: Sequence[float],
    habitat_local_anatomical_forward_axis: Sequence[float],
    ue_asset_local_forward_yaw_degrees: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    local_forward = _unit_forward_axis(
        habitat_local_anatomical_forward_axis,
        owner="Habitat local anatomical forward axis",
    )
    world_h = _rotate_vector_xyzw(rotation_xyzw, local_forward)
    horizontal_norm = math.hypot(world_h[0], world_h[2])
    if horizontal_norm <= 1.0e-12:
        raise SpearVisualPlanError(
            "Timeline rotation makes the anatomical forward vertical"
        )
    world_h_unit = (
        world_h[0] / horizontal_norm,
        0.0,
        world_h[2] / horizontal_norm,
    )
    # The legacy UE glTF import swaps Habitat Y/Z.  Anatomical forward is
    # horizontal, so the UE horizontal vector is (H.x, H.z).
    world_ue = (world_h_unit[0], world_h_unit[2], 0.0)
    desired_world_yaw = math.degrees(math.atan2(world_ue[1], world_ue[0]))
    asset_yaw = _number(
        ue_asset_local_forward_yaw_degrees,
        owner="UE asset local anatomical forward yaw",
    )
    actor_yaw = _wrap_yaw_degrees(desired_world_yaw - asset_yaw)
    return world_h_unit, world_ue, actor_yaw


def actor_ue_yaw_degrees(
    rotation_xyzw: Sequence[float],
    habitat_local_anatomical_forward_axis: Sequence[float],
    ue_asset_local_forward_yaw_degrees: float,
) -> float:
    """Return the UE actor yaw implied by an authoritative Timeline rotation."""

    return _actor_forward_and_yaw(
        rotation_xyzw,
        habitat_local_anatomical_forward_axis,
        ue_asset_local_forward_yaw_degrees,
    )[2]


def _animation_for_action(action_id: str, binding: SpearActorBinding) -> str:
    normalized = action_id.casefold()
    if normalized in {"walk", "walking"}:
        return binding.walking_animation
    if normalized in {"idle", "static"}:
        return binding.idle_animation
    raise SpearVisualPlanError(
        f"Timeline action_id {action_id!r} has no Idle/Walking SPEAR binding"
    )


def _actor_declarations(timeline: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    actors = timeline.get("actors")
    if not isinstance(actors, list) or not actors:
        raise SpearVisualPlanError("Timeline actors must be a non-empty list")
    if not all(isinstance(actor, Mapping) for actor in actors):
        raise SpearVisualPlanError("Timeline actors must contain mappings")
    actor_ids = [
        _nonempty_string(actor.get("actor_id"), owner="Timeline actor_id")
        for actor in actors
    ]
    if len(actor_ids) != len(set(actor_ids)):
        raise SpearVisualPlanError("Timeline actor_id values must be unique")
    for index, actor in enumerate(actors):
        for field in ("asset_id", "template_id", "body_plan_id"):
            _nonempty_string(
                actor.get(field), owner=f"Timeline actors[{index}].{field}"
            )
    return actors


def _validate_source_and_gate_closure(
    source_manifest: Mapping[str, Any],
    flags: Mapping[str, Any],
    qualification: Mapping[str, Any],
    actors_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SpearVisualPlanError("source_manifest.sources must be non-empty")
    if not all(isinstance(source, Mapping) for source in sources):
        raise SpearVisualPlanError("source_manifest.sources must contain mappings")
    source_ids = [
        _nonempty_string(
            source.get("source_endpoint_id"),
            owner="source_manifest source_endpoint_id",
        )
        for source in sources
    ]
    if len(source_ids) != len(set(source_ids)):
        raise SpearVisualPlanError("source_manifest source_endpoint_id values must be unique")

    source_summary: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        trajectory = source.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise SpearVisualPlanError(
                f"source_manifest.sources[{index}].trajectory must be a mapping"
            )
        if trajectory.get("frame_count") != FRAME_COUNT:
            raise SpearVisualPlanError(
                f"source_manifest source {source_ids[index]!r} must contain 75 frames"
            )
        positions = trajectory.get("positions_m")
        if not isinstance(positions, list) or len(positions) != FRAME_COUNT:
            raise SpearVisualPlanError(
                f"source_manifest source {source_ids[index]!r} positions must contain 75 frames"
            )
        for frame_index, point in enumerate(positions):
            _finite_vector(
                point,
                3,
                owner=(
                    f"source_manifest source {source_ids[index]!r} "
                    f"positions_m[{frame_index}]"
                ),
            )
        endpoint = source.get("endpoint")
        binding = endpoint.get("binding") if isinstance(endpoint, Mapping) else None
        if not isinstance(binding, Mapping):
            raise SpearVisualPlanError(
                f"source_manifest source {source_ids[index]!r} lacks endpoint.binding"
            )
        entity_actor_id = binding.get("entity_instance_id")
        if entity_actor_id is not None:
            if entity_actor_id not in actors_by_id:
                raise SpearVisualPlanError(
                    f"source_manifest actor {entity_actor_id!r} does not resolve in Timeline"
                )
            entity_asset_id = binding.get("entity_asset_id")
            if entity_asset_id != actors_by_id[entity_actor_id]["asset_id"]:
                raise SpearVisualPlanError(
                    f"source_manifest actor {entity_actor_id!r} asset differs from Timeline"
                )
        source_summary.append(
            {
                "source_endpoint_id": source_ids[index],
                "activation": source.get("activation"),
                "entity_actor_id": entity_actor_id,
            }
        )

    source_flags = flags.get("source_flags")
    if not isinstance(source_flags, Mapping) or set(source_flags) != set(source_ids):
        raise SpearVisualPlanError(
            "flags.source_flags must close exactly over source_manifest sources"
        )

    if qualification.get("status") != "pass":
        raise SpearVisualPlanError("room qualification status must be pass")
    gate = qualification.get("source_center_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "pass":
        raise SpearVisualPlanError("source-center gate status must be pass")
    failed = gate.get("failed_source_frame_indices")
    if not isinstance(failed, Mapping):
        raise SpearVisualPlanError(
            "source-center gate failed_source_frame_indices must be a mapping"
        )
    gate_sources = gate.get("sources")
    if not isinstance(gate_sources, Mapping):
        raise SpearVisualPlanError("source-center gate sources must be a mapping")
    for source_id in source_ids:
        record = gate_sources.get(source_id)
        if not isinstance(record, Mapping) or record.get("status") != "pass":
            raise SpearVisualPlanError(
                f"source-center gate for {source_id!r} must be pass"
            )
        if failed.get(source_id):
            raise SpearVisualPlanError(
                f"source-center gate for {source_id!r} retains failed frames"
            )
        failed_indices = record.get("failed_frame_indices")
        if failed_indices not in (None, []):
            raise SpearVisualPlanError(
                f"source-center gate for {source_id!r} retains failed frames"
            )
        frames = record.get("frames")
        if isinstance(frames, list) and any(
            not isinstance(frame, Mapping) or frame.get("status") != "pass"
            for frame in frames
        ):
            raise SpearVisualPlanError(
                f"source-center gate for {source_id!r} contains a failed frame"
            )
    return source_summary


def _compact_flag_summary(flags: Mapping[str, Any]) -> dict[str, Any]:
    clip_flags = flags.get("clip_flags")
    if not isinstance(clip_flags, Mapping):
        raise SpearVisualPlanError("flags.clip_flags must be a mapping")
    compact: dict[str, dict[str, Any]] = {}
    for flag_id, assessment in clip_flags.items():
        if not isinstance(flag_id, str) or not isinstance(assessment, Mapping):
            raise SpearVisualPlanError("flags.clip_flags contains an invalid assessment")
        status = assessment.get("status")
        value = assessment.get("value")
        if status not in {"present", "absent", "not_evaluated"}:
            raise SpearVisualPlanError(
                f"flags.clip_flags[{flag_id!r}] has an invalid status"
            )
        if value is not None and not isinstance(value, bool):
            raise SpearVisualPlanError(
                f"flags.clip_flags[{flag_id!r}].value must be boolean or null"
            )
        compact[flag_id] = {"status": status, "value": value}
    return compact


def _compiled_camera_states(
    *,
    timeline_frames: Sequence[Mapping[str, Any]],
    sensor_rig_trajectory: Mapping[str, Any] | None,
    qualification_position_m: Sequence[float],
    qualification_yaw_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compile exact per-frame Habitat and UE camera states.

    A supplied SensorRigTrajectory is authoritative and must be cross-bound to
    Timeline ``view0``.  Legacy inputs without that sidecar retain their old
    fixed qualification pose, but it is materialized explicitly on every
    frame so the UE executor has only one state-consumption path.
    """

    # Keep this dependency lazy: sensor_rig_trajectory imports the M6x
    # trajectory package, whose public package surface includes optional
    # backend registries.  A module-level import here would make importing the
    # standalone SensorRig contract depend on SPEAR import order.
    from avengine.sensor_rig_trajectory import (
        compute_sensor_rig_pose_hash,
        materialize_sensor_rig_trajectory,
        validate_sensor_rig_trajectory,
    )

    explicit_trajectory = sensor_rig_trajectory is not None
    if explicit_trajectory:
        if not isinstance(sensor_rig_trajectory, Mapping):
            raise SpearVisualPlanError("sensor_rig_trajectory must be a mapping")
        errors = validate_sensor_rig_trajectory(sensor_rig_trajectory)
        if errors:
            raise SpearVisualPlanError(
                "sensor_rig_trajectory is invalid: " + "; ".join(errors)
            )
        trajectory = deepcopy(dict(sensor_rig_trajectory))
        state_source = "SensorRigTrajectory_v1"
    else:
        trajectory = materialize_sensor_rig_trajectory(
            trajectory_id="qualification_listener_static_compatibility_v1",
            program={
                "kind": "HOLD",
                "position_m": list(qualification_position_m),
                "yaw_deg": qualification_yaw_deg,
            },
        )
        state_source = "room_qualification_static_compatibility"

    trajectory_frames = trajectory.get("frames")
    if not isinstance(trajectory_frames, list) or len(trajectory_frames) != FRAME_COUNT:
        raise SpearVisualPlanError(
            "sensor_rig_trajectory frames must contain exactly 75 frames"
        )

    compiled: list[dict[str, Any]] = []
    for frame_index, (timeline_frame, trajectory_frame) in enumerate(
        zip(timeline_frames, trajectory_frames)
    ):
        if not isinstance(timeline_frame, Mapping):
            raise SpearVisualPlanError(
                f"Timeline frames[{frame_index}] is not a mapping"
            )
        if not isinstance(trajectory_frame, Mapping):
            raise SpearVisualPlanError(
                f"sensor_rig_trajectory frames[{frame_index}] must be a mapping"
            )
        if (
            trajectory_frame.get("frame_index") != frame_index
            or trajectory_frame.get("pts_ticks") != frame_index * 3_200
        ):
            raise SpearVisualPlanError(
                f"sensor_rig_trajectory frame {frame_index} is off the Timeline clock"
            )
        world_from_rig = trajectory_frame.get("world_from_rig")
        if not isinstance(world_from_rig, Mapping):
            raise SpearVisualPlanError(
                f"sensor_rig_trajectory frame {frame_index} lacks world_from_rig"
            )
        pose_hash = trajectory_frame.get("pose_hash")
        if pose_hash != compute_sensor_rig_pose_hash(world_from_rig):
            raise SpearVisualPlanError(
                f"sensor_rig_trajectory frame {frame_index} pose_hash does not bind "
                "world_from_rig"
            )
        position = _finite_vector(
            world_from_rig.get("translation_m"),
            3,
            owner=(
                f"sensor_rig_trajectory frame {frame_index} "
                "world_from_rig.translation_m"
            ),
        )
        rotation = _finite_vector(
            world_from_rig.get("rotation_xyzw"),
            4,
            owner=(
                f"sensor_rig_trajectory frame {frame_index} "
                "world_from_rig.rotation_xyzw"
            ),
        )
        yaw = habitat_yaw_degrees_from_xyzw(rotation)
        if explicit_trajectory:
            view_pose_hashes = timeline_frame.get("view_pose_hashes")
            if (
                not isinstance(view_pose_hashes, Mapping)
                or view_pose_hashes.get("view0") != pose_hash
            ):
                raise SpearVisualPlanError(
                    f"Timeline frame {frame_index} view0 pose hash differs from "
                    "SensorRigTrajectory"
                )
        compiled.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3_200,
                "world_from_rig": deepcopy(dict(world_from_rig)),
                "habitat_position_m": list(position),
                "habitat_yaw_deg": yaw,
                "ue_position_cm": list(habitat_point_to_apartment_ue_cm(position)),
                "ue_yaw_deg": camera_ue_yaw_degrees(yaw),
                "pose_hash": pose_hash,
            }
        )

    return (
        {
            "state_source": state_source,
            "trajectory_schema": trajectory.get("schema"),
            "trajectory_id": trajectory.get("trajectory_id"),
            "pose_hash_algorithm": trajectory.get("pose_hash_algorithm"),
            "timeline_pose_hash_crosscheck": explicit_trajectory,
        },
        compiled,
    )


def build_spear_visual_plan(
    *,
    timeline: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    flags: Mapping[str, Any],
    room_capsule: Mapping[str, Any],
    qualification: Mapping[str, Any],
    actor_bindings: Mapping[str, SpearActorBinding | Mapping[str, Any]],
    body_plan_forward_axes: Mapping[str, Sequence[float]] | None = None,
    sensor_rig_trajectory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached 75-frame visual plan without loading SPEAR.

    ``actor_bindings`` is keyed by Timeline ``asset_id``.  It provides only
    UE implementation details; it cannot replace Timeline transforms/actions
    or RoomCapsule room identity.
    """

    named_inputs = {
        "timeline": timeline,
        "source_manifest": source_manifest,
        "flags": flags,
        "room_capsule": room_capsule,
        "qualification": qualification,
    }
    for name, value in named_inputs.items():
        if not isinstance(value, Mapping):
            raise SpearVisualPlanError(f"{name} must be a mapping")
        _assert_finite_tree(value, owner=name)
    if timeline.get("schema") != TIMELINE_SCHEMA:
        raise SpearVisualPlanError("Timeline schema must be Timeline v2")
    if room_capsule.get("schema") != ROOM_CAPSULE_SCHEMA:
        raise SpearVisualPlanError("room_capsule schema is not M6.x RoomCapsule v1")
    if not isinstance(actor_bindings, Mapping):
        raise SpearVisualPlanError("actor_bindings must be keyed by asset_id")

    video = timeline.get("video")
    if not isinstance(video, Mapping) or video.get("frame_count") != FRAME_COUNT:
        raise SpearVisualPlanError("Timeline video must declare exactly 75 frames")
    if (
        video.get("fps_num") != 15
        or video.get("fps_den") != 1
        or video.get("ticks_per_frame") != 3_200
    ):
        raise SpearVisualPlanError("Timeline 75-frame clock is not the frozen v2 clock")
    frames = timeline.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise SpearVisualPlanError("Timeline frames must contain exactly 75 frames")

    actors = _actor_declarations(timeline)
    actor_ids = [actor["actor_id"] for actor in actors]
    actors_by_id = {actor["actor_id"]: actor for actor in actors}

    normalized_bindings: dict[str, SpearActorBinding] = {}
    forward_axes: dict[str, tuple[float, float, float]] = {
        key: _unit_forward_axis(
            value, owner=f"body plan {key!r} local anatomical forward axis"
        )
        for key, value in DEFAULT_BODY_PLAN_FORWARD_AXES.items()
    }
    if body_plan_forward_axes is not None:
        if not isinstance(body_plan_forward_axes, Mapping):
            raise SpearVisualPlanError("body_plan_forward_axes must be a mapping")
        for body_plan_id, axis in body_plan_forward_axes.items():
            body_plan = _nonempty_string(
                body_plan_id, owner="body_plan_forward_axes key"
            )
            forward_axes[body_plan] = _unit_forward_axis(
                axis,
                owner=f"body plan {body_plan!r} local anatomical forward axis",
            )

    actor_plan_records: list[dict[str, Any]] = []
    for actor in actors:
        asset_id = actor["asset_id"]
        if asset_id not in actor_bindings:
            raise SpearVisualPlanError(
                f"Timeline asset {asset_id!r} has no SPEAR actor binding"
            )
        binding = SpearActorBinding.from_value(
            actor_bindings[asset_id], asset_id=asset_id
        )
        normalized_bindings[asset_id] = binding
        body_plan_id = actor["body_plan_id"]
        if body_plan_id not in forward_axes:
            raise SpearVisualPlanError(
                f"Timeline body plan {body_plan_id!r} has no Habitat forward-axis binding"
            )
        actor_plan_records.append(
            {
                "actor_id": actor["actor_id"],
                "asset_id": asset_id,
                "template_id": actor["template_id"],
                "body_plan_id": body_plan_id,
                "blueprint_class_path": binding.blueprint_class_path,
                "idle_animation": binding.idle_animation,
                "walking_animation": binding.walking_animation,
                "habitat_local_anatomical_forward_axis": list(
                    forward_axes[body_plan_id]
                ),
                "ue_anatomical_forward_yaw_deg": (
                    binding.ue_anatomical_forward_yaw_deg
                ),
            }
        )

    source_summary = _validate_source_and_gate_closure(
        source_manifest, flags, qualification, actors_by_id
    )
    room_capsule_id = _nonempty_string(
        room_capsule.get("room_capsule_id"), owner="room_capsule_id"
    )
    room_ref = room_capsule.get("room_registry_ref")
    if not isinstance(room_ref, Mapping):
        raise SpearVisualPlanError("room_capsule.room_registry_ref must be a mapping")
    room_id = _nonempty_string(room_ref.get("room_id"), owner="RoomCapsule room_id")
    if qualification.get("room_id") != room_id:
        raise SpearVisualPlanError(
            "room qualification room_id differs from the authoritative RoomCapsule"
        )

    listener = qualification.get("listener")
    if not isinstance(listener, Mapping):
        raise SpearVisualPlanError("room qualification listener must be a mapping")
    camera_habitat_position = _finite_vector(
        listener.get("position_m"), 3, owner="qualification listener.position_m"
    )
    camera_habitat_yaw = _number(
        listener.get("yaw_deg"), owner="qualification listener.yaw_deg"
    )
    camera_hfov = _number(
        listener.get("camera_hfov_degrees"),
        owner="qualification listener.camera_hfov_degrees",
    )
    if not 0.0 < camera_hfov < 180.0:
        raise SpearVisualPlanError("qualification listener camera HFOV is invalid")
    camera_state_metadata, camera_states = _compiled_camera_states(
        timeline_frames=frames,
        sensor_rig_trajectory=sensor_rig_trajectory,
        qualification_position_m=camera_habitat_position,
        qualification_yaw_deg=camera_habitat_yaw,
    )

    compiled_frames: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise SpearVisualPlanError(f"Timeline frames[{frame_index}] is not a mapping")
        if frame.get("frame_index") != frame_index:
            raise SpearVisualPlanError(
                f"Timeline frames[{frame_index}].frame_index is not exact"
            )
        if frame.get("pts_ticks") != frame_index * 3_200:
            raise SpearVisualPlanError(
                f"Timeline frames[{frame_index}].pts_ticks is not exact"
            )
        states = frame.get("actor_states")
        if not isinstance(states, list):
            raise SpearVisualPlanError(
                f"Timeline frames[{frame_index}].actor_states must be a list"
            )
        state_ids = [
            state.get("actor_id") if isinstance(state, Mapping) else None
            for state in states
        ]
        if state_ids != actor_ids:
            raise SpearVisualPlanError(
                f"Timeline actor closure fails at frame {frame_index}"
            )
        compiled_states: list[dict[str, Any]] = []
        for actor, state in zip(actors, states):
            assert isinstance(state, Mapping)
            transform = state.get("root_transform")
            if not isinstance(transform, Mapping):
                raise SpearVisualPlanError(
                    f"Timeline frame {frame_index} actor {actor['actor_id']!r} lacks root_transform"
                )
            translation = _finite_vector(
                transform.get("translation_m"),
                3,
                owner=(
                    f"Timeline frame {frame_index} actor {actor['actor_id']!r} "
                    "translation_m"
                ),
            )
            rotation = _finite_vector(
                transform.get("rotation_xyzw"),
                4,
                owner=(
                    f"Timeline frame {frame_index} actor {actor['actor_id']!r} "
                    "rotation_xyzw"
                ),
            )
            action_id = _nonempty_string(
                state.get("action_id"),
                owner=f"Timeline frame {frame_index} action_id",
            )
            action_phase = _number(
                state.get("action_phase"),
                owner=f"Timeline frame {frame_index} action_phase",
            )
            if not 0.0 <= action_phase < 1.0:
                raise SpearVisualPlanError(
                    f"Timeline frame {frame_index} action_phase must be in [0,1)"
                )
            binding = normalized_bindings[actor["asset_id"]]
            world_h, world_ue, actor_yaw = _actor_forward_and_yaw(
                rotation,
                forward_axes[actor["body_plan_id"]],
                binding.ue_anatomical_forward_yaw_deg,
            )
            compiled_states.append(
                {
                    "actor_id": actor["actor_id"],
                    "asset_id": actor["asset_id"],
                    "blueprint_class_path": binding.blueprint_class_path,
                    "action_id": action_id,
                    "action_phase": action_phase,
                    "action_time_ticks": state.get("action_time_ticks"),
                    "ue_animation": _animation_for_action(action_id, binding),
                    "translation_m": list(translation),
                    "translation_ue_cm": list(
                        habitat_point_to_apartment_ue_cm(translation)
                    ),
                    "rotation_xyzw": list(rotation),
                    "anatomical_forward_habitat_world": list(world_h),
                    "anatomical_forward_ue_world": list(world_ue),
                    "actor_yaw_ue_deg": actor_yaw,
                }
            )
        compiled_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame["pts_ticks"],
                "camera_state": camera_states[frame_index],
                "actor_states": compiled_states,
            }
        )

    source_manifest_listener = source_manifest.get("listener")
    capsule_rig = room_capsule.get("camera_listener_rig")
    if not isinstance(source_manifest_listener, Mapping) or not isinstance(
        capsule_rig, Mapping
    ):
        raise SpearVisualPlanError("listener bindings are missing")
    if source_manifest_listener.get("listener_id") != capsule_rig.get("listener_id"):
        raise SpearVisualPlanError(
            "source_manifest listener differs from the authoritative RoomCapsule"
        )
    trajectory_ref = source_manifest_listener.get("sensor_rig_trajectory")
    if trajectory_ref is not None:
        if not isinstance(trajectory_ref, Mapping):
            raise SpearVisualPlanError(
                "source_manifest listener sensor-rig reference must be a mapping"
            )
        if sensor_rig_trajectory is None:
            raise SpearVisualPlanError(
                "source_manifest declares a SensorRigTrajectory but no sidecar "
                "was supplied"
            )
        if (
            trajectory_ref.get("trajectory_id")
            != sensor_rig_trajectory.get("trajectory_id")
            or trajectory_ref.get("content_sha256")
            != canonical_json_sha256(sensor_rig_trajectory)
        ):
            raise SpearVisualPlanError(
                "source_manifest SensorRigTrajectory reference differs from "
                "the supplied sidecar"
            )

    return {
        "schema": PLAN_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "authority": {
            "actor_state": "Timeline_v2",
            "room_identity_and_layout": "RoomCapsule",
            "source_logic": "source_manifest_and_flags",
            "source_center_placement": "room_qualification",
            "camera_listener_state": camera_state_metadata["state_source"],
            "backend_may_replan": False,
        },
        "coordinate_contract": {
            "habitat_to_apartment_ue_position": (
                "U_cm=(100*H_x,100*H_z,100*H_y)"
            ),
            "camera_yaw": "UE_yaw_deg=-90-Habitat_yaw_deg",
            "actor_yaw": (
                "yaw(UE(world_from_actor*Habitat_local_anatomical_forward))"
                "-UE_asset_local_forward_yaw"
            ),
        },
        "room": {
            "room_capsule_id": room_capsule_id,
            "room_capsule_revision": room_capsule.get("revision"),
            "room_id": room_id,
            "source_scene_provenance": deepcopy(
                room_capsule.get("source_scene_provenance")
            ),
        },
        "camera": {
            "listener_id": capsule_rig.get("listener_id"),
            "horizontal_fov_deg": camera_hfov,
            "calibration_scope": "static_all_frames",
            "per_frame_state_field": "frames[].camera_state",
            "default_pose_scope": "frame_zero_compatibility_only",
            "habitat_position_m": deepcopy(
                camera_states[0]["habitat_position_m"]
            ),
            "habitat_yaw_deg": camera_states[0]["habitat_yaw_deg"],
            "ue_position_cm": deepcopy(camera_states[0]["ue_position_cm"]),
            "ue_yaw_deg": camera_states[0]["ue_yaw_deg"],
            **camera_state_metadata,
        },
        "render": {
            "frame_count": FRAME_COUNT,
            "fps_num": 15,
            "fps_den": 1,
            "ticks_per_frame": 3_200,
        },
        "actors": actor_plan_records,
        "source_logic": {
            "source_manifest_schema": source_manifest.get("schema"),
            "scenario_id": source_manifest.get("scenario_id"),
            "variant_id": source_manifest.get("variant_id"),
            "sources": source_summary,
            "flag_report_schema": flags.get("schema"),
            "clip_flags": _compact_flag_summary(flags),
        },
        "qualification": {
            "status": "pass",
            "source_center_gate_status": "pass",
            "claim_boundary": "source_center_only",
        },
        "frames": compiled_frames,
    }


def _load_mapping(path: str | Path, *, owner: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpearVisualPlanError(f"cannot load {owner}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SpearVisualPlanError(f"{owner} JSON root must be an object")
    return value


def build_spear_visual_plan_from_files(
    *,
    timeline_path: str | Path,
    source_manifest_path: str | Path,
    flags_path: str | Path,
    room_capsule_path: str | Path,
    qualification_path: str | Path,
    actor_bindings: Mapping[str, SpearActorBinding | Mapping[str, Any]],
    body_plan_forward_axes: Mapping[str, Sequence[float]] | None = None,
    sensor_rig_trajectory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load bounded JSON inputs and compile a path-free plan."""

    return build_spear_visual_plan(
        timeline=_load_mapping(timeline_path, owner="timeline"),
        source_manifest=_load_mapping(
            source_manifest_path, owner="source_manifest"
        ),
        flags=_load_mapping(flags_path, owner="flags"),
        room_capsule=_load_mapping(room_capsule_path, owner="room_capsule"),
        qualification=_load_mapping(qualification_path, owner="qualification"),
        actor_bindings=actor_bindings,
        body_plan_forward_axes=body_plan_forward_axes,
        sensor_rig_trajectory=(
            None
            if sensor_rig_trajectory_path is None
            else _load_mapping(
                sensor_rig_trajectory_path, owner="sensor_rig_trajectory"
            )
        ),
    )
