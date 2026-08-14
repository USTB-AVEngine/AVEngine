"""Habitat-native two-human MP3D capture authority and runtime.

The retained SPEAR suite is consumed only as a frozen timeline/state source.
It remains comparison evidence; this module never promotes its visual role.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import _resolved_scene, discover_runtime_root
from avengine.m6x.rir_cache import RIRCacheError, validate_semantic_rir_job_plan
from avengine.runtime_profiles import (
    RuntimeProfileError,
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
)


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TIME_BASE_HZ = 48_000
TICKS_PER_FRAME = 3_200
DURATION_TICKS = 240_000
ACTOR_IDS = ("source1_actor", "source2_actor")
SOURCE_SLOTS = ("source1", "source2")
SEMANTIC_IDS = (62_000, 62_001)
PACKAGE_STEMS = ("human0", "human1")
COMPARISON_VISUAL_ROLE = "comparison_visual"


class TwoHumanCaptureError(RuntimeError):
    """The two-human request, state join, or Habitat capture is invalid."""


@dataclass(frozen=True)
class HumanActorAuthority:
    actor_id: str
    source_slot_id: str
    asset_id: str
    asset_revision: str
    source_glb: Path
    semantic_id: int
    package_stem: str
    walking_profile_sample_count: int | None
    emitter_offset_m: tuple[float, float, float]
    anatomical_forward_axis: tuple[float, float, float]
    anatomical_forward_source: str


@dataclass(frozen=True)
class PlannedHumanFrame:
    frame_index: int
    pts_ticks: int
    action_id: str
    action_time_ticks: int
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TwoHumanCaptureAuthority:
    episode_id: str
    room_id: str
    room_revision: str
    actors: tuple[HumanActorAuthority, HumanActorAuthority]
    actor_frames: tuple[tuple[PlannedHumanFrame, ...], tuple[PlannedHumanFrame, ...]]
    rig_frames: tuple[Mapping[str, Any], ...]
    resolution_hw: tuple[int, int]
    horizontal_fov_deg: float
    suite_visual_role: str
    qualification_claim: bool
    formal_dataset_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TwoHumanCaptureError(message)


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{owner} must be an object")
    return value


def _sequence(value: Any, *, owner: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{owner} must be an array",
    )
    return value


def _finite_vector(value: Any, length: int, *, owner: str) -> tuple[float, ...]:
    items = _sequence(value, owner=owner)
    _require(len(items) == length, f"{owner} must contain {length} numbers")
    result: list[float] = []
    for item in items:
        _require(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item)),
            f"{owner} must contain finite numbers",
        )
        result.append(float(item))
    return tuple(result)


def _vec3(value: Any, *, owner: str) -> tuple[float, float, float]:
    return _finite_vector(value, 3, owner=owner)  # type: ignore[return-value]


def _quat(value: Any, *, owner: str) -> tuple[float, float, float, float]:
    result = _finite_vector(value, 4, owner=owner)
    norm = math.sqrt(sum(item * item for item in result))
    _require(math.isclose(norm, 1.0, abs_tol=1.0e-7), f"{owner} must be unit")
    return result  # type: ignore[return-value]


def _transform(value: Any, *, owner: str) -> dict[str, list[float]]:
    transform = _mapping(value, owner=owner)
    return {
        "translation_m": list(
            _vec3(transform.get("translation_m"), owner=f"{owner}.translation_m")
        ),
        "rotation_xyzw": list(
            _quat(transform.get("rotation_xyzw"), owner=f"{owner}.rotation_xyzw")
        ),
    }


def _same_transform(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return bool(
        np.allclose(
            first["translation_m"], second["translation_m"], rtol=0.0, atol=1.0e-9
        )
        and np.allclose(
            first["rotation_xyzw"], second["rotation_xyzw"], rtol=0.0, atol=1.0e-9
        )
    )


def _identity_transform(value: Any, *, owner: str) -> bool:
    return _same_transform(
        _transform(value, owner=owner),
        {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )


def _safe_regular_path(value: Any, *, owner: str) -> Path:
    raw = Path(str(value))
    _require(raw.is_absolute(), f"{owner} must be absolute")
    candidates = (raw, *raw.parents)
    _require(
        not any(candidate.is_symlink() for candidate in candidates),
        f"{owner} cannot contain symlinks",
    )
    _require(raw.is_file(), f"{owner} must be a regular file")
    return raw.resolve(strict=True)


def _resolved_regular_path(value: Any, *, owner: str) -> Path:
    """Resolve a controlled runtime alias to its regular-file target."""

    raw = Path(str(value))
    _require(raw.is_absolute(), f"{owner} must be absolute")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TwoHumanCaptureError(f"{owner} must resolve to a regular file") from exc
    _require(resolved.is_file(), f"{owner} must resolve to a regular file")
    return resolved


def _require_same_runtime_file(declared: Any, expected: Any, *, owner: str) -> None:
    declared_path = _resolved_regular_path(declared, owner=f"atom {owner}")
    expected_path = _resolved_regular_path(expected, owner=f"resolved M1 {owner}")
    _require(
        declared_path.samefile(expected_path),
        f"atom {owner} differs from resolved M1 runtime",
    )


def _validate_camera_runtime_navigation(
    camera_runtime: Mapping[str, Any], navigation: Mapping[str, Any]
) -> None:
    atom_height = float(camera_runtime.get("agent_height_m", math.nan))
    atom_radius = float(camera_runtime.get("agent_radius_m", math.nan))
    m1_height = float(navigation.get("agent_height_m", math.nan))
    m1_radius = float(navigation.get("agent_radius_m", math.nan))
    # The atom radius belongs to the camera-candidate clearance solver.  The
    # M1 radius configures the render agent.  They are independent positive
    # parameters; only their common physical height is expected to agree.
    _require(
        all(
            math.isfinite(value) and value > 0.0
            for value in (atom_height, atom_radius, m1_height, m1_radius)
        )
        and atom_height == m1_height,
        "atom solver and M1 render-agent navigation values are invalid",
    )


def _build_actor_authorities(
    atom: Mapping[str, Any], plan: Mapping[str, Any], registry: Mapping[str, Any]
) -> tuple[HumanActorAuthority, HumanActorAuthority]:
    declarations = _sequence(plan.get("actors"), owner="suite plan actors")
    _require(
        [
            item.get("actor_id") if isinstance(item, Mapping) else None
            for item in declarations
        ]
        == list(ACTOR_IDS),
        "suite actors must be source1_actor then source2_actor",
    )
    framing = _mapping(atom.get("actor_framing"), owner="atom actor_framing")
    bindings = _sequence(framing.get("actor_bindings"), owner="actor bindings")
    by_id = {
        item.get("actor_id"): item for item in bindings if isinstance(item, Mapping)
    }
    _require(
        set(by_id) == set(ACTOR_IDS) and len(bindings) == 2,
        "atom must bind exactly both human actors",
    )
    result: list[HumanActorAuthority] = []
    for index, (actor_id, slot, semantic_id, package_stem) in enumerate(
        zip(ACTOR_IDS, SOURCE_SLOTS, SEMANTIC_IDS, PACKAGE_STEMS, strict=True)
    ):
        declaration = _mapping(declarations[index], owner=f"{actor_id} declaration")
        binding = _mapping(by_id[actor_id], owner=f"{actor_id} binding")
        asset_id = str(declaration.get("asset_id", ""))
        revision = str(declaration.get("asset_revision", ""))
        _require(
            asset_id
            and revision
            and binding.get("asset_id") == asset_id
            and binding.get("asset_revision") == revision,
            f"{actor_id} asset identity differs across atom and suite",
        )
        try:
            profile = resolve_source_asset_runtime_profile(registry, asset_id, revision)
        except RuntimeProfileError as exc:
            raise TwoHumanCaptureError(str(exc)) from exc
        identity = _mapping(profile.get("identity"), owner=f"{actor_id} identity")
        _require(
            profile.get("entity_class") == "articulated_human"
            and identity.get("species_id") == "human",
            f"{actor_id} is not an articulated human",
        )
        timeline = _mapping(profile.get("timeline"), owner=f"{actor_id} timeline")
        period = timeline.get("walk_phase_period_frames")
        _require(period in {16, 19}, f"{actor_id} walking period must be 16 or 19")
        _require(
            timeline.get("body_plan_id")
            == declaration.get("body_plan_id")
            == "biped_human"
            and timeline.get("template_id") == declaration.get("template_id")
            and timeline.get("idle_action_id") == "idle"
            and timeline.get("walking_action_id") == "walk"
            and binding.get("skin_index") == 0,
            f"{actor_id} human skin/body/timeline binding drift",
        )
        _require(
            binding.get("action_name_by_action_id")
            == {"idle": "Standing_Idle", "walk": "Walking"},
            f"{actor_id} action mapping drift",
        )
        anchors = _sequence(
            profile.get("emitter_anchors"), owner=f"{actor_id} emitter anchors"
        )
        anchor = [
            item
            for item in anchors
            if isinstance(item, Mapping)
            and item.get("anchor_id") == profile.get("default_emitter_anchor_id")
        ]
        _require(len(anchor) == 1, f"{actor_id} default emitter anchor is not unique")
        emitter = _vec3(anchor[0].get("offset_m"), owner=f"{actor_id} emitter offset")
        _require(
            tuple(declaration.get("emitter_offset_m", ())) == emitter,
            f"{actor_id} suite emitter differs from runtime profile",
        )
        source_glb = _safe_regular_path(
            binding.get("source_asset_path"), owner=f"{actor_id} source GLB"
        )
        axis = _vec3(
            timeline.get("local_anatomical_forward_axis"),
            owner=f"{actor_id} forward axis",
        )
        _require(
            axis == (0.0, 0.0, 1.0), f"{actor_id} anatomical forward must remain +Z"
        )
        result.append(
            HumanActorAuthority(
                actor_id=actor_id,
                source_slot_id=slot,
                asset_id=asset_id,
                asset_revision=revision,
                source_glb=source_glb,
                semantic_id=semantic_id,
                package_stem=package_stem,
                walking_profile_sample_count=19 if period == 19 else None,
                emitter_offset_m=emitter,
                anatomical_forward_axis=axis,
                anatomical_forward_source=(
                    f"runtime_profile:{registry.get('registry_id')}/{asset_id}@{revision}"
                    "/timeline.local_anatomical_forward_axis"
                ),
            )
        )
    _require(
        result[0].semantic_id != result[1].semantic_id
        and result[0].asset_id != result[1].asset_id
        and result[0].source_glb != result[1].source_glb,
        "two humans must have distinct semantics, assets, and source paths",
    )
    return result[0], result[1]


def _build_actor_frames(
    plan: Mapping[str, Any],
) -> tuple[tuple[PlannedHumanFrame, ...], tuple[PlannedHumanFrame, ...]]:
    frames = _sequence(plan.get("frames"), owner="suite plan frames")
    _require(len(frames) == FRAME_COUNT, "suite plan must contain exactly 75 frames")
    collected: list[list[PlannedHumanFrame]] = [[], []]
    for ordinal, raw_frame in enumerate(frames):
        frame = _mapping(raw_frame, owner=f"suite frame {ordinal}")
        _require(
            frame.get("frame_index") == ordinal
            and frame.get("pts_ticks") == ordinal * TICKS_PER_FRAME,
            f"suite frame {ordinal} Timeline drift",
        )
        states = _sequence(
            frame.get("actor_states"), owner=f"suite frame {ordinal} actors"
        )
        _require(
            [
                item.get("actor_id") if isinstance(item, Mapping) else None
                for item in states
            ]
            == list(ACTOR_IDS),
            f"suite frame {ordinal} actor order drift",
        )
        for actor_index, raw_state in enumerate(states):
            state = _mapping(
                raw_state, owner=f"suite frame {ordinal} actor {actor_index}"
            )
            action_id = state.get("action_id")
            action_ticks = state.get("action_time_ticks")
            action_phase = state.get("action_phase")
            _require(
                action_id in {"idle", "walk"}, f"frame {ordinal} action is invalid"
            )
            _require(
                state.get("frame_index") == ordinal
                and state.get("asset_id") == plan["actors"][actor_index].get("asset_id")
                and not isinstance(action_phase, bool)
                and isinstance(action_phase, (int, float))
                and math.isfinite(float(action_phase))
                and 0.0 <= float(action_phase) < 1.0
                and isinstance(action_ticks, int)
                and not isinstance(action_ticks, bool)
                and action_ticks >= 0
                and action_ticks % TICKS_PER_FRAME == 0,
                f"frame {ordinal} action_time_ticks is off the 15 Hz grid",
            )
            _require(
                action_id == "idle"
                and float(action_phase) == 0.0
                and action_ticks == ordinal * TICKS_PER_FRAME,
                f"strict static actor action drift at frame {ordinal}",
            )
            collected[actor_index].append(
                PlannedHumanFrame(
                    frame_index=ordinal,
                    pts_ticks=ordinal * TICKS_PER_FRAME,
                    action_id=str(action_id),
                    action_time_ticks=action_ticks,
                    translation_m=_vec3(
                        state.get("translation_m"), owner=f"frame {ordinal} translation"
                    ),
                    rotation_xyzw=_quat(
                        state.get("rotation_xyzw"), owner=f"frame {ordinal} rotation"
                    ),
                )
            )
    for actor_index, actor_frames in enumerate(collected):
        first = actor_frames[0]
        _require(
            all(
                frame.translation_m == first.translation_m
                and frame.rotation_xyzw == first.rotation_xyzw
                for frame in actor_frames
            ),
            f"{ACTOR_IDS[actor_index]} strict static root/rotation must remain frozen",
        )
    return tuple(collected[0]), tuple(collected[1])


def validate_two_human_authority_documents(
    *,
    atom: Mapping[str, Any],
    suite: Mapping[str, Any],
    sensor_rig: Mapping[str, Any],
    trajectory_bank: Mapping[str, Any],
    rir_plan: Mapping[str, Any],
    runtime_profiles: Mapping[str, Any],
    room: Mapping[str, Any],
    m1_request: Mapping[str, Any],
) -> TwoHumanCaptureAuthority:
    """Join all pre-existing authorities without promoting UE visual evidence."""

    try:
        normalized_rir_jobs = validate_semantic_rir_job_plan(rir_plan)
    except RIRCacheError as exc:
        raise TwoHumanCaptureError(str(exc)) from exc

    _require(
        atom.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_atom_request_v2",
        "atom request schema drift",
    )
    _require(
        atom.get("qualification_claim") is False
        and atom.get("formal_dataset_count") == 0,
        "atom must remain non-formal",
    )
    episode_id = str(atom.get("episode_id", ""))
    _require(bool(episode_id), "atom episode_id is missing")
    scenarios = _sequence(suite.get("scenarios"), owner="suite scenarios")
    _require(len(scenarios) == 1, "suite must contain one scenario")
    scenario = _mapping(scenarios[0], owner="suite scenario")
    plan = _mapping(scenario.get("plan"), owner="suite plan")
    _require(
        suite.get("schema") == "avengine_optional_spear_imported_glb_suite_v1"
        and scenario.get("schema") == "avengine_optional_spear_imported_glb_scenario_v1"
        and plan.get("schema") == "avengine_optional_spear_visual_plan_v1"
        and suite.get("backend_role")
        == scenario.get("backend_role")
        == plan.get("backend_role")
        == COMPARISON_VISUAL_ROLE,
        "retained UE suite schemas/roles must remain exact comparison evidence",
    )
    scenario_render = _mapping(scenario.get("render"), owner="scenario render")
    plan_render = _mapping(plan.get("render"), owner="plan render")
    qualification = _mapping(plan.get("qualification"), owner="plan qualification")
    _require(
        suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0
        and scenario_render.get("frame_count") == FRAME_COUNT
        and scenario_render.get("frame_rate_hz") == FRAME_RATE_HZ
        and plan_render
        == {
            "fps_den": 1,
            "fps_num": FRAME_RATE_HZ,
            "frame_count": FRAME_COUNT,
            "ticks_per_frame": TICKS_PER_FRAME,
        }
        and qualification.get("qualification_claim") is False
        and qualification.get("formal_dataset_count") == 0,
        "suite/scenario/plan Timeline or non-formal boundary drift",
    )
    _require(
        scenario.get("scenario_id") == episode_id, "suite and atom episode IDs differ"
    )
    atom_room = _mapping(atom.get("room"), owner="atom room")
    plan_room = _mapping(plan.get("room"), owner="suite plan room")
    _require(
        plan_room.get("room_id") == atom_room.get("room_id")
        and plan_room.get("room_revision") == atom_room.get("room_revision")
        and plan_room.get("scene_id") == atom_room.get("scene_id"),
        "suite room/revision/scene differs from atom",
    )
    actors = _build_actor_authorities(atom, plan, runtime_profiles)
    actor_frames = _build_actor_frames(plan)

    rig_frames = _sequence(sensor_rig.get("frames"), owner="sensor rig frames")
    _require(
        sensor_rig.get("schema") == "avengine_sensor_rig_trajectory_v1"
        and sensor_rig.get("trajectory_id")
        == plan.get("camera", {}).get("sensor_rig_trajectory_id")
        and sensor_rig.get("rig_id") == "camera_rig_0"
        and sensor_rig.get("listener_id")
        == plan.get("camera", {}).get("listener_id")
        == "listener0"
        and sensor_rig.get("formal_view_id") == "view0"
        and sensor_rig.get("camera_listener_coupling") == "rigid_colocated_cooriented"
        and sensor_rig.get("time_base_hz") == TIME_BASE_HZ
        and sensor_rig.get("ticks_per_frame") == TICKS_PER_FRAME
        and sensor_rig.get("coordinate_frame") == "avengine_world_right_handed_y_up_m"
        and sensor_rig.get("pose_model") == "yaw_only_about_world_positive_y"
        and sensor_rig.get("frame_count") == FRAME_COUNT
        and sensor_rig.get("frame_rate_hz") == FRAME_RATE_HZ
        and sensor_rig.get("duration_ticks") == DURATION_TICKS
        and len(rig_frames) == FRAME_COUNT,
        "sensor rig identity or Timeline drift",
    )
    _require(
        _identity_transform(sensor_rig.get("rig_from_camera"), owner="rig_from_camera")
        and _identity_transform(
            sensor_rig.get("rig_from_listener"), owner="rig_from_listener"
        ),
        "sensor rig camera/listener offsets must remain identity",
    )
    normalized_rig: list[Mapping[str, Any]] = []
    for ordinal, raw_rig in enumerate(rig_frames):
        rig_frame = _mapping(raw_rig, owner=f"rig frame {ordinal}")
        expected_transform = _transform(
            rig_frame.get("world_from_rig"), owner=f"rig frame {ordinal}"
        )
        plan_frame = _mapping(plan["frames"][ordinal], owner=f"plan frame {ordinal}")
        camera_state = _mapping(
            plan_frame.get("camera_state"), owner=f"camera state {ordinal}"
        )
        _require(
            rig_frame.get("frame_index") == ordinal
            and rig_frame.get("pts_ticks") == ordinal * TICKS_PER_FRAME
            and camera_state.get("frame_index") == ordinal
            and camera_state.get("pts_ticks") == ordinal * TICKS_PER_FRAME
            and _same_transform(
                expected_transform,
                _transform(
                    camera_state.get("world_from_rig"), owner=f"camera state {ordinal}"
                ),
            ),
            f"camera and sensor rig differ at frame {ordinal}",
        )
        normalized_rig.append({**rig_frame, "world_from_rig": expected_transform})
    first_rig = normalized_rig[0]["world_from_rig"]
    _require(
        all(
            _same_transform(first_rig, item["world_from_rig"])
            for item in normalized_rig
        ),
        "current two-human capture requires the selected HOLD rig",
    )
    program = _mapping(sensor_rig.get("program"), owner="sensor rig program")
    yaw_degrees = float(program.get("yaw_deg", math.nan))
    half_yaw = math.radians(yaw_degrees) / 2.0
    _require(
        program.get("kind") == "HOLD"
        and _vec3(program.get("position_m"), owner="HOLD position")
        == tuple(first_rig["translation_m"])
        and np.allclose(
            first_rig["rotation_xyzw"],
            [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)],
            rtol=0.0,
            atol=1.0e-9,
        ),
        "sensor rig HOLD program differs from its 75 poses",
    )

    episodes = _sequence(trajectory_bank.get("episodes"), owner="trajectory episodes")
    _require(
        trajectory_bank.get("schema") == "avengine_room_trajectory_bank_v2"
        and trajectory_bank.get("frame_count") == FRAME_COUNT
        and trajectory_bank.get("frame_rate_hz") == FRAME_RATE_HZ
        and trajectory_bank.get("seconds_per_episode") == 5.0
        and trajectory_bank.get("source_slots") == list(SOURCE_SLOTS)
        and len(episodes) == 1
        and isinstance(episodes[0], Mapping)
        and episodes[0].get("episode_id") == episode_id,
        "trajectory bank identity or Timeline drift",
    )
    episode = episodes[0]
    _require(
        episode.get("motion_case") == "strict_two_human_static_mp3d",
        "trajectory motion case drift",
    )
    roots = _mapping(episode.get("source_root_paths_m"), owner="trajectory roots")
    centers = _mapping(episode.get("source_center_paths_m"), owner="trajectory centers")
    expected_centers: dict[str, list[list[float]]] = {}
    for index, actor in enumerate(actors):
        expected_roots = [list(frame.translation_m) for frame in actor_frames[index]]
        expected_centers[actor.source_slot_id] = [
            [root[axis] + actor.emitter_offset_m[axis] for axis in range(3)]
            for root in expected_roots
        ]
        _require(
            roots.get(actor.source_slot_id) == expected_roots,
            f"{actor.source_slot_id} trajectory roots differ from suite",
        )
        _require(
            centers.get(actor.source_slot_id) == expected_centers[actor.source_slot_id],
            f"{actor.source_slot_id} trajectory centers differ from suite",
        )

    jobs = normalized_rir_jobs
    _require(
        rir_plan.get("schema") == "avengine_room_rir_job_plan_v2"
        and rir_plan.get("status") == "planned_not_run"
        and rir_plan.get("producer_backend") == "RLR Audio Propagation"
        and rir_plan.get("listener_pose_mode") == "per_episode_frame"
        and rir_plan.get("dry_audio_independent") is True
        and rir_plan.get("slot_identity_affects_cache_key") is False
        and rir_plan.get("cache_key_fields")
        == [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ]
        and rir_plan.get("stride_frames") == 1
        and rir_plan.get("requested_pair_state_count") == 2 * FRAME_COUNT
        and rir_plan.get("unique_rir_job_count") == 2
        and rir_plan.get("cache_reuse_count") == 148
        and rir_plan.get("unique_listener_pose_count") == 1,
        "RIR plan Timeline drift",
    )
    _require(len(jobs) == 2, "static two-human RIR plan must contain exactly two jobs")
    use_keys: list[tuple[str, int]] = []
    for raw_job in jobs:
        job = _mapping(raw_job, owner="RIR job")
        listener_position = _vec3(
            job.get("listener_position_m"), owner="RIR listener position"
        )
        listener_rotation = _finite_vector(
            job.get("listener_orientation_wxyz"), 4, owner="RIR listener orientation"
        )
        for raw_use in _sequence(job.get("uses"), owner="RIR uses"):
            use = _mapping(raw_use, owner="RIR use")
            slot = str(use.get("source_slot_id", ""))
            frame_index = use.get("frame_index")
            _require(
                slot in SOURCE_SLOTS
                and isinstance(frame_index, int)
                and 0 <= frame_index < FRAME_COUNT,
                "RIR use identity is invalid",
            )
            rig_transform = normalized_rig[frame_index]["world_from_rig"]
            xyzw = rig_transform["rotation_xyzw"]
            expected_wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
            _require(
                use.get("episode_id") == episode_id
                and listener_position == tuple(rig_transform["translation_m"])
                and np.allclose(listener_rotation, expected_wxyz, rtol=0.0, atol=1.0e-9)
                and tuple(job.get("source_position_m", ()))
                == tuple(expected_centers[slot][frame_index]),
                f"RIR state differs from suite/rig at {slot} frame {frame_index}",
            )
            use_keys.append((slot, frame_index))
    expected_use_keys = [
        (slot, frame) for slot in SOURCE_SLOTS for frame in range(FRAME_COUNT)
    ]
    _require(
        len(use_keys) == 2 * FRAME_COUNT
        and len(set(use_keys)) == len(use_keys)
        and sorted(use_keys) == sorted(expected_use_keys)
        and all(
            sum(slot == owner for slot, _ in use_keys) == FRAME_COUNT
            for owner in SOURCE_SLOTS
        ),
        "RIR uses must uniquely cover exactly 75 frames for each source",
    )

    room_id = str(atom_room.get("room_id", ""))
    room_revision = str(atom_room.get("room_revision", ""))
    scene_id = str(atom_room.get("scene_id", ""))
    room_scene = _mapping(room.get("scene"), owner="M1 room scene")
    coordinate_system = _mapping(
        room.get("coordinate_system"), owner="M1 room coordinate system"
    )
    _require(
        room.get("room_id") == room_id
        and room.get("room_kind") == "habitat_native"
        and scene_id
        and Path(str(room_scene.get("scene_id", ""))).stem == scene_id,
        "M1 room/scene differs from atom MP3D room",
    )
    _require(
        room.get("geometry_representation") == "real_surface_mesh"
        and coordinate_system.get("handedness") == "right"
        and coordinate_system.get("up_axis") == "+Y"
        and coordinate_system.get("forward_axis") == "-Z"
        and coordinate_system.get("linear_unit") == "meter"
        and coordinate_system.get("quaternion_order") == "xyzw"
        and room_scene.get("navmesh_policy") == "load_declared"
        and room_scene.get("load_semantic_mesh") is True
        and room_scene.get("enable_physics") is True,
        "M1 room coordinate/scene production semantics drift",
    )
    _require(m1_request.get("room_id") == room_id, "M1 request room differs from atom")
    m1_rig = _mapping(m1_request.get("primary_camera_rig"), owner="M1 camera rig")
    calibration = _mapping(m1_rig.get("shared_calibration"), owner="M1 calibration")
    atom_camera_framing = _mapping(
        atom.get("camera_framing"), owner="atom camera framing"
    )
    atom_calibration = _mapping(
        atom_camera_framing.get("calibration"), owner="atom camera calibration"
    )
    render = _mapping(scenario.get("render"), owner="suite render")
    height, width = calibration.get("resolution_hw", (None, None))
    _require(
        (height, width) == (render.get("height"), render.get("width"))
        and (height, width) == (720, 1280)
        and float(calibration.get("hfov_degrees"))
        == float(render.get("horizontal_fov_deg"))
        and calibration.get("projection") == "pinhole"
        and calibration.get("near_m") == atom_calibration.get("near_m") == 0.05
        and calibration.get("far_m") == 100.0
        and _same_transform(
            _transform(m1_rig.get("world_from_rig"), owner="M1 rig pose"), first_rig
        ),
        "M1 camera does not match selected suite rig/calibration",
    )
    modalities = _sequence(m1_rig.get("modalities"), owner="M1 modalities")
    _require(
        [item.get("modality") for item in modalities if isinstance(item, Mapping)]
        == ["rgb", "depth", "semantic"],
        "M1 modalities must remain ordered rgb/depth/semantic",
    )
    listener = _mapping(m1_request.get("listener"), owner="M1 listener")
    plan_camera = _mapping(plan.get("camera"), owner="suite plan camera")
    _require(
        m1_rig.get("rig_id") == "camera_rig_0"
        and m1_rig.get("view_id") == "view0"
        and listener.get("listener_id") == plan_camera.get("listener_id") == "listener0"
        and listener.get("attached_to") == m1_rig.get("rig_id"),
        "M1/suite listener or camera rig identity drift",
    )
    _require(
        _same_transform(
            _transform(calibration.get("rig_from_sensor"), owner="M1 rig_from_sensor"),
            {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        )
        and _same_transform(
            _transform(listener.get("rig_from_listener"), owner="M1 rig_from_listener"),
            {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        ),
        "camera sensors and listener must be rigidly co-located/co-oriented",
    )
    m1_sources = _sequence(m1_request.get("sources"), owner="M1 sources")
    _require(
        len(m1_sources) == 2
        and [item.get("source_id") for item in m1_sources if isinstance(item, Mapping)]
        == list(SOURCE_SLOTS),
        "M1 sources must be source1 then source2",
    )
    for index, source in enumerate(m1_sources):
        source_transform = _mapping(
            _mapping(source, owner=f"M1 source {index}").get("world_from_source"),
            owner=f"M1 source {index} transform",
        )
        _require(
            _vec3(
                source_transform.get("translation_m"),
                owner=f"M1 source {index} position",
            )
            == tuple(expected_centers[SOURCE_SLOTS[index]][0]),
            f"M1 {SOURCE_SLOTS[index]} position differs from trajectory/RIR",
        )
        _require(
            _quat(
                source_transform.get("rotation_xyzw"),
                owner=f"M1 source {index} rotation",
            )
            == actor_frames[index][0].rotation_xyzw,
            f"M1 {SOURCE_SLOTS[index]} rotation differs from frozen suite actor",
        )
    return TwoHumanCaptureAuthority(
        episode_id=episode_id,
        room_id=room_id,
        room_revision=room_revision,
        actors=actors,
        actor_frames=actor_frames,
        rig_frames=tuple(normalized_rig),
        resolution_hw=(int(height), int(width)),
        horizontal_fov_deg=float(calibration["hfov_degrees"]),
        suite_visual_role=COMPARISON_VISUAL_ROLE,
        qualification_claim=False,
        formal_dataset_count=0,
    )


def load_two_human_capture_authority(
    *,
    atom_request_path: str | Path,
    suite_plan_path: str | Path,
    sensor_rig_path: str | Path,
    trajectory_bank_path: str | Path,
    rir_plan_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    runtime_root: str | Path | None = None,
) -> TwoHumanCaptureAuthority:
    atom = load_json(atom_request_path)
    registry_path = _safe_regular_path(
        atom.get("actor_framing", {}).get("runtime_profile_registry"),
        owner="runtime profile registry",
    )
    m1_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    authority = validate_two_human_authority_documents(
        atom=atom,
        suite=load_json(suite_plan_path),
        sensor_rig=load_json(sensor_rig_path),
        trajectory_bank=load_json(trajectory_bank_path),
        rir_plan=load_json(rir_plan_path),
        runtime_profiles=load_source_asset_runtime_registry(registry_path),
        room=m1_inputs.room,
        m1_request=m1_inputs.request,
    )
    selected_runtime = discover_runtime_root(runtime_root)
    resolved = _resolved_scene(m1_inputs, selected_runtime)
    camera_runtime = _mapping(atom.get("camera_runtime"), owner="atom camera runtime")
    atom_room = _mapping(atom.get("room"), owner="atom room")
    path_pairs = {
        "scene": (camera_runtime.get("scene_path"), resolved.get("scene_id")),
        "dataset": (
            camera_runtime.get("dataset_config_path"),
            resolved.get("dataset_config"),
        ),
        "navmesh": (atom_room.get("navmesh_path"), resolved.get("navmesh")),
        "physics": (
            camera_runtime.get("physics_config_path"),
            selected_runtime / "data/default.physics_config.json",
        ),
    }
    for owner, (declared, expected) in path_pairs.items():
        _require_same_runtime_file(declared, expected, owner=owner)
    navigation = _mapping(m1_inputs.room.get("navigation"), owner="M1 navigation")
    _require(
        camera_runtime.get("loaded_scene_id") == atom_room.get("scene_id"),
        "atom camera runtime scene identity differs from M1 room",
    )
    _validate_camera_runtime_navigation(camera_runtime, navigation)
    return authority


__all__ = [
    "HumanActorAuthority",
    "PlannedHumanFrame",
    "TwoHumanCaptureAuthority",
    "TwoHumanCaptureError",
    "load_two_human_capture_authority",
    "validate_two_human_authority_documents",
]
