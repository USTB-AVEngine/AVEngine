"""CPU-only, source-bound actor motion profiles.

The profile is intentionally data driven: it binds a proposed candidate, the
selected row that preceded it, and the materialized base suite without knowing
anything about a particular room or mechanism name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.m6x.room_feasibility import (
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)

PROFILE_SCHEMA = "avengine_actor_motion_profile_v1"
FRAME_SCHEMA = "avengine_actor_motion_profile_frame_v1"


class ActorMotionProfileError(ValueError):
    """A profile or one of its immutable authorities is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActorMotionProfileError(message)


def bind_planning_episode(
    *,
    planning_manifest_path: str | Path,
    episode_id: str,
) -> dict[str, Any]:
    """Bind one planning row by absolute regular path and unique selector."""

    path = Path(planning_manifest_path).resolve()
    _require(path.is_file(), f"planning manifest is not a file: {path}")
    document = load_json(path)
    _require(isinstance(document, Mapping), "planning manifest is not an object")
    episodes = document.get("episodes")
    _require(isinstance(episodes, list), "planning manifest episodes are missing")
    matches = [
        (index, value)
        for index, value in enumerate(episodes)
        if isinstance(value, Mapping) and value.get("episode_id") == episode_id
    ]
    _require(
        len(matches) == 1,
        f"planning episode selector must resolve exactly once: {episode_id!r}",
    )
    index, row = matches[0]
    return {
        "path": str(path),
        "json_pointer": f"/episodes/{index}",
        "value": deepcopy(dict(row)),
    }


def _source_binding(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    json_pointer: str,
) -> dict[str, Any]:
    source_path = Path(path).resolve()
    _require(source_path.is_file(), f"authority is not a file: {source_path}")
    return {
        "path": str(source_path),
        "document_sha256": sha256_file(source_path),
        "json_pointer": json_pointer,
        "canonical_value_sha256": canonical_json_sha256(value),
        "value": deepcopy(dict(value)),
    }


def _candidate(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    authorities = profile.get("authorities")
    _require(isinstance(authorities, Mapping), "authorities are missing")
    binding = authorities.get("candidate")
    _require(isinstance(binding, Mapping), "candidate authority is missing")
    value = binding.get("value")
    _require(isinstance(value, Mapping), "candidate value is missing")
    return value


def _as_mapping(value: object, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    return value


def _as_list(value: object, message: str) -> list[Any]:
    _require(isinstance(value, list), message)
    return value


def _actor_paths(
    actor: Mapping[str, Any], slot: str, count: int
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for key in (
        "root_path_m",
        "translation_ue_cm_path",
        "action_id_path",
        "ue_animation_path",
        "action_phase_path",
        "action_time_ticks_path",
        "animation_timing_mode_path",
        "native_source_frame_index_path",
        "actor_yaw_ue_deg_path",
    ):
        values = _as_list(actor.get(key), f"{key} missing for {slot!r}")
        _require(len(values) == count, f"{key} length drift for {slot!r}")
        result[key] = values
    return result


def _validate_actor_semantics(
    actor: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    slot: str,
    count: int,
    rate: int,
    frame_ticks: int,
    speech_window: list[int],
) -> None:
    _require(
        actor.get("slot_id") == slot
        and actor.get("actor_id") == declaration.get("actor_id")
        and actor.get("asset_id") == declaration.get("asset_id"),
        f"actor declaration drift for {slot!r}",
    )
    runtime = _as_mapping(
        declaration.get("runtime_asset_expectation"),
        f"runtime declaration missing for {slot!r}",
    )
    _require(
        runtime.get("source_slot_id") == slot
        and runtime.get("asset_id") == declaration.get("asset_id")
        and runtime.get("asset_revision") == declaration.get("asset_revision"),
        f"runtime declaration drift for {slot!r}",
    )
    animations = _as_mapping(
        declaration.get("animation_paths_by_action_id"),
        f"animation declaration missing for {slot!r}",
    )
    paths = _actor_paths(actor, slot, count)
    for action, animation in zip(
        paths["action_id_path"], paths["ue_animation_path"], strict=True
    ):
        _require(
            action in animations and animation == animations[action],
            f"action/animation declaration drift for {slot!r}",
        )
    moving = actor.get("moving")
    _require(type(moving) is bool, f"moving flag missing for {slot!r}")
    if not moving:
        _require(
            actor.get("native_rate_active_interval") is None
            and all(root == paths["root_path_m"][0] for root in paths["root_path_m"])
            and set(paths["action_id_path"]) == {"idle"}
            and set(paths["action_phase_path"]) == {0}
            and set(paths["action_time_ticks_path"]) == {0}
            and set(paths["native_source_frame_index_path"]) == {None}
            and len(set(paths["animation_timing_mode_path"])) == 1,
            f"static actor is not held Idle for {slot!r}",
        )
        return

    interval = _as_mapping(
        actor.get("native_rate_active_interval"),
        f"active interval missing for {slot!r}",
    )
    output_range = _as_list(
        interval.get("output_frame_range_inclusive"),
        f"output range missing for {slot!r}",
    )
    native_range = _as_list(
        interval.get("native_source_frame_range_inclusive"),
        f"native range missing for {slot!r}",
    )
    _require(
        len(output_range) == 2
        and len(native_range) == 2
        and all(type(value) is int for value in output_range + native_range),
        f"active range shape drift for {slot!r}",
    )
    start, end = output_range
    native_start, native_end = native_range
    intervals = end - start
    outside_action = interval.get("outside_action_id")
    _require(
        0 <= start < end < count
        and native_start >= 0
        and native_end - native_start == intervals
        and interval.get("output_interval_count") == intervals
        and interval.get("native_interval_count") == intervals
        and interval.get("output_sample_count") == intervals + 1
        and interval.get("native_sample_count") == intervals + 1
        and interval.get("output_frame_rate_hz") == rate
        and interval.get("native_frame_rate_hz") == rate
        and interval.get("time_scale") == 1
        and interval.get("global_time_stretch_applied") is False
        and interval.get("outside_root_policy") == "hold_nearest_boundary_root",
        f"native-rate active interval drift for {slot!r}",
    )
    actions = paths["action_id_path"]
    active_action = actions[start]
    _require(
        active_action != outside_action
        and actions[:start] == [outside_action] * start
        and actions[start : end + 1] == [active_action] * (intervals + 1)
        and actions[end + 1 :] == [outside_action] * (count - end - 1),
        f"active/outside action drift for {slot!r}",
    )
    roots = paths["root_path_m"]
    _require(
        all(root == roots[start] for root in roots[:start])
        and all(root == roots[end] for root in roots[end + 1 :]),
        f"outside roots do not hold active boundaries for {slot!r}",
    )
    native_indices = paths["native_source_frame_index_path"]
    _require(
        native_indices[:start] == [native_start] * start
        and native_indices[start : end + 1] == list(range(native_start, native_end + 1))
        and native_indices[end + 1 :] == [native_end] * (count - end - 1),
        f"native source frame mapping drift for {slot!r}",
    )
    ticks = paths["action_time_ticks_path"]
    active_ticks = ticks[start : end + 1]
    _require(
        all(type(value) is int for value in active_ticks)
        and all(
            current - previous == frame_ticks
            for previous, current in pairwise(active_ticks)
        )
        and set(ticks[:start] + ticks[end + 1 :]) <= {0},
        f"native-rate animation tick drift for {slot!r}",
    )
    trajectory = _as_mapping(
        actor.get("trajectory_preflight"),
        f"trajectory preflight missing for {slot!r}",
    )
    cycle_ticks = trajectory.get("animation_ticks_per_phase_cycle")
    _require(
        type(cycle_ticks) is int
        and cycle_ticks > 0
        and all(
            np.isclose(
                paths["action_phase_path"][index],
                (ticks[index] / cycle_ticks) % 1.0,
            )
            for index in range(start, end + 1)
        )
        and set(
            paths["action_phase_path"][:start] + paths["action_phase_path"][end + 1 :]
        )
        <= {0},
        f"animation phase/tick drift for {slot!r}",
    )
    overlap = len(
        set(range(start, end + 1)) & set(range(speech_window[0], speech_window[1] + 1))
    )
    _require(
        interval.get("speech_window_inclusive") == speech_window
        and interval.get("speech_overlap_frame_count") == overlap
        and overlap > 0,
        f"active interval lost speech overlap for {slot!r}",
    )


def validate_actor_motion_authorities(
    candidate: Mapping[str, Any],
    selected_old_row: Mapping[str, Any],
    base_suite: Mapping[str, Any],
) -> None:
    """Fail closed on motion semantics and three-way authority drift."""

    _require(
        candidate.get("qualification_claim") is False
        and candidate.get("formal_episode_count") == 0
        and candidate.get("gpu_launch_authorized") is False,
        "candidate CPU/formal claim boundary drift",
    )
    legacy_id = candidate.get("legacy_episode_id")
    mechanism = candidate.get("mechanism")
    _require(
        selected_old_row.get("episode_id") == legacy_id
        and selected_old_row.get("mechanism") == mechanism
        and selected_old_row.get("target_side") == candidate.get("target_side"),
        "candidate/old-row identity drift",
    )
    scenarios = _as_list(base_suite.get("scenarios"), "base-suite scenarios missing")
    _require(len(scenarios) == 1, "base suite must contain exactly one scenario")
    scenario = _as_mapping(scenarios[0], "base-suite scenario is invalid")
    plan = _as_mapping(scenario.get("plan"), "base-suite plan missing")
    _require(
        scenario.get("scenario_id") == legacy_id
        and scenario.get("variant_id") == mechanism,
        "candidate/base-suite identity drift",
    )
    actors = _as_mapping(candidate.get("actors"), "candidate actors missing")
    declarations = _as_mapping(
        candidate.get("actor_declarations"), "candidate declarations missing"
    )
    plan_actors = _as_list(plan.get("actors"), "base-suite actors missing")
    plan_declarations = {
        declaration["actor_id"]: declaration
        for declaration in (
            _as_mapping(value, "base-suite actor declaration invalid")
            for value in plan_actors
        )
    }
    _require(
        len(plan_declarations) == len(plan_actors)
        and declarations == plan_declarations,
        "candidate/base-suite actor declaration drift",
    )
    for role in ("target", "distractor"):
        slot = candidate.get(f"{role}_slot")
        old_role = _as_mapping(
            selected_old_row.get(role), f"old {role} authority missing"
        )
        _require(
            isinstance(slot, str)
            and slot in actors
            and old_role.get("source_slot_id") == slot,
            f"{role} slot cross-authority drift",
        )
        actor = _as_mapping(actors[slot], f"candidate {role} actor invalid")
        declaration = _as_mapping(
            declarations.get(actor.get("actor_id")),
            f"candidate {role} declaration missing",
        )
        _require(
            actor.get("asset_id") == old_role.get("runtime_asset_id")
            and declaration.get("asset_revision") == old_role.get("runtime_revision"),
            f"{role} asset/revision cross-authority drift",
        )

    camera = _as_mapping(candidate.get("camera"), "candidate camera missing")
    old_camera = _as_mapping(selected_old_row.get("camera"), "old camera missing")
    old_yaws = _as_list(old_camera.get("yaw_path_deg"), "old camera yaw path missing")
    _require(
        plan.get("camera") == camera
        and old_camera.get("translation_m") == camera.get("habitat_position_m")
        and old_camera.get("horizontal_fov_deg") == camera.get("horizontal_fov_deg")
        and old_yaws
        and all(yaw == camera.get("habitat_yaw_deg") for yaw in old_yaws),
        "camera cross-authority drift",
    )
    activation = _as_mapping(
        candidate.get("source_activation_contract"),
        "source activation contract missing",
    )
    source_logic = _as_mapping(
        activation.get("source_logic"), "candidate source logic missing"
    )
    _require(
        activation.get("modified") is False
        and source_logic == plan.get("source_logic"),
        "source activation/base-suite drift",
    )
    source_rows = _as_list(source_logic.get("sources"), "source rows missing")
    target_slot = str(candidate.get("target_slot"))
    distractor_slot = str(candidate.get("distractor_slot"))
    expected_activation = {
        actors[target_slot]["actor_id"]: "active",
        actors[distractor_slot]["actor_id"]: "silent",
    }
    observed_activation = {
        row["entity_actor_id"]: row.get("activation")
        for row in (
            _as_mapping(value, "source activation row invalid") for value in source_rows
        )
    }
    _require(
        len(observed_activation) == len(source_rows)
        and observed_activation == expected_activation,
        "source activation actor/role drift",
    )

    old_target = _as_mapping(selected_old_row.get("target"), "old target missing")
    old_distractor = _as_mapping(
        selected_old_row.get("distractor"), "old distractor missing"
    )
    contract = _as_mapping(
        candidate.get("audio_event_contract"), "audio event contract missing"
    )
    audio = _as_mapping(contract.get("audio_program"), "audio program missing")
    validation = _as_mapping(audio.get("validation"), "audio validation missing")
    sample_rate = audio.get("sample_rate_hz")
    sample_count = audio.get("sample_count")
    start_sample = audio.get("target_speech_start_sample")
    active_samples = old_target.get("speech_sample_count")
    _require(
        all(
            type(value) is int
            for value in (sample_rate, sample_count, start_sample, active_samples)
        )
        and sample_rate > 0
        and sample_count > 0
        and active_samples > 0
        and 0 <= start_sample < start_sample + active_samples <= sample_count,
        "audio sample authority drift",
    )
    rate = candidate.get("frame_rate_hz")
    _require(type(rate) is int and rate > 0, "candidate frame rate drift")
    speech_window = [
        start_sample * rate // sample_rate,
        ((start_sample + active_samples) * rate + sample_rate - 1) // sample_rate - 1,
    ]
    _require(
        contract.get("sound_event_content_and_timing_modified") is False
        and contract.get("source_activation_modified") is False
        and contract.get("existing_exact_rir_reuse_authorized") is False
        and contract.get("fresh_exact_rir_required") is True
        and audio.get("target_source_slot") == target_slot
        and audio.get("distractor_source_slot") == distractor_slot
        and audio.get("target_event_count") == 1
        and audio.get("distractor_event_count") == 0
        and contract.get("target_speech_start_sample") == start_sample
        and contract.get("speech_frame_window_inclusive") == speech_window
        and validation.get("speech_frame_window_inclusive") == speech_window
        and old_target.get("speech_frame_window_inclusive") == speech_window
        and old_target.get("voice_policy") == "speaking"
        and old_distractor.get("voice_policy") == "silent",
        "audio role/timing cross-authority drift",
    )
    if "target_sound_asset_id" in audio:
        _require(
            audio.get("target_sound_asset_id") == old_target.get("sound_asset_id"),
            "audio sound asset cross-authority drift",
        )
    if "target_active_sample_count" in validation:
        _require(
            validation.get("target_active_sample_count") == active_samples,
            "audio active-sample cross-authority drift",
        )

    count = candidate.get("frame_count")
    tick_rate = candidate.get("timeline_ticks_per_second")
    _require(
        type(count) is int
        and count > 0
        and type(tick_rate) is int
        and tick_rate > 0
        and tick_rate % rate == 0,
        "candidate timeline authority drift",
    )
    frame_ticks = tick_rate // rate
    _require(
        candidate.get("frame_ticks") == frame_ticks and len(old_yaws) == count,
        "candidate frame-tick/camera closure drift",
    )
    frames = _as_list(candidate.get("frames"), "candidate frames missing")
    _require(len(frames) == count, "candidate frame count drift")
    actor_order = list(actors)
    for index, frame_value in enumerate(frames):
        frame = _as_mapping(frame_value, f"candidate frame {index} invalid")
        states = _as_list(
            frame.get("actor_states"), f"actor states missing at f{index}"
        )
        _require(
            frame.get("frame_index") == index
            and frame.get("pts_ticks") == index * frame_ticks
            and frame.get("frame_coverage_end_ticks") == (index + 1) * frame_ticks
            and [state.get("slot_id") for state in states] == actor_order,
            f"frame timeline/actor order drift at f{index}",
        )
        for state, slot in zip(states, actor_order, strict=True):
            actor = actors[slot]
            expected = {
                "actor_id": actor["actor_id"],
                "slot_id": slot,
                "translation_m": actor["root_path_m"][index],
                "translation_ue_cm": actor["translation_ue_cm_path"][index],
                "action_id": actor["action_id_path"][index],
                "ue_animation": actor["ue_animation_path"][index],
                "action_phase": actor["action_phase_path"][index],
                "action_time_ticks": actor["action_time_ticks_path"][index],
                "animation_timing_mode": actor["animation_timing_mode_path"][index],
                "native_source_frame_index": actor["native_source_frame_index_path"][
                    index
                ],
                "actor_yaw_ue_deg": actor["actor_yaw_ue_deg_path"][index],
            }
            _require(
                dict(state) == expected, f"frame/actor path drift at f{index} {slot}"
            )

    moving_slots: list[str] = []
    for slot, actor_value in actors.items():
        actor = _as_mapping(actor_value, f"candidate actor {slot!r} invalid")
        actor_id = actor.get("actor_id")
        declaration = _as_mapping(
            declarations.get(actor_id), f"candidate declaration {actor_id!r} missing"
        )
        _validate_actor_semantics(
            actor,
            declaration,
            slot=str(slot),
            count=count,
            rate=rate,
            frame_ticks=frame_ticks,
            speech_window=speech_window,
        )
        if actor.get("moving") is True:
            moving_slots.append(str(slot))
    mechanism_preflight = _as_mapping(
        candidate.get("mechanism_preflight"), "mechanism preflight missing"
    )
    _require(
        mechanism_preflight.get("expected_moving_slots") == moving_slots
        and mechanism_preflight.get("observed_moving_slots") == moving_slots
        and mechanism_preflight.get("mechanism_speech_overlap_frame_count", 0) > 0,
        "mechanism/actor motion semantics drift",
    )


def materialize_profile_frames(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return canonical, hash-bound frames from a validated profile."""

    candidate = _candidate(profile)
    frames = candidate.get("frames")
    _require(isinstance(frames, list) and bool(frames), "candidate frames are missing")
    result: list[dict[str, Any]] = []
    for frame_index, source in enumerate(frames):
        _require(isinstance(source, Mapping), f"frame {frame_index} is not an object")
        _require(
            source.get("frame_index") == frame_index, "frame indices are not exact"
        )
        core = {
            "schema": FRAME_SCHEMA,
            "frame_index": frame_index,
            "pts_ticks": source.get("pts_ticks"),
            "actor_states": deepcopy(source.get("actor_states")),
        }
        result.append({**core, "canonical_frame_sha256": canonical_json_sha256(core)})
    return result


def source_center_paths(profile: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    """Derive emitter centers by adding declared offsets to candidate roots."""

    candidate = _candidate(profile)
    declarations = candidate.get("actor_declarations")
    actors = candidate.get("actors")
    _require(isinstance(declarations, Mapping), "actor declarations are missing")
    _require(isinstance(actors, Mapping) and bool(actors), "actors are missing")
    result: dict[str, list[list[float]]] = {}
    for slot, actor in sorted(actors.items()):
        _require(isinstance(actor, Mapping), f"actor {slot!r} is invalid")
        actor_id = actor.get("actor_id")
        declaration = declarations.get(actor_id)
        _require(
            isinstance(declaration, Mapping), f"declaration for {actor_id!r} is missing"
        )
        offset = declaration.get("emitter_offset_m")
        roots = actor.get("root_path_m")
        _require(
            isinstance(offset, Sequence)
            and len(offset) == 3
            and isinstance(roots, Sequence),
            f"source-center inputs for {slot!r} are invalid",
        )
        result[str(slot)] = [
            [float(root[axis]) + float(offset[axis]) for axis in range(3)]
            for root in roots
        ]
    return result


def _rir_expectation(profile: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _candidate(profile)
    frame_count = int(candidate["frame_count"])
    episode_id = str(candidate["candidate_episode_id"])
    centers = source_center_paths(profile)
    roots = {
        str(slot): np.asarray(actor["root_path_m"], dtype=np.float64)
        for slot, actor in candidate["actors"].items()
    }
    episode = TrajectoryEpisode(
        episode_id=episode_id,
        motion_case=str(candidate["mechanism"]),
        source_root_paths_m=roots,
        source_center_paths_m={
            slot: np.asarray(path, dtype=np.float64) for slot, path in centers.items()
        },
        statistics={},
    )
    bank = TrajectoryBank(
        episodes=(episode,),
        frame_count=frame_count,
        frame_rate_hz=int(candidate["frame_rate_hz"]),
        seed=0,
    )
    base_suite = profile["authorities"]["base_suite"]["value"]
    scenario = base_suite["scenarios"][0]
    frames = scenario["plan"]["frames"]
    positions = [frame["camera_state"]["habitat_position_m"] for frame in frames]
    orientations = []
    for frame in frames:
        xyzw = frame["camera_state"]["world_from_rig"]["rotation_xyzw"]
        orientations.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    plan = build_rir_job_plan(
        bank,
        listener_positions_m_by_episode={episode_id: positions},
        listener_orientations_wxyz_by_episode={episode_id: orientations},
        stride_frames=1,
    )
    return {
        "builder": "avengine.m6x.room_feasibility.build_rir_job_plan",
        "stride_frames": plan["stride_frames"],
        "requested_pair_state_count": plan["requested_pair_state_count"],
        "unique_rir_job_count": plan["unique_rir_job_count"],
        "canonical_plan_sha256": canonical_json_sha256(plan),
    }


def build_actor_motion_profile(
    *,
    candidate_path: str | Path,
    candidate: Mapping[str, Any],
    old_preflight_path: str | Path,
    selected_old_row: Mapping[str, Any],
    base_suite_path: str | Path,
    base_suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a generic immutable profile from three supplied authorities."""

    validate_actor_motion_authorities(candidate, selected_old_row, base_suite)

    core: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "status": "pass_cpu_bound_actor_motion_profile",
        "qualification_claim": False,
        "formal_episode_count": 0,
        "authorities": {
            "candidate": _source_binding(candidate_path, candidate, json_pointer=""),
            "selected_old_row": _source_binding(
                old_preflight_path, selected_old_row, json_pointer="/canaries/0"
            ),
            "base_suite": _source_binding(base_suite_path, base_suite, json_pointer=""),
        },
    }
    core["frames"] = materialize_profile_frames(core)
    core["rir_expectation"] = _rir_expectation(core)
    profile = {**core, "profile_content_sha256": canonical_json_sha256(core)}
    validate_actor_motion_profile(profile)
    return profile


def validate_actor_motion_profile(profile: Mapping[str, Any]) -> None:
    """Fail closed on source drift and basic frame/profile inconsistency."""

    _require(profile.get("schema") == PROFILE_SCHEMA, "profile schema is invalid")
    _require(
        profile.get("qualification_claim") is False, "qualification claim is forbidden"
    )
    _require(
        profile.get("formal_episode_count") == 0, "formal episode count must be zero"
    )
    core = dict(profile)
    declared_hash = core.pop("profile_content_sha256", None)
    _require(
        declared_hash == canonical_json_sha256(core), "profile content hash mismatch"
    )
    authorities = profile.get("authorities")
    _require(isinstance(authorities, Mapping), "authorities are missing")
    for name, pointer in (
        ("candidate", ""),
        ("selected_old_row", "/canaries/0"),
        ("base_suite", ""),
    ):
        binding = authorities.get(name)
        _require(isinstance(binding, Mapping), f"{name} authority is missing")
        path = Path(str(binding.get("path", "")))
        _require(path.is_file(), f"{name} authority file is missing")
        _require(
            binding.get("document_sha256") == sha256_file(path),
            f"{name} file hash drift",
        )
        document = load_json(path)
        actual = document if pointer == "" else document["canaries"][0]
        _require(binding.get("json_pointer") == pointer, f"{name} JSON pointer drift")
        _require(binding.get("value") == actual, f"{name} bound value drift")
        _require(
            binding.get("canonical_value_sha256") == canonical_json_sha256(actual),
            f"{name} value hash drift",
        )
    validate_actor_motion_authorities(
        authorities["candidate"]["value"],
        authorities["selected_old_row"]["value"],
        authorities["base_suite"]["value"],
    )
    _require(
        profile.get("frames") == materialize_profile_frames(profile),
        "frame materialization drift",
    )
    _require(
        profile.get("rir_expectation") == _rir_expectation(profile),
        "RIR expectation drift",
    )
