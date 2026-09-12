"""Fixed-camera native actor-motion groups for cross-time state questions."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.dataset import binding_group_native as native
from avengine.capture.qa_plan_adapters import load_planning_resources
from avengine.rooms.conditioned_sampler import load_conditioned_sound_pool, sound_matches
from avengine.routes.trajectory import resample_polyline_by_arc_length
from avengine.qa.answerability import MeshHandle, line_of_sight


DEFAULT_LATE_PATH_ANSWER_MARGIN_DEG = 3.0


def _bearing(point, camera):
    delta = np.asarray(point) - camera["position_m"]
    return math.degrees(math.atan2(float(delta @ camera["basis"]["right"]),
                                   float(delta @ camera["basis"]["forward"])))


def _wrap(value):
    return (value + 180) % 360 - 180


def _settings(request):
    value = deepcopy(request.get("binding_motion") or {})
    required = ("walk_speed_range_mps", "minimum_motion_s", "end_hold_s",
                "minimum_entity_separation_m", "angle_tolerance_deg", "source_start_s")
    if any(key not in value for key in required):
        raise native.BindingNativeError("binding_motion must explicitly configure timing, speed, separation and angle tolerance")
    speeds = value["walk_speed_range_mps"]
    if len(speeds) != 2 or not all(math.isfinite(float(speed)) for speed in speeds) or not 0 < speeds[0] <= speeds[1]:
        raise native.BindingNativeError("walk_speed_range_mps must contain two positive ordered speeds")
    if any(not math.isfinite(float(value[key])) or float(value[key]) < 0 for key in required[1:]):
        raise native.BindingNativeError("binding motion values must be finite and nonnegative")
    offset = value.get("route_seed_offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise native.BindingNativeError("route_seed_offset must be a nonnegative integer")

    margin = value.get("answer_margin_deg", DEFAULT_LATE_PATH_ANSWER_MARGIN_DEG)
    try:
        margin_value = float(margin)
    except (TypeError, ValueError):
        margin_value = math.nan
    if isinstance(margin, bool) or not math.isfinite(margin_value) or margin_value < 0:
        raise native.BindingNativeError(
            "answer_margin_deg must be a finite nonnegative number"
        )
    value["answer_margin_deg"] = margin_value

    return value



def _late_plan_repair_inputs(
    plan: Mapping[str, Any], camera: Mapping[str, Any], repair: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    """Load a captured repair witness without changing its historical files.

    The prior v1 plan supplies the path that is kept, while the prior capture's
    pixel preflight supplies the measured root-to-pixel drift. Both are explicit
    inputs to a new late plan; no drift is inferred from a different world or
    from a planner-only bearing.
    """
    if not isinstance(repair, Mapping):
        raise native.BindingNativeError("late_plan_repair must be a mapping")
    plan_value = repair.get("source_plan_path")
    preflight_value = repair.get("capture_preflight_path")
    if not isinstance(plan_value, str) or not plan_value.strip():
        raise native.BindingNativeError(
            "late_plan_repair needs source_plan_path"
        )
    if not isinstance(preflight_value, str) or not preflight_value.strip():
        raise native.BindingNativeError(
            "late_plan_repair needs capture_preflight_path"
        )
    source_plan_path = Path(plan_value).expanduser().resolve()
    preflight_path = Path(preflight_value).expanduser().resolve()
    if not source_plan_path.is_file() or not preflight_path.is_file():
        raise native.BindingNativeError(
            "late_plan_repair historical plan/preflight input is unavailable: "
            f"{source_plan_path}, {preflight_path}"
        )
    historical_plan = native._load(source_plan_path)
    historical_frames = (historical_plan.get("visual_plan") or {}).get("frames")
    historical_actors = (historical_plan.get("visual_plan") or {}).get("actors")
    current_frames = (plan.get("visual_plan") or {}).get("frames")
    current_actors = (plan.get("visual_plan") or {}).get("actors")
    if not isinstance(historical_frames, list) or not isinstance(current_frames, list):
        raise native.BindingNativeError(
            "late_plan_repair source and current plans need visual frames"
        )
    if len(historical_frames) != len(current_frames):
        raise native.BindingNativeError(
            "late_plan_repair source plan frame count differs from current plan"
        )
    historical_ids = [
        str(actor.get("actor_id")) for actor in historical_actors or ()
        if isinstance(actor, Mapping) and actor.get("actor_id")
    ]
    current_ids = [
        str(actor.get("actor_id")) for actor in current_actors or ()
        if isinstance(actor, Mapping) and actor.get("actor_id")
    ]
    if historical_ids != current_ids or not current_ids:
        raise native.BindingNativeError(
            "late_plan_repair source/current actor slots differ"
        )

    def frame_states(frame: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        states = frame.get("actor_states")
        if isinstance(states, Mapping):
            values = states.values()
        elif isinstance(states, Sequence) and not isinstance(states, (str, bytes)):
            values = states
        else:
            values = ()
        return {
            str(state.get("actor_id")): state
            for state in values
            if isinstance(state, Mapping) and state.get("actor_id")
        }

    historical_paths: dict[str, np.ndarray] = {}
    for actor_id in current_ids:
        points = []
        for frame in historical_frames:
            state = frame_states(frame).get(actor_id)
            if state is None:
                raise native.BindingNativeError(
                    f"late_plan_repair source plan lacks {actor_id} in a frame"
                )
            points.append(state.get("root_transform", {}).get("translation_m"))
        try:
            path = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as error:
            raise native.BindingNativeError(
                f"late_plan_repair source path is not numeric for {actor_id}"
            ) from error
        if path.shape != (len(current_frames), 3) or not np.isfinite(path).all():
            raise native.BindingNativeError(
                f"late_plan_repair source path has invalid shape for {actor_id}"
            )
        historical_paths[actor_id] = path

    current_initial = frame_states(current_frames[0])
    for actor_id, path in historical_paths.items():
        current_state = current_initial.get(actor_id)
        if current_state is None or not np.allclose(
            path[0], np.asarray(
                current_state.get("root_transform", {}).get("translation_m"),
                dtype=float,
            ), atol=1e-5,
        ):
            raise native.BindingNativeError(
                f"late_plan_repair source path start differs for {actor_id}"
            )

    preflight = native._load(preflight_path)
    precise = preflight.get("angles_full_precision_deg")
    published = preflight.get("angles_deg")
    v1_precise = precise.get("v1") if isinstance(precise, Mapping) else None
    v1_published = published.get("v1") if isinstance(published, Mapping) else None
    if not isinstance(v1_precise, Mapping) or not isinstance(v1_published, Mapping):
        raise native.BindingNativeError(
            "late_plan_repair capture preflight lacks v1 measured bearings"
        )
    drift: dict[str, float] = {}
    for actor_id in current_ids:
        value = v1_precise.get(actor_id)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise native.BindingNativeError(
                f"late_plan_repair capture preflight lacks numeric v1 drift for {actor_id}"
            )
        drift[actor_id] = float(value) - _bearing(
            historical_paths[actor_id][-1], camera
        )

    locked_raw = repair.get("locked_sources")
    if isinstance(locked_raw, str):
        locked_raw = [locked_raw]
    if (
        not isinstance(locked_raw, Sequence)
        or isinstance(locked_raw, (str, bytes))
        or not locked_raw
    ):
        raise native.BindingNativeError(
            "late_plan_repair needs one or more locked_sources"
        )
    locked_ids = [str(value) for value in locked_raw]
    if any(value not in historical_paths for value in locked_ids):
        raise native.BindingNativeError(
            f"late_plan_repair locked_sources are not actor slots: {locked_ids}"
        )
    evidence = {
        "source_plan_path": str(source_plan_path),
        "capture_preflight_path": str(preflight_path),
        "locked_sources": locked_ids,
        "drift_source": "capture_preflight.angles_full_precision_deg.v1",
        "drift_deg": {key: round(float(value), 6) for key, value in drift.items()},
        "measured_v1_published_deg": {
            key: v1_published.get(key) for key in current_ids
        },
        "source_plan_frame_count": len(historical_frames),
    }
    return (
        {actor_id: historical_paths[actor_id] for actor_id in locked_ids},
        drift,
        evidence,
    )


def _state_member_request(
    context: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    requests = context.get("member_requests")
    if not isinstance(requests, Mapping):
        raise native.BindingNativeError(
            "cross_time_state context has no member_requests"
        )
    member_ids = list(item.get("member_request_ids") or ())
    if not member_ids:
        member_ids = list(
            (context.get("contract") or {}).get("member_request_ids") or ()
        )
    for member_id in member_ids:
        request = requests.get(str(member_id))
        if isinstance(request, Mapping):
            return deepcopy(dict(request))
    raise native.BindingNativeError(
        f"cross_time_state has no request for stage members {member_ids}"
    )


def _state_retained_root(
    context: Mapping[str, Any], unit_id: str
) -> Path | None:
    try:
        value = native._retained_root_for(context, unit_id)
    except (KeyError, TypeError):
        value = None
    return None if value is None else Path(value).expanduser().resolve()


def _state_done(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return native._result_rows(results)


def _state_result(
    done: Mapping[str, Mapping[str, Any]], unit_id: str
) -> dict[str, Any]:
    row = done.get(unit_id)
    if row is None:
        raise native.BindingNativeError(
            f"cross_time_state is missing a passed result for {unit_id}"
        )
    return dict(row)


def _state_plan_path(row: Mapping[str, Any], *, unit_id: str) -> Path:
    facts = row.get("facts") or {}
    outputs = row.get("outputs") or {}
    value = facts.get("episode_plan_path") or outputs.get("episode_plan")
    if not isinstance(value, str) or not value.strip():
        raise native.BindingNativeError(
            f"{unit_id} result has no episode_plan path"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise native.BindingNativeError(
            f"{unit_id} episode_plan is unavailable: {path}"
        )
    return path


def _state_capture_output(
    row: Mapping[str, Any], *, unit_id: str
) -> dict[str, Any]:
    outputs = row.get("outputs") or {}
    capture = outputs.get("capture")
    neutral = outputs.get("neutral_readback")
    if (
        not isinstance(capture, str)
        or not Path(capture).expanduser().resolve().is_dir()
        or not isinstance(neutral, str)
        or not Path(neutral).expanduser().resolve().is_file()
    ):
        raise native.BindingNativeError(
            f"{unit_id} result has no readable capture and neutral_readback"
        )
    return {
        "capture": str(Path(capture).expanduser().resolve()),
        "neutral_readback": str(Path(neutral).expanduser().resolve()),
        "visual_video": outputs.get("visual_video"),
    }


def _state_base_plan_path(
    context: Mapping[str, Any], request: Mapping[str, Any]
) -> Path:
    candidates = [
        context.get("state_base_plan_path"),
        context.get("base_plan_path"),
        context.get("ordinary_plan_path"),
        request.get("state_base_plan_path"),
        request.get("ordinary_plan_path"),
        request.get("retained_plan_path"),
    ]
    retained = _state_retained_root(context, "v0")
    if retained is not None:
        candidates.insert(0, retained / "plan/episode_plan.json")
    for value in candidates:
        if isinstance(value, Path):
            path = value.expanduser().resolve()
        elif isinstance(value, str) and value.strip():
            path = Path(value).expanduser().resolve()
        else:
            continue
        if path.is_file():
            return path
    raise native.BindingNativeError(
        "cross_time_state v0 plan needs an existing ordinary or retained "
        "static episode_plan.json; no native planner is started implicitly"
    )


def _state_materialization_base(
    context: Mapping[str, Any], plan_path: Path
) -> Path:
    for key in ("state_base_episode_root", "base_episode_root"):
        value = context.get(key)
        if isinstance(value, (str, Path)):
            candidate = Path(value).expanduser().resolve()
            if (candidate / "plan").is_dir():
                return candidate
    for parent in plan_path.parents:
        plan_dir = parent / "plan"
        if not plan_dir.is_dir():
            continue
        if any(
            (plan_dir / name).exists()
            for name in (
                "room_package.json",
                "habitat_room_manifest.json",
                "path_bindings.json",
                "room_layout.json",
            )
        ):
            return parent
    raise native.BindingNativeError(
        f"cross_time_state cannot find a plan resource root for {plan_path}"
    )


def _frame_states(frame: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    states = frame.get("actor_states")
    if isinstance(states, Mapping):
        return [value for value in states.values() if isinstance(value, Mapping)]
    if isinstance(states, Sequence) and not isinstance(states, (str, bytes)):
        return [value for value in states if isinstance(value, Mapping)]
    return []


def _validate_stationary_plan(plan: Mapping[str, Any]) -> None:
    visual = plan.get("visual_plan")
    if not isinstance(visual, Mapping) or not isinstance(visual.get("frames"), list):
        raise native.BindingNativeError(
            "cross_time_state v0 base plan lacks visual frames"
        )
    frames = [frame for frame in visual["frames"] if isinstance(frame, Mapping)]
    if not frames:
        raise native.BindingNativeError(
            "cross_time_state v0 base plan has no frames"
        )
    first = {
        str(state.get("actor_id")): state
        for state in _frame_states(frames[0])
        if state.get("actor_id")
    }
    if not first:
        raise native.BindingNativeError(
            "cross_time_state v0 base plan has no actor states"
        )
    for frame_index, frame in enumerate(frames):
        states = {
            str(state.get("actor_id")): state
            for state in _frame_states(frame)
            if state.get("actor_id")
        }
        if set(states) != set(first):
            raise native.BindingNativeError(
                f"cross_time_state v0 base plan changes actor slots at frame {frame_index}"
            )
        for actor_id, state in states.items():
            if bool(state.get("moving")):
                raise native.BindingNativeError(
                    f"cross_time_state v0 base plan is not stationary for {actor_id}"
                )
            for field in ("root_transform", "planned_emitter_m"):
                if state.get(field) != first[actor_id].get(field):
                    raise native.BindingNativeError(
                        f"cross_time_state v0 base plan moves {actor_id} before "
                        "the measured wet tail"
                    )


def measured_motion_window_from_facts(
    facts_by_assignment: Mapping[str, Mapping[str, Any]],
    *,
    clock: Mapping[str, Any],
    binding_motion: Mapping[str, Any],
    reserve_tail_s: float | None = None,
) -> dict[str, Any]:
    """Compute the late-plan window from actual early-audio facts."""
    required = ("minimum_motion_s", "end_hold_s")
    missing = [key for key in required if key not in binding_motion]
    if missing:
        raise native.BindingNativeError(
            f"cross_time_state late plan lacks binding_motion fields {missing}"
        )
    ends: list[float] = []
    for assignment, facts in facts_by_assignment.items():
        audio = facts.get("audio") if isinstance(facts, Mapping) else None
        tails = audio.get("wet_tail_intervals") if isinstance(audio, Mapping) else None
        if not isinstance(tails, list) or not tails:
            raise native.BindingNativeError(
                f"{assignment} facts lack measured wet_tail_intervals"
            )
        for row in tails:
            if not isinstance(row, Mapping):
                raise native.BindingNativeError(
                    f"{assignment} has an invalid measured wet-tail row"
                )
            end_s = row.get("end_s")
            if (
                isinstance(end_s, bool)
                or not isinstance(end_s, (int, float))
                or not math.isfinite(float(end_s))
                or float(end_s) < 0
            ):
                raise native.BindingNativeError(
                    f"{assignment} has an invalid measured wet-tail end_s"
                )
            ends.append(float(end_s))
    if not ends:
        raise native.BindingNativeError(
            "cross_time_state has no measured wet-tail endpoints"
        )
    try:
        fps = float(clock["frame_rate_hz"])
        frame_count = int(clock["frame_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise native.BindingNativeError(
            "cross_time_state late plan needs frame_count and frame_rate_hz"
        ) from exc
    if not math.isfinite(fps) or fps <= 0 or frame_count < 1:
        raise native.BindingNativeError(
            "cross_time_state late plan has an invalid video clock"
        )
    wet_end_s = max(ends)
    first_motion_frame = math.ceil(wet_end_s * fps) + 1
    last_motion_frame = frame_count - 1 - math.ceil(
        float(binding_motion["end_hold_s"]) * fps
    )
    minimum_motion_frames = math.ceil(
        float(binding_motion["minimum_motion_s"]) * fps
    )
    available_motion_frames = last_motion_frame - first_motion_frame + 1
    if available_motion_frames < minimum_motion_frames:
        raise native.BindingNativeError(
            "measured reverberation leaves insufficient declared movement time: "
            f"first_motion_frame={first_motion_frame}, "
            f"last_motion_frame={last_motion_frame}, "
            f"minimum_motion_frames={minimum_motion_frames}"
        )
    return {
        "first_motion_frame": int(first_motion_frame),
        "last_motion_frame": int(last_motion_frame),
        "measured_wet_end_s": float(wet_end_s),
        "minimum_motion_frames": int(minimum_motion_frames),
        "available_motion_frames": int(available_motion_frames),
        "wet_tail_end_s_by_assignment": {
            str(assignment): max(
                float(row["end_s"])
                for row in (facts.get("audio") or {}).get(
                    "wet_tail_intervals", []
                )
                if isinstance(row, Mapping)
            )
            for assignment, facts in facts_by_assignment.items()
        },
        "boundary_formula": "ceil(max(wet_end_s) * fps) + 1",
        "authority": "actual early binaural readback facts",
        "requested_terminal_tail_s": float(
            binding_motion.get("reserve_tail_s", 0.0)
            if reserve_tail_s is None else reserve_tail_s
        ),
    }


def motion_window_from_stage_results(
    results: Sequence[Mapping[str, Any]],
    *,
    clock: Mapping[str, Any],
    binding_motion: Mapping[str, Any],
    reserve_tail_s: float | None = None,
) -> dict[str, Any]:
    """Read the two passed early audio fact files and compute the late window."""
    done = _state_done(results)
    facts_by_assignment: dict[str, dict[str, Any]] = {}
    for assignment in ("a0", "a1"):
        row = _state_result(done, f"v0_{assignment}")
        facts = row.get("facts") or {}
        value = facts.get("facts_path")
        if not isinstance(value, str) or not Path(value).expanduser().resolve().is_file():
            raise native.BindingNativeError(
                f"v0_{assignment} has no readable facts_path"
            )
        facts_by_assignment[assignment] = native._load(
            Path(value).expanduser().resolve()
        )
    return measured_motion_window_from_facts(
        facts_by_assignment,
        clock=clock,
        binding_motion=binding_motion,
        reserve_tail_s=reserve_tail_s,
    )


def select_early_audio(plan, request, rng, *, geometry_clip_budget_s=None):
    """Filter by the declared temporal budget, then choose uniformly."""
    value = deepcopy(plan)
    request = deepcopy(request)
    config = _settings(request)
    clock = value["clock"]
    if request.get("camera", {}).get("motion") != "static":
        raise native.BindingNativeError("fixed-camera motion groups reject moving cameras")
    simulation_path = native._file(
        request.get("simulation_request") or request.get("runtime", {}).get("simulation_request")
        or str(native.REPOSITORY / "examples/runtime/rir_cache_simulation_request_v2.json"),
        base=native.REPOSITORY, owner="RIR simulation request")
    simulation = native._load(simulation_path)["simulation"]
    maximum_ir = float(simulation["max_ir_seconds"])
    request["simulation_request"] = str(simulation_path)
    duration = float(clock["sample_count"]) / float(clock["sample_rate_hz"])
    start = float(config["source_start_s"])
    bound = min(duration - float(request["profile"]["reserve_tail_s"]),
                duration - float(config["end_hold_s"]) - float(config["minimum_motion_s"]) - 1/float(clock["frame_rate_hz"]))
    budget = min(bound - start, float(request.get("sound_selection", {}).get("max_clip_s", duration)))
    if geometry_clip_budget_s is not None:
        budget = min(budget, float(geometry_clip_budget_s))
    # The actual early render determines when reverberation ends. An SDK's
    # maximum IR allocation is retained as provenance, not treated as a
    # measured four-second audible tail. The requested terminal tail is intact.
    pool_path = native._file(request["sound_pool"], base=native.REPOSITORY, owner="current sound pool")
    sounds = load_conditioned_sound_pool(native._load(pool_path), source_path=str(pool_path))
    registry = {a["asset_id"]: a for a in native._load(native._file(
        request["source_registry"], base=native.REPOSITORY, owner="source registry"))["assets"]}
    actors = value["visual_plan"]["actors"]
    if len(actors) != 2:
        raise native.BindingNativeError("first fixed-camera state recipe requires exactly two actors")
    if any(registry[a["asset_id"]]["entity_class"] != "articulated_human"
           and registry[a["asset_id"]]["entity_class"] != "articulated_animal" for a in actors):
        raise native.BindingNativeError("this motion recipe requires registered articulated sources")
    sr, tb = int(clock["sample_rate_hz"]), int(clock["time_base_hz"])
    first = int(round(start*sr))
    columns = {}
    selection = {}
    for index, actor in enumerate(actors):
        asset = registry[actor["asset_id"]]
        eligible = [sound for sound in sounds
                    if int(sound["sample_rate_hz"]) == sr
                    and int(sound["sample_count"])/sr <= budget
                    and (int(sound["audible_end_sample_exclusive"]) - int(sound["audible_start_sample"]))/sr
                        >= float(request.get("sound_selection", {}).get("min_audible_s", 0))
                    and sound_matches(asset, sound)
                    and (not sound.get("compatible_asset_ids")
                         or actor["asset_id"] in sound["compatible_asset_ids"])]
        if not eligible:
            raise native.BindingNativeError(
                f"no complete compatible sound for {actor['actor_id']} fits the requested motion and tail budgets")
        sound = deepcopy(eligible[int(rng.integers(len(eligible)))])
        last = first + int(sound["sample_count"])
        event = {**sound, "event_id": "event_001", "actor_id": actor["actor_id"],
                 "start_sample": first, "end_sample": last, "end_sample_exclusive": last,
                 "start_tick": round(first*tb/sr), "end_tick": round(last*tb/sr),
                 "end_tick_exclusive": round(last*tb/sr), "source_start_sample": 0,
                 "source_end_sample_exclusive": int(sound["sample_count"]), "linear_gain": 1.0,
                 "planned_audible_interval_samples": [first+int(sound["audible_start_sample"]),
                                                       first+int(sound["audible_end_sample_exclusive"])]}
        columns[f"a{index}"] = {"audio_events": [event],
                                "voice_bindings": [{**sound, "actor_id": actor["actor_id"]}]}
        selection[f"a{index}"] = {"eligible_count": len(eligible),
                                  "selected_sound_asset_id": sound["sound_asset_id"]}
    value.update(deepcopy(columns["a0"]))
    value["binding_motion_audio_columns"] = columns
    request["entities"]["silent_count"] = 1
    request["profile"]["anchor_count"] = 1
    request["binding_motion"]["sound_selection"] = {
        "policy": "uniform_over_each_target_compatible_complete_clips_within_declared_budget",
        "columns": selection, "maximum_clip_seconds": budget,
        "maximum_ir_seconds": maximum_ir,
    }
    value["request"] = request
    latest_end = max(row["audio_events"][0]["end_sample_exclusive"] for row in columns.values())
    begin_frame = math.ceil((latest_end/sr + maximum_ir)*float(clock["frame_rate_hz"])) + 1
    end_frame = int(clock["frame_count"]) - 1 - math.ceil(config["end_hold_s"]*float(clock["frame_rate_hz"]))
    return value, request, begin_frame, end_frame


def _root_state(actor, initial, path, fps, ticks_per_frame):
    """Use registered native walk/idle actions without changing their clip period."""
    timeline = actor.get("timeline")
    if not timeline or not timeline.get("walking_action_id"):
        raise native.BindingNativeError("source has no registered walk action")
    period = int(timeline["walk_phase_period_frames"]) * ticks_per_frame
    axis = np.asarray(timeline.get("local_anatomical_forward_axis", [1, 0, 0]), dtype=float)
    anatomical_yaw = math.atan2(-axis[2], axis[0])
    delta = np.vstack([np.diff(path, axis=0), np.zeros((1, 3))])
    initial_q = initial["root_transform"]["rotation_xyzw"]
    heading = 2*math.atan2(initial_q[1], initial_q[3])
    motion_tick = 0
    output = []
    for index, (point, direction) in enumerate(zip(path, delta)):
        moving = bool(np.linalg.norm(direction)*fps > .05)
        if moving:
            heading = math.atan2(-direction[2], direction[0]) - anatomical_yaw
            motion_tick += ticks_per_frame
        c, s = math.cos(heading), math.sin(heading)
        matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        state = deepcopy(initial)
        state.update(frame_index=index, moving=moving,
                     action_id=timeline["walking_action_id"] if moving else timeline["idle_action_id"],
                     action_time_ticks=motion_tick if moving else 0,
                     action_phase=(motion_tick % period)/period if moving else 0.)
        state["root_transform"] = {"translation_m": point.tolist(),
                                  "rotation_xyzw": [0., math.sin(heading/2), 0., math.cos(heading/2)],
                                  "scale": deepcopy(initial["root_transform"]["scale"])}
        offset = np.asarray(actor["emitter_binding"]["emitter_offset_m"], dtype=float)
        state["planned_emitter_m"] = (point + matrix @ offset).tolist()
        output.append(state)
    return output


def _visible_path(states, camera, mesh, margin_deg):
    # Every camera-to-emitter segment lies inside this union bounding box.
    # Retain exactly the potentially intersecting triangles once per path,
    # avoiding a whole-room bounding-box scan for every stationary frame.
    if isinstance(mesh, MeshHandle):
        endpoints = np.asarray([camera["position_m"],
                                *[state["planned_emitter_m"] for state in states]])
        low, high = endpoints.min(axis=0) - 1e-4, endpoints.max(axis=0) + 1e-4
        relevant = np.all(mesh.maximum >= low, axis=1) & np.all(mesh.minimum <= high, axis=1)
        mesh = MeshHandle(mesh.vertices, mesh.triangles[relevant], mesh.source)
    seen = set()
    for state in states:
        position = tuple(state["planned_emitter_m"])
        if position in seen:
            continue
        seen.add(position)
        point = np.asarray(state["planned_emitter_m"])
        delta = point - camera["position_m"]
        if delta @ camera["basis"]["forward"] <= 0:
            return "behind_camera"
        if abs(_bearing(point, camera)) >= camera["horizontal_fov_deg"]/2 - margin_deg:
            return "frustum"
        if line_of_sight(mesh, camera["position_m"], point) != "clear":
            return "blocked"
    return None



def _native_segments(space, start, minimum_frames, maximum_frames, fps):
    """Contiguous retained Recast points at the original spacing and frame rate.

    A static anchor can be either route endpoint. Walking the retained geometry
    in either direction updates the actor's heading; it does not interpolate,
    accelerate, teleport or modify the original route bank.
    """
    if not math.isclose(float(space.frame_rate_hz), fps, rel_tol=0, abs_tol=1e-9):
        raise native.BindingNativeError("native route clock differs from requested video clock")
    choices = []
    for route in space.route_bank():
        points = np.asarray(route["points_m"], dtype=float)
        starts = np.flatnonzero(np.linalg.norm(points-start, axis=1) <= 1e-5)
        for first in starts:
            for direction in (1, -1):
                available = len(points)-int(first) if direction == 1 else int(first)+1
                for length in range(minimum_frames, min(maximum_frames, available)+1):
                    indices = int(first)+np.arange(length)*direction
                    sampled = points[indices]
                    choices.append((sampled, {
                        "authority": space.metadata["route_authority"], "route_id": route["route_id"],
                        "source_frame_indices": indices.tolist(), "native_frame_rate_hz": fps,
                        "native_route_frame_count": len(points), "selected_frame_count": length,
                        "direction": "forward" if direction == 1 else "reverse_retained_geometry",
                        "time_stretch_applied": False, "coordinate_interpolation_applied": False}))
    return choices

def native_common_endpoint_paths(space, starts, *, frame_rate_hz, minimum_motion_s,
                                 path_length_range_m, walk_speed_range_mps,
                                 minimum_entity_separation_m, same_floor_tolerance_m):
    """Find two unchanged Recast segments ending at the same retained point.

    Only one actor moves in each identity visual variant. Each path therefore
    stays separated from the other actor's fixed start. Returned frame counts
    are native timing and must not be resampled by the caller.
    """
    if len(starts) != 2 or space.route_bank() is None:
        raise native.BindingNativeError("common-endpoint native paths require two starts and a route bank")
    fps = float(frame_rate_hz)
    lower, upper = map(float, path_length_range_m)
    speed_lower, speed_upper = map(float, walk_speed_range_mps)
    if not (0 < lower <= upper and 0 < speed_lower <= speed_upper):
        raise native.BindingNativeError("native path lengths and speeds must be positive and ordered")
    minimum_frames = max(2, math.ceil(float(minimum_motion_s) * fps) + 1)
    maximum_frames = math.floor(upper / speed_lower * fps) + 1
    actor_ids = list(starts)
    floor = sum(float(starts[actor][1]) for actor in actor_ids) / 2
    endpoints = []
    for actor_id, other_id in (actor_ids, actor_ids[::-1]):
        by_end, seen = {}, set()
        for points, record in _native_segments(
                space, np.asarray(starts[actor_id]), minimum_frames, maximum_frames, fps):
            signature = tuple(map(tuple, points))
            if signature in seen:
                continue
            seen.add(signature)
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            moving = steps[steps * fps > .05]
            length = float(steps.sum())
            if (not lower <= length <= upper
                    or not len(moving)
                    or len(moving) < math.ceil(float(minimum_motion_s) * fps)
                    or (moving * fps).min() < speed_lower - 1e-5
                    or (moving * fps).max() > speed_upper + 1e-5
                    or np.max(np.abs(points[:, 1] - floor)) > same_floor_tolerance_m
                    or np.min(np.linalg.norm(points - starts[other_id], axis=1)) < minimum_entity_separation_m):
                continue
            by_end.setdefault(tuple(points[-1]), []).append(
                (points, {**record, "path_length_m": length,
                          "minimum_step_speed_mps": float((moving * fps).min()),
                          "maximum_step_speed_mps": float((moving * fps).max())}))
        endpoints.append(by_end)
    result = []
    for target in sorted(set(endpoints[0]) & set(endpoints[1])):
        for first in endpoints[0][target]:
            for second in endpoints[1][target]:
                result.append({
                    "target_m": list(target),
                    "paths": {actor_ids[0]: first[0], actor_ids[1]: second[0]},
                    "route_records": {actor_ids[0]: first[1], actor_ids[1]: second[1]},
                    "native_frame_counts": {actor_ids[0]: len(first[0]), actor_ids[1]: len(second[0])},
                    "native_timing_preserved": True,
                })
    return result




#: How many times the late-path search may spend the profile's per-actor
#: sampling budget before giving up. A pair search needs candidates on both
#: sides at once, and a measured sweep of a real world that does contain legal
#: pairs produced roughly one usable candidate per hundred attempts for the
#: more constrained source. One round of two hundred attempts therefore fails
#: on a world that is perfectly solvable; eight rounds reaches the density the
#: sweep needed. The cap is what stops an impossible world being searched for
#: ever, so it is a number and not an absence.
#:
#: Sixteen rather than eight because the pair count grows with the product of
#: the two pools, and the separated pairs are about one per cent of pairs in
#: the world this was measured on: eight rounds reached four hundred pairs and
#: found none, the sweep reached seventeen hundred and found eighteen.
LATE_PATH_SEARCH_ROUNDS = 16


def _destinations_changing_the_bearing(
    space, rng, start, camera, initial_bearing, low, high,
    minimum_change_deg, bounds, tries: int = 16, preferred_sign: int = 0,
    avoid_paths=None, avoid_radius: float = 0.0,
):
    """Navigable points that both change the answer and stay within reach.

    The question this intervention has to change is "where is the source, from
    the camera", so the destination is proposed by the bearing it would give,
    not by a direction in the room. Sampling uniformly around the actor sent
    most proposals to places that were perfectly walkable and left the answer
    where it was; nearly half of one actor's attempts died on
    ``answer_change_too_small``.

    The bearing is measured in the camera's own frame, so a target bearing is
    a ray from the camera, and the reach band is a pair of circles around the
    actor's start. Their intersection is solved for directly instead of being
    hunted for by rejection. Nothing here decides legality: the route, its
    speed, its navigability and the final bearing are all still checked by the
    caller on the route that comes back.
    """

    import numpy as _np

    start = _np.asarray(start, dtype=float)
    origin = _np.asarray(camera["position_m"], dtype=float)
    forward = _np.asarray(camera["basis"]["forward"], dtype=float)
    right = _np.asarray(camera["basis"]["right"], dtype=float)
    offset = start - origin
    preferred: list = []
    fallback: list = []
    for _ in range(int(tries)):
        swing = float(rng.uniform(minimum_change_deg + 2.0,
                                  minimum_change_deg + 70.0))
        # Two sources that both swing the same way end up in the same part of
        # the room: the pair search then has plenty of candidates and no
        # separated pair among them. Each actor is given a side and keeps to
        # it most of the time, which is also the shape the endpoint-gap
        # constraint is asking for. The rest of the draws still cross over, so
        # a room where the preferred side is unwalkable is not a dead end.
        if preferred_sign and rng.random() < 0.65:
            sign = 1.0 if preferred_sign > 0 else -1.0
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
        target = math.radians(initial_bearing + sign * swing)
        direction = math.cos(target) * forward + math.sin(target) * right
        norm = float(_np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        direction = direction / norm
        # |origin + r*direction - start| in [low, high] -> a quadratic in r.
        along = float(direction @ offset)
        gap = float(offset @ offset)
        spans = []
        for radius in (low, high):
            disc = along * along - (gap - radius * radius)
            if disc < 0:
                continue
            root = math.sqrt(disc)
            spans.extend([along - root, along + root])
        spans = [value for value in spans if value > 0.0]
        if len(spans) < 2:
            continue
        distance = float(rng.uniform(min(spans), max(spans)))
        point = origin + distance * direction
        point[1] = start[1]
        if _np.any(point < bounds[0]) or _np.any(point > bounds[1]):
            continue
        reach = float(_np.linalg.norm(point - start))
        if not low <= reach <= high:
            continue
        try:
            if not space.is_navigable(point):
                continue
        except Exception:
            continue
        # The pair has to end up apart, so a destination away from the routes
        # the other source is already considering is worth preferring. It is a
        # preference and not a filter: insisting on it drove one source's
        # destinations out of the camera's view entirely and left it with no
        # candidate at all, which is a worse answer than a close one.
        clear = True
        if avoid_paths is not None and avoid_radius > 0.0:
            clear = not any(
                float(_np.min(_np.linalg.norm(other - point, axis=1))) < avoid_radius
                for other in avoid_paths
            )
        (preferred if clear else fallback).append(point)
    return preferred or fallback


def _sample_destination_within_reach(space, rng, start, bounds, low, high,
                                     tries: int = 12):
    """A navigable point roughly ``low``..``high`` metres from ``start``.

    Drawn as a direction and a distance rather than as a point in a box, so
    the proposal lands where a walk of the permitted length can end. The
    straight-line distance is only a proxy for the route length -- the route
    is never shorter -- so this narrows the proposal and never decides
    legality; every constraint is still checked on the route itself.

    Falls back to the plain navigable draw, so a room where the annulus is
    mostly unnavigable still proposes something rather than nothing.
    """

    import numpy as _np

    start = _np.asarray(start, dtype=float)
    for _ in range(int(tries)):
        angle = float(rng.uniform(0.0, 2.0 * _np.pi))
        radius = float(rng.uniform(low, high))
        point = start.copy()
        point[0] = start[0] + radius * _np.cos(angle)
        point[2] = start[2] + radius * _np.sin(angle)
        if _np.any(point < bounds[0]) or _np.any(point > bounds[1]):
            continue
        try:
            if space.is_navigable(point):
                return point
        except Exception:
            continue
    try:
        return space.sample_navigable(rng, bounds)
    except ValueError:
        return None

def sample_late_paths(plan, request, begin_frame, end_frame, *,
                      initial_visual_bearings=None, locked_paths=None,
                      bearing_drift=None):
    """Draw legal endpoint changes with the original camera left untouched.

    ``locked_paths`` pins one source to a trajectory that has already been
    captured, so only the source that failed is re-solved. That is the shape a
    repair takes after a native capture: three of the four answer comparisons
    passed and re-drawing both sources throws away the one that worked. It is
    also the only shape that solves the M23 world -- re-solving both sources
    there produced eight candidates for the constrained one and no separated
    pair among six hundred and eighty-eight, while pinning it found repairs
    1.48 m clear.

    ``bearing_drift`` is the per-source difference between the bearing the
    planner computes from a root transform and the bearing the authority reads
    from the visible pixel centroid, measured on a real capture of this same
    world. It is applied before the published-degree comparisons so the
    planner predicts what will actually be measured rather than what it would
    like to be measured.
    """
    config = _settings(request)
    camera = plan["visual_plan"]["camera"]
    frames = plan["visual_plan"]["frames"]
    fps, count = float(plan["clock"]["frame_rate_hz"]), int(plan["clock"]["frame_count"])
    actors = plan["visual_plan"]["actors"]
    initial = {row["actor_id"]: row for row in frames[0]["actor_states"]}
    space, mesh, _ = load_planning_resources(plan["resources"], request)
    selection_seed = int(request["seed"]) + 810 + int(config.get("route_seed_offset", 0))
    rng = np.random.default_rng(selection_seed)
    maximum_frames = end_frame - begin_frame + 1
    minimum_frames = max(2, math.ceil(float(config["minimum_motion_s"])*fps) + 1)
    attempts = int(request["profile"]["retry_budget_within_profile"])
    # The post-capture authority asks for more than twice the tolerance,
    # measured on rounded pixel bearings. The planner asks for that plus a
    # margin, on the same rounding, so a candidate that only just clears the
    # rule on paper is not taken to a native capture.
    answer_margin_deg = float(config["answer_margin_deg"])
    answer_threshold_deg = (
        2 * float(config["angle_tolerance_deg"]) + answer_margin_deg
    )
    locked_paths = dict(locked_paths or {})
    bearing_drift = dict(bearing_drift or {})
    all_candidates: list[list] = [[] for _ in actors]
    per_actor_rejections: dict[str, dict[str, int]] = {}
    # One pass over both actors. The pair search below may ask for
    # another: the budget in the profile is a per-actor sampling budget,
    # and a *pair* search needs candidates on both sides at once. A world
    # where one source can only stand in a few places yielded one
    # candidate against seven, and no separated pair among them, while a
    # wider sweep of the same world showed legal pairs do exist in it.
    def _sample_one_round() -> None:
        for actor_index, actor in enumerate(actors):
            aid = actor["actor_id"]
            start = np.asarray(initial[aid]["root_transform"]["translation_m"], dtype=float)
            initial_angle = (
                float(initial_visual_bearings[aid]) if initial_visual_bearings is not None
                else _bearing(initial[aid]["root_transform"]["translation_m"], camera)
            )
            candidates = all_candidates[actor_index]
            rejected = Counter(per_actor_rejections.get(aid) or {})
            if aid in locked_paths:
                if not candidates:
                    pinned = np.asarray(locked_paths[aid], dtype=float)
                    if pinned.shape != (count, 3):
                        raise native.BindingNativeError(
                            f"locked path for {aid} is {pinned.shape}, expected "
                            f"{(count, 3)}")
                    pinned_states = _root_state(
                        actor, initial[aid], pinned, fps,
                        int(plan["clock"]["ticks_per_frame"]))
                    candidates.append((pinned, pinned_states, {
                        "authority": "captured_trajectory_pinned",
                        "path_length_m": float(np.linalg.norm(
                            np.diff(pinned[begin_frame:], axis=0), axis=1).sum()),
                        "selected_frame_count": int(count - begin_frame),
                    }))
                per_actor_rejections[aid] = dict(rejected)
                continue
            speed_lower, speed_upper = (
                float(config["walk_speed_range_mps"][0]),
                float(config["walk_speed_range_mps"][1]),
            )
            # Give each actor its own side to swing towards, so the two do not
            # walk into the same corner of the room.
            actor_swing_sign = 1 if actor_index % 2 == 0 else -1
            # The routes the actors already planned. Only the part of each route
            # that moves is worth avoiding: before the window everyone is still
            # standing where the captured world put them.
            already_routed = [
                candidate[0][begin_frame:]
                for group in all_candidates for candidate in group
            ] or None
            bank = space.route_bank()
            native_choices = None
            if bank is not None:
                native_choices = _native_segments(space, start, minimum_frames, maximum_frames, fps)
                if not native_choices:
                    rejected["no_retained_native_segment_at_initial_position"] += 1
                rng.shuffle(native_choices)
            for attempt in range(attempts):
                if native_choices is not None:
                    if attempt >= len(native_choices):
                        break
                    sampled, record = native_choices[attempt]
                else:
                    # Propose a destination at a distance this profile can
                    # actually walk inside the motion window, instead of anywhere
                    # in a three-metre box. Half of every actor's attempts were
                    # being spent on destinations no declared speed could reach in
                    # time, which left one actor with a single candidate and the
                    # pair search with nothing to choose between. The distance band
                    # is derived from the same declared speeds and the same window;
                    # nothing about what counts as a legal path changes here.
                    reach_low = speed_lower * max(minimum_frames - 1, 1) / fps
                    reach_high = speed_upper * (maximum_frames - 1) / fps
                    bounds = space.bounds().copy()
                    bounds[0] = np.maximum(bounds[0], start - [reach_high, 0.3, reach_high])
                    bounds[1] = np.minimum(bounds[1], start + [reach_high, 0.3, reach_high])
                    # A path is never shorter than the straight line, so the low
                    # edge is relaxed a little: a route that detours can still be
                    # long enough from a nearer point.
                    reach_span = (reach_low * 0.85, reach_high)
                    # Draw plainly, at a distance this profile can walk in
                    # the window. An earlier version also aimed half the draws
                    # at a bearing the answer would have to change to. That
                    # raised how many candidates each source found and lost
                    # the pair: aimed destinations pass the per-source answer
                    # check by construction, so they come to dominate the
                    # accepted pool, and they all point into the wedge the
                    # camera can see, which is where the two sources stand
                    # closest together. Measured on this world, plain draws
                    # reached pairs more than a metre apart and the mixture
                    # never passed 0.63 m against a 0.95 m rule. The aiming
                    # helper is kept for a caller that wants it and is not
                    # used here.
                    destination = _sample_destination_within_reach(
                        space, rng, start, bounds, reach_span[0], reach_span[1])
                    if destination is None:
                        rejected["no_navigable_destination_within_reach"] += 1
                        continue
                    try:
                        if abs(destination[1] - start[1]) > 0.3:
                            continue
                        poly = space.shortest_path(start, destination)
                    except ValueError:
                        rejected["no_sample_or_path"] += 1
                        continue
                    if poly is None or len(poly) < 2:
                        rejected["no_path"] += 1
                        continue
                    length = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
                    # Draw the speed from the band that actually fits this path in
                    # the motion window, instead of drawing from the whole declared
                    # band and throwing the result away. Same constraints, same
                    # attempt budget: only the proposal changes. Drawing blind lost
                    # about two thirds of every actor's attempts here, and left one
                    # actor with a single candidate to pair against.
                    #
                    #   n = ceil(length / speed * fps) + 1  must lie in
                    #   [minimum_frames, maximum_frames], so with
                    #   k = length / speed * fps that is k in
                    #   (minimum_frames - 2, maximum_frames - 1], and therefore
                    #   speed in [length*fps/(maximum_frames-1),
                    #             length*fps/(minimum_frames-2)).
                    speed_lower_declared, speed_upper_declared = speed_lower, speed_upper
                    fitting_low = length * fps / max(maximum_frames - 1, 1)
                    fitting_high = (
                        length * fps / (minimum_frames - 2)
                        if minimum_frames > 2 else float("inf")
                    )
                    low = max(speed_lower_declared, fitting_low)
                    high = min(speed_upper_declared, fitting_high)
                    if not low <= high:
                        # No speed this profile permits can walk this path inside
                        # the window. That is a real rejection, not a wasted draw.
                        rejected["path_length_cannot_fit_any_declared_speed"] += 1
                        continue
                    speed = float(rng.uniform(low, high)) if high > low else float(low)
                    n = math.ceil(length/speed*fps) + 1
                    if not minimum_frames <= n <= maximum_frames:
                        rejected["path_does_not_fit_motion_window"] += 1
                        continue
                    sampled = resample_polyline_by_arc_length(poly, n)
                    record = {"authority": space.metadata["authority"], "polyline_m": poly.tolist(),
                              "speed_mps": length/max((n-1)/fps, 1/fps), "time_stretch_applied": False,
                              "selected_frame_count": len(sampled)}
                motion_speeds = np.linalg.norm(np.diff(sampled, axis=0), axis=1)*fps
                moving_speeds = motion_speeds[motion_speeds > 0.05]
                if (not len(moving_speeds)
                        or np.min(moving_speeds) < config["walk_speed_range_mps"][0] - 1e-5
                        or np.max(moving_speeds) > config["walk_speed_range_mps"][1] + 1e-5):
                    rejected["outside_declared_walk_speed"] += 1
                    continue
                # Native segments above are exact retained Recast samples. The
                # auxiliary raster is a camera-placement approximation, not the
                # authority for rejecting those already-native route coordinates.
                if bank is None and not all(space.is_navigable(point) for point in sampled):
                    rejected["non_navigable_interpolated_path"] += 1
                    continue
                path = np.repeat(start[None], count, axis=0)
                path[begin_frame:begin_frame+len(sampled)] = sampled
                path[begin_frame+len(sampled):] = sampled[-1]
                states = _root_state(actor, initial[aid], path, fps, int(plan["clock"]["ticks_per_frame"]))
                planned_change = abs(_wrap(
                    _public_degrees(_bearing(
                        states[-1]["root_transform"]["translation_m"], camera)
                        + float(bearing_drift.get(aid, 0.0)))
                    - _public_degrees(initial_angle)))
                if planned_change <= answer_threshold_deg:
                    rejected["answer_change_too_small"] += 1
                    continue
                visibility_failure = _visible_path(
                    states[begin_frame:], camera, mesh,
                    margin_deg=LATE_PATH_VISIBILITY_MARGIN_DEG)
                if visibility_failure:
                    rejected[visibility_failure] += 1
                    continue
                candidates.append((path, states, record))
            per_actor_rejections[aid] = dict(rejected)

    # Bounded extension. Each round spends the declared per-actor budget
    # again; the cap is what keeps a genuinely impossible world from
    # being searched forever, and the number of rounds actually used is
    # reported so a solution found on the sixth round does not look like
    # one found on the first.
    pair_actor_ids = [actor["actor_id"] for actor in actors]
    rounds_used = 0
    legal: list = []
    pair_rejected = Counter()
    closest_separation = None
    widest_gap_among_separated = None
    for _ in range(LATE_PATH_SEARCH_ROUNDS):
        rounds_used += 1
        _sample_one_round()
        if any(not group for group in all_candidates):
            continue
        legal = []
        pair_rejected = Counter()
        closest_separation = None
        widest_gap_among_separated = None
        for first in all_candidates[0]:
            for second in all_candidates[1]:
                separation = float(np.min(np.linalg.norm(first[0]-second[0],axis=1)))
                if closest_separation is None or separation > closest_separation:
                    closest_separation = separation
                if separation < config["minimum_entity_separation_m"]:
                    pair_rejected["closer_than_minimum_entity_separation"] += 1
                    continue
                endpoint_angles = [
                    _public_degrees(
                        _bearing(item[1][-1]["root_transform"]["translation_m"], camera)
                        + float(bearing_drift.get(aid_of, 0.0)))
                    for item, aid_of in zip((first, second), pair_actor_ids)]
                gap = abs(_wrap(endpoint_angles[1]-endpoint_angles[0]))
                if widest_gap_among_separated is None or gap > widest_gap_among_separated:
                    widest_gap_among_separated = float(gap)
                if gap <= answer_threshold_deg:
                    pair_rejected["endpoint_bearings_too_similar"] += 1
                    continue
                legal.append((first, second))
        if legal:
            break
    for actor_index, actor in enumerate(actors):
        if not all_candidates[actor_index]:
            aid = actor["actor_id"]
            raise native.BindingNativeError(
                f"no same-condition legal late path for {aid}: "
                f"{per_actor_rejections.get(aid, {})} after {rounds_used} "
                f"rounds of {attempts} attempts")
    if not legal:
        raise native.BindingNativeError(
            "no collision-separated pair of late paths meets the answer "
            "separation: "
            + json.dumps({
                "candidates_per_actor": [len(group) for group in all_candidates],
                "pairs_examined": sum(pair_rejected.values()),
                "pair_rejections": dict(pair_rejected),
                "widest_separation_m": (
                    None if closest_separation is None
                    else round(closest_separation, 3)),
                "required_separation_m": config["minimum_entity_separation_m"],
                "widest_endpoint_gap_deg_among_separated_pairs": (
                    None if widest_gap_among_separated is None
                    else round(widest_gap_among_separated, 2)),
                "required_endpoint_gap_deg": answer_threshold_deg,
                "post_capture_authority_requires_more_than_deg": (
                    2 * float(config["angle_tolerance_deg"])),
                "planner_margin_deg": answer_margin_deg,
                "answer_threshold_deg": answer_threshold_deg,
                "per_actor_rejections": per_actor_rejections,
                "retry_budget_within_profile": attempts,
                "search_rounds_used": rounds_used,
                "search_round_cap": LATE_PATH_SEARCH_ROUNDS,
            }, sort_keys=True)
        )
    selected = legal[int(rng.integers(len(legal)))]
    return {a["actor_id"]: selected[i][1] for i, a in enumerate(actors)}, {
        "selection": "uniform_over_legal_sampled_native_path_pairs",
        "selection_seed": selection_seed,
        "answer_geometry_proxy": "actor_root_projection; actual visible centroids remain the final authority",
        "initial_bearing_source": "native_visible_centroids" if initial_visual_bearings is not None else "actor_root_projection",
        "legal_pair_count": len(legal), "begin_frame": begin_frame, "latest_end_frame": end_frame,
        "minimum_required_window_frames": min(
            max(item[2]["selected_frame_count"] for item in pair) for pair in legal),
        "camera_motion": "static", "sources": {a["actor_id"]: selected[i][2] for i,a in enumerate(actors)},
    }



def _native_geometry_clip_budget(plan, request):
    package = plan.get("resources", {}).get("room_package", {})
    if package.get("walkable_space", {}).get("kind") != "route_bank":
        return None
    config = _settings(request)
    fps = float(plan["clock"]["frame_rate_hz"])
    end = int(plan["clock"]["frame_count"])-1-math.ceil(config["end_hold_s"]*fps)
    _, geometry = sample_late_paths(plan, request, 1, end)
    latest_wet_end = (end-geometry["minimum_required_window_frames"]-1)/fps
    # Reserve the caller's tail inside the audio-to-motion schedule. This
    # filters whole clips to a geometry-derived slot; it never changes the
    # caller's maximum clip setting, tail reservation or actual audio samples.
    return latest_wet_end-float(request["profile"]["reserve_tail_s"])-config["source_start_s"]

def materialize_motion_visual(base_root, output, plan):
    base, output = Path(base_root).resolve(), Path(output).resolve()
    if output.exists():
        raise native.BindingNativeError(f"refusing existing visual output: {output}")
    (output/"plan").mkdir(parents=True)
    for name in ("room_package.json", "path_bindings.json", "room_layout.json", "navigation.npz"):
        source = base/"plan"/name
        if source.is_file():
            shutil.copy2(source, output/"plan"/name)
    native._write(output/"request.json", plan["request"])
    native._write(output/"plan/episode_plan.json", plan)
    native._write(output/"plan/audio_events.json", plan["audio_events"])
    native._write(output/"plan/voice_bindings.json", plan["voice_bindings"])
    if native.room_family_from_plan(plan) in {"hm3d", "mp3d"}:
        from avengine.assets.mp3d_region_actor_tracks import materialize_common_plan_habitat
        shutil.copy2(base/"plan/habitat_room_manifest.json", output/"plan/habitat_room_manifest.json")
        materialize_common_plan_habitat(
            plan=plan, room_manifest=output/"plan/habitat_room_manifest.json",
            runtime_registry=plan["request"]["source_registry"], output=output/"plan/habitat_execution",
            allow_research_candidate=bool(plan["request"].get("allow_research_candidate_assets",False)))
    return output


def _render_state_column(
    plan,
    captured,
    root,
    visual_id,
    assignment,
    *,
    request: Mapping[str, Any] | None = None,
    shared_visual_root: str | Path | None = None,
):
    endpoints = native._neutral_endpoint_bindings(captured["neutral_readback"], plan=plan)
    member_id = f"{visual_id}_{assignment}"
    column_plan = deepcopy(plan)
    if "binding_motion_audio_columns" in plan:
        column_plan.update(deepcopy(plan["binding_motion_audio_columns"][assignment]))
    audio_request = request if request is not None else column_plan.get("request")
    if not isinstance(audio_request, Mapping):
        raise native.BindingNativeError(
            f"{member_id} has no request for audio assignment"
        )
    column_plan["request"] = deepcopy(dict(audio_request))
    assigned, req = native.build_audio_assignment_plan(
        column_plan, audio_request, assignment,
        assignment_targets={"a0": ["source1"], "a1": ["source2"]},
        expected_event_count=1, endpoint_by_actor=endpoints)
    output = native.materialize_audio_variant(
        captured, root / "variants" / member_id, assigned, req, member_id=member_id
    )
    result = native.finalize_audio_assignment(
        output, req, shared_visual_root=shared_visual_root
    )
    result = {
        **result,
        "variant_root": str(output),
        "assignment_plan_path": str(output / "plan/episode_plan.json"),
        "assignment_request_path": str(output / "request.json"),
        "assignment": assignment,
        "visual_id": visual_id,
    }
    native._write(root / f"{member_id}_audio.json", result)
    if not captured.get("visual_video"):
        captured["visual_video"] = result.get("visual_video") or result["preview"]
    return result


def _native_visual_endpoint_bearings(capture, facts, actor_ids, *, owner):
    from avengine.qa.angular_questions import public_camera_calibration
    calibration = public_camera_calibration(facts)
    if calibration is None:
        raise native.BindingNativeError("state visual preflight lacks public camera calibration")
    last_frame = int(facts["time"]["frame_count"]) - 1
    path = Path(capture["capture"]) / "pixel_visibility_truth.json"
    pixels = native._load(path)
    if pixels.get("resolution_hw") != [calibration["height_px"], calibration["width_px"]]:
        raise native.BindingNativeError("state visual preflight camera resolution differs")
    angles = {}
    for actor_id in actor_ids:
        rows = pixels.get("per_instance", {}).get(actor_id, {}).get("frames", [])
        row = next((item for item in rows if item.get("frame_index") == last_frame), None)
        if row is None or row.get("state") not in {"visible_clear", "visible_occluded"}:
            raise native.BindingNativeError(f"state endpoint is not visible: {owner}/{actor_id}")
        centroid = row.get("visible_centroid_xy_px")
        if centroid is None and row.get("visible_fraction") == 1.0:
            centroid = row.get("target_centroid_xy_px")
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            raise native.BindingNativeError("state endpoint lacks an actual visible centroid")
        x, y = map(float, centroid)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < calibration["width_px"] and 0 <= y < calibration["height_px"]):
            raise native.BindingNativeError("state endpoint centroid escapes the actual image")
        angles[actor_id] = math.degrees(math.atan2(x-calibration["cx_px"], calibration["fx_px"]))
    return angles


def native_state_visual_history_preflight(captured, facts_by_assignment):
    """Apply the final question's visibility-history requirement before late RLR."""
    from avengine.qa.binding_questions import _at_event, _event
    from avengine.qa.unified_catalog import VISIBLE_STATES

    checks = []
    for visual_id, capture in captured.items():
        pixels = native._load(Path(capture["capture"]) / "pixel_visibility_truth.json")
        visibility = {
            actor_id: {int(row["frame_index"]): row for row in record["frames"]}
            for actor_id, record in pixels["per_instance"].items()
        }
        for assignment, facts in facts_by_assignment.items():
            visual_facts = dict(facts, visibility=visibility)
            anchor, candidates = _at_event(visual_facts, _event(visual_facts, 1))
            end = int(facts["time"]["frame_count"])
            for actor_id in candidates:
                invalid = [frame for frame in range(anchor, end)
                           if visibility.get(actor_id, {}).get(frame, {}).get("state")
                           not in VISIBLE_STATES]
                checks.append({"visual_id": visual_id, "assignment": assignment,
                               "actor_id": actor_id, "anchor_frame": anchor,
                               "query_frame": end - 1, "invalid_frames": invalid,
                               "pass": not invalid})
    return {"status": "pass" if all(row["pass"] for row in checks) else "fail",
            "authority": "actual native pixel states and the final question anchor rule",
            "checks": checks}


def _public_degrees(angle):
    """The integer degree the dataset publishes for a bearing."""
    return (int(round(float(angle))) + 180) % 360 - 180


#: How far inside the camera frustum a late path must stay while it is being
#: planned. This is a planning inset, not the authority: the authority is the
#: pixel visibility read back from the actual capture. Measured on the M23
#: world, the inset at five degrees refused every repair of the one path that
#: failed -- seven thousand eight hundred attempts, no solution -- while the
#: same search at three degrees found six, the best of them 1.48 m clear of
#: the other source and passing the published-angle rule by 22 and 52 degrees
#: against a rule of more than 20. Five degrees was rejecting paths the real
#: visibility test accepts, and a conservative proxy that refuses legal work
#: is not being careful, it is being wrong. Three keeps a real margin against
#: the frustum edge without standing in for the authority.
LATE_PATH_VISIBILITY_MARGIN_DEG = 3.0

#: Extra degrees the late-path planner demands on top of what the post-capture
#: authority demands. The planner measures a planned root position; the
#: authority measures the actual visible pixel centroid and then rounds to the
#: published integer degree. A pair that clears the rule by a fraction of a
#: degree in the plan can land exactly on it once captured -- the first real
#: v1 capture of this group failed one of its four comparisons by a single
#: degree, twenty against a rule of more than twenty, after a native world had
#: already been opened. The margin is here so that drift is paid for in CPU
#: sampling rather than in a spent native capture.
LATE_PATH_ANSWER_MARGIN_DEG = DEFAULT_LATE_PATH_ANSWER_MARGIN_DEG


def native_state_visual_angle_preflight(captured, facts, actor_ids, tolerance_deg):
    """Check actual rounded visual answers before rendering late audio."""
    last_frame = int(facts["time"]["frame_count"]) - 1
    precise = {v: _native_visual_endpoint_bearings(captured[v], facts, actor_ids, owner=v)
               for v in ("v0", "v1")}
    angles = {v: {actor: _public_degrees(angle) for actor, angle in values.items()}
              for v, values in precise.items()}
    first, second = actor_ids
    pairs = [
        ("audio_a0", ("v0", first), ("v1", first)),
        ("audio_a1", ("v0", second), ("v1", second)),
        ("video_v0", ("v0", first), ("v0", second)),
        ("video_v1", ("v1", first), ("v1", second)),
    ]
    comparisons = [{
        "comparison": label,
        "answer_separation_deg": abs(_wrap(angles[a[0]][a[1]]-angles[b[0]][b[1]])),
    } for label, a, b in pairs]
    for row in comparisons:
        row["pass"] = row["answer_separation_deg"] > 2*float(tolerance_deg)
    return {
        "status": "pass" if all(row["pass"] for row in comparisons) else "fail",
        "authority": "actual native visible pixel centroids and public camera calibration",
        "query_frame": last_frame, "angles_deg": angles, "angles_full_precision_deg": precise,
        "angle_tolerance_deg": float(tolerance_deg), "comparisons": comparisons,
        "claim": "visual separation only; actual audio and full group validation remain required",
    }


def prepare_fixed_camera_state_group(*, base_episode_root, output_root, group_id, world_id,
                                     reuse_static_group_root=None, rpc_port=None, graphics_adapter=None,
                                     resample_early_audio=False, route_seed_offset=None):
    base, root = Path(base_episode_root).resolve(), Path(output_root).resolve()
    if root.exists():
        raise native.BindingNativeError(f"refusing existing group output: {root}")
    root.mkdir(parents=True)
    base_plan = native._load(base/"plan/episode_plan.json")
    request = deepcopy(base_plan["request"])
    runtime_overrides = {}
    for key, value in (("rpc_port", rpc_port), ("graphics_adapter", graphics_adapter)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "rpc_port" else 0):
                raise native.BindingNativeError(f"invalid runtime {key}")
            runtime_overrides[key] = value
    request.setdefault("runtime", {}).update(runtime_overrides)
    retained = Path(reuse_static_group_root).resolve() if reuse_static_group_root else None
    if retained:
        # Continue into a fresh output after a route/capture failure, retaining
        # the two expensive, completed native audio columns unchanged.
        plan = native._load(retained/"visual/v0/plan/episode_plan.json")
        from avengine.qa.binding_conditions import validate_binding_episode
        for key in ("frame_count", "frame_rate_hz", "sample_rate_hz", "room_id",
                    "source_registry", "source_asset_ids", "camera", "seed"):
            if request.get(key) != plan["request"].get(key):
                raise native.BindingNativeError(f"retained static request differs in {key}")
        for assignment in ("a0", "a1"):
            previous_path = retained/f"v0_{assignment}_audio.json"
            if previous_path.is_file():
                previous = native._load(previous_path)
                facts_path = native._file(previous["facts"], base=retained, owner="retained audio facts")
                validate_binding_episode(native._load(facts_path), request=request,
                                         base=facts_path.parent, cache={})
        request = deepcopy(plan["request"])
        request.setdefault("runtime", {}).update(runtime_overrides)
        plan["request"] = deepcopy(request)
        config = _settings(request)
        end = int(plan["clock"]["frame_count"]) - 1 - math.ceil(
            config["end_hold_s"]*float(plan["clock"]["frame_rate_hz"]))
        if resample_early_audio:
            budget = _native_geometry_clip_budget(plan, request)
            plan, request, _, end = select_early_audio(
                plan, request, np.random.default_rng(int(request["seed"])+800),
                geometry_clip_budget_s=budget)
        native._write(root/"reused_static_group.json", {"source": str(retained),
                      "reuse_completed_audio": not resample_early_audio,
                      "reason": "resume native static capture into fresh downstream output"})
    else:
        rng = np.random.default_rng(int(request["seed"])+800)
        budget = _native_geometry_clip_budget(base_plan, request)
        plan, request, _, end = select_early_audio(base_plan, request, rng, geometry_clip_budget_s=budget)
    if route_seed_offset is not None:
        if isinstance(route_seed_offset, bool) or not isinstance(route_seed_offset, int) or route_seed_offset < 0:
            raise native.BindingNativeError("route_seed_offset must be a nonnegative integer")
        request.setdefault("binding_motion", {})["route_seed_offset"] = route_seed_offset
        plan["request"] = deepcopy(request)
    native._write(root/"request.json",request)
    native._write(root/"base_plan_source.json",{"path":str(base/"plan/episode_plan.json")})
    common = deepcopy(plan)
    if "base_layout_conditions" not in common:
        common["base_layout_conditions"] = {key: common.pop(key,None) for key in
                                            ("condition_profile","planned_conditions","planning_result","activity_plan")}
    common["request"]["sampling_policy"] = "binding_fixed_camera_motion_v1"
    v0 = deepcopy(common)
    v0["episode_id"] = f"{group_id}_v0"
    v0["request"]["episode_id"] = v0["episode_id"]
    v0["binding_motion"] = {"variant":"v0","motion":"stationary"}
    if retained:
        captured = {"v0":native._load(retained/"v0_capture.json")}
        # Keep a local plan for another explicit continuation; the capture
        # receipt still points to the original native files.
        (root/"visual/v0/plan").mkdir(parents=True)
        native._write(root/"visual/v0/plan/episode_plan.json", v0)
    else:
        output = materialize_motion_visual(base,root/"visual/v0",v0)
        captured = {"v0":native.capture_visual_plan(v0["request"],output,label="v0")}
    native._write(root/"v0_capture.json",captured["v0"])
    variants = {}
    wet_ends = []
    for assignment in ("a0","a1"):
        member_id = f"v0_{assignment}"
        if retained and not resample_early_audio and (retained/f"{member_id}_audio.json").is_file():
            variants[member_id] = native._load(retained/f"{member_id}_audio.json")
            native._write(root/f"{member_id}_audio.json",variants[member_id])
        else:
            variants[member_id] = _render_state_column(v0,captured["v0"],root,"v0",assignment)
        facts = native._load(Path(variants[member_id]["facts"]))
        tails = facts["audio"].get("wet_tail_intervals")
        if not tails:
            raise native.BindingNativeError("early native audio lacks wet-tail measurements")
        wet_ends.extend(float(row["end_s"]) for row in tails)
    fps = float(plan["clock"]["frame_rate_hz"])
    begin = math.ceil(max(wet_ends)*fps)+1
    minimum_frames = math.ceil(float(request["binding_motion"]["minimum_motion_s"])*fps)
    if end-begin+1 < minimum_frames:
        raise native.BindingNativeError("measured reverberation leaves insufficient declared movement time")
    native._write(root/"measured_motion_window.json",{
        "first_motion_frame":begin,"last_motion_frame":end,"measured_wet_end_s":max(wet_ends),
        "authority":"actual early binaural readbacks","requested_terminal_tail_s":request["profile"]["reserve_tail_s"]})
    initial_visual_bearings = _native_visual_endpoint_bearings(
        captured["v0"], native._load(Path(variants["v0_a0"]["facts"])),
        [actor["actor_id"] for actor in plan["visual_plan"]["actors"]], owner="v0",
    )
    states,route = sample_late_paths(
        plan,request,begin,end,initial_visual_bearings=initial_visual_bearings)
    v1 = deepcopy(common)
    v1["episode_id"] = f"{group_id}_v1"
    v1["request"]["episode_id"] = v1["episode_id"]
    v1["binding_motion"] = {"variant":"v1","route":route}
    for frame in v1["visual_plan"]["frames"]:
        frame["actor_states"] = [deepcopy(states[a["actor_id"]][frame["frame_index"]])
                                  for a in v1["visual_plan"]["actors"]]
    output = materialize_motion_visual(base,root/"visual/v1",v1)
    captured["v1"] = native.capture_visual_plan(v1["request"],output,label="v1")
    native._write(root/"v1_capture.json",captured["v1"])
    history_check = native_state_visual_history_preflight(
        captured, {assignment: native._load(Path(variants[f"v0_{assignment}"]["facts"]))
                   for assignment in ("a0", "a1")})
    native._write(root/"native_visual_history_preflight.json", history_check)
    if history_check["status"] != "pass":
        raise native.BindingNativeError("actual visual identity history is interrupted before the state query")
    visual_check = native_state_visual_angle_preflight(
        captured, native._load(Path(variants["v0_a0"]["facts"])),
        [actor["actor_id"] for actor in plan["visual_plan"]["actors"]],
        request["binding_motion"]["angle_tolerance_deg"],
    )
    native._write(root/"native_visual_angle_preflight.json", visual_check)
    if visual_check["status"] != "pass":
        raise native.BindingNativeError(
            f"actual native visual answers are not separated under the declared tolerance: {visual_check['angles_deg']}")
    for assignment in ("a0","a1"):
        member_id = f"v1_{assignment}"
        variants[member_id] = _render_state_column(v1,captured["v1"],root,"v1",assignment)
        facts = native._load(Path(variants[member_id]["facts"]))
        if max(float(row["end_s"]) for row in facts["audio"]["wet_tail_intervals"]) >= begin/fps:
            raise native.BindingNativeError("actual wet tail reaches the actor-motion window")
    spec = native._group_spec(group_id,world_id,native.room_family_from_plan(plan),plan["scene"]["room_id"],captured,variants)
    spec["request"] = request
    spec["groups"][0].update(task_family="cross_time_state",query={"event_number":1,"query_anchor":"clip_end"},
                             angle_tolerance_deg=request["binding_motion"]["angle_tolerance_deg"])
    native._write(root/"group_spec.json",spec)
    from avengine.qa.binding_groups import assemble_binding_dataset
    result = assemble_binding_dataset(spec,input_base=root,output=root/"assembled",seed=group_id)
    summary = {"status":"pass","captured":captured,"variants":variants,"route":route,
               "group_count":result["group_count"],"sample_count":result["sample_count"],
               "validation":result["validation"],"assembled":str(root/"assembled/binding_groups.json")}
    native._write(root/"summary.json",summary)
    return summary


# ---------------------------------------------------------------------------
# Cross-time state recipe: the same native save/resume dispatcher with
# recipe-owned plan, late-plan, audio and assembly stage functions.
# ---------------------------------------------------------------------------

STATE_TASK_FAMILY = "cross_time_state"


def _state_request_from_result(row: Mapping[str, Any], *, unit_id: str) -> dict[str, Any]:
    outputs = row.get("outputs") or {}
    value = outputs.get("request_path")
    if not isinstance(value, str) or not Path(value).expanduser().resolve().is_file():
        raise native.BindingNativeError(
            f"{unit_id} result has no readable request_path"
        )
    return native._load(Path(value).expanduser().resolve())


def _state_early_facts(
    done: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    facts_by_assignment: dict[str, dict[str, Any]] = {}
    for assignment in ("a0", "a1"):
        row = _state_result(done, f"v0_{assignment}")
        facts = row.get("facts") or {}
        value = facts.get("facts_path")
        if (
            not isinstance(value, str)
            or not Path(value).expanduser().resolve().is_file()
        ):
            raise native.BindingNativeError(
                f"v0_{assignment} has no readable facts_path"
            )
        facts_by_assignment[assignment] = native._load(
            Path(value).expanduser().resolve()
        )
    return facts_by_assignment


def _state_plan_asset_order(plan: Mapping[str, Any]) -> list[str]:
    visual = plan.get("visual_plan")
    actors = visual.get("actors") if isinstance(visual, Mapping) else None
    if not isinstance(actors, list):
        return []
    return [
        str(actor.get("asset_id"))
        for actor in actors
        if isinstance(actor, Mapping) and actor.get("actor_id")
    ]


def _state_validate_static_base(
    plan: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    visual = plan.get("visual_plan")
    if not isinstance(visual, Mapping):
        raise native.BindingNativeError(
            "cross_time_state base plan lacks visual_plan"
        )
    camera = visual.get("camera")
    if not isinstance(camera, Mapping) or camera.get("motion") != "static":
        raise native.BindingNativeError(
            "cross_time_state base plan must use a static camera"
        )
    scene = plan.get("scene")
    if isinstance(scene, Mapping) and scene.get("room_id") not in {
        None, request.get("room_id")
    }:
        raise native.BindingNativeError(
            "cross_time_state base plan scene differs from the requested room"
        )
    clock = plan.get("clock")
    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz"):
        if key in request and isinstance(clock, Mapping):
            if float(clock.get(key, -1)) != float(request[key]):
                raise native.BindingNativeError(
                    f"cross_time_state base plan clock differs in {key}"
                )
    declared_assets = request.get("source_asset_ids")
    planned_assets = _state_plan_asset_order(plan)
    if (
        isinstance(declared_assets, Sequence)
        and not isinstance(declared_assets, (str, bytes))
        and planned_assets
        and planned_assets != [str(value) for value in declared_assets]
    ):
        raise native.BindingNativeError(
            "cross_time_state base plan asset order differs from its request"
        )


def _state_bind_plan_request(
    plan: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    value = deepcopy(dict(plan))
    visual = value.get("visual_plan")
    if not isinstance(visual, Mapping):
        return value
    declared_assets = request.get("source_asset_ids")
    if (
        not isinstance(declared_assets, Sequence)
        or isinstance(declared_assets, (str, bytes))
    ):
        return value
    assets = [str(item) for item in declared_assets]
    by_actor = {
        f"source{index + 1}": asset
        for index, asset in enumerate(assets)
    }
    actors = visual.get("actors")
    if isinstance(actors, list):
        for actor in actors:
            if isinstance(actor, Mapping):
                actor_id = str(actor.get("actor_id") or "")
                if actor_id in by_actor:
                    actor["asset_id"] = by_actor[actor_id]
    frames = visual.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            states = frame.get("actor_states")
            if isinstance(states, Mapping):
                iterable = states.values()
            elif isinstance(states, list):
                iterable = states
            else:
                iterable = ()
            for state in iterable:
                if isinstance(state, Mapping):
                    actor_id = str(state.get("actor_id") or "")
                    if actor_id in by_actor:
                        state["asset_id"] = by_actor[actor_id]
    return value


def _state_shared_visual_root(
    context: Mapping[str, Any], output_root: str | Path
) -> str:
    value = context.get("shared_visual_root") or context.get(
        "t01_shared_visual_root"
    )
    if isinstance(value, (str, Path)) and str(value).strip():
        return str(Path(value).expanduser().resolve())
    return str(
        native.shared_visual_evidence_root(output_root, context["group_id"])
    )


# --- state variant execution materialization ---------------------------------
# capture_command reads its Habitat execution inputs out of the plan directory a
# plan unit reports, because run_visual_capture_unit symlinks that directory. A
# plan unit that reports a directory holding only episode_plan.json therefore
# fails at the capture boundary before any renderer starts. The base plan is not
# a substitute: the variant plan differs from it, so the execution inputs have
# to be materialized from the variant.
STATE_VARIANT_PLAN_FILES = (
    "episode_plan.json", "audio_events.json", "voice_bindings.json",
)
STATE_HABITAT_EXECUTION_FILES = (
    "habitat_room_manifest.json",
    "habitat_execution/case_manifest.json",
    "habitat_execution/m1_capture_request.json",
)
STATE_UE_EXECUTION_FILES = ("room_package.json",)


def state_variant_execution_requirements(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Files capture_command needs beside this plan, by its real room family."""
    family = native.room_family_from_plan(plan)
    if family in {"hm3d", "mp3d"}:
        return STATE_VARIANT_PLAN_FILES + STATE_HABITAT_EXECUTION_FILES
    if family in {"kujiale", "apartment"}:
        return STATE_VARIANT_PLAN_FILES + STATE_UE_EXECUTION_FILES
    raise native.BindingNativeError(
        f"cross_time_state does not support room family {family!r}"
    )


def state_variant_materialization_status(
    plan_path: str | Path, *, plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report whether a reported state plan directory can drive a capture.

    Read-only. ``missing`` is what capture_command would refuse on, named before
    an attempt is charged a world instead of after.
    """
    path = Path(plan_path).expanduser().resolve()
    if not path.is_file():
        raise native.BindingNativeError(
            f"state variant plan is unavailable: {path}"
        )
    value = dict(plan) if isinstance(plan, Mapping) else native._load(path)
    plan_dir = path.parent
    required = state_variant_execution_requirements(value)
    missing = [name for name in required if not (plan_dir / name).is_file()]
    return {
        "schema": "avengine_state_variant_materialization_status_v1",
        "episode_plan_path": str(path),
        "plan_dir": str(plan_dir),
        "room_family": native.room_family_from_plan(value),
        "plan_coordinates": value.get("plan_coordinates"),
        "required": list(required),
        "missing": missing,
        "present": [name for name in required if (plan_dir / name).is_file()],
        "status": "materialized" if not missing else "incomplete",
        "base_plan_source": value.get("base_plan_source"),
    }


def _state_repair_base_root(
    plan: Mapping[str, Any],
    plan_path: Path,
    *,
    base_root: str | Path | None,
    context: Mapping[str, Any] | None,
) -> Path:
    """Resolve the base episode whose room resources the variant was planned on.

    The variant plan records its own ``base_plan_source``, so that recorded link
    is preferred over walking the directory tree, which can otherwise climb into
    an unrelated attempt.
    """
    candidates: list[Path] = []
    if base_root is not None:
        candidates.append(Path(base_root).expanduser().resolve())
    for value in (
        plan.get("base_plan_source"),
        (plan.get("state_recipe") or {}).get("base_plan_source")
        if isinstance(plan.get("state_recipe"), Mapping) else None,
    ):
        if isinstance(value, str) and value.strip():
            recorded = Path(value).expanduser().resolve()
            candidates.append(
                recorded.parent.parent if recorded.name.endswith(".json")
                else recorded
            )
    for candidate in candidates:
        if (candidate / "plan").is_dir():
            return candidate
    if context is not None:
        return _state_materialization_base(context, plan_path)
    raise native.BindingNativeError(
        "cross_time_state repair cannot find the base episode of "
        f"{plan_path}; looked at "
        + ", ".join(str(path) for path in candidates)
    )


def repair_state_variant_materialization(
    plan_path: str | Path,
    output: str | Path,
    *,
    base_root: str | Path | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete one thin state variant plan directory into a fresh tree.

    The reported plan file and its directory stay read-only: the variant plan is
    read out of it and materialized again, in full, under a fresh no-clobber
    output. Only the room resources the base episode owns come from the base;
    camera, clock, audio program, voice bindings and tracks are all the
    variant's, and the returned checks prove that rather than asserting it.
    Pure CPU: no renderer, no RLR, no capture.
    """
    path = Path(plan_path).expanduser().resolve()
    plan = native._load(path)
    before = state_variant_materialization_status(path, plan=plan)
    base = _state_repair_base_root(
        plan, path, base_root=base_root, context=context,
    )
    base_plan_path = base / "plan/episode_plan.json"
    if not base_plan_path.is_file():
        raise native.BindingNativeError(
            f"cross_time_state repair base episode has no plan: {base_plan_path}"
        )
    base_plan = native._load(base_plan_path)
    for key in ("room_id", "scene_id"):
        variant_room = (plan.get("scene") or {}).get(key)
        base_room = (base_plan.get("scene") or {}).get(key)
        if variant_room is not None and base_room is not None:
            if str(variant_room) != str(base_room):
                raise native.BindingNativeError(
                    "cross_time_state repair refuses a base episode from another "
                    f"room: variant {variant_room!r} != base {base_room!r}"
                )
    target = Path(output).expanduser().resolve()
    materialized = materialize_motion_visual(base, target, plan)
    repaired_plan_path = materialized / "plan/episode_plan.json"
    after = state_variant_materialization_status(repaired_plan_path)
    if after["missing"]:
        raise native.BindingNativeError(
            "cross_time_state repair did not complete the plan directory: "
            f"{after['missing']}"
        )
    checks = _state_variant_materialization_checks(
        plan, base_plan, materialized,
    )
    failures = [name for name, ok in checks.items() if ok is False]
    if failures:
        raise native.BindingNativeError(
            f"cross_time_state repair produced a base-derived tree: {failures}"
        )
    return {
        "schema": "avengine_state_variant_repair_v1",
        "status": "pass",
        "materialized_from": str(path),
        "base_episode_root": str(base),
        "plan_root": str(materialized),
        "episode_plan_path": str(repaired_plan_path),
        "request_path": str(materialized / "request.json"),
        "before": before,
        "after": after,
        "variant_evidence": checks,
        "native_visual_worlds_created": 0,
        "native_acoustic_contexts_created": 0,
    }


def _state_variant_materialization_checks(
    plan: Mapping[str, Any],
    base_plan: Mapping[str, Any],
    materialized: Path,
) -> dict[str, Any]:
    """Prove the materialized tree carries the variant, not the base episode."""
    written = native._load(materialized / "plan/episode_plan.json")
    checks: dict[str, Any] = {
        "episode_plan_equals_variant": _state_json_equal(written, plan),
        "episode_plan_differs_from_base": not _state_json_equal(written, base_plan),
    }
    for name, key in (("audio_events", "audio_events"),
                      ("voice_bindings", "voice_bindings")):
        side = materialized / f"plan/{name}.json"
        if not side.is_file():
            checks[f"{name}_written"] = False
            continue
        # these two are JSON arrays, so they are read plainly rather than
        # through the object-only loader
        value = json.loads(side.read_text(encoding="utf-8"))
        checks[f"{name}_equals_variant"] = _state_json_equal(value, plan.get(key))
        if key in base_plan:
            checks[f"{name}_differs_from_base"] = not _state_json_equal(
                value, base_plan.get(key)
            )
    camera = (written.get("visual_plan") or {}).get("camera")
    checks["camera_equals_variant"] = _state_json_equal(
        camera, (plan.get("visual_plan") or {}).get("camera")
    )
    checks["clock_equals_variant"] = _state_json_equal(
        written.get("clock"), plan.get("clock")
    )
    case = materialized / "plan/habitat_execution/case_manifest.json"
    if case.is_file():
        case_clock = native._load(case).get("clock") or {}
        variant_clock = plan.get("clock") or {}
        checks["case_manifest_clock_equals_variant"] = all(
            case_clock.get(key) == variant_clock.get(key)
            for key in ("frame_count", "frame_rate_hz", "sample_rate_hz",
                        "sample_count", "time_base_hz", "ticks_per_frame")
        )
    tracks_dir = materialized / "plan/habitat_execution/tracks"
    if tracks_dir.is_dir():
        checks["tracks"] = _state_track_variant_evidence(plan, base_plan, tracks_dir)
    return checks


def _state_json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _state_plan_translations(plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows: dict[str, list[list[float]]] = {}
    for frame in (plan.get("visual_plan") or {}).get("frames") or ():
        if not isinstance(frame, Mapping):
            continue
        for state in _frame_states(frame):
            actor_id = state.get("actor_id")
            transform = state.get("root_transform") or {}
            point = transform.get("translation_m")
            if not isinstance(actor_id, str) or point is None:
                continue
            rows.setdefault(actor_id, []).append(list(point))
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def _state_track_variant_evidence(
    plan: Mapping[str, Any], base_plan: Mapping[str, Any], tracks_dir: Path,
) -> dict[str, Any]:
    """Compare each materialized track against the variant and the base plan."""
    variant = _state_plan_translations(plan)
    base = _state_plan_translations(base_plan)
    rows: dict[str, Any] = {}
    for track_path in sorted(tracks_dir.glob("*.json")):
        track = json.loads(track_path.read_text(encoding="utf-8"))
        if not isinstance(track, Mapping):
            continue
        actor_id = str(track.get("actor_id") or track_path.stem)
        points = []
        for frame in track.get("frames") or ():
            if not isinstance(frame, Mapping):
                continue
            point = None
            for key in ("planned_world_from_actor", "root_transform"):
                transform = frame.get(key)
                if isinstance(transform, Mapping) and transform.get("translation_m") is not None:
                    point = transform["translation_m"]
                    break
            if point is None:
                point = frame.get("planned_route_center_m") or frame.get("position_m")
            if point is not None:
                points.append(list(point))
        row: dict[str, Any] = {"frame_count": len(points)}
        if points and actor_id in variant:
            observed = np.asarray(points, dtype=float)
            if observed.shape == variant[actor_id].shape:
                row["max_delta_to_variant_m"] = float(
                    np.max(np.linalg.norm(observed - variant[actor_id], axis=1))
                )
                row["matches_variant"] = row["max_delta_to_variant_m"] <= 1.0e-6
            if actor_id in base and observed.shape == base[actor_id].shape:
                row["max_delta_to_base_m"] = float(
                    np.max(np.linalg.norm(observed - base[actor_id], axis=1))
                )
        rows[actor_id] = row
    matched = [row.get("matches_variant") for row in rows.values()]
    return {
        "by_actor": rows,
        "all_tracks_match_the_variant": bool(matched) and all(
            value is True for value in matched
        ),
    }


def _state_plan_result(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    plan: Mapping[str, Any],
    *,
    base_root: str | Path | None = None,
) -> dict[str, Any]:
    request = plan.get("request")
    if not isinstance(request, Mapping):
        raise native.BindingNativeError(
            f"{item['unit_id']} plan has no complete request"
        )
    request_path = native._write(unit_root / "request.json", request)
    original_request_path = None
    original_request = plan.get("state_original_request")
    if isinstance(original_request, Mapping):
        original_request_path = native._write(
            unit_root / "state_original_request.json", original_request
        )
    # The capture unit symlinks the directory of the plan reported here, so the
    # variant's execution inputs have to live in it. Writing only the episode
    # plan leaves capture_command with no case manifest and no m1 request.
    base = _state_repair_base_root(
        plan, unit_root / "episode/plan/episode_plan.json",
        base_root=base_root, context=context,
    )
    materialized = materialize_motion_visual(base, unit_root / "episode", plan)
    plan_path = materialized / "plan/episode_plan.json"
    status = state_variant_materialization_status(plan_path, plan=plan)
    if status["missing"]:
        raise native.BindingNativeError(
            f"{item['unit_id']} plan directory is not materialized for capture: "
            f"{status['missing']}"
        )
    renderer = native._plan_renderer(plan)
    source_assets = _state_plan_asset_order(plan)
    return native._stage_result(
        item,
        status="pass",
        facts={
            "episode_plan_path": str(plan_path),
            "renderer": renderer,
            "clock": deepcopy(plan.get("clock")),
        },
        outputs={
            "plan_root": str(plan_path.parent.parent),
            "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "variant_materialization": status,
            "base_episode_root_used": str(base),
            "original_request_path": (
                None if original_request_path is None
                else str(original_request_path)
            ),
            "source_asset_ids": source_assets,
            "state_recipe": STATE_TASK_FAMILY,
            "base_episode_root": context.get("state_base_episode_root"),
        },
    )


def _run_state_visual_plan_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del lease
    unit_id = str(item.get("unit_id") or "")
    if unit_id == "v1":
        return _run_state_late_plan_unit(
            item, context, unit_root, output_root=output_root, results=results
        )
    if unit_id != "v0":
        raise native.BindingNativeError(
            f"cross_time_state visual_plan runner cannot handle {unit_id!r}"
        )
    request = _state_member_request(context, item)
    base_plan_bootstrap = None
    try:
        base_plan_path = _state_base_plan_path(context, request)
        base_plan_root = _state_materialization_base(context, base_plan_path)
    except native.BindingNativeError:
        # A fresh core group has no retained/ordinary base episode yet. Use the
        # existing ordinary plan-only entry point to create a CPU plan input;
        # this never invokes visual capture, GPU, or RLR.
        bootstrap_request_path = native._write(
            unit_root / "base_plan_request.json", request
        )
        planned = native.plan_visual_variant(
            bootstrap_request_path,
            unit_root / "base_plan",
            label=f"{unit_id}_base_plan",
            log=unit_root / f"{unit_id}.base_plan.log",
        )
        base_plan_root = Path(planned["output"]).expanduser().resolve()
        base_plan_path = Path(planned["plan"]).expanduser().resolve()
        base_plan_bootstrap = {
            "source": "ordinary_plan_only",
            "entrypoint": "avengine.dataset.binding_group_native.plan_visual_variant",
            "plan_path": str(base_plan_path),
            "base_episode_root": str(base_plan_root),
            "native_visual_worlds_created": 0,
            "native_acoustic_contexts_created": 0,
        }
    base_plan = native._load(base_plan_path)
    _state_validate_static_base(base_plan, request)
    budget = context.get("state_geometry_clip_budget_s")
    if budget is None:
        budget = _native_geometry_clip_budget(base_plan, request)
    early_plan, early_request, _begin, _end = select_early_audio(
        base_plan,
        request,
        np.random.default_rng(int(request.get("seed", 0)) + 800),
        geometry_clip_budget_s=budget,
    )
    _validate_stationary_plan(early_plan)
    early_plan["request"] = deepcopy(early_request)
    early_plan["state_recipe"] = {
        "task_family": STATE_TASK_FAMILY,
        "variant": "v0",
        "motion": "stationary",
        "base_plan_source": str(base_plan_path),
        "geometry_bootstrap": True,
        "original_request_preserved": True,
        "base_plan_bootstrap": deepcopy(base_plan_bootstrap),
    }
    early_plan["state_original_request"] = deepcopy(request)
    early_plan["base_plan_source"] = str(base_plan_path)
    early_plan["base_plan_bootstrap"] = deepcopy(base_plan_bootstrap)
    early_plan["base_layout_conditions"] = {
        key: deepcopy(early_plan[key])
        for key in (
            "condition_profile",
            "planned_conditions",
            "planning_result",
            "activity_plan",
        )
        if key in early_plan
    }
    materialization_base = _state_materialization_base(context, base_plan_path)
    context["state_base_episode_root"] = str(materialization_base)
    return _state_plan_result(
        item, context, unit_root, early_plan, base_root=materialization_base,
    )


def _run_state_late_plan_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del output_root
    done = _state_done(results)
    v0_row = _state_result(done, "v0")
    v0_plan_path = _state_plan_path(v0_row, unit_id="v0")
    v0_plan = native._load(v0_plan_path)
    v0_capture = _state_capture_output(
        _state_result(done, "v0_capture"), unit_id="v0_capture"
    )
    v0_request = _state_request_from_result(v0_row, unit_id="v0")
    base_hint = (v0_row.get("outputs") or {}).get("base_episode_root")
    if isinstance(base_hint, str) and base_hint.strip():
        context["state_base_episode_root"] = str(
            Path(base_hint).expanduser().resolve()
        )
    v1_request = _state_member_request(context, item)
    motion_request = deepcopy(v0_request)
    if isinstance(v1_request.get("binding_motion"), Mapping):
        motion_request["binding_motion"] = deepcopy(
            v1_request["binding_motion"]
        )
    facts_by_assignment = _state_early_facts(done)
    window = measured_motion_window_from_facts(
        facts_by_assignment,
        clock=v0_plan["clock"],
        binding_motion=motion_request.get("binding_motion") or {},
        reserve_tail_s=(motion_request.get("profile") or {}).get(
            "reserve_tail_s"
        ),
    )
    actor_ids = [
        str(actor.get("actor_id"))
        for actor in (v0_plan.get("visual_plan") or {}).get("actors", [])
        if isinstance(actor, Mapping) and actor.get("actor_id")
    ]
    initial_bearings = _native_visual_endpoint_bearings(
        v0_capture,
        facts_by_assignment["a0"],
        actor_ids,
        owner="v0",
    )
    repair = context.get("late_plan_repair")
    locked_paths = None
    bearing_drift = None
    repair_evidence = None
    if repair is not None:
        locked_paths, bearing_drift, repair_evidence = _late_plan_repair_inputs(
            v0_plan, v0_plan["visual_plan"]["camera"], repair
        )
    states, route = sample_late_paths(
        v0_plan,
        motion_request,
        window["first_motion_frame"],
        window["last_motion_frame"],
        initial_visual_bearings=initial_bearings,
        locked_paths=locked_paths,
        bearing_drift=bearing_drift,
    )
    v1_plan = _state_bind_plan_request(v0_plan, v1_request)
    v1_plan["episode_id"] = f"{context['group_id']}_v1"
    v1_plan["request"] = deepcopy(v1_request)
    v1_plan["request"]["episode_id"] = v1_plan["episode_id"]
    v1_plan["state_recipe"] = {
        "task_family": STATE_TASK_FAMILY,
        "variant": "v1",
        "motion": "declared_after_wet_tail",
        "measured_motion_window": deepcopy(window),
        "route": deepcopy(route),
        "base_plan_source": str(v0_plan_path),
        "original_request_preserved": True,
    }
    if repair_evidence is not None:
        v1_plan["state_recipe"]["late_plan_repair"] = repair_evidence
    v1_plan["state_recipe"]["answer_margin_deg"] = float(
        _settings(motion_request)["answer_margin_deg"]
    )
    for frame in (v1_plan.get("visual_plan") or {}).get("frames", []):
        if not isinstance(frame, Mapping):
            continue
        frame_index = int(frame.get("frame_index", 0))
        states_by_actor = [
            deepcopy(states[actor_id][frame_index])
            for actor_id in actor_ids
            if actor_id in states and frame_index < len(states[actor_id])
        ]
        frame["actor_states"] = states_by_actor
    _state_validate_static_base(v1_plan, v1_request)
    native._write(unit_root / "measured_motion_window.json", window)
    base_root = _state_materialization_base(context, v0_plan_path)
    output = unit_root / "episode"
    materialized = materialize_motion_visual(base_root, output, v1_plan)
    plan_path = materialized / "plan/episode_plan.json"
    request_path = materialized / "request.json"
    return native._stage_result(
        item,
        status="pass",
        facts={
            "episode_plan_path": str(plan_path),
            "renderer": native._plan_renderer(v1_plan),
            "clock": deepcopy(v1_plan.get("clock")),
            "first_motion_frame": window["first_motion_frame"],
            "last_motion_frame": window["last_motion_frame"],
            "measured_wet_end_s": window["measured_wet_end_s"],
        },
        outputs={
            "plan_root": str(materialized),
            "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "source_asset_ids": _state_plan_asset_order(v1_plan),
            "measured_motion_window": window,
            "route": route,
            "state_recipe": STATE_TASK_FAMILY,
        },
    )


def _state_capture_item_with_materialized_plan(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Give the capture a complete plan directory, repairing a thin one first.

    A plan unit that already passed keeps its reported directory read-only, so a
    resume cannot simply re-plan without re-sampling the variant. Instead the
    variant plan is materialized again under this capture attempt and the
    upstream input is pointed at that fresh tree. Pure CPU; the renderer has not
    started at this point.
    """
    unit_spec = _unit_row_for_state(context, str(item["unit_id"]))
    plan_units = [str(name) for name in unit_spec.get("depends_on_units") or ()]
    if len(plan_units) != 1:
        return dict(item), None
    plan_unit = plan_units[0]
    upstream = ((item.get("inputs") or {}).get(plan_unit)) or {}
    plan_value = (upstream.get("facts") or {}).get("episode_plan_path")
    if not isinstance(plan_value, str) or not Path(plan_value).is_file():
        return dict(item), None
    status = state_variant_materialization_status(plan_value)
    if not status["missing"]:
        return dict(item), None
    repaired = repair_state_variant_materialization(
        plan_value,
        unit_root / "repaired_plan/episode",
        context=context,
    )
    patched_item = deepcopy(dict(item))
    inputs = dict(patched_item.get("inputs") or {})
    row = deepcopy(dict(inputs.get(plan_unit) or {}))
    facts = dict(row.get("facts") or {})
    facts["episode_plan_path"] = repaired["episode_plan_path"]
    facts["variant_materialization_repaired_from"] = plan_value
    row["facts"] = facts
    outputs = dict(row.get("outputs") or {})
    outputs["episode_plan"] = repaired["episode_plan_path"]
    outputs["plan_root"] = repaired["plan_root"]
    if not isinstance(outputs.get("request_path"), str) or not Path(
        str(outputs.get("request_path"))
    ).is_file():
        outputs["request_path"] = repaired["request_path"]
    row["outputs"] = outputs
    inputs[plan_unit] = row
    patched_item["inputs"] = inputs
    return patched_item, repaired


def _unit_row_for_state(
    context: Mapping[str, Any], unit_id: str
) -> dict[str, Any]:
    for row in (context.get("group_spec") or {}).get("stage_units") or ():
        if str(row.get("unit_id")) == unit_id:
            return dict(row)
    raise native.BindingNativeError(
        f"{unit_id} is not a unit of {context.get('group_id')}"
    )


def _run_state_visual_capture_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(item.get("unit_id") or "")
    if unit_id == "v0_capture":
        retained = _state_retained_root(context, unit_id)
        if retained is not None:
            done = _state_done(results)
            plan_path = _state_plan_path(
                _state_result(done, "v0"), unit_id="v0"
            )
            entry = context["contract"]["visual_units"][unit_id]
            adopted = native._retained_visual_entry(
                retained,
                expected_assets=entry["source_asset_ids"],
                label=unit_id,
            )
            if Path(adopted["plan"]).resolve() != plan_path.resolve():
                native.compare_group_visual_plans(
                    plan_path,
                    adopted["plan"],
                    contract=context["contract"],
                )
            frame_count = native._captured_frame_count(
                Path(adopted["capture"]), native._load(plan_path)
            )
            return native._stage_result(
                item,
                status="pass",
                facts={
                    "capture_receipt_path": str(
                        Path(adopted["capture"]) / "research_receipt.json"
                    ),
                    "captured_frame_count": frame_count,
                },
                outputs={
                    "capture": str(Path(adopted["capture"]).resolve()),
                    "capture_root": str(Path(adopted["output"]).resolve()),
                    "episode_plan": str(Path(adopted["plan"]).resolve()),
                    "request_path": str(
                        (retained / "request.json").resolve()
                    ),
                    "neutral_readback": str(
                        Path(adopted["neutral_readback"]).resolve()
                    ),
                    "frame_readbacks": adopted.get("frame_readbacks"),
                    "visual_video": adopted.get("visual_video"),
                    "source_asset_ids": list(entry["source_asset_ids"]),
                    "reused_retained_visual_root": str(retained),
                    "native_visual_worlds_created": 0,
                },
            )
    capture_item, repaired = _state_capture_item_with_materialized_plan(
        item, context, unit_root,
    )
    result = native.run_visual_capture_unit(
        capture_item,
        context,
        unit_root,
        output_root=output_root,
        results=results,
        lease=lease,
    )
    if repaired is not None:
        outputs = dict(result.get("outputs") or {})
        outputs["variant_materialization_repair"] = repaired
        result["outputs"] = outputs
    if unit_id != "v1_capture":
        return result
    done = _state_done(list(results) + [result])
    v0_capture = _state_capture_output(
        _state_result(done, "v0_capture"), unit_id="v0_capture"
    )
    v1_capture = _state_capture_output(result, unit_id=unit_id)
    facts_by_assignment = _state_early_facts(done)
    captured = {"v0": v0_capture, "v1": v1_capture}
    history = native_state_visual_history_preflight(
        captured, facts_by_assignment
    )
    history_path = native._write(
        unit_root / "native_visual_history_preflight.json", history
    )
    if history["status"] != "pass":
        raise native.BindingNativeError(
            "actual visual identity history is interrupted before the state query"
        )
    actor_ids = [
        str(actor.get("actor_id"))
        for actor in (native._load(
            _state_plan_path(_state_result(done, "v0"), unit_id="v0")
        ).get("visual_plan") or {}).get("actors", [])
        if isinstance(actor, Mapping) and actor.get("actor_id")
    ]
    angle_tolerance = (
        _state_member_request(context, item)
        .get("binding_motion", {})
        .get("angle_tolerance_deg")
    )
    if (
        isinstance(angle_tolerance, bool)
        or not isinstance(angle_tolerance, (int, float))
        or not math.isfinite(float(angle_tolerance))
        or float(angle_tolerance) < 0
    ):
        raise native.BindingNativeError(
            "cross_time_state v1 capture needs a finite nonnegative "
            "binding_motion.angle_tolerance_deg"
        )
    angle = native_state_visual_angle_preflight(
        captured,
        facts_by_assignment["a0"],
        actor_ids,
        float(angle_tolerance),
    )
    angle_path = native._write(
        unit_root / "native_visual_angle_preflight.json", angle
    )
    if angle["status"] != "pass":
        raise native.BindingNativeError(
            "actual native visual answers are not separated under the declared tolerance"
        )
    result["facts"] = {
        **(result.get("facts") or {}),
        "state_visual_history_preflight": str(history_path),
        "state_visual_angle_preflight": str(angle_path),
    }
    result["outputs"] = {
        **(result.get("outputs") or {}),
        "state_visual_history_preflight": str(history_path),
        "state_visual_angle_preflight": str(angle_path),
    }
    return result


def _run_state_audio_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del lease
    unit_id = str(item.get("unit_id") or "")
    unit_spec = native._unit_row(context, unit_id)
    visual_unit_id = str(unit_spec.get("visual_unit_id") or "")
    if not visual_unit_id:
        raise native.BindingNativeError(
            f"{unit_id} declares no visual unit"
        )
    done = _state_done(results)
    capture_row = _state_result(done, visual_unit_id)
    capture = _state_capture_output(capture_row, unit_id=visual_unit_id)
    plan_path = _state_plan_path(
        capture_row, unit_id=visual_unit_id
    )
    request_path = (capture_row.get("outputs") or {}).get("request_path")
    if not isinstance(request_path, str) or not Path(request_path).is_file():
        raise native.BindingNativeError(
            f"{visual_unit_id} capture has no saved request"
        )
    plan = native._load(plan_path)
    capture_request = native._load(Path(request_path))
    member_request_id, member_request = native._member_request_for_audio_unit(
        context, item, unit_spec
    )
    request, audio_view_fields = native._apply_member_audio_view_fields(
        capture_request, member_request, label=unit_id
    )
    visual_id = visual_unit_id.split("_", 1)[0]
    assignment = unit_id.rsplit("_", 1)[-1]
    result = _render_state_column(
        plan,
        capture,
        unit_root,
        visual_id,
        assignment,
        request=request,
        shared_visual_root=_state_shared_visual_root(context, output_root),
    )
    visual_video = result.get("visual_video") or capture.get("visual_video")
    if (
        not isinstance(visual_video, str)
        or not Path(visual_video).expanduser().resolve().is_file()
    ):
        raise native.BindingNativeError(
            f"{unit_id} audio finalization published no readable visual_video"
        )
    facts_path = Path(result["facts"]).expanduser().resolve()
    facts = native._load(facts_path)
    tails = (facts.get("audio") or {}).get("wet_tail_intervals")
    if not isinstance(tails, list) or not tails:
        raise native.BindingNativeError(
            f"{unit_id} audio facts lack measured wet tails"
        )
    if visual_id == "v1":
        late = _state_result(done, "v1")
        late_facts = late.get("facts") or {}
        first_frame = late_facts.get("first_motion_frame")
        if not isinstance(first_frame, int):
            raise native.BindingNativeError(
                "v1 audio needs late_plan.first_motion_frame"
            )
        boundary_s = first_frame / float(plan["clock"]["frame_rate_hz"])
        if any(float(row["end_s"]) >= boundary_s for row in tails):
            raise native.BindingNativeError(
                f"{unit_id} measured wet tail reaches the actor-motion window"
            )
    outputs = {
        "variant_root": result["variant_root"],
        "audio": result.get("audio"),
        "questions": result.get("questions"),
        "visual_video": visual_video,
        "capture": capture["capture"],
        "episode_plan": str(plan_path),
        "request_path": request_path,
        "neutral_readback": capture["neutral_readback"],
        "assignment_plan_path": result["assignment_plan_path"],
        "assignment_request_path": result["assignment_request_path"],
        "assignment_column": assignment,
        "visual_unit_id": visual_unit_id,
        "member_request_id": member_request_id,
        "audio_view_fields": audio_view_fields,
        "capture_request_path": request_path,
        "shared_visual_root": result.get("shared_visual_root"),
        "audio_report": result.get("audio_report"),
        "declared_audio_delivery": result.get("declared_audio_delivery"),
        "delivered_audio_layouts": deepcopy(
            result.get("delivered_audio_layouts") or {}
        ),
        "ancillary_audio_outputs": deepcopy(
            (result.get("result") or {}).get("ancillary_audio_outputs")
        ),
    }
    facts_out = {
        "facts_path": str(facts_path),
        "audio_report_path": str(Path(result["audio_report"]).resolve()),
        "wet_tail_intervals": deepcopy(tails),
    }
    delivered_layouts = result.get("delivered_audio_layouts") or {}
    if delivered_layouts.get("status") != "pass":
        return native._stage_result(
            item,
            status="blocked",
            facts=facts_out,
            outputs=outputs,
            reason=delivered_layouts.get("reason")
            or "declared audio layouts were not delivered",
        )
    return native._stage_result(
        item,
        status="pass",
        facts=facts_out,
        outputs=outputs,
    )


def _run_state_assembly_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del lease
    done = _state_done(results)
    contract = context["contract"]
    visual_units = contract["visual_units"]
    visual: dict[str, dict[str, Any]] = {}
    plans: dict[str, str] = {}
    readbacks: dict[str, str] = {}
    requests: dict[str, dict[str, Any]] = {}
    for unit_id in sorted(visual_units):
        row = _state_result(done, unit_id)
        outputs = row.get("outputs") or {}
        plan_path = _state_plan_path(row, unit_id=unit_id)
        capture = _state_capture_output(row, unit_id=unit_id)
        visual[unit_id] = {
            "capture": capture["capture"],
            "visual_video": capture.get("visual_video"),
            "neutral_readback": capture["neutral_readback"],
        }
        plans[unit_id] = str(plan_path)
        readbacks[unit_id] = capture["neutral_readback"]
        requests[unit_id] = _state_request_from_result(row, unit_id=unit_id)
    visual_ids = sorted(visual_units)
    if len(visual_ids) != 2:
        raise native.BindingNativeError(
            f"cross_time_state assembly needs two visual units, got {visual_ids}"
        )
    for left_id, right_id in [(visual_ids[0], visual_ids[1])]:
        native.compare_group_visual_plans(
            plans[left_id], plans[right_id], contract=contract
        )
        native.compare_group_native_visuals(
            {"neutral_readback": readbacks[left_id]},
            {"neutral_readback": readbacks[right_id]},
            contract=contract,
            left_unit_id=left_id,
            right_unit_id=right_id,
        )
    member_units: list[tuple[str, str, str]] = []
    variants: dict[str, dict[str, Any]] = {}
    for _column, unit_ids in sorted(contract["audio_columns"].items()):
        for audio_unit_id in unit_ids:
            row = _state_result(done, audio_unit_id)
            facts = row.get("facts") or {}
            outputs = row.get("outputs") or {}
            ancillary = deepcopy(outputs.get("ancillary_audio_outputs") or [])
            delivered_layouts = deepcopy(
                outputs.get("delivered_audio_layouts")
                or facts.get("delivered_audio_layouts")
                or {}
            )
            if delivered_layouts.get("status") != "pass":
                raise native.BindingNativeError(
                    f"{audio_unit_id} audio layout delivery is not valid: "
                    f"{delivered_layouts}"
                )
            visual_unit_id = str(outputs["visual_unit_id"])
            audio_delivery = {
                "audio_report_path": str(
                    Path(facts["audio_report_path"]).expanduser().resolve()
                ),
                "assignment_request_path": str(
                    Path(
                        outputs.get("assignment_request_path")
                        or outputs.get("request_path")
                    ).expanduser().resolve()
                ),
                "primary_audio_path": str(
                    Path(outputs["audio"]).expanduser().resolve()
                ),
                "declared_audio_delivery": deepcopy(
                    outputs.get("declared_audio_delivery") or {}
                ),
                "delivered_audio_layouts": delivered_layouts,
                "ancillary_audio_outputs": ancillary,
            }
            variants[audio_unit_id] = {
                "facts": str(facts["facts_path"]),
                "audio": str(outputs["audio"]),
                "visual_video": outputs.get("visual_video"),
                "audio_report": str(facts["audio_report_path"]),
                "visual_capture_root": str(outputs["capture"]),
                "audio_delivery": audio_delivery,
            }
            member_units.append(
                (audio_unit_id, visual_unit_id, audio_unit_id)
            )
    anchor_id = visual_ids[0]
    member_ids = [
        str(value) for value in contract.get("member_request_ids") or ()
    ]
    if not member_ids:
        raise native.BindingNativeError(
            "cross_time_state assembly has no original member request IDs"
        )
    original_requests = context.get("member_requests")
    if not isinstance(original_requests, Mapping):
        raise native.BindingNativeError(
            "cross_time_state assembly has no original member requests"
        )
    anchor_request = original_requests.get(member_ids[0])
    if not isinstance(anchor_request, Mapping):
        raise native.BindingNativeError(
            f"cross_time_state assembly cannot read original request {member_ids[0]}"
        )
    anchor_request = deepcopy(dict(anchor_request))
    anchor_plan = native._load(Path(plans[anchor_id]))
    tolerance = (
        _state_member_request(context, item)
        .get("binding_motion", {})
        .get("angle_tolerance_deg")
    )
    if tolerance is None:
        raise native.BindingNativeError(
            "cross_time_state assembly needs binding_motion.angle_tolerance_deg"
        )
    profile = {
        "task_family": STATE_TASK_FAMILY,
        "source_count": len(contract["world_population"]),
        "camera": deepcopy(
            (anchor_plan.get("visual_plan") or {}).get("camera", {})
        ),
        "camera_motion": "static",
        "audio": {
            "rir_stride": anchor_request.get("rir_stride"),
            "post_assembly_convolution_gain": anchor_request.get(
                "post_assembly_convolution_gain"
            ),
            "declared_delivery": native.declared_audio_delivery(anchor_request),
        },
        "reserve_tail_s": (anchor_request.get("profile") or {}).get(
            "reserve_tail_s"
        ),
        "sound_pool": anchor_request.get("sound_pool"),
        "motion_timing": "after_wet_tail",
    }
    room_family = native.room_family_from_plan(anchor_plan)
    spec = native._group_spec(
        context["group_id"],
        context["world_id"],
        room_family,
        context["room_id"],
        visual,
        variants,
        request=anchor_request,
        profile=profile,
        member_units=member_units,
        split=context.get("split", "pilot"),
        task_family=STATE_TASK_FAMILY,
    )
    spec["request"] = deepcopy(anchor_request)
    spec["groups"][0].update(
        task_family=STATE_TASK_FAMILY,
        query={"event_number": 1, "query_anchor": "clip_end"},
        angle_tolerance_deg=float(tolerance),
    )
    spec_path = native._write(unit_root / "group_spec.json", spec)
    from avengine.qa.binding_groups import assemble_binding_dataset
    assembled = assemble_binding_dataset(
        spec,
        input_base=Path(output_root).expanduser().resolve(),
        output=unit_root / "assembled",
        seed=f"{context['group_id']}-assembly",
        verify_media=True,
    )
    facts = {
        "group_spec_path": str(spec_path),
        "assembled_path": str((unit_root / "assembled").resolve()),
        "validation": {
            "status": "pass",
            "query": {"event_number": 1, "query_anchor": "clip_end"},
            "controlled_world_checks": [
                {
                    "check": "planned_world_and_native_readback",
                    "status": "pass",
                }
            ],
            "group_count": assembled.get("group_count"),
            "world_count": assembled.get("world_count"),
            "sample_count": assembled.get("sample_count"),
            "media_validation": assembled.get("validation"),
            "public_payload_check": deepcopy(
                assembled.get("public_payload_check")
            ),
        },
    }
    return native._stage_result(
        item,
        status="pass",
        facts=facts,
        outputs={
            "assembled_root": str((unit_root / "assembled").resolve()),
            "group_spec": str(spec_path),
            "member_sample_ids": {
                str(row.get("member_id")): row.get("sample_id")
                for group in assembled.get("groups", [])
                for row in group.get("members", [])
            },
            "member_units": [list(row) for row in member_units],
            "world_id": context["world_id"],
            "world_id_source": context["world_id_source"],
            "native_visual_worlds_created": 0,
        },
    )


def group_stage_context(
    *args: Any,
    state_base_episode_root: str | Path | None = None,
    shared_visual_root: str | Path | None = None,
    state_geometry_clip_budget_s: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reuse native group context and attach state-only source hints."""
    context = native.group_stage_context(*args, **kwargs)
    if state_base_episode_root is not None:
        context["state_base_episode_root"] = str(
            Path(state_base_episode_root).expanduser().resolve()
        )
    if shared_visual_root is not None:
        context["shared_visual_root"] = str(
            Path(shared_visual_root).expanduser().resolve()
        )
    if state_geometry_clip_budget_s is not None:
        context["state_geometry_clip_budget_s"] = float(
            state_geometry_clip_budget_s
        )
    return context


def load_group_stage_results(
    output_root: str | Path, group_id: str
) -> list[dict[str, Any]]:
    """Reuse native result loading without changing its save format."""
    return native.load_group_stage_results(output_root, group_id)


STATE_STAGE_RUNNERS = {
    "visual_plan": _run_state_visual_plan_unit,
    "visual_capture": _run_state_visual_capture_unit,
    "audio": _run_state_audio_unit,
    "assembly": _run_state_assembly_unit,
}
STAGE_RUNNERS = STATE_STAGE_RUNNERS


def run_group_stage_work_item(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run state units through native's fresh/resume/save dispatcher."""
    return native.run_group_stage_work_item(
        item,
        context,
        output_root=output_root,
        results=results,
        lease=lease,
        resume=resume,
        stage_runners=STATE_STAGE_RUNNERS,
    )


def _recipe_audio_attempt_root(
    previous_attempt_root: str | Path,
    item: Mapping[str, Any],
    lineage: Mapping[str, Any] | None = None,
) -> Path:
    """Find the materialized audio root inside one of this recipe's attempts.

    The generic recovery in binding_group_native accepts ``episode`` or the
    attempt root itself. This recipe materializes its audio under
    ``variants/<unit_id>`` instead, so the unit id resolves the member before
    the shared implementation reads it. Nothing is copied or written here.
    """
    previous = Path(previous_attempt_root).expanduser().resolve()
    unit_ids = []
    for source in (item, lineage or {}):
        if isinstance(source, Mapping):
            for key in ("unit_id", "member_request_id", "request_id"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    unit_ids.append(value.strip())
    candidates = [previous / "episode"]
    candidates.extend(previous / "variants" / unit_id for unit_id in unit_ids)
    candidates.append(previous)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "request.json").is_file():
            return candidate
    raise native.BindingNativeError(
        "previous audio attempt has no materialized audio root; looked at "
        + ", ".join(str(path) for path in candidates)
    )


def recover_rendered_audio_attempt(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
    previous_attempt_root: str | Path,
    lineage: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU-finalize one interrupted audio attempt of this recipe.

    The ordinary runner looks this name up on the recipe module, so the module
    has to carry it or an interrupted member of this family silently gets no
    recovery at all. Only the attempt layout differs from the shared
    implementation, which then does the finalize without launching RLR.
    """
    return native.recover_rendered_audio_attempt(
        item,
        context,
        output_root=output_root,
        previous_attempt_root=_recipe_audio_attempt_root(
            previous_attempt_root, item, lineage,
        ),
        lineage=lineage,
        results=results,
        lease=lease,
    )


__all__ = [
    "STATE_TASK_FAMILY",
    "recover_rendered_audio_attempt",
    "state_variant_execution_requirements",
    "state_variant_materialization_status",
    "repair_state_variant_materialization",
    "STATE_STAGE_RUNNERS",
    "STAGE_RUNNERS",
    "group_stage_context",
    "load_group_stage_results",
    "run_group_stage_work_item",
    "select_early_audio",
    "sample_late_paths",
    "materialize_motion_visual",
    "_render_state_column",
    "native_common_endpoint_paths",
    "native_state_visual_history_preflight",
    "native_state_visual_angle_preflight",
    "measured_motion_window_from_facts",
    "motion_window_from_stage_results",
    "prepare_fixed_camera_state_group",
]
