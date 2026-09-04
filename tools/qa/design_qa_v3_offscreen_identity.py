#!/usr/bin/env python3
"""Produce CPU research candidates for the off-screen-to-on-screen identity task.

Two selected human voices each speak one complete train utterance while their
planned route is outside the camera cone, then repeat that same utterance after
entering the cone. Gate A rebinds both occurrences of each voice to the other
audio endpoint while visual identity, content and timing stay fixed. Gate B
swaps visual appearance bindings while keeping the main audio program.
The route and visibility labels are geometric plans; pixel visibility and
recognisability remain pending native media review.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import scene_sampler as SS  # noqa: E402
from audio_profiles import (  # noqa: E402
    AudioProfileError,
    AudioProfileSearchExhausted,
)
from avengine.assets.sound_pool import (  # noqa: E402
    SoundPoolError,
    SpeechSelectionSearchExhausted,
    clip_source_from_params,
)
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    author_current_n_actor_visual_timeline,
)
from build_qa_v3_n_actor_canary import (  # noqa: E402
    _actor_entry,
    build_endpoint_registry,
    seed_uint64,
)
from build_qa_v3_programs import (  # noqa: E402
    build_program,
    program_request_fields,
    validate_m6_audio_program,
)
from design_qa_v3_scene_batch import resolve_scene_render_context  # noqa: E402
from qa_v3_request import (  # noqa: E402
    answer_forms_from_params,
    read_qa_params,
    write_requested_questions,
)


SCHEMA = "avengine_qa_v3_offscreen_identity_candidate_v1"
BATCH_SCHEMA = "avengine_qa_v3_offscreen_identity_batch_v1"
GATEB_SCHEMA = "avengine_qa_v3_f2_gateb_intervention_v1"


class OffscreenIdentityError(ValueError):
    """The explicit F2 profile cannot produce a truthful candidate."""


class OffscreenIdentitySearchExhausted(OffscreenIdentityError):
    """The finite route search ended without a geometry candidate."""

    def __init__(self, message: str, *, attempts: int):
        super().__init__(message)
        self.attempts = int(attempts)


def _read(path: str | Path) -> Any:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise OffscreenIdentityError(f"missing regular JSON file: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OffscreenIdentityError(f"cannot read JSON {source}: {exc}") from exc


def _write(path: str | Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise OffscreenIdentityError(f"refusing to replace output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _fresh_directory(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise OffscreenIdentityError(f"refusing to replace output directory: {destination}")
    destination.mkdir(parents=True)
    return destination


def _profile(profile_path: str | Path) -> dict[str, Any]:
    value = _read(profile_path)
    if not isinstance(value, Mapping) or value.get("schema") != "avengine_qa_v3_offscreen_identity_profile_v1":
        raise OffscreenIdentityError("profile has the wrong schema")
    result = dict(value)
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise OffscreenIdentityError("profile id must be non-empty")
    if result.get("actor_count") != 2:
        raise OffscreenIdentityError("the current offscreen identity profile requires actor_count=2")
    colors = result.get("appearance_colors")
    if (
        not isinstance(colors, list)
        or len(colors) != 2
        or any(not isinstance(color, str) or not color.strip() for color in colors)
        or len(set(colors)) != 2
    ):
        raise OffscreenIdentityError("profile appearance_colors must contain two distinct colors")
    for key in ("early_window_seconds", "late_window_seconds"):
        window = result.get(key)
        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in window
            )
            or float(window[1]) <= float(window[0])
        ):
            raise OffscreenIdentityError(f"profile {key} must be an increasing finite pair")
    for key in ("gap_seconds", "tail_seconds", "min_pairwise_azimuth_deg", "visibility_margin_deg"):
        value = result.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise OffscreenIdentityError(f"profile {key} must be finite and non-negative")
    attempts = result.get("max_search_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise OffscreenIdentityError("profile max_search_attempts must be positive")
    visibility_rule = result.get("visibility_rule")
    if (
        not isinstance(visibility_rule, Mapping)
        or visibility_rule.get("early") != "out_of_view"
        or visibility_rule.get("late") != "in_view"
    ):
        raise OffscreenIdentityError(
            "profile visibility_rule must declare early=out_of_view and late=in_view"
        )
    return result


def _window_samples(
    profile: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        clip_seconds = float(params["CLIP_SECONDS"])
        rate = int(params["SAMPLE_RATE_HZ"])
        frame_count = int(params["FRAME_COUNT"])
        fps = float(params["VIDEO_FPS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OffscreenIdentityError(
            "params need CLIP_SECONDS, SAMPLE_RATE_HZ, FRAME_COUNT, and VIDEO_FPS"
        ) from exc
    if (
        not math.isfinite(clip_seconds)
        or clip_seconds <= 0.0
        or rate <= 0
        or frame_count <= 0
        or not math.isfinite(fps)
        or fps <= 0.0
    ):
        raise OffscreenIdentityError("params clock values must be positive and finite")
    gap_seconds = float(profile["gap_seconds"])
    tail_seconds = float(profile["tail_seconds"])
    windows: dict[str, tuple[float, float]] = {}
    for name in ("early", "late"):
        raw = profile[f"{name}_window_seconds"]
        lo, hi = float(raw[0]), float(raw[1])
        if lo < 0.0 or hi > clip_seconds:
            raise OffscreenIdentityError(
                f"{name} speech window {raw} lies outside CLIP_SECONDS={clip_seconds}"
            )
        windows[name] = (lo, hi)
    if windows["early"][1] > windows["late"][0]:
        raise OffscreenIdentityError("early and late speech windows overlap")
    if windows["late"][1] + tail_seconds > clip_seconds + 1e-9:
        raise OffscreenIdentityError("late window leaves less than the declared tail")
    if gap_seconds < 0.0:
        raise OffscreenIdentityError("gap_seconds must be non-negative")
    return {
        "clip_seconds": clip_seconds,
        "sample_rate_hz": rate,
        "frame_count": frame_count,
        "frame_rate_hz": fps,
        "gap_seconds": gap_seconds,
        "gap_samples": int(round(gap_seconds * rate)),
        "tail_seconds": tail_seconds,
        "tail_samples": int(math.ceil(tail_seconds * rate)),
        "windows_seconds": windows,
        "windows_samples": {
            name: (int(round(lo * rate)), int(round(hi * rate)))
            for name, (lo, hi) in windows.items()
        },
    }


def _select_speech_pair(
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    rng: np.random.Generator,
    max_pair_duration_samples: int,
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    if str(params.get("SOUND_SOURCE_MODE")) != "event_pool":
        raise OffscreenIdentityError(
            "offscreen identity requires SOUND_SOURCE_MODE=event_pool"
        )
    try:
        source = clip_source_from_params(params, rng, pair_kind="human")
    except (SoundPoolError, OSError, TypeError, ValueError) as exc:
        raise OffscreenIdentityError(f"cannot load the speech event pool: {exc}") from exc
    if source is None:
        raise OffscreenIdentityError("event_pool did not produce a speech source")
    details: dict[str, Any] = {}
    try:
        clips = source.select_distinct_speech_clips(
            2,
            split=str(profile.get("speech_split", params.get("SPEECH_SPLIT", "train"))),
            max_total_duration_samples=max_pair_duration_samples,
            selection_attempts=int(profile.get("speech_selection_attempts", 64)),
            selection_candidate_window=profile.get("speech_selection_candidate_window", 8),
            selection_strategy=str(profile.get("speech_selection_strategy", "bounded_random")),
            selection_fallback_strategy=profile.get("speech_selection_fallback_strategy"),
            distinct_transcripts=bool(profile.get("distinct_transcripts", True)),
            selection_details=details,
        )
    except SpeechSelectionSearchExhausted as exc:
        raise OffscreenIdentityError(
            f"complete speech pair did not fit the available window budget: {exc}"
        ) from exc
    except SoundPoolError as exc:
        raise OffscreenIdentityError(f"complete speech pair unavailable: {exc}") from exc
    if len(clips) != 2:
        raise OffscreenIdentityError(f"speech pool returned {len(clips)} clips, need two")
    if len({clip.speaker_id for clip in clips}) != 2:
        raise OffscreenIdentityError("speech pair does not have two distinct speakers")
    if len({clip.utterance_id for clip in clips}) != 2:
        raise OffscreenIdentityError("speech pair does not have two distinct utterances")
    if len({" ".join(str(clip.transcript).split()).casefold() for clip in clips}) != 2:
        raise OffscreenIdentityError("speech pair does not have two distinct transcripts")
    rate = int(params["SAMPLE_RATE_HZ"])
    for clip in clips:
        if int(clip.sample_rate_hz) != rate:
            raise OffscreenIdentityError(
                f"speech clip {clip.sound_asset_id} rate {clip.sample_rate_hz} != {rate}"
            )
        if int(clip.source_end_sample_exclusive) - int(clip.source_start_sample) != int(clip.duration_samples):
            raise OffscreenIdentityError(
                f"speech clip {clip.sound_asset_id} does not expose its complete source window"
            )
        if any(not getattr(clip, key, None) for key in ("speaker_id", "utterance_id", "transcript", "split")):
            raise OffscreenIdentityError(
                f"speech clip {clip.sound_asset_id} lacks identity metadata"
            )
    order = [int(index) for index in rng.permutation(2)]
    ordered = tuple(clips[index] for index in order)
    details.update(
        {
            "max_pair_duration_samples": int(max_pair_duration_samples),
            "max_pair_duration_seconds": max_pair_duration_samples / rate,
            "assignment_permutation": order,
            "selected": [
                {
                    "sound_asset_id": clip.sound_asset_id,
                    "speaker_id": clip.speaker_id,
                    "utterance_id": clip.utterance_id,
                    "transcript": clip.transcript,
                    "split": clip.split,
                    "duration_samples": int(clip.duration_samples),
                    "source_start_sample": int(clip.source_start_sample),
                    "source_end_sample_exclusive": int(clip.source_end_sample_exclusive),
                }
                for clip in ordered
            ],
        }
    )
    return ordered, details


def _place_pair(
    clips: Sequence[Any],
    *,
    window_samples: tuple[int, int],
    gap_samples: int,
    rng: np.random.Generator,
    phase: str,
) -> list[dict[str, Any]]:
    if len(clips) != 2:
        raise OffscreenIdentityError("exactly two clips are required per speech phase")
    lo, hi = window_samples
    durations = [int(clip.duration_samples) for clip in clips]
    required = sum(durations) + gap_samples
    if required > hi - lo:
        raise OffscreenIdentityError(
            f"two complete {phase} utterances need {required} samples but its window has {hi - lo}; no clipping is allowed"
        )
    start = int(rng.integers(lo, hi - required + 1))
    events: list[dict[str, Any]] = []
    cursor = start
    for index, clip in enumerate(clips):
        end = cursor + int(clip.duration_samples)
        events.append(
            {
                "phase": phase,
                "occurrence_index": index,
                "event_id": f"speech_{phase}_event_{index + 1}",
                "slot": f"source{index + 1}",
                "start_sample": cursor,
                "duration_samples": int(clip.duration_samples),
                "source_start_sample": int(clip.source_start_sample),
                "source_end_sample_exclusive": int(clip.source_end_sample_exclusive),
                "sound_asset_id": str(clip.sound_asset_id),
                "speaker_id": str(clip.speaker_id),
                "utterance_id": str(clip.utterance_id),
                "transcript": str(clip.transcript),
                "split": str(clip.split),
            }
        )
        cursor = end + gap_samples
    return events


def _speech_occurrences(
    params: Mapping[str, Any], profile: Mapping[str, Any], *, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clock = _window_samples(profile, params)
    rate = int(clock["sample_rate_hz"])
    early_lo, early_hi = clock["windows_samples"]["early"]
    late_lo, late_hi = clock["windows_samples"]["late"]
    pair_budget = min(early_hi - early_lo, late_hi - late_lo) - int(clock["gap_samples"])
    if pair_budget <= 0:
        raise OffscreenIdentityError(
            "speech windows are shorter than one two-utterance pair plus its gap"
        )
    rng = np.random.default_rng(seed_uint64(seed))
    clips, selection = _select_speech_pair(
        params,
        profile,
        rng=rng,
        max_pair_duration_samples=pair_budget,
    )
    early = _place_pair(
        clips,
        window_samples=(early_lo, early_hi),
        gap_samples=int(clock["gap_samples"]),
        rng=rng,
        phase="early_offscreen",
    )
    late = _place_pair(
        clips,
        window_samples=(late_lo, late_hi),
        gap_samples=int(clock["gap_samples"]),
        rng=rng,
        phase="late_onscreen_repeat",
    )
    events = early + late
    return events, {
        "clock": clock,
        "seed": seed,
        "selection": selection,
        "pair_budget_samples": pair_budget,
        "pair_budget_seconds": pair_budget / rate,
        "events": copy.deepcopy(events),
    }


def _event_frame_indices(
    event: Mapping[str, Any], *, sample_rate_hz: int, frame_rate_hz: float, frame_count: int
) -> list[int]:
    start = int(event["start_sample"])
    end = start + int(event["duration_samples"])
    first = max(0, min(frame_count - 1, math.floor(start * frame_rate_hz / sample_rate_hz)))
    last = max(
        first,
        min(frame_count - 1, math.ceil(end * frame_rate_hz / sample_rate_hz) - 1),
    )
    return list(range(first, last + 1))


def _norm_angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _check_route(
    route: Any,
    *,
    camera: Sequence[float],
    yaw: float,
    half_fov: float,
    events: Sequence[Mapping[str, Any]],
    phase: str,
    clock: Mapping[str, Any],
    min_distance_cm: float,
    slot: str | None = None,
    visibility: str | None = None,
    visibility_margin_deg: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    frames: list[int] = []
    for event in events:
        if event["phase"] == phase and (slot is None or event.get("slot") == slot):
            frames.extend(
                _event_frame_indices(
                    event,
                    sample_rate_hz=int(clock["sample_rate_hz"]),
                    frame_rate_hz=float(clock["frame_rate_hz"]),
                    frame_count=int(clock["frame_count"]),
                )
            )
    frames = sorted(set(frames))
    azimuths = [float(SS.relative_azimuth_deg(camera, yaw, route.at(frame))) for frame in frames]
    distances = [float(math.dist(camera, route.at(frame))) for frame in frames]
    if visibility == "out_of_view":
        visibility_ok = all(
            abs(value) > half_fov + visibility_margin_deg for value in azimuths
        )
    elif visibility == "in_view":
        visibility_ok = all(
            abs(value) <= half_fov - visibility_margin_deg for value in azimuths
        )
    else:
        raise OffscreenIdentityError(
            f"unknown planned visibility {visibility!r} for speech phase {phase!r}"
        )
    distance_ok = all(value >= min_distance_cm for value in distances)
    return visibility_ok and distance_ok, {
        "phase": phase,
        "frames": frames,
        "azimuth_min_deg": min((abs(value) for value in azimuths), default=None),
        "azimuth_max_deg": max((abs(value) for value in azimuths), default=None),
        "distance_min_cm": min(distances, default=None),
        "planned_visibility": visibility,
        "visibility_margin_deg": visibility_margin_deg,
        "visibility_basis": "camera_cone_geometry_only",
        "visibility_ok": visibility_ok,
        "distance_ok": distance_ok,
    }


def _find_offscreen_entry_plan(
    scene: Any,
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    clock: Mapping[str, Any],
    *,
    seed: str,
) -> dict[str, Any]:
    moving = [route for route in scene.routes if route.displacement_cm > 1.0e-6]
    actor_count = int(profile["actor_count"])
    if len(moving) < actor_count:
        raise OffscreenIdentityError(
            f"scene has {len(moving)} moving routes, need {actor_count}"
        )
    try:
        half_fov = SS.effective_half_fov(scene, params)
        min_distance_raw = profile.get("min_camera_distance_cm")
        if min_distance_raw is None:
            min_distance_raw = params["MIN_CAMERA_DISTANCE_CM"]
        min_distance_cm = float(min_distance_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise OffscreenIdentityError(f"invalid camera geometry parameters: {exc}") from exc
    if not math.isfinite(min_distance_cm) or min_distance_cm < 0.0:
        raise OffscreenIdentityError("minimum camera distance must be finite and non-negative")
    rng = np.random.default_rng(seed_uint64(f"route|{seed}"))
    attempts = int(profile["max_search_attempts"])
    min_pairwise = float(profile["min_pairwise_azimuth_deg"])
    visibility_margin = float(profile.get("visibility_margin_deg", 0.0))
    if not math.isfinite(visibility_margin) or not 0.0 <= visibility_margin < half_fov:
        raise OffscreenIdentityError(
            "visibility_margin_deg must be finite and smaller than the effective half-FOV"
        )
    visibility_rule = profile.get("visibility_rule") or {
        "early": "out_of_view",
        "late": "in_view",
    }
    for attempt in range(1, attempts + 1):
        camera = scene.camera_points[int(rng.integers(len(scene.camera_points)))]
        picked = SS.sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, None)
        if picked is None:
            continue
        yaw, clearance = picked
        indices = [int(index) for index in rng.permutation(len(moving))]
        chosen: list[Any] = []
        reports: list[dict[str, Any]] = []
        for index in indices:
            route = moving[index]
            source_slot_id = f"source{len(chosen) + 1}"
            early_ok, early_report = _check_route(
                route,
                camera=camera,
                yaw=float(yaw),
                half_fov=half_fov,
                events=events,
                phase="early_offscreen",
                clock=clock,
                min_distance_cm=min_distance_cm,
                slot=source_slot_id,
                visibility=str(visibility_rule["early"]),
                visibility_margin_deg=visibility_margin,
            )
            late_ok, late_report = _check_route(
                route,
                camera=camera,
                yaw=float(yaw),
                half_fov=half_fov,
                events=events,
                phase="late_onscreen_repeat",
                clock=clock,
                min_distance_cm=min_distance_cm,
                slot=source_slot_id,
                visibility=str(visibility_rule["late"]),
                visibility_margin_deg=visibility_margin,
            )
            if not (early_ok and late_ok):
                continue
            if any(
                any(
                    _norm_angle_delta(
                        SS.relative_azimuth_deg(camera, yaw, route.at(frame)),
                        SS.relative_azimuth_deg(camera, yaw, prior.at(frame)),
                    )
                    < min_pairwise
                    for frame in set(early_report["frames"] + late_report["frames"])
                )
                for prior in chosen
            ):
                continue
            chosen.append(route)
            reports.append(
                {
                    "source_slot_id": source_slot_id,
                    "route_id": route.route_id,
                    "route_source": route.source,
                    "route_provenance": route.source_record,
                    "early": early_report,
                    "late": late_report,
                }
            )
            if len(chosen) == actor_count:
                return {
                    "camera_xy": [float(value) for value in camera],
                    "camera_yaw_deg": float(yaw),
                    "camera_clearance": clearance,
                    "half_fov_deg": half_fov,
                    "min_camera_distance_cm": min_distance_cm,
                    "min_pairwise_azimuth_deg": min_pairwise,
                    "search_attempts": attempt,
                    "routes": chosen,
                    "route_reports": reports,
                    "line_of_sight_screened": bool(scene.line_of_sight_screened),
                    "physical_visibility_status": "pending_native_pixel_join",
                }
    raise OffscreenIdentitySearchExhausted(
        f"no {actor_count}-route offscreen-entry plan within {attempts} finite attempts",
        attempts=attempts,
    )


def _asset_registry() -> tuple[Path, dict[str, dict[str, Any]]]:
    path = REPO / "examples/runtime/source_asset_runtime_profiles.json"
    value = _read(path)
    if not isinstance(value, Mapping) or not isinstance(value.get("assets"), list):
        raise OffscreenIdentityError("source asset registry has no assets list")
    by_id = {
        str(item["asset_id"]): dict(item)
        for item in value["assets"]
        if isinstance(item, Mapping) and item.get("asset_id")
    }
    return path, by_id


def _human_assets(by_id: Mapping[str, Mapping[str, Any]], colors: Sequence[str]) -> list[str]:
    ids: list[str] = []
    for color in colors:
        matches = [
            str(asset_id)
            for asset_id, item in by_id.items()
            if item.get("entity_class") == "articulated_human"
            and (item.get("realized_attributes") or {}).get("top_color") == color
        ]
        if len(matches) != 1:
            raise OffscreenIdentityError(
                f"expected one registered controlled human for {color!r}, found {matches}"
            )
        ids.append(matches[0])
    return ids


def _selection(
    asset_ids: Sequence[str],
    *,
    by_id: Mapping[str, Mapping[str, Any]],
    snapshot_content: str | Path,
) -> dict[str, Any]:
    return {
        "schema": "avengine_n_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "offscreen identity CPU geometry candidate only",
        "actors": [
            _actor_entry(f"source{index}", asset_id, by_id, str(snapshot_content))
            for index, asset_id in enumerate(asset_ids, start=1)
        ],
    }


def _timeline_routes(scene: Any, plan: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    ground = float(scene.render_config["ground_z_ue_cm"])
    return {
        f"source{index}": [
            [float(x), float(y), ground] for x, y in route.samples_xy
        ]
        for index, route in enumerate(plan["routes"], start=1)
    }


def _audio_signature(
    program: Mapping[str, Any], *, include_endpoint: bool = True
) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    for event in program["events"]:
        values: tuple[Any, ...] = (
            event["sound_asset_id"],
            event["start_sample"],
            event["end_sample_exclusive"],
            event["source_start_sample"],
            event["source_end_sample_exclusive"],
        )
        if include_endpoint:
            values = (event["source_endpoint_id"], *values)
        signature.append(values)
    return signature


def _gateb_descriptor() -> dict[str, str]:
    """Name the independent visual Gate B artifacts explicitly."""

    return {
        "actor_selection": "actor_selection_gateB.json",
        "source_endpoints": "source_endpoints_gateB.json",
        "timeline": "timeline_gateB.json",
        "audio_program": "audio_program.json",
    }


def _swap_audio_slots(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rebind every voice occurrence to the other endpoint slot."""

    slot_map = {"source1": "source2", "source2": "source1"}
    result: list[dict[str, Any]] = []
    for event in events:
        slot = event.get("slot")
        if slot not in slot_map:
            raise OffscreenIdentityError(
                f"Gate A audio rebind cannot map endpoint slot {slot!r}"
            )
        result.append({**dict(event), "slot": slot_map[slot]})
    return result


def _speech_binding_rows(
    events: Sequence[Mapping[str, Any]], color_by_slot: Mapping[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "phase": event["phase"],
            "occurrence_index": event["occurrence_index"],
            "slot": event["slot"],
            "appearance": color_by_slot[event["slot"]],
            "sound_asset_id": event["sound_asset_id"],
            "speaker_id": event["speaker_id"],
            "utterance_id": event["utterance_id"],
            "transcript": event["transcript"],
            "split": event["split"],
            "start_sample": event["start_sample"],
            "duration_samples": event["duration_samples"],
            "source_start_sample": event["source_start_sample"],
            "source_end_sample_exclusive": event["source_end_sample_exclusive"],
            "complete_source_window": True,
        }
        for event in events
    ]


def _question(gold: str, options: Sequence[str]) -> dict[str, Any]:
    stem = "What colour was the person who spoke first?"
    return {
        "mcq": {"stem": stem, "options_space": list(options), "truth_option": gold},
        "open": {"stem": stem, "truth_value": gold, "scoring": "closed_set"},
    }


def _gatea_fact(
    main_fact: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    point_id: str,
    gold: str,
    appearance_by_slot: Mapping[str, str],
    speech_bindings: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    target_slot: str,
) -> dict[str, Any]:
    """Return the Gate A fact with its changed gold and unchanged question form."""

    result = copy.deepcopy(dict(main_fact))
    result.update(
        {
            **copy.deepcopy(dict(question)),
            "variant": "gateA",
            "gatea_of": point_id,
            "fact_record": "fact_record_gateA.json",
            "target": {
                **copy.deepcopy(dict(main_fact["target"])),
                "first_speaker_slot": target_slot,
                "first_speaker_appearance": gold,
            },
            "appearance_by_slot": dict(appearance_by_slot),
            "speech_bindings": copy.deepcopy(list(speech_bindings)),
            "audio": {
                "program": "audio_program_gateA.json",
                "schedule": copy.deepcopy(dict(schedule)),
                "voice_content_repeated_late": True,
            },
        }
    )
    return result


def _write_scene_candidate(
    scene_dir: Path,
    *,
    profile: Mapping[str, Any],
    params: Mapping[str, Any],
    scene_config_path: Path,
    scene: Any,
    plan: Mapping[str, Any],
    speech_events: Sequence[Mapping[str, Any]],
    speech_meta: Mapping[str, Any],
    registry_path: Path,
    by_id: Mapping[str, Mapping[str, Any]],
    snapshot_content: str | Path,
    seed: str,
    cell_index: int,
) -> dict[str, Any]:
    colors = list(profile["appearance_colors"])
    appearance_rng = np.random.default_rng(seed_uint64(f"appearance|{seed}"))
    color_order = [colors[int(index)] for index in appearance_rng.permutation(2)]
    main_asset_ids = _human_assets(by_id, color_order)
    gateb_asset_ids = list(reversed(main_asset_ids))
    main_color_by_slot = {
        f"source{index + 1}": color
        for index, color in enumerate(color_order)
    }
    point_id = f"{scene.scene_id}_f2_offscreen_identity_{cell_index + 1:03d}"
    point_dir = scene_dir / point_id
    point_dir.mkdir(parents=True)

    main_selection = _selection(
        main_asset_ids, by_id=by_id, snapshot_content=snapshot_content
    )
    gateb_selection = _selection(
        gateb_asset_ids, by_id=by_id, snapshot_content=snapshot_content
    )
    main_selection_path = _write(point_dir / "actor_selection.json", main_selection)
    gateb_selection_path = _write(point_dir / "actor_selection_gateB.json", gateb_selection)
    main_endpoint_path = point_dir / "source_endpoints.json"
    gateb_endpoint_path = point_dir / "source_endpoints_gateB.json"
    allowed = {"source1": ["speech_playback"], "source2": ["speech_playback"]}
    _, main_endpoint_records = build_endpoint_registry(
        main_selection,
        by_id,
        main_endpoint_path,
        allowed_sound_classes_by_slot=allowed,
        selection_path=main_selection_path,
    )
    _, gateb_endpoint_records = build_endpoint_registry(
        gateb_selection,
        by_id,
        gateb_endpoint_path,
        allowed_sound_classes_by_slot=allowed,
        selection_path=gateb_selection_path,
    )
    main_endpoints = {
        actor["source_slot_id"]: endpoint["source_endpoint_id"]
        for actor, endpoint in zip(main_selection["actors"], main_endpoint_records, strict=True)
    }
    gateb_endpoints = {
        actor["source_slot_id"]: endpoint["source_endpoint_id"]
        for actor, endpoint in zip(gateb_selection["actors"], gateb_endpoint_records, strict=True)
    }
    if main_endpoints != gateb_endpoints:
        raise OffscreenIdentityError("Gate B changed endpoint IDs while swapping appearances")

    render = resolve_scene_render_context(scene)
    routes = _timeline_routes(scene, plan)
    camera_ue = [
        float(plan["camera_xy"][0]),
        float(plan["camera_xy"][1]),
        float(render["ground_z_ue_cm"] + float(plan["camera_clearance"]["camera_height_m"]) * 100.0),
    ]
    main_timeline_path = point_dir / "timeline.json"
    main_timeline = author_current_n_actor_visual_timeline(
        actor_selection_path=main_selection_path,
        source_asset_registry_path=registry_path,
        output_path=main_timeline_path,
        camera_position_ue_cm=camera_ue,
        camera_yaw_deg=float(plan["camera_yaw_deg"]),
        routes_by_slot_ue_cm=routes,
        native_map=str(render["native_map"]),
        room_profile_id=str(render["room_profile_id"]),
        hfov_degrees=scene.hfov_deg,
        frame_count=int(speech_meta["clock"]["frame_count"]),
        frame_rate_hz=float(speech_meta["clock"]["frame_rate_hz"]),
        ticks_per_frame=int(params["TICKS_PER_FRAME"]),
    )
    gateb_timeline_path = point_dir / "timeline_gateB.json"
    gateb_timeline = author_current_n_actor_visual_timeline(
        actor_selection_path=gateb_selection_path,
        source_asset_registry_path=registry_path,
        output_path=gateb_timeline_path,
        camera_position_ue_cm=camera_ue,
        camera_yaw_deg=float(plan["camera_yaw_deg"]),
        routes_by_slot_ue_cm=routes,
        native_map=str(render["native_map"]),
        room_profile_id=str(render["room_profile_id"]),
        hfov_degrees=scene.hfov_deg,
        frame_count=int(speech_meta["clock"]["frame_count"]),
        frame_rate_hz=float(speech_meta["clock"]["frame_rate_hz"]),
        ticks_per_frame=int(params["TICKS_PER_FRAME"]),
    )
    del main_timeline, gateb_timeline

    base_request = _read(scene_config_path and SS.read_scene_config(scene_config_path)["camera_base_request"])
    m1_request = apply_camera_listener_pose_ue(
        base_request,
        request_id=f"qa_v3_{scene.scene_id}_{point_id}",
        position_m=render["world_transform"](camera_ue),
        ue_yaw_degrees=float(plan["camera_yaw_deg"]),
        horizontal_fov_deg=float(scene.hfov_deg),
    )
    m1_path = _write(point_dir / "m1_capture_request.json", m1_request)

    request_common = {
        "pair_kind": "f2_offscreen_identity",
        "point_id": point_id,
        "slot_endpoints": main_endpoints,
        **program_request_fields(params, include_mode=False),
        "mode": "sequential_sources",
    }
    gatea_audio_events = _swap_audio_slots(speech_events)
    main_program = build_program(
        request_common,
        list(speech_events),
        revision="v1",
    )
    gatea_program = build_program(
        request_common,
        gatea_audio_events,
        revision="gateA_v1",
    )
    validate_m6_audio_program(main_program)
    validate_m6_audio_program(gatea_program)
    main_program_path = _write(point_dir / "audio_program.json", main_program)
    gatea_program_path = _write(point_dir / "audio_program_gateA.json", gatea_program)

    option_rng = np.random.default_rng(seed_uint64(f"options|{seed}"))
    option_order = [int(index) for index in option_rng.permutation(2)]
    options = [color_order[index] for index in option_order]
    main_gold = main_color_by_slot["source1"]
    gatea_gold = main_color_by_slot["source2"]
    main_question = _question(main_gold, options)
    gatea_question = _question(gatea_gold, options)
    main_bindings = _speech_binding_rows(speech_events, main_color_by_slot)
    gatea_bindings = _speech_binding_rows(gatea_audio_events, main_color_by_slot)
    main_audio_content_signature = _audio_signature(main_program, include_endpoint=False)
    gatea_audio_content_signature = _audio_signature(gatea_program, include_endpoint=False)
    main_audio_endpoint_signature = _audio_signature(main_program, include_endpoint=True)
    gatea_audio_endpoint_signature = _audio_signature(gatea_program, include_endpoint=True)
    checks = {
        "audio_content_and_timing_preserved": main_audio_content_signature == gatea_audio_content_signature,
        "audio_endpoint_assignment_changed": main_audio_endpoint_signature != gatea_audio_endpoint_signature,
        "audio_program_actual_differs": main_program["events"] != gatea_program["events"],
        "question_stem_preserved": main_question["mcq"]["stem"] == gatea_question["mcq"]["stem"],
        "question_options_preserved": main_question["mcq"]["options_space"] == gatea_question["mcq"]["options_space"],
        "gold_changes": main_gold != gatea_gold,
        "main_voice_actor_stable_early_late": all(
            row["appearance"] == main_color_by_slot[row["slot"]]
            for row in main_bindings
        ),
        "gatea_voice_actor_stable_early_late": all(
            row["appearance"] == main_color_by_slot[row["slot"]]
            for row in gatea_bindings
        ),
        "gatea_rebinds_both_occurrences_per_voice": all(
            main_bindings[index]["appearance"] != gatea_bindings[index]["appearance"]
            and main_bindings[index]["slot"] != gatea_bindings[index]["slot"]
            and main_bindings[index]["speaker_id"] == gatea_bindings[index]["speaker_id"]
            and main_bindings[index]["utterance_id"] == gatea_bindings[index]["utterance_id"]
            for index in range(len(main_bindings))
        ),
        "complete_utterance_windows_preserved": all(
            row["complete_source_window"]
            and row["duration_samples"]
            == row["source_end_sample_exclusive"] - row["source_start_sample"]
            for row in main_bindings + gatea_bindings
        ),
    }
    if not all(checks.values()):
        raise OffscreenIdentityError(f"F2 identity checks failed: {checks}")

    main_fact = {
        "schema": SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "truth_status": "pending_native_pixel_join",
        "evidence_class": "cpu_geometry_candidate",
        "profile_id": profile["id"],
        "scene_id": scene.scene_id,
        "point_id": point_id,
        "answer_forms": answer_forms_from_params(params),
        **main_question,
        "target": {
            "first_speaker_slot": "source1",
            "first_speaker_appearance": main_gold,
            "speaker_id": speech_events[0]["speaker_id"],
            "utterance_id": speech_events[0]["utterance_id"],
            "transcript": speech_events[0]["transcript"],
        },
        "appearance_by_slot": main_color_by_slot,
        "speech_bindings": main_bindings,
        "geometry": {
            "camera_xy": plan["camera_xy"],
            "camera_yaw_deg": plan["camera_yaw_deg"],
            "half_fov_deg": plan["half_fov_deg"],
            "route_reports": plan["route_reports"],
            "early_visibility": (
                f"{profile['visibility_rule']['early']}_by_camera_cone_geometry"
            ),
            "late_visibility": (
                f"{profile['visibility_rule']['late']}_by_camera_cone_geometry"
            ),
            "pixel_visibility": "pending_native_pixel_join",
            "line_of_sight_screened": plan["line_of_sight_screened"],
        },
        "audio": {
            "program": "audio_program.json",
            "schedule": speech_meta,
            "voice_content_repeated_late": True,
        },
        "gatea_checks": checks,
        "artifacts": {
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
            "m1_request": "m1_capture_request.json",
            "source_endpoints": "source_endpoints.json",
            "audio_program": "audio_program.json",
            "audio_program_gateA": "audio_program_gateA.json",
            "actor_selection_gateB": "actor_selection_gateB.json",
            "source_endpoints_gateB": "source_endpoints_gateB.json",
            "timeline_gateB": "timeline_gateB.json",
            "gateB": "gateB_intervention.json",
        },
    }
    gatea_fact = _gatea_fact(
        main_fact,
        question=gatea_question,
        point_id=point_id,
        gold=gatea_gold,
        appearance_by_slot=main_color_by_slot,
        speech_bindings=gatea_bindings,
        schedule={
            **copy.deepcopy(dict(speech_meta)),
            "events": copy.deepcopy(gatea_audio_events),
        },
        target_slot="source2",
    )
    gateb = {
        "schema": GATEB_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "variant": "gateB",
        "kept_fixed": ["audio_program", "event_times", "question_stem", "question_options"],
        "changed_visual_fact": "appearance_to_route_binding",
        **_gateb_descriptor(),
        "audio_unchanged": True,
        "media_status": "pending_native_pixel_join",
        "claim_boundary": "descriptive visual exchange only; native media must verify it",
    }
    _write(point_dir / "fact_record.json", main_fact)
    _write(point_dir / "fact_record_gateA.json", gatea_fact)
    _write(point_dir / "gateB_intervention.json", gateb)
    return {
        "point_id": point_id,
        "status": "research_candidate",
        "truth_status": "pending_native_pixel_join",
        "artifacts": {
            "selection": str(main_selection_path),
            "selection_gateB": str(gateb_selection_path),
            "endpoints": str(main_endpoint_path.resolve()),
            "endpoints_gateB": str(gateb_endpoint_path.resolve()),
            "timeline": str(main_timeline_path.resolve()),
            "timeline_gateB": str(gateb_timeline_path.resolve()),
            "m1_request": str(m1_path),
            "audio_program": str(main_program_path),
            "audio_program_gateA": str(gatea_program_path),
            "fact": str((point_dir / "fact_record.json").resolve()),
            "fact_gateA": str((point_dir / "fact_record_gateA.json").resolve()),
            "gateB": str((point_dir / "gateB_intervention.json").resolve()),
        },
        "geometry": {
            "search_attempts": int(plan["search_attempts"]),
            "route_ids": [route.route_id for route in plan["routes"]],
            "route_sources": [route.source for route in plan["routes"]],
        },
        "gatea_checks": checks,
    }


def _scene_candidate(
    *,
    scene_config_path: Path,
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    out_root: Path,
    snapshot_content: str | Path,
    seed: str,
    cell_index: int,
    registry_path: Path,
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    config = SS.read_scene_config(scene_config_path)
    clock = SS.validate_frame_clock(params, require_clip_seconds=True)
    scene = SS.load_scene(
        config,
        frame_count=clock["frame_count"],
        frame_rate_hz=clock["frame_rate_hz"],
    )
    SS.require_camera_clearance(scene, params)
    render = resolve_scene_render_context(scene)
    del render
    speech_events, speech_meta = _speech_occurrences(params, profile, seed=seed)
    plan = _find_offscreen_entry_plan(
        scene,
        params,
        profile,
        speech_events,
        speech_meta["clock"],
        seed=seed,
    )
    scene_dir = out_root / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    result = _write_scene_candidate(
        scene_dir,
        profile=profile,
        params=params,
        scene_config_path=scene_config_path,
        scene=scene,
        plan=plan,
        speech_events=speech_events,
        speech_meta=speech_meta,
        registry_path=registry_path,
        by_id=by_id,
        snapshot_content=snapshot_content,
        seed=seed,
        cell_index=cell_index,
    )
    return {
        "scene_id": scene.scene_id,
        "scene_config": str(scene_config_path.resolve()),
        "profile_id": profile["id"],
        **result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", action="append", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--snapshot-content", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--seed", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if isinstance(args.cells, bool) or args.cells <= 0:
        parser.error("--cells must be positive")
    if args.out_root.exists() or args.out_root.is_symlink():
        parser.error(f"refusing to overwrite output: {args.out_root.resolve()}")
    profile = _profile(args.profile)
    params = read_qa_params(args.params)
    registry_path, by_id = _asset_registry()
    snapshot = Path(args.snapshot_content).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.is_symlink():
        parser.error(f"--snapshot-content must be a regular directory: {snapshot}")
    out_root = _fresh_directory(args.out_root)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for scene_config in args.scene_config:
        for cell_index in range(args.cells):
            cell_seed = f"{args.seed}|{scene_config.stem}|{cell_index}"
            try:
                records.append(
                    _scene_candidate(
                        scene_config_path=scene_config,
                        params=params,
                        profile=profile,
                        out_root=out_root,
                        snapshot_content=snapshot,
                        seed=cell_seed,
                        cell_index=cell_index,
                        registry_path=registry_path,
                        by_id=by_id,
                    )
                )
            except (OffscreenIdentityError, AudioProfileError) as exc:
                rejected.append(
                    {
                        "scene_config": str(scene_config.resolve()),
                        "cell_index": cell_index,
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
    question_result = None
    if records:
        question_result = write_requested_questions(
            out_root,
            (record["artifacts"]["fact"] for record in records),
            params,
        )
    manifest = {
        "schema": BATCH_SCHEMA,
        "status": "research_candidate" if records else "resource_unavailable",
        "qualification_claim": False,
        "profile_id": profile["id"],
        "params": str(Path(args.params).expanduser().resolve()),
        "snapshot_content": str(snapshot),
        "seed": args.seed,
        "scenes": [str(path.expanduser().resolve()) for path in args.scene_config],
        "counts": {
            "scenes_requested": len(args.scene_config),
            "cells_requested": len(args.scene_config) * args.cells,
            "candidates": len(records),
            "rejected": len(rejected),
        },
        "records": records,
        "rejected": rejected,
        "question_request": question_result,
        "pixel_status": "pending_native_pixel_join",
        "claim_boundary": (
            "CPU route and complete speech planning only. Early/late visibility labels "
            "come from camera-cone geometry; native pixels, recognisability, and media "
            "answerability remain pending."
        ),
    }
    _write(out_root / "batch_manifest.json", manifest)
    print(
        json.dumps(
            {
                "out": str(out_root),
                "candidates": len(records),
                "rejected": len(rejected),
                "scenes": [record["scene_id"] for record in records],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
