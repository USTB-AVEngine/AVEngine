#!/usr/bin/env python3
"""Author backend-independent F2 direction facts from observed native frames.

The author consumes actual ``frame_records.json`` emitter positions, the M1
listener request, and an AudioProgram.  It never reads actor roots or planned
routes.  A profile supplies the query event/window and answer domain.  The
result is a main-only research fact; counterfactual audio remains pending until
an actual route-swapped recording is supplied by a separate capture.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from avengine.capture.orientation import habitat_basis_from_xyzw  # noqa: E402
from avengine.contracts.transforms import (  # noqa: E402
    compose_transforms,
    normalized_quaternion_xyzw,
)


PROFILE_SCHEMA = "avengine_qa_v3_observed_f2_direction_profile_v1"
FACT_SCHEMA = "avengine_qa_v3_observed_f2_direction_fact_v1"
BATCH_SCHEMA = "avengine_qa_v3_observed_f2_direction_batch_v1"


class ObservedF2DirectionError(ValueError):
    """Observed frame/program inputs cannot support the requested fact."""


def _read(path: str | Path) -> Any:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ObservedF2DirectionError(f"missing regular JSON file: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObservedF2DirectionError(f"cannot read JSON {source}: {exc}") from exc


def _write(path: str | Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ObservedF2DirectionError(f"refusing to replace output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservedF2DirectionError(f"{owner} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ObservedF2DirectionError(f"{owner} must be finite")
    return result


def _vector3(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ObservedF2DirectionError(f"{owner} must be a finite 3-vector")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ObservedF2DirectionError(f"{owner} must be a finite 3-vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ObservedF2DirectionError(f"{owner} must be a finite 3-vector")
    return np.ascontiguousarray(result)


def _transform(value: Any, *, owner: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping):
        raise ObservedF2DirectionError(f"{owner} must be a transform object")
    translation = _vector3(value.get("translation_m"), owner=f"{owner}.translation_m")
    try:
        quaternion = normalized_quaternion_xyzw(value.get("rotation_xyzw"))
    except (TypeError, ValueError) as exc:
        raise ObservedF2DirectionError(f"{owner}.rotation_xyzw is invalid") from exc
    return {
        "translation_m": [float(item) for item in translation],
        "rotation_xyzw": [float(item) for item in quaternion],
    }


def _transform_error(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_t = _transform(left, owner="left transform")
    right_t = _transform(right, owner="right transform")
    translation_error = float(
        np.linalg.norm(
            np.asarray(left_t["translation_m"]) - np.asarray(right_t["translation_m"])
        )
    )
    q_left = np.asarray(left_t["rotation_xyzw"], dtype=np.float64)
    q_right = np.asarray(right_t["rotation_xyzw"], dtype=np.float64)
    rotation_error = min(float(np.linalg.norm(q_left - q_right)), float(np.linalg.norm(q_left + q_right)))
    return max(translation_error, rotation_error)


def _listener_from_m1(m1: Mapping[str, Any]) -> dict[str, list[float]]:
    try:
        rig = m1["primary_camera_rig"]
        listener = m1["listener"]
        world_from_rig = _transform(rig["world_from_rig"], owner="M1 world_from_rig")
        rig_from_listener = _transform(listener["rig_from_listener"], owner="M1 rig_from_listener")
    except (KeyError, TypeError) as exc:
        raise ObservedF2DirectionError("M1 request lacks primary camera/listener transforms") from exc
    return compose_transforms(world_from_rig, rig_from_listener)


def relative_azimuth_from_listener(
    source_position_m: Sequence[float], listener_pose: Mapping[str, Any]
) -> float:
    """Return signed degrees from listener forward (-Z) toward right (+X)."""

    source = _vector3(source_position_m, owner="source position")
    listener = _transform(listener_pose, owner="listener pose")
    listener_position = np.asarray(listener["translation_m"], dtype=np.float64)
    vector = source - listener_position
    horizontal = np.asarray([vector[0], vector[2]], dtype=np.float64)
    if float(np.linalg.norm(horizontal)) <= 1.0e-12:
        raise ObservedF2DirectionError("source and listener occupy the same horizontal point")
    try:
        basis = habitat_basis_from_xyzw(listener["rotation_xyzw"])
    except (TypeError, ValueError) as exc:
        raise ObservedF2DirectionError(f"listener orientation is invalid: {exc}") from exc
    forward = np.asarray(basis.forward_xz, dtype=np.float64)
    right = np.asarray(basis.right_xz, dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    right_norm = float(np.linalg.norm(right))
    if forward_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        raise ObservedF2DirectionError("listener orientation has no horizontal basis")
    forward /= forward_norm
    right /= right_norm
    return math.degrees(
        math.atan2(float(np.dot(horizontal, right)), float(np.dot(horizontal, forward)))
    )


def _load_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != PROFILE_SCHEMA:
        raise ObservedF2DirectionError(f"profile schema must be {PROFILE_SCHEMA!r}")
    result = dict(value)
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise ObservedF2DirectionError("profile id must be non-empty")
    domain = result.get("answer_domain")
    if domain not in {"full_circle", "front_back"}:
        raise ObservedF2DirectionError("profile answer_domain must be full_circle or front_back")
    event_index = result.get("query_event_index")
    if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
        raise ObservedF2DirectionError("query_event_index must be non-negative")
    window = result.get("query_window")
    if not isinstance(window, Mapping) or window.get("kind") != "audio_event":
        raise ObservedF2DirectionError("profile query_window must select an audio_event")
    for key in ("start_padding_samples", "end_padding_samples"):
        value = window.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ObservedF2DirectionError(f"query_window.{key} must be non-negative")
    if domain == "full_circle":
        shape = result.get("answer_shape")
        if not isinstance(shape, Mapping):
            raise ObservedF2DirectionError("full_circle profile needs answer_shape")
        bands = shape.get("equal_bands")
        if isinstance(bands, bool) or not isinstance(bands, int) or bands < 2:
            raise ObservedF2DirectionError("full_circle equal_bands must be an integer >= 2")
    else:
        split = result.get("front_back_split_deg")
        if isinstance(split, bool) or not isinstance(split, (int, float)):
            raise ObservedF2DirectionError("front_back_split_deg must be finite")
        split = float(split)
        if not math.isfinite(split) or not 0.0 < split < 180.0:
            raise ObservedF2DirectionError("front_back_split_deg must lie in (0,180)")
    return result


def load_profiles(path: str | Path) -> list[dict[str, Any]]:
    value = _read(path)
    if not isinstance(value, list) or not value:
        raise ObservedF2DirectionError("profile file must contain a non-empty list")
    profiles = [_load_profile(item) for item in value if isinstance(item, Mapping)]
    if len(profiles) != len(value):
        raise ObservedF2DirectionError("every profile entry must be an object")
    ids = [profile["id"] for profile in profiles]
    if len(set(ids)) != len(ids):
        raise ObservedF2DirectionError("profile ids must be unique")
    return profiles


def _clock_fraction(value: Any, *, owner: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ObservedF2DirectionError(f"{owner} must be a positive rational number")
    try:
        result = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ObservedF2DirectionError(
            f"{owner} must be a positive rational number"
        ) from exc
    if result <= 0:
        raise ObservedF2DirectionError(f"{owner} must be a positive rational number")
    return result


def _clock_value_equal(
    left: Any, right: Any, *, left_owner: str, right_owner: str
) -> bool:
    return _clock_fraction(left, owner=left_owner) == _clock_fraction(
        right, owner=right_owner
    )


def _load_program_and_frames(
    *,
    frame_records_path: str | Path,
    m1_path: str | Path,
    audio_program_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, Path]:
    records = _read(frame_records_path)
    m1 = _read(m1_path)
    program = _read(audio_program_path)
    if not isinstance(records, Mapping) or not isinstance(m1, Mapping) or not isinstance(program, Mapping):
        raise ObservedF2DirectionError("frame records, M1 request, and AudioProgram must be objects")
    frames = records.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ObservedF2DirectionError("frame records has no frames")
    render = records.get("render")
    timeline = program.get("timeline")
    if not isinstance(render, Mapping) or not isinstance(timeline, Mapping):
        raise ObservedF2DirectionError("frame records/program lack render timeline metadata")
    for key in ("frame_count", "sample_count"):
        if render.get(key) != timeline.get(key):
            raise ObservedF2DirectionError(
                f"AudioProgram and observed render differ at timeline.{key}: "
                f"{timeline.get(key)!r} vs {render.get(key)!r}"
            )
    for key in ("sample_rate_hz", "ticks_per_frame", "time_base_hz"):
        if not _clock_value_equal(
            render.get(key),
            timeline.get(key),
            left_owner=f"observed render {key}",
            right_owner=f"AudioProgram {key}",
        ):
            raise ObservedF2DirectionError(
                f"AudioProgram and observed render differ at timeline.{key}: "
                f"{timeline.get(key)!r} vs {render.get(key)!r}"
            )
    time_base = _clock_fraction(
        timeline.get("time_base_hz"), owner="AudioProgram time_base_hz"
    )
    sample_rate = _clock_fraction(
        timeline.get("sample_rate_hz"), owner="AudioProgram sample_rate_hz"
    )
    ticks_per_sample = _clock_fraction(
        timeline.get("ticks_per_sample"), owner="AudioProgram ticks_per_sample"
    )
    ticks_per_frame = _clock_fraction(
        timeline.get("ticks_per_frame"), owner="AudioProgram ticks_per_frame"
    )
    video_fps = _clock_fraction(
        timeline.get("video_fps"), owner="AudioProgram video_fps"
    )
    if ticks_per_sample != time_base / sample_rate:
        raise ObservedF2DirectionError(
            "AudioProgram ticks_per_sample disagrees with time_base_hz/sample_rate_hz"
        )
    if video_fps != time_base / ticks_per_frame:
        raise ObservedF2DirectionError(
            "AudioProgram video_fps disagrees with time_base_hz/ticks_per_frame"
        )
    if not _clock_value_equal(
        render.get("frame_rate_hz"),
        timeline.get("video_fps"),
        left_owner="observed render frame_rate_hz",
        right_owner="AudioProgram video_fps",
    ):
        raise ObservedF2DirectionError(
            "observed render frame_rate_hz differs from AudioProgram video_fps"
        )
    if len(frames) != int(timeline["frame_count"]):
        raise ObservedF2DirectionError("observed frame count differs from AudioProgram timeline")
    expected_listener = _listener_from_m1(m1)
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
            raise ObservedF2DirectionError(f"frame records index {index} is invalid")
        if frame.get("pts_ticks") != index * int(timeline["ticks_per_frame"]):
            raise ObservedF2DirectionError(f"frame {index} pts_ticks differs from timeline")
        camera = frame.get("camera_readback")
        if not isinstance(camera, Mapping) or not isinstance(camera.get("agent"), Mapping):
            raise ObservedF2DirectionError(f"frame {index} lacks actual listener/camera readback")
        actual_listener = _transform(camera["agent"], owner=f"frame {index} listener readback")
        if _transform_error(actual_listener, expected_listener) > 2.0e-4:
            raise ObservedF2DirectionError(
                f"frame {index} listener readback differs from M1 listener pose"
            )
        sources = frame.get("source_positions_m")
        actors = frame.get("actor_readbacks")
        if not isinstance(sources, list) or not sources:
            raise ObservedF2DirectionError(f"frame {index} lacks actual source_positions_m")
        if not isinstance(actors, list) or len(actors) != len(sources):
            raise ObservedF2DirectionError(
                f"frame {index} actor readbacks do not align with source_positions_m"
            )
        for source in sources:
            _vector3(source, owner=f"frame {index} source position")
        for actor_index, actor in enumerate(actors):
            if not isinstance(actor, Mapping):
                raise ObservedF2DirectionError(f"frame {index} actor readback is invalid")
            endpoint = actor.get("source_endpoint_id")
            if not isinstance(endpoint, str) or not endpoint:
                raise ObservedF2DirectionError(f"frame {index} actor endpoint ID is missing")
            emitter = _vector3(
                actor.get("emitter_world_position_m"),
                owner=f"frame {index} actor emitter readback",
            )
            if float(np.max(np.abs(emitter - np.asarray(sources[actor_index], dtype=np.float64)))) > 2.0e-4:
                raise ObservedF2DirectionError(
                    f"frame {index} source_positions_m differs from actual emitter readback"
                )
    events = program.get("events")
    endpoints = program.get("candidate_source_endpoint_ids")
    if not isinstance(events, list) or not events:
        raise ObservedF2DirectionError("AudioProgram has no events")
    if not isinstance(endpoints, list) or not endpoints:
        raise ObservedF2DirectionError("AudioProgram has no source endpoints")
    return (
        dict(records),
        dict(m1),
        dict(program),
        Path(frame_records_path).expanduser().resolve(),
        Path(m1_path).expanduser().resolve(),
        Path(audio_program_path).expanduser().resolve(),
    )


def _event_window_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    event: Mapping[str, Any],
    sample_count: int,
    frame_count: int,
    sample_rate_hz: int,
    window: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], tuple[int, int]]:
    start = int(event["start_sample"]) - int(window.get("start_padding_samples", 0))
    end = int(event["end_sample_exclusive"]) + int(window.get("end_padding_samples", 0))
    if start < 0 or end > sample_count or end <= start:
        raise ObservedF2DirectionError("profile query window falls outside the audio timeline")
    selected: list[Mapping[str, Any]] = []
    selected_indices: list[int] = []
    for index, frame in enumerate(frames):
        frame_start = (index * sample_count) // frame_count
        frame_end = ((index + 1) * sample_count + frame_count - 1) // frame_count
        if frame_end > start and frame_start < end:
            selected.append(frame)
            selected_indices.append(index)
    if not selected:
        raise ObservedF2DirectionError("query window contains no observed frame samples")
    del sample_rate_hz
    return selected, (selected_indices[0], selected_indices[-1])


def _endpoint_index(frame: Mapping[str, Any], endpoint_id: str) -> int:
    actors = frame.get("actor_readbacks")
    if not isinstance(actors, list):
        raise ObservedF2DirectionError("frame has no actor_readbacks")
    matches = [
        index
        for index, actor in enumerate(actors)
        if isinstance(actor, Mapping) and actor.get("source_endpoint_id") == endpoint_id
    ]
    if len(matches) != 1:
        raise ObservedF2DirectionError(
            f"frame has {len(matches)} readbacks for source endpoint {endpoint_id!r}"
        )
    return matches[0]


def _classify_full_circle(angle: float, band_count: int) -> tuple[int, float, float, str]:
    width = 360.0 / band_count
    normalized = ((float(angle) + 180.0) % 360.0) - 180.0
    index = min(band_count - 1, int(math.floor((normalized + 180.0) / width)))
    lower = -180.0 + index * width
    upper = lower + width
    label = f"[{lower:g}, {upper:g})"
    return index, lower, upper, label


def _classify_front_back(angle: float, split_deg: float) -> tuple[str, dict[str, Any]]:
    absolute = abs(float(angle))
    if absolute <= split_deg:
        return "front", {"start_deg": -split_deg, "end_deg": split_deg}
    return "back", {
        "start_deg": split_deg,
        "end_deg": 360.0 - split_deg,
        "representation": "absolute_azimuth_greater_than_split",
    }


def _audio_event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "event_id",
        "source_endpoint_id",
        "sound_asset_id",
        "start_sample",
        "end_sample_exclusive",
        "source_start_sample",
        "source_end_sample_exclusive",
    )
    missing = [key for key in required if key not in event]
    if missing:
        raise ObservedF2DirectionError(f"AudioProgram event missing {missing}")
    start = int(event["start_sample"])
    end = int(event["end_sample_exclusive"])
    source_start = int(event["source_start_sample"])
    source_end = int(event["source_end_sample_exclusive"])
    if end <= start or source_end <= source_start:
        raise ObservedF2DirectionError("AudioProgram event has an empty window")
    return {
        "event_id": str(event["event_id"]),
        "source_endpoint_id": str(event["source_endpoint_id"]),
        "sound_asset_id": str(event["sound_asset_id"]),
        "start_sample": start,
        "end_sample_exclusive": end,
        "duration_samples": end - start,
        "source_start_sample": source_start,
        "source_end_sample_exclusive": source_end,
        "source_duration_samples": source_end - source_start,
    }


def _receipt_geometry_matches(
    receipt_frame_path: str | Path,
    records: Mapping[str, Any],
) -> bool:
    try:
        receipt_records = _read(receipt_frame_path)
    except ObservedF2DirectionError:
        return False
    if not isinstance(receipt_records, Mapping):
        return False
    left_frames = records.get("frames")
    right_frames = receipt_records.get("frames")
    if not isinstance(left_frames, list) or not isinstance(right_frames, list):
        return False
    if len(left_frames) != len(right_frames):
        return False
    for left, right in zip(left_frames, right_frames, strict=True):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if left.get("frame_index") != right.get("frame_index"):
            return False
        try:
            if not np.allclose(
                np.asarray(left["source_positions_m"], dtype=np.float64),
                np.asarray(right["source_positions_m"], dtype=np.float64),
                rtol=0.0,
                atol=2.0e-4,
            ):
                return False
            left_agent = left["camera_readback"]["agent"]
            right_agent = right["camera_readback"]["agent"]
            if _transform_error(left_agent, right_agent) > 2.0e-4:
                return False
            left_endpoints = [
                actor["source_endpoint_id"]
                for actor in left["actor_readbacks"]
                if isinstance(actor, Mapping)
            ]
            right_endpoints = [
                actor["source_endpoint_id"]
                for actor in right["actor_readbacks"]
                if isinstance(actor, Mapping)
            ]
            if left_endpoints != right_endpoints:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _validate_audio_receipt(
    receipt_path: str | Path,
    *,
    records: Mapping[str, Any],
    m1_path: Path,
    program_path: Path,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "pending",
        "receipt": str(Path(receipt_path).expanduser().resolve()),
        "checks": {},
    }
    try:
        receipt = _read(receipt_path)
    except ObservedF2DirectionError as exc:
        result["reason"] = str(exc)
        return result
    if not isinstance(receipt, Mapping):
        result["reason"] = "audio receipt must be a JSON object"
        return result
    result["receipt_status"] = receipt.get("status")
    audio_program = receipt.get("audio_program")
    inputs = receipt.get("inputs")
    sources = receipt.get("sources")
    if not isinstance(audio_program, Mapping) or not isinstance(inputs, Mapping):
        result["reason"] = "audio receipt lacks audio_program or inputs"
        return result
    checks: dict[str, bool] = {
        "receipt_status_pass": receipt.get("status") == "pass",
        "program_path_matches": (
            isinstance(audio_program.get("path"), str)
            and Path(audio_program["path"]).expanduser().resolve() == program_path.resolve()
        ),
        "program_id_matches": audio_program.get("program_id") == program.get("program_id"),
        "program_timeline_matches": audio_program.get("timeline") == program.get("timeline"),
        "m1_path_matches": (
            isinstance(inputs.get("m1_request"), Mapping)
            and inputs["m1_request"].get("path") is not None
            and Path(inputs["m1_request"]["path"]).expanduser().resolve() == m1_path.resolve()
        ),
    }
    observed_endpoint_ids = sorted(
        {
            actor.get("source_endpoint_id")
            for frame in records.get("frames", [])
            if isinstance(frame, Mapping)
            for actor in frame.get("actor_readbacks", [])
            if isinstance(actor, Mapping) and isinstance(actor.get("source_endpoint_id"), str)
        }
    )
    receipt_source_ids = (
        sorted(str(value) for value in sources.get("source_ids", []))
        if isinstance(sources, Mapping) and isinstance(sources.get("source_ids"), list)
        else []
    )
    checks["source_ids_match"] = observed_endpoint_ids == receipt_source_ids
    checks["source_clock_matches"] = (
        isinstance(sources, Mapping)
        and sources.get("frame_count") == program.get("timeline", {}).get("frame_count")
        and sources.get("frame_rate_hz") == program.get("timeline", {}).get("video_fps")
        and sources.get("ticks_per_frame") == program.get("timeline", {}).get("ticks_per_frame")
    )
    receipt_frames = inputs.get("visual_capture_frame_records")
    if isinstance(receipt_frames, Mapping) and isinstance(receipt_frames.get("path"), str):
        checks["frame_records_geometry_matches"] = _receipt_geometry_matches(
            receipt_frames["path"], records
        )
    else:
        checks["frame_records_geometry_matches"] = False
    result["checks"] = checks
    if all(checks.values()):
        result["status"] = "pass"
        result["reason"] = (
            "audio receipt passed and its program, M1 request, source IDs, clock, "
            "and observed frame geometry agree with the supplied inputs"
        )
    else:
        result["reason"] = "audio receipt did not agree with every supplied geometry/program input"
    return result


def author_observed_f2_direction(
    *,
    frame_records_path: str | Path,
    m1_path: str | Path,
    audio_program_path: str | Path,
    profile: Mapping[str, Any],
    output_directory: str | Path,
    audio_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one main-only observed direction fact and questions."""

    profile = _load_profile(profile)
    records, m1, program, frame_path, m1_file, program_path = _load_program_and_frames(
        frame_records_path=frame_records_path,
        m1_path=m1_path,
        audio_program_path=audio_program_path,
    )
    timeline = program["timeline"]
    events = program["events"]
    event_index = int(profile["query_event_index"])
    if event_index >= len(events):
        raise ObservedF2DirectionError(
            f"profile query_event_index={event_index} exceeds {len(events)} AudioProgram events"
        )
    event = _audio_event_summary(events[event_index])
    endpoint = event["source_endpoint_id"]
    if endpoint not in {str(item) for item in program["candidate_source_endpoint_ids"]}:
        raise ObservedF2DirectionError(
            f"AudioProgram event endpoint {endpoint!r} is not a candidate endpoint"
        )
    selected_frames, frame_span = _event_window_frames(
        records["frames"],
        event=event,
        sample_count=int(timeline["sample_count"]),
        frame_count=int(timeline["frame_count"]),
        sample_rate_hz=int(timeline["sample_rate_hz"]),
        window=profile["query_window"],
    )
    angles: list[float] = []
    observed_rows: list[dict[str, Any]] = []
    for frame in selected_frames:
        index = int(frame["frame_index"])
        actor_index = _endpoint_index(frame, endpoint)
        source = frame["source_positions_m"][actor_index]
        listener = frame["camera_readback"]["agent"]
        angle = relative_azimuth_from_listener(source, listener)
        angles.append(angle)
        observed_rows.append(
            {
                "frame_index": index,
                "pts_ticks": int(frame["pts_ticks"]),
                "source_endpoint_id": endpoint,
                "source_position_m": [float(value) for value in source],
                "listener_position_m": [float(value) for value in listener["translation_m"]],
                "listener_rotation_xyzw": [float(value) for value in listener["rotation_xyzw"]],
                "relative_azimuth_deg": angle,
            }
        )
    domain = profile["answer_domain"]
    if domain == "full_circle":
        band_count = int(profile["answer_shape"]["equal_bands"])
        classified = [_classify_full_circle(angle, band_count) for angle in angles]
        if len({item[0] for item in classified}) != 1:
            raise ObservedF2DirectionError(
                "query event spans more than one full-circle answer band"
            )
        band_index, lower, upper, label = classified[0]
        truth_option = label
        truth_payload = {
            "answer_domain": "full_circle",
            "band_index": band_index,
            "interval_deg": {"start_deg": lower, "end_deg": upper, "width_deg": upper - lower},
            "convention": "listener_forward_minus_z_right_plus_x",
        }
        options = [
            _classify_full_circle(-180.0 + index * 360.0 / band_count, band_count)[3]
            for index in range(band_count)
        ]
    else:
        split = float(profile["front_back_split_deg"])
        classes = [_classify_front_back(angle, split)[0] for angle in angles]
        if len(set(classes)) != 1:
            raise ObservedF2DirectionError(
                "query event spans both front and back answer domains"
            )
        truth_option = classes[0]
        truth_payload = {
            "answer_domain": "front_back",
            "split_deg": split,
            "interval_deg": _classify_front_back(angles[0], split)[1],
            "convention": "listener_forward_minus_z_right_plus_x",
        }
        options = ["front", "back"]
    if audio_receipt_path is None:
        audio_status = {
            "status": "pending",
            "reason": "no audio receipt supplied; only geometry/AudioProgram consistency was authored",
            "receipt": None,
        }
    else:
        audio_status = _validate_audio_receipt(
            audio_receipt_path,
            records=records,
            m1_path=m1_file,
            program_path=program_path,
            program=program,
        )
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ObservedF2DirectionError(f"refusing to replace output directory: {output}")
    output.mkdir(parents=True)
    fact = {
        "schema": FACT_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "truth_status": (
            "observed_geometry_and_audio_receipt_pending_media"
            if audio_status.get("status") == "pass"
            else "observed_geometry_pending_audio_and_media"
        ),
        "evidence_class": "observed_native_geometry",
        "profile_id": profile["id"],
        "answer_domain": domain,
        "question": {
            "mcq": {
                "stem": "Which direction did the sound source come from?",
                "options_space": options,
                "truth_option": truth_option,
            },
            "open": {
                "stem": "Which direction did the sound source come from?",
                "truth_value": truth_option,
                "scoring": "configured_direction_domain",
            },
        },
        "truth": truth_payload,
        "event": {
            **event,
            "query_event_index": event_index,
            "query_frame_span": {"first": frame_span[0], "last": frame_span[1]},
            "source_endpoint_id": endpoint,
        },
        "observed_geometry": {
            "frame_count": len(observed_rows),
            "angle_min_deg": min(angles),
            "angle_max_deg": max(angles),
            "angle_mean_deg": sum(angles) / len(angles),
            "rows": observed_rows,
            "source_authority": "frame_records.source_positions_m plus listener camera_readback",
            "planned_routes_used": False,
            "actor_root_used": False,
        },
        "inputs": {
            "frame_records": str(frame_path),
            "m1_request": str(m1_file),
            "audio_program": str(program_path),
            "audio_receipt": audio_status["receipt"],
        },
        "audio_validation": audio_status,
        "counterfactual": {
            "status": "pending",
            "reason": "no real route-swap audio/observed frame set was supplied; no GA fact was fabricated",
        },
    }
    fact_path = _write(output / "fact_record.json", fact)
    questions = [
        {
            "point_id": f"{program.get('program_id', 'program')}:{event['event_id']}",
            "profile_id": profile["id"],
            "variant": "main",
            "form": form,
            "question": fact["question"][form]["stem"],
            "answer": {
                key: value for key, value in fact["question"][form].items() if key != "stem"
            },
        }
        for form in ("mcq", "open")
    ]
    questions_path = _write(output / "questions.json", questions)
    return {
        "profile_id": profile["id"],
        "status": fact["status"],
        "fact": str(fact_path),
        "questions": str(questions_path),
        "event_id": event["event_id"],
        "source_endpoint_id": endpoint,
        "frame_span": {"first": frame_span[0], "last": frame_span[1]},
        "truth_option": truth_option,
        "angle_min_deg": min(angles),
        "angle_max_deg": max(angles),
        "audio_status": audio_status,
        "counterfactual_status": "pending",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-records", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--audio-program", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile-id")
    parser.add_argument("--audio-receipt", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    if args.profile_id is not None:
        profiles = [profile for profile in profiles if profile["id"] == args.profile_id]
        if not profiles:
            raise SystemExit(f"unknown --profile-id {args.profile_id!r}")
    root = Path(args.out_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise SystemExit(f"refusing to replace output: {root}")
    root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for profile in profiles:
        destination = root / profile["id"]
        try:
            records.append(
                author_observed_f2_direction(
                    frame_records_path=args.frame_records,
                    m1_path=args.m1_request,
                    audio_program_path=args.audio_program,
                    profile=profile,
                    output_directory=destination,
                    audio_receipt_path=args.audio_receipt,
                )
            )
        except ObservedF2DirectionError as exc:
            rejected.append(
                {"profile_id": profile["id"], "reason": type(exc).__name__, "detail": str(exc)}
            )
    manifest = {
        "schema": BATCH_SCHEMA,
        "status": "research_candidate" if records else "rejected",
        "qualification_claim": False,
        "records": records,
        "rejected": rejected,
        "inputs": {
            "frame_records": str(Path(args.frame_records).expanduser().resolve()),
            "m1_request": str(Path(args.m1_request).expanduser().resolve()),
            "audio_program": str(Path(args.audio_program).expanduser().resolve()),
            "profiles": str(Path(args.profiles).expanduser().resolve()),
            "audio_receipt": None if args.audio_receipt is None else str(args.audio_receipt.expanduser().resolve()),
        },
        "claim_boundary": (
            "Main-only observed native geometry and AudioProgram consistency. "
            "No GA route swap, audio rendering, model result, or formal admission claim."
        ),
    }
    _write(root / "batch_manifest.json", manifest)
    print(json.dumps({"out": str(root), "records": len(records), "rejected": len(rejected)}, sort_keys=True))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
