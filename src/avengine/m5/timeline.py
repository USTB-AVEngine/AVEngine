"""Exact M5 timeline and audio-only counterfactual contracts.

The frozen v2 timeline remains the frame/sample authority.  M5 adds the
cross-field checks that JSON Schema cannot express and keeps source routing in
a separate dynamic-audio manifest because v2 audio events intentionally do
not contain ``source_id``.  All builders in this module are pure: they return
detached JSON-compatible values and never write files.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import canonical_json_sha256, load_json


TIMELINE_SCHEMA = "avengine_authoritative_timeline_v2"
EPISODE_REQUEST_SCHEMA = "avengine_m5_episode_request_v1"
DYNAMIC_AUDIO_MANIFEST_SCHEMA = "avengine_m5_dynamic_audio_render_manifest_v1"

TIME_BASE_HZ = 48_000
DURATION_TICKS = 240_000
VIDEO_FPS_NUM = 15
VIDEO_FPS_DEN = 1
FRAME_COUNT = 75
TICKS_PER_FRAME = 3_200
AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_SAMPLE_COUNT = 80_000
TICKS_PER_SAMPLE = 3
FORMAL_VIEW_IDS = ("view0",)

VISUAL_VOCAL_ARTICULATION = {
    "mode": "disabled_for_shortcut_control",
    "mouth_motion_present": False,
}

FOA_AUTHORITY: dict[str, Any] = {
    "format_id": "rlr_foa_acn_n3d_world_v1",
    "ambisonic_order": 1,
    "channel_count": 4,
    "raw_channel_order": ["W", "Y", "Z", "X"],
    "acn_indices": [0, 1, 2, 3],
    "normalization": "N3D",
    "coordinate_frame": "avengine_world",
    "handedness": "right",
    "axes": {"right": "+X", "up": "+Y", "back": "+Z", "forward": "-Z"},
    "raw_array_layout": "channel_major_[channels,samples]",
    "dtype": "float32_le",
}

FROZEN_COUNTERFACTUAL_FIELDS = (
    "timeline.video",
    "timeline.actors",
    "timeline.frames",
    "timeline.audio_events_except_audio_asset_sha256",
    "visual_vocal_articulation",
    "listener",
    "actor_source_event_ids",
)
ALLOWED_COUNTERFACTUAL_FIELDS = (
    "request.events[*].dry_audio_asset_sha256",
    "timeline.audio_events[*].audio_asset_sha256",
    "dynamic_audio_render_manifest.source_routes[*].dry_audio_asset_sha256",
)
DERIVED_COUNTERFACTUAL_FIELDS = (
    "request.request_content_sha256",
    "dynamic_audio_render_manifest.timeline_content_sha256",
    "dynamic_audio_render_manifest.manifest_content_sha256",
)

_SCHEMA_FILES = {
    TIMELINE_SCHEMA: "avengine_timeline_v2.schema.json",
    EPISODE_REQUEST_SCHEMA: "m5_episode_request_v1.schema.json",
    DYNAMIC_AUDIO_MANIFEST_SCHEMA: (
        "m5_dynamic_audio_render_manifest_v1.schema.json"
    ),
}
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class M5TimelineError(ValueError):
    """One or more exact-timeline or counterfactual invariants failed."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def sample_boundary(frame_boundary_index: int) -> int:
    """Return exact audio boundary ``B(f)=(3200*f+1)//3`` for ``0 <= f <= 75``."""

    if (
        isinstance(frame_boundary_index, bool)
        or not isinstance(frame_boundary_index, int)
        or not 0 <= frame_boundary_index <= FRAME_COUNT
    ):
        raise ValueError("frame_boundary_index must be an integer in 0..75")
    return (TICKS_PER_FRAME * frame_boundary_index + 1) // TICKS_PER_SAMPLE


def frame_sample_interval(frame_index: int) -> tuple[int, int]:
    """Return the adjacent, gap-free audio interval owned by one video frame."""

    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or not 0 <= frame_index < FRAME_COUNT
    ):
        raise ValueError("frame_index must be an integer in 0..74")
    return sample_boundary(frame_index), sample_boundary(frame_index + 1)


def _schema_path(schema_name: str) -> Path:
    try:
        filename = _SCHEMA_FILES[schema_name]
    except KeyError as exc:
        raise ValueError(f"unknown M5 schema: {schema_name!r}") from exc
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine schema is unavailable: {filename}")
    return path


def json_schema_errors(value: Any, schema_name: str) -> list[str]:
    schema = load_json(_schema_path(schema_name))
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def _content_hash_errors(value: Mapping[str, Any], field: str) -> list[str]:
    try:
        actual = canonical_json_sha256(
            {key: item for key, item in value.items() if key != field}
        )
    except (TypeError, ValueError) as exc:
        return [f"{field} cannot be recomputed: {exc}"]
    if value.get(field) != actual:
        return [f"{field} does not match canonical document content"]
    return []


def _dedupe(errors: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(errors))


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        return []
    return list(value)


def _stable_ids(
    values: Sequence[Mapping[str, Any]], field: str, owner: str, errors: list[str]
) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(values):
        item = value.get(field)
        if not isinstance(item, str) or not _STABLE_ID.fullmatch(item):
            errors.append(f"{owner}[{index}].{field} must be a stable ID")
        else:
            result.append(item)
    if len(result) != len(set(result)):
        errors.append(f"{owner} {field} values must be unique")
    return result


def validate_episode_request(request: Mapping[str, Any]) -> list[str]:
    """Validate the structural and cross-reference M5 episode request contract."""

    if not isinstance(request, Mapping):
        return ["episode request must be a mapping"]
    errors = json_schema_errors(request, EPISODE_REQUEST_SCHEMA)
    if not _all_numbers_finite(request):
        errors.append("episode request contains a non-finite number")
    errors.extend(_content_hash_errors(request, "request_content_sha256"))

    actors = _mapping_list(request.get("actors"))
    sources = _mapping_list(request.get("sources"))
    events = _mapping_list(request.get("events"))
    actor_ids = _stable_ids(actors, "actor_id", "actors", errors)
    source_ids = _stable_ids(sources, "source_id", "sources", errors)
    event_ids = _stable_ids(events, "event_id", "events", errors)

    if len(actors) != 2:
        errors.append("M5 canary requires exactly two actors")
    if len(sources) != 2:
        errors.append("M5 canary requires exactly two sources")
    if len(events) != 2:
        errors.append("M5 canary requires exactly two audio events")
    if actor_ids and actor_ids != sorted(actor_ids, key=lambda item: item.encode("ascii")):
        errors.append("actors must use canonical stable-ID order")
    if source_ids and source_ids != sorted(source_ids, key=lambda item: item.encode("ascii")):
        errors.append("sources must use canonical stable-ID order")

    source_actor_ids = [source.get("actor_id") for source in sources]
    if len(actor_ids) == 2 and source_actor_ids != actor_ids:
        errors.append("sources must bind one-to-one to actors in canonical order")
    source_by_id = {
        source.get("source_id"): source
        for source in sources
        if isinstance(source.get("source_id"), str)
    }

    event_actor_ids = [event.get("actor_id") for event in events]
    event_source_ids = [event.get("source_id") for event in events]
    if len(actor_ids) == 2 and event_actor_ids != actor_ids:
        errors.append("events must bind one-to-one to actors in canonical order")
    if len(source_ids) == 2 and event_source_ids != source_ids:
        errors.append("events must bind one-to-one to sources in canonical order")
    for index, event in enumerate(events):
        raw_source_id = event.get("source_id")
        source = source_by_id.get(raw_source_id) if isinstance(raw_source_id, str) else None
        if source is None:
            errors.append(f"events[{index}].source_id does not resolve")
            continue
        for field in ("actor_id", "emitter_bone", "emitter_path_sha256"):
            if event.get(field) != source.get(field):
                errors.append(f"events[{index}].{field} differs from its source")

    intervals = [
        (event.get("start_sample"), event.get("end_sample")) for event in events
    ]
    if len(intervals) == 2 and intervals[0] != intervals[1]:
        errors.append("the two source events must have exactly the same interval")
    for index, (start, end) in enumerate(intervals):
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= AUDIO_SAMPLE_COUNT
        ):
            errors.append(f"events[{index}] must satisfy 0 <= start_sample < end_sample <= 80000")
    if intervals and isinstance(intervals[0][0], int) and isinstance(intervals[0][1], int):
        start, end = intervals[0]
        if not any(start <= sample_boundary(frame) < end for frame in range(FRAME_COUNT)):
            errors.append("the simultaneous event interval must be active at a video PTS")

    dry_hashes = [event.get("dry_audio_asset_sha256") for event in events]
    if len(dry_hashes) == 2 and dry_hashes[0] == dry_hashes[1]:
        errors.append("counterfactual dry audio assets must have distinct SHA-256 values")
    if event_ids and len(event_ids) == 2 and event_ids != sorted(
        event_ids, key=lambda item: item.encode("ascii")
    ):
        errors.append("events must use canonical stable-ID order")

    if request.get("visual_vocal_articulation") != VISUAL_VOCAL_ARTICULATION:
        errors.append(
            "visual vocal articulation must be disabled_for_shortcut_control"
        )
    profile = request.get("timeline_profile")
    audio = profile.get("audio") if isinstance(profile, Mapping) else None
    if not isinstance(audio, Mapping) or audio.get("authority") != FOA_AUTHORITY:
        errors.append("authoritative audio must be four-channel ACN/N3D FOA")
    return _dedupe(errors)


def _timeline_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "actor_id": event["actor_id"],
        "event_type": event["event_type"],
        "start_sample": event["start_sample"],
        "end_sample": event["end_sample"],
        "emitter_bone": event["emitter_bone"],
        "emitter_path_sha256": event["emitter_path_sha256"],
        "audio_asset_sha256": event["dry_audio_asset_sha256"],
        "semantic_sync_required": event["semantic_sync_required"],
    }


def _request_with_events(
    request: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = deepcopy(dict(request))
    result["events"] = deepcopy(list(events))
    result.pop("request_content_sha256", None)
    result["request_content_sha256"] = canonical_json_sha256(result)
    return result


def _copy_actor_state(
    value: Mapping[str, Any], *, vocalizing: bool
) -> dict[str, Any]:
    return {
        "actor_id": value["actor_id"],
        "root_transform": deepcopy(value["root_transform"]),
        "action_id": value["action_id"],
        "action_time_ticks": value["action_time_ticks"],
        "action_phase": value["action_phase"],
        "pose_hash": value["pose_hash"],
        "contacts": deepcopy(value["contacts"]),
        "mouth_state": {"open_ratio": 0.0, "vocalizing": vocalizing},
    }


def build_timeline(
    request: Mapping[str, Any],
    visual_frames: Sequence[Mapping[str, Any]],
    *,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one exact v2 timeline from fixed visual states and an M5 request.

    ``visual_frames`` contains exactly 75 objects with ``actor_states`` and a
    sole ``view_pose_hashes.view0``.  Timing and mouth labels are derived here;
    callers cannot supply an independent frame/sample clock.
    """

    if not isinstance(request, Mapping):
        raise M5TimelineError(["episode request must be a mapping"])
    variant_request = (
        deepcopy(dict(request))
        if events is None
        else _request_with_events(request, events)
    )
    request_errors = validate_episode_request(variant_request)
    if request_errors:
        raise M5TimelineError(request_errors)
    if isinstance(visual_frames, (str, bytes)) or len(visual_frames) != FRAME_COUNT:
        raise M5TimelineError(["visual_frames must contain exactly 75 frames"])

    actor_declarations = deepcopy(variant_request["actors"])
    actor_ids = [actor["actor_id"] for actor in actor_declarations]
    timeline_events = [_timeline_event(event) for event in variant_request["events"]]
    events_by_actor: dict[str, list[dict[str, Any]]] = {actor_id: [] for actor_id in actor_ids}
    for event in timeline_events:
        events_by_actor[event["actor_id"]].append(event)

    frames: list[dict[str, Any]] = []
    build_errors: list[str] = []
    for frame_index, raw_frame in enumerate(visual_frames):
        if not isinstance(raw_frame, Mapping):
            build_errors.append(f"visual_frames[{frame_index}] must be a mapping")
            continue
        if set(raw_frame) != {"actor_states", "view_pose_hashes"}:
            build_errors.append(
                f"visual_frames[{frame_index}] must contain only actor_states and view_pose_hashes"
            )
        raw_states = _mapping_list(raw_frame.get("actor_states"))
        state_ids = [state.get("actor_id") for state in raw_states]
        if state_ids != actor_ids:
            build_errors.append(
                f"visual_frames[{frame_index}].actor_states must match canonical actor order"
            )
            continue
        view_hashes = raw_frame.get("view_pose_hashes")
        if not isinstance(view_hashes, Mapping) or set(view_hashes) != {"view0"}:
            build_errors.append(
                f"visual_frames[{frame_index}].view_pose_hashes must contain only view0"
            )
            continue
        for actor_index, state in enumerate(raw_states):
            supplied_mouth = state.get("mouth_state")
            if isinstance(supplied_mouth, Mapping) and supplied_mouth.get("open_ratio") != 0.0:
                build_errors.append(
                    f"visual_frames[{frame_index}].actor_states[{actor_index}] has mouth motion"
                )
        start_sample, end_sample = frame_sample_interval(frame_index)
        try:
            actor_states = [
                _copy_actor_state(
                    state,
                    vocalizing=any(
                        event["start_sample"] <= start_sample < event["end_sample"]
                        for event in events_by_actor[state["actor_id"]]
                    ),
                )
                for state in raw_states
            ]
        except KeyError as exc:
            build_errors.append(
                f"visual_frames[{frame_index}] actor state lacks {exc.args[0]}"
            )
            continue
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "sample_start": start_sample,
                "sample_end": end_sample,
                "actor_states": actor_states,
                "view_pose_hashes": deepcopy(dict(view_hashes)),
            }
        )
    if build_errors:
        raise M5TimelineError(_dedupe(build_errors))

    timeline = {
        "schema": TIMELINE_SCHEMA,
        "time_base_hz": TIME_BASE_HZ,
        "duration_ticks": DURATION_TICKS,
        "video": {
            "fps_num": VIDEO_FPS_NUM,
            "fps_den": VIDEO_FPS_DEN,
            "frame_count": FRAME_COUNT,
            "ticks_per_frame": TICKS_PER_FRAME,
            "view_ids": list(FORMAL_VIEW_IDS),
        },
        "audio": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "sample_count": AUDIO_SAMPLE_COUNT,
            "ticks_per_sample": TICKS_PER_SAMPLE,
            "channel_count": 4,
        },
        "actors": actor_declarations,
        "frames": frames,
        "audio_events": timeline_events,
    }
    errors = validate_timeline_semantics(timeline, episode_request=variant_request)
    if errors:
        raise M5TimelineError(errors)
    return timeline


build_authoritative_timeline = build_timeline


def validate_timeline_semantics(
    timeline: Mapping[str, Any],
    *,
    episode_request: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return every v2/M5 cross-field error without mutating the document."""

    if not isinstance(timeline, Mapping):
        return ["timeline must be a mapping"]
    errors = json_schema_errors(timeline, TIMELINE_SCHEMA)
    if not _all_numbers_finite(timeline):
        errors.append("timeline contains a non-finite number")

    if timeline.get("time_base_hz") != TIME_BASE_HZ:
        errors.append("timeline time_base_hz must equal 48000")
    if timeline.get("duration_ticks") != DURATION_TICKS:
        errors.append("timeline duration_ticks must equal 240000")
    video = timeline.get("video")
    if not isinstance(video, Mapping) or video.get("view_ids") != ["view0"]:
        errors.append("video.view_ids must equal exactly ['view0']")
    audio = timeline.get("audio")
    if not isinstance(audio, Mapping) or audio.get("channel_count") != 4:
        errors.append("timeline authoritative audio must have four FOA channels")

    actors = _mapping_list(timeline.get("actors"))
    actor_ids = _stable_ids(actors, "actor_id", "timeline.actors", errors)
    if len(actors) != 2:
        errors.append("timeline must declare exactly two actors")
    if actor_ids and actor_ids != sorted(actor_ids, key=lambda item: item.encode("ascii")):
        errors.append("timeline actors must use canonical stable-ID order")

    events = _mapping_list(timeline.get("audio_events"))
    event_ids = _stable_ids(events, "event_id", "timeline.audio_events", errors)
    if len(events) != 2:
        errors.append("timeline must contain exactly two audio events")
    event_actor_ids = [event.get("actor_id") for event in events]
    if len(actor_ids) == 2 and event_actor_ids != actor_ids:
        errors.append("timeline events must bind one-to-one to canonical actors")
    intervals = [(event.get("start_sample"), event.get("end_sample")) for event in events]
    if len(intervals) == 2 and intervals[0] != intervals[1]:
        errors.append("timeline source events must start and end together")
    if event_ids and len(event_ids) == 2 and event_ids != sorted(
        event_ids, key=lambda item: item.encode("ascii")
    ):
        errors.append("timeline events must use canonical stable-ID order")
    for index, event in enumerate(events):
        start = event.get("start_sample")
        end = event.get("end_sample")
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= AUDIO_SAMPLE_COUNT
        ):
            errors.append(
                f"timeline.audio_events[{index}] must satisfy start_sample < end_sample"
            )
        if event.get("actor_id") not in actor_ids:
            errors.append(f"timeline.audio_events[{index}].actor_id does not resolve")

    frames = _mapping_list(timeline.get("frames"))
    if len(frames) != FRAME_COUNT:
        errors.append("timeline must contain exactly 75 frames")
    simultaneous_vocal_frame = False
    events_by_actor: dict[str, list[Mapping[str, Any]]] = {
        actor_id: [] for actor_id in actor_ids
    }
    for event in events:
        actor_id = event.get("actor_id")
        if isinstance(actor_id, str) and actor_id in events_by_actor:
            events_by_actor[actor_id].append(event)
    for frame_index, frame in enumerate(frames):
        if frame_index >= FRAME_COUNT:
            errors.append(f"frames[{frame_index}] exceeds the exact 75-frame timeline")
            continue
        expected_start, expected_end = frame_sample_interval(frame_index)
        if frame.get("frame_index") != frame_index:
            errors.append(f"frames[{frame_index}].frame_index must equal its array index")
        if frame.get("pts_ticks") != frame_index * TICKS_PER_FRAME:
            errors.append(f"frames[{frame_index}].pts_ticks is not exact")
        if frame.get("sample_start") != expected_start:
            errors.append(f"frames[{frame_index}].sample_start must equal B({frame_index})")
        if frame.get("sample_end") != expected_end:
            errors.append(f"frames[{frame_index}].sample_end must equal B({frame_index + 1})")
        view_hashes = frame.get("view_pose_hashes")
        if not isinstance(view_hashes, Mapping) or set(view_hashes) != {"view0"}:
            errors.append(f"frames[{frame_index}].view_pose_hashes must contain only view0")
        states = _mapping_list(frame.get("actor_states"))
        state_ids = [state.get("actor_id") for state in states]
        if state_ids != actor_ids:
            errors.append(f"frames[{frame_index}].actor_states does not match actors")
        vocal_flags: list[bool] = []
        for state_index, state in enumerate(states):
            mouth = state.get("mouth_state")
            if not isinstance(mouth, Mapping) or mouth.get("open_ratio") != 0.0:
                errors.append(
                    f"frames[{frame_index}].actor_states[{state_index}] mouth open_ratio must be 0"
                )
                continue
            actor_id = state.get("actor_id")
            expected_vocalizing = any(
                isinstance(event.get("start_sample"), int)
                and isinstance(event.get("end_sample"), int)
                and event["start_sample"] <= expected_start < event["end_sample"]
                for event in (
                    events_by_actor.get(actor_id, [])
                    if isinstance(actor_id, str)
                    else []
                )
            )
            if mouth.get("vocalizing") is not expected_vocalizing:
                errors.append(
                    f"frames[{frame_index}].actor_states[{state_index}] vocalizing disagrees with events"
                )
            vocal_flags.append(mouth.get("vocalizing") is True)
        if len(vocal_flags) == 2 and all(vocal_flags):
            simultaneous_vocal_frame = True
    if len(events) == 2 and not simultaneous_vocal_frame:
        errors.append("the two actors must vocalize simultaneously at a video PTS")
    if frames and frames[0].get("sample_start") != 0:
        errors.append("the first frame audio interval must start at sample 0")
    if frames and frames[-1].get("sample_end") != AUDIO_SAMPLE_COUNT:
        errors.append("the final frame audio interval must end at sample 80000")

    if episode_request is not None:
        request_errors = validate_episode_request(episode_request)
        errors.extend(f"episode request: {error}" for error in request_errors)
        if not request_errors:
            if actors != episode_request["actors"]:
                errors.append("timeline actors differ from the episode request")
            requested_events = [_timeline_event(event) for event in episode_request["events"]]
            if events != requested_events:
                errors.append("timeline audio events differ from the episode request")
            if episode_request["visual_vocal_articulation"] != VISUAL_VOCAL_ARTICULATION:
                errors.append("episode request does not declare disabled mouth articulation")
    return _dedupe(errors)


def _visual_timeline_projection(timeline: Mapping[str, Any]) -> dict[str, Any]:
    frames = _mapping_list(timeline.get("frames"))
    return {
        "video": deepcopy(timeline.get("video")),
        "actors": deepcopy(timeline.get("actors")),
        "frames": [
            {
                "frame_index": frame.get("frame_index"),
                "pts_ticks": frame.get("pts_ticks"),
                "actor_states": deepcopy(frame.get("actor_states")),
                "view_pose_hashes": deepcopy(frame.get("view_pose_hashes")),
            }
            for frame in frames
        ],
    }


def _derived_stable_id(prefix: str, value: str) -> str:
    candidate = f"{prefix}.{value}"
    if len(candidate) <= 128 and _STABLE_ID.fullmatch(candidate):
        return candidate
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}.{digest}"


def _build_dynamic_audio_manifest(
    request: Mapping[str, Any], timeline: Mapping[str, Any]
) -> dict[str, Any]:
    sources = {source["source_id"]: source for source in request["sources"]}
    routes = []
    for event in request["events"]:
        source = sources[event["source_id"]]
        routes.append(
            {
                "route_id": _derived_stable_id("route", event["source_id"]),
                "actor_id": event["actor_id"],
                "source_id": event["source_id"],
                "event_id": event["event_id"],
                "emitter_path_sha256": source["emitter_path_sha256"],
                "dry_audio_asset_sha256": event["dry_audio_asset_sha256"],
                "start_sample": event["start_sample"],
                "end_sample": event["end_sample"],
            }
        )
    manifest = {
        "schema": DYNAMIC_AUDIO_MANIFEST_SCHEMA,
        "manifest_id": _derived_stable_id(
            "dynamic_audio", request["counterfactual_pair_id"]
        ),
        "request_id": request["request_id"],
        "timeline_content_sha256": canonical_json_sha256(timeline),
        "visual_timeline_sha256": canonical_json_sha256(
            _visual_timeline_projection(timeline)
        ),
        "time_base_hz": TIME_BASE_HZ,
        "duration_ticks": DURATION_TICKS,
        "frame_count": FRAME_COUNT,
        "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
        "sample_count": AUDIO_SAMPLE_COUNT,
        "frame_sample_boundary": {
            "algorithm": "nearest_integer_tick_boundary_v1",
            "formula": "B(f)=(3200*f+1)//3",
        },
        "authority": deepcopy(FOA_AUTHORITY),
        "listener_id": request["listener"]["listener_id"],
        "canonical_source_order": [source["source_id"] for source in request["sources"]],
        "source_routes": routes,
        "render_policy": {
            "source_pose_evaluation": "timeline_frame_fixed_state",
            "rir_application": "raised_cosine_source_time_partition_v1",
            "accumulation": "independent_stems_then_linear_sum",
            "tail_policy": "retain_full_tail_then_crop_half_open_0_80000",
            "normalization_policy": "forbidden",
            "limiter_policy": "forbidden",
        },
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_dynamic_audio_render_manifest(
    manifest: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
    timeline: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate exact timing, FOA authority and named two-source routing."""

    if not isinstance(manifest, Mapping):
        return ["dynamic audio render manifest must be a mapping"]
    errors = json_schema_errors(manifest, DYNAMIC_AUDIO_MANIFEST_SCHEMA)
    if not _all_numbers_finite(manifest):
        errors.append("dynamic audio render manifest contains a non-finite number")
    errors.extend(_content_hash_errors(manifest, "manifest_content_sha256"))
    routes = _mapping_list(manifest.get("source_routes"))
    route_ids = _stable_ids(routes, "route_id", "source_routes", errors)
    source_ids = _stable_ids(routes, "source_id", "source_routes", errors)
    event_ids = _stable_ids(routes, "event_id", "source_routes", errors)
    if len(routes) != 2:
        errors.append("dynamic audio manifest must contain exactly two source routes")
    if source_ids and source_ids != manifest.get("canonical_source_order"):
        errors.append("source routes do not match canonical_source_order")
    intervals = [(route.get("start_sample"), route.get("end_sample")) for route in routes]
    if len(intervals) == 2 and intervals[0] != intervals[1]:
        errors.append("dynamic source routes must be active over the same interval")
    if len(route_ids) == 2 and len(event_ids) != 2:
        errors.append("every source route must retain one stable event ID")
    if manifest.get("authority") != FOA_AUTHORITY:
        errors.append("dynamic render authority must be four-channel ACN/N3D FOA")
    if timeline is not None:
        if manifest.get("timeline_content_sha256") != canonical_json_sha256(timeline):
            errors.append("timeline_content_sha256 does not bind the supplied timeline")
        visual_hash = canonical_json_sha256(_visual_timeline_projection(timeline))
        if manifest.get("visual_timeline_sha256") != visual_hash:
            errors.append("visual_timeline_sha256 does not bind the visual timeline")
    if request is not None:
        request_errors = validate_episode_request(request)
        errors.extend(f"episode request: {error}" for error in request_errors)
        if not request_errors:
            if manifest.get("request_id") != request["request_id"]:
                errors.append("dynamic manifest request_id differs from request")
            expected = [
                (
                    event["actor_id"],
                    event["source_id"],
                    event["event_id"],
                    event["emitter_path_sha256"],
                    event["dry_audio_asset_sha256"],
                    event["start_sample"],
                    event["end_sample"],
                )
                for event in request["events"]
            ]
            actual = [
                (
                    route.get("actor_id"),
                    route.get("source_id"),
                    route.get("event_id"),
                    route.get("emitter_path_sha256"),
                    route.get("dry_audio_asset_sha256"),
                    route.get("start_sample"),
                    route.get("end_sample"),
                )
                for route in routes
            ]
            if actual != expected:
                errors.append("dynamic source routes differ from the episode request")
    return _dedupe(errors)


def _swap_request_dry_routes(request: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(request))
    first = result["events"][0]["dry_audio_asset_sha256"]
    second = result["events"][1]["dry_audio_asset_sha256"]
    result["events"][0]["dry_audio_asset_sha256"] = second
    result["events"][1]["dry_audio_asset_sha256"] = first
    result.pop("request_content_sha256", None)
    result["request_content_sha256"] = canonical_json_sha256(result)
    return result


def _difference_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_difference_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        if len(left) != len(right):
            paths.append(f"{prefix}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                _difference_paths(left_item, right_item, f"{prefix}[{index}]")
            )
        return paths
    return [] if left == right else [prefix]


_ALLOWED_DIFFERENCE_PATTERNS = (
    re.compile(r"^request\.events\[\d+\]\.dry_audio_asset_sha256$"),
    re.compile(r"^timeline\.audio_events\[\d+\]\.audio_asset_sha256$"),
    re.compile(
        r"^dynamic_audio_render_manifest\.source_routes\[\d+\]\.dry_audio_asset_sha256$"
    ),
)
_DERIVED_DIFFERENCE_PATHS = {
    "request.request_content_sha256",
    "dynamic_audio_render_manifest.timeline_content_sha256",
    "dynamic_audio_render_manifest.manifest_content_sha256",
}


def _matches_allowed_difference(path: str) -> bool:
    return any(pattern.fullmatch(path) for pattern in _ALLOWED_DIFFERENCE_PATTERNS)


def _variant_lineage_errors(episode: Mapping[str, Any], label: str) -> list[str]:
    request = episode.get("request")
    timeline = episode.get("timeline")
    manifest = episode.get("dynamic_audio_render_manifest")
    if not all(isinstance(value, Mapping) for value in (request, timeline, manifest)):
        return [f"episode {label} lacks request/timeline/dynamic manifest mappings"]
    request_events = _mapping_list(request.get("events"))
    timeline_events = _mapping_list(timeline.get("audio_events"))
    routes = _mapping_list(manifest.get("source_routes"))
    if not (len(request_events) == len(timeline_events) == len(routes) == 2):
        return [f"episode {label} lineage does not contain exactly two routes"]
    errors: list[str] = []
    for index, (event, timeline_event, route) in enumerate(
        zip(request_events, timeline_events, routes)
    ):
        dry = event.get("dry_audio_asset_sha256")
        if timeline_event.get("audio_asset_sha256") != dry:
            errors.append(f"episode {label} route {index} timeline dry SHA differs")
        if route.get("dry_audio_asset_sha256") != dry:
            errors.append(f"episode {label} route {index} manifest dry SHA differs")
        for field in ("actor_id", "event_id"):
            if event.get(field) != timeline_event.get(field) or event.get(field) != route.get(field):
                errors.append(f"episode {label} route {index} {field} lineage differs")
        if event.get("source_id") != route.get("source_id"):
            errors.append(f"episode {label} route {index} source_id lineage differs")
    return errors


def compare_counterfactual_pair(
    episode_a: Mapping[str, Any],
    episode_b: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove visual invariance and reject every non-whitelisted A/B change.

    The one-argument form accepts the object returned by
    :func:`build_counterfactual_pair`; the two-argument form compares its two
    episode payloads directly.
    """

    if episode_b is None:
        episodes = episode_a.get("episodes") if isinstance(episode_a, Mapping) else None
        if not isinstance(episodes, Mapping):
            return {
                "schema": "avengine_m5_counterfactual_comparison_v1",
                "status": "fail",
                "errors": ["pair must contain episodes.A and episodes.B"],
                "visual_invariant": False,
                "dry_audio_source_routing_swap": False,
                "allowed_difference_whitelist": list(ALLOWED_COUNTERFACTUAL_FIELDS),
                "observed_allowed_differences": [],
                "observed_derived_differences": [],
                "unexpected_differences": [],
            }
        episode_a = episodes.get("A")  # type: ignore[assignment]
        episode_b = episodes.get("B")  # type: ignore[assignment]
    if not isinstance(episode_a, Mapping) or not isinstance(episode_b, Mapping):
        return {
            "schema": "avengine_m5_counterfactual_comparison_v1",
            "status": "fail",
            "errors": ["both counterfactual episodes must be mappings"],
            "visual_invariant": False,
            "dry_audio_source_routing_swap": False,
            "allowed_difference_whitelist": list(ALLOWED_COUNTERFACTUAL_FIELDS),
            "observed_allowed_differences": [],
            "observed_derived_differences": [],
            "unexpected_differences": [],
        }

    errors: list[str] = []
    for label, episode in (("A", episode_a), ("B", episode_b)):
        request = episode.get("request")
        timeline = episode.get("timeline")
        manifest = episode.get("dynamic_audio_render_manifest")
        if not isinstance(request, Mapping):
            errors.append(f"episode {label} request is missing")
            continue
        if not isinstance(timeline, Mapping):
            errors.append(f"episode {label} timeline is missing")
            continue
        if not isinstance(manifest, Mapping):
            errors.append(f"episode {label} dynamic audio manifest is missing")
            continue
        errors.extend(
            f"episode {label}: {error}"
            for error in validate_timeline_semantics(
                timeline, episode_request=request
            )
        )
        errors.extend(
            f"episode {label}: {error}"
            for error in validate_dynamic_audio_render_manifest(
                manifest, request=request, timeline=timeline
            )
        )
        errors.extend(_variant_lineage_errors(episode, label))

    differences = _difference_paths(episode_a, episode_b)
    observed_allowed = sorted(
        path for path in differences if _matches_allowed_difference(path)
    )
    observed_derived = sorted(
        path for path in differences if path in _DERIVED_DIFFERENCE_PATHS
    )
    unexpected = sorted(
        path
        for path in differences
        if not _matches_allowed_difference(path)
        and path not in _DERIVED_DIFFERENCE_PATHS
    )
    if unexpected:
        errors.append("counterfactual contains non-whitelisted differences")

    timeline_a = episode_a.get("timeline")
    timeline_b = episode_b.get("timeline")
    visual_a = (
        _visual_timeline_projection(timeline_a)
        if isinstance(timeline_a, Mapping)
        else None
    )
    visual_b = (
        _visual_timeline_projection(timeline_b)
        if isinstance(timeline_b, Mapping)
        else None
    )
    visual_hash_a = canonical_json_sha256(visual_a) if visual_a is not None else None
    visual_hash_b = canonical_json_sha256(visual_b) if visual_b is not None else None
    declared_visual_a = episode_a.get("visual_state_sha256")
    declared_visual_b = episode_b.get("visual_state_sha256")
    visual_invariant = bool(
        visual_a == visual_b
        and visual_hash_a == declared_visual_a
        and visual_hash_b == declared_visual_b
    )
    if not visual_invariant:
        errors.append("counterfactual visual timeline state is not identical")

    def route_hashes(episode: Mapping[str, Any]) -> list[Any]:
        manifest = episode.get("dynamic_audio_render_manifest")
        routes = _mapping_list(
            manifest.get("source_routes") if isinstance(manifest, Mapping) else None
        )
        return [route.get("dry_audio_asset_sha256") for route in routes]

    hashes_a = route_hashes(episode_a)
    hashes_b = route_hashes(episode_b)
    routing_swap = bool(
        len(hashes_a) == len(hashes_b) == 2
        and hashes_a[0] != hashes_a[1]
        and hashes_b == list(reversed(hashes_a))
    )
    if not routing_swap:
        errors.append("B must exactly swap A's two dry-audio source routes")
    if not observed_allowed:
        errors.append("counterfactual pair contains no causal dry-audio change")

    errors = _dedupe(errors)
    return {
        "schema": "avengine_m5_counterfactual_comparison_v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "visual_invariant": visual_invariant,
        "visual_timeline_sha256": visual_hash_a if visual_invariant else None,
        "dry_audio_source_routing_swap": routing_swap,
        "allowed_difference_whitelist": list(ALLOWED_COUNTERFACTUAL_FIELDS),
        "derived_difference_whitelist": list(DERIVED_COUNTERFACTUAL_FIELDS),
        "observed_allowed_differences": observed_allowed,
        "observed_derived_differences": observed_derived,
        "unexpected_differences": unexpected,
    }


def build_counterfactual_pair(
    request: Mapping[str, Any],
    visual_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build A/B episodes whose only causal change is dry-audio source routing."""

    request_errors = validate_episode_request(request)
    if request_errors:
        raise M5TimelineError(request_errors)
    request_a = deepcopy(dict(request))
    request_b = _swap_request_dry_routes(request)
    timeline_a = build_timeline(request_a, visual_frames)
    timeline_b = build_timeline(request_b, visual_frames)
    manifest_a = _build_dynamic_audio_manifest(request_a, timeline_a)
    manifest_b = _build_dynamic_audio_manifest(request_b, timeline_b)
    visual_hash_a = canonical_json_sha256(_visual_timeline_projection(timeline_a))
    visual_hash_b = canonical_json_sha256(_visual_timeline_projection(timeline_b))
    pair: dict[str, Any] = {
        "schema": "avengine_m5_counterfactual_pair_v1",
        "counterfactual_pair_id": request["counterfactual_pair_id"],
        "frozen_fields": list(FROZEN_COUNTERFACTUAL_FIELDS),
        "allowed_changed_fields": list(ALLOWED_COUNTERFACTUAL_FIELDS),
        "derived_changed_fields": list(DERIVED_COUNTERFACTUAL_FIELDS),
        "episodes": {
            "A": {
                "request": request_a,
                "timeline": timeline_a,
                "dynamic_audio_render_manifest": manifest_a,
                "visual_state_sha256": visual_hash_a,
            },
            "B": {
                "request": request_b,
                "timeline": timeline_b,
                "dynamic_audio_render_manifest": manifest_b,
                "visual_state_sha256": visual_hash_b,
            },
        },
    }
    comparison = compare_counterfactual_pair(pair)
    if comparison["status"] != "pass":
        raise M5TimelineError(comparison["errors"])
    pair["comparison"] = comparison
    return pair
