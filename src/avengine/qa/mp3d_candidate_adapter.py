"""Adapt planned MP3D actor tracks to the common QA candidate shape.

The adapter keeps the route's planned centres explicitly labelled and leaves
native emitter readbacks, pixel truth and rendered audio as missing inputs.
It accepts the case/room/M1 paths used by the native capture entrypoint and
small in-memory mappings for tests.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


MP3D_CANDIDATE_ADAPTER_SCHEMA = "avengine_qa_mp3d_candidate_adapter_v1"
MP3D_BACKEND_ID = "habitat_native"


class MP3DCandidateAdapterError(ValueError):
    """A planned MP3D candidate cannot be represented safely."""


def _document(value: Any, *, owner: str) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value)), None
    if value is None:
        raise MP3DCandidateAdapterError(f"{owner} is required")
    raw_path = Path(value).expanduser()
    if raw_path.is_symlink():
        raise MP3DCandidateAdapterError(f"{owner} must not be a symlink: {raw_path}")
    path = raw_path.resolve()
    if not path.is_file():
        raise MP3DCandidateAdapterError(f"{owner} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MP3DCandidateAdapterError(f"cannot read {owner}: {error}") from error
    if not isinstance(value, Mapping):
        raise MP3DCandidateAdapterError(f"{owner} must contain a JSON object")
    return deepcopy(dict(value)), path


def _optional_document(value: Any, *, owner: str) -> tuple[dict[str, Any] | None, Path | None]:
    if value is None:
        return None, None
    return _document(value, owner=owner)


def _one_of(primary: Any, alias: Any, *, owner: str) -> Any:
    if primary is not None and alias is not None:
        if isinstance(primary, (str, Path)) and isinstance(alias, (str, Path)):
            same = Path(primary).expanduser().resolve() == Path(alias).expanduser().resolve()
        else:
            same = primary == alias
        if not same:
            raise MP3DCandidateAdapterError(f"{owner} was supplied twice with different values")
    return primary if primary is not None else alias


def _path_from_case(case_path: Path | None, raw: Any, *, owner: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise MP3DCandidateAdapterError(f"{owner} must be a nonempty path")
    path = Path(raw).expanduser()
    relative_to_case = not path.is_absolute() and case_path is not None
    if relative_to_case:
        path = case_path.parent / path
    if path.is_symlink():
        raise MP3DCandidateAdapterError(f"{owner} must not be a symlink: {path}")
    resolved = path.resolve()
    if relative_to_case:
        case_root = case_path.parent.resolve()
        try:
            resolved.relative_to(case_root)
        except ValueError as error:
            raise MP3DCandidateAdapterError(
                f"{owner} escapes the case directory: {resolved}"
            ) from error
    return resolved


def _track_inputs(
    case: Mapping[str, Any],
    *,
    case_path: Path | None,
    explicit: Sequence[Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: Any = explicit
    if records is None:
        records = case.get("actor_tracks", case.get("tracks"))
    if isinstance(records, Mapping):
        records = list(records.values())
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise MP3DCandidateAdapterError("case must contain an actor_tracks list")
    tracks: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MP3DCandidateAdapterError(f"actor_tracks[{index}] must be an object")
        value: Mapping[str, Any] = record
        raw_path = record.get("track_path")
        if raw_path is not None:
            path = _path_from_case(
                case_path, raw_path, owner=f"actor_tracks[{index}].track_path"
            )
            value, _ = _document(path, owner=f"actor track {index}")
            references.append({"track_path": str(path), "source_record": deepcopy(dict(record))})
        elif isinstance(record.get("track"), Mapping):
            value = record["track"]
            references.append({"track_path": None, "source_record": deepcopy(dict(record))})
        else:
            references.append({"track_path": None, "source_record": deepcopy(dict(record))})
        tracks.append(deepcopy(dict(value)))
    if not tracks:
        raise MP3DCandidateAdapterError(
            "MP3D candidates require at least one actor track"
        )
    return tracks, references


def _actor_records(
    tracks: Sequence[Mapping[str, Any]],
    missing: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    actors: list[dict[str, Any]] = []
    frames_by_slot: dict[str, list[dict[str, Any]]] = {}
    slots: list[str] = []
    endpoints: set[str] = set()
    for index, track in enumerate(tracks):
        asset = track.get("asset")
        emitter = track.get("emitter")
        slot = track.get("source_slot_id")
        if not isinstance(slot, str) or not slot:
            slot = f"source{index + 1}"
            missing["selection"].append(f"selection.actors[{index}].source_slot_id")
        if slot in slots:
            raise MP3DCandidateAdapterError(f"duplicate actor source slot: {slot!r}")
        slots.append(slot)
        endpoint = track.get("source_endpoint_id")
        if not isinstance(endpoint, str) or not endpoint:
            endpoint = None
            missing["selection"].append(f"selection.actors[{index}].source_endpoint_id")
        elif endpoint in endpoints:
            raise MP3DCandidateAdapterError(f"duplicate actor source endpoint: {endpoint!r}")
        endpoints.add(endpoint) if endpoint is not None else None
        actor_id = track.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id:
            actor_id = f"{slot}_actor"
            missing["selection"].append(f"selection.actors[{index}].actor_id")
        asset_id = asset.get("asset_id") if isinstance(asset, Mapping) else None
        revision = asset.get("revision") if isinstance(asset, Mapping) else None
        if not isinstance(asset_id, str) or not asset_id:
            asset_id = None
            missing["selection"].append(f"selection.actors[{index}].asset.asset_id")
        if not isinstance(revision, str) or not revision:
            revision = None
            missing["selection"].append(f"selection.actors[{index}].asset.revision")
        semantic_id = track.get("semantic_id")
        if not isinstance(semantic_id, int) or isinstance(semantic_id, bool) or semantic_id < 0:
            semantic_id = None
            missing["selection"].append(f"selection.actors[{index}].semantic_id")
        if not isinstance(emitter, Mapping):
            emitter = None
            missing["selection"].append(f"selection.actors[{index}].emitter")
        actors.append(
            {
                "actor_id": actor_id,
                "source_slot_id": slot,
                "source_endpoint_id": endpoint,
                "asset_id": asset_id,
                "asset_revision": revision,
                "semantic_id": semantic_id,
                "emitter": deepcopy(dict(emitter)) if emitter is not None else None,
            }
        )
        raw_frames = track.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            missing["timeline"].append(f"timeline.tracks.{slot}.frames")
            frames_by_slot[slot] = []
            continue
        normalized: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(raw_frames):
            if not isinstance(frame, Mapping):
                missing["timeline"].append(f"timeline.tracks.{slot}.frames[{frame_index}]")
                continue
            normalized.append(
                {
                    "frame_index": frame.get("frame_index", frame_index),
                    "pts_ticks": frame.get("pts_ticks"),
                    "action_id": frame.get("action_id"),
                    "action_time_ticks": frame.get("action_time_ticks"),
                    "action_sample_index": frame.get("action_sample_index"),
                    "planned_route_center_m": deepcopy(frame.get("planned_route_center_m")),
                    "planned_world_from_actor": deepcopy(frame.get("planned_world_from_actor")),
                    "planned_world_from_skin_root": deepcopy(frame.get("planned_world_from_skin_root")),
                    "joint_targets": deepcopy(frame.get("joint_targets")),
                }
            )
        frames_by_slot[slot] = normalized
    return actors, frames_by_slot


def _clock(case: Mapping[str, Any], timeline: Mapping[str, Any] | None) -> dict[str, Any] | None:
    value = case.get("clock")
    if not isinstance(value, Mapping) and isinstance(timeline, Mapping):
        value = timeline.get("render", timeline.get("clock"))
    return (
        {key: deepcopy(item) for key, item in value.items()}
        if isinstance(value, Mapping)
        else None
    )


def _timeline(
    *,
    case: Mapping[str, Any],
    room: Mapping[str, Any],
    m1: Mapping[str, Any] | None,
    clock: Mapping[str, Any] | None,
    actors: Sequence[Mapping[str, Any]],
    frames_by_slot: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_index: dict[int, dict[str, Any]] = {}
    for actor in actors:
        slot = actor["source_slot_id"]
        for ordinal, frame in enumerate(frames_by_slot.get(slot, ())):
            index = frame.get("frame_index", ordinal)
            if not isinstance(index, int) or isinstance(index, bool):
                index = ordinal
            target = by_index.setdefault(
                index,
                {
                    "frame_index": index,
                    "pts_ticks": frame.get("pts_ticks"),
                    "planned_camera_pose": None,
                    "actor_states": [],
                },
            )
            target["actor_states"].append(
                {
                    "source_slot_id": slot,
                    "actor_id": actor["actor_id"],
                    "source_endpoint_id": actor["source_endpoint_id"],
                    "asset_id": actor["asset_id"],
                    "asset_revision": actor["asset_revision"],
                    "planned_route_center_m": deepcopy(frame.get("planned_route_center_m")),
                    "action_id": frame.get("action_id"),
                    "action_time_ticks": frame.get("action_time_ticks"),
                    "action_sample_index": frame.get("action_sample_index"),
                }
            )
    rig = m1.get("primary_camera_rig") if isinstance(m1, Mapping) else None
    if isinstance(rig, Mapping) and isinstance(rig.get("world_from_rig"), Mapping):
        camera = {
            "world_from_rig": deepcopy(dict(rig["world_from_rig"])),
            "hfov_degrees": (
                rig.get("shared_calibration", {}).get("hfov_degrees")
                if isinstance(rig.get("shared_calibration"), Mapping)
                else None
            ),
        }
        for frame in by_index.values():
            frame["planned_camera_pose"] = camera
    render = dict(clock or {})
    if not render and by_index:
        render["frame_count"] = len(by_index)
    return {
        "schema": "avengine_mp3d_region_planned_timeline_v1",
        "artifact_role": "planned_timeline_not_native_capture",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "backend_id": MP3D_BACKEND_ID,
        "room": {
            "room_id": room.get("room_id"),
            "house_id": case.get("region", {}).get("house_id")
            if isinstance(case.get("region"), Mapping)
            else None,
        },
        "region": deepcopy(case.get("region")),
        "route_family_id": case.get("route_family_id"),
        "motion_case": case.get("motion_case"),
        "render": render,
        "source_endpoint_ids": [actor["source_endpoint_id"] for actor in actors],
        "actors": [
            {
                key: actor.get(key)
                for key in (
                    "source_slot_id",
                    "actor_id",
                    "asset_id",
                    "asset_revision",
                    "source_endpoint_id",
                )
            }
            for actor in actors
        ],
        "frames": [by_index[index] for index in sorted(by_index)],
    }


def _fact(
    *,
    case: Mapping[str, Any],
    case_path: Path | None,
    room: Mapping[str, Any],
    room_path: Path | None,
    m1: Mapping[str, Any] | None,
    m1_path: Path | None,
    audio_program_path: Path | None,
    clock: Mapping[str, Any] | None,
    actors: Sequence[Mapping[str, Any]],
    frames_by_slot: Mapping[str, Sequence[Mapping[str, Any]]],
    audio_program: Mapping[str, Any] | None,
    missing: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    scene = room.get("scene")
    instances = [
        {
            "instance_id": actor["source_slot_id"],
            **{
                key: actor.get(key)
                for key in (
                    "actor_id",
                    "source_slot_id",
                    "asset_id",
                    "asset_revision",
                    "semantic_id",
                    "emitter",
                )
            },
        }
        for actor in actors
    ]
    tracks: dict[str, Any] = {}
    for actor in actors:
        slot = actor["source_slot_id"]
        values = frames_by_slot.get(slot, ())
        tracks[slot] = {
            "position_semantics": "planned_route_center_not_emitter_readback",
            "planned_route_center_m_by_frame": [
                deepcopy(frame.get("planned_route_center_m")) for frame in values
            ],
            "planned_world_from_skin_root_by_frame": [
                deepcopy(frame.get("planned_world_from_skin_root")) for frame in values
            ],
            "observed_emitter_position_m_by_frame": None,
            "observed_root_readback_by_frame": None,
        }
    endpoint_ids = [actor["source_endpoint_id"] for actor in actors]
    return {
        "schema": "avengine_qa_mp3d_candidate_fact_v1",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "backend_id": MP3D_BACKEND_ID,
        "runtime_consumer_status": "pending_question_facts",
        "backend_inputs": {
            "case_manifest": None if case_path is None else str(case_path),
            "room_manifest": None if room_path is None else str(room_path),
            "m1_request": None if m1_path is None else str(m1_path),
            "audio_program": (
                None
                if audio_program_path is None
                else str(audio_program_path)
            ),
        },
        "scene_id": scene.get("scene_id") if isinstance(scene, Mapping) else None,
        "room": {
            "room_id": room.get("room_id"),
            "room_kind": room.get("room_kind"),
            "room_manifest_path": None if room_path is None else str(room_path),
        },
        "case": {
            "region": deepcopy(case.get("region")),
            "route_family_id": case.get("route_family_id"),
            "motion_case": case.get("motion_case"),
        },
        "time": deepcopy(dict(clock)) if isinstance(clock, Mapping) else None,
        "listener": {
            "camera_rig": deepcopy(m1.get("primary_camera_rig"))
            if isinstance(m1, Mapping)
            else None,
            "listener": deepcopy(m1.get("listener")) if isinstance(m1, Mapping) else None,
            "authority": "declared M1 request; observed camera readback pending",
            "m1_request_path": None if m1_path is None else str(m1_path),
        },
        "instances": instances,
        "tracks": tracks,
        "visibility": {
            "status": "pending_observed_native_capture",
            "planned_route_center_is_not_pixel_truth": True,
            "pixel_truth": None,
        },
        "audio": {
            "status": "available" if audio_program is not None else "missing",
            "program_id": (
                audio_program.get("program_id")
                if isinstance(audio_program, Mapping)
                else None
            ),
            "candidate_source_endpoint_ids": (
                deepcopy(audio_program.get("candidate_source_endpoint_ids"))
                if isinstance(audio_program, Mapping)
                else endpoint_ids
            ),
            "observed_render_receipt": None,
            "mixture": None,
        },
        "missing": {key: list(values) for key, values in missing.items() if values},
    }


def adapt_mp3d_candidate(
    case_manifest_path: str | Path | Mapping[str, Any] | None = None,
    room_manifest_path: str | Path | Mapping[str, Any] | None = None,
    m1_request_path: str | Path | Mapping[str, Any] | None = None,
    *,
    case_manifest: str | Path | Mapping[str, Any] | None = None,
    room_manifest: str | Path | Mapping[str, Any] | None = None,
    m1_request: str | Path | Mapping[str, Any] | None = None,
    track_paths: Sequence[str | Path | Mapping[str, Any]] | None = None,
    planned_timeline_path: str | Path | Mapping[str, Any] | None = None,
    selection_path: str | Path | Mapping[str, Any] | None = None,
    audio_program_path: str | Path | Mapping[str, Any] | None = None,
    backend_id: str = MP3D_BACKEND_ID,
    case_path: str | Path | Mapping[str, Any] | None = None,
    room_path: str | Path | Mapping[str, Any] | None = None,
    m1_path: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt one case/track/room/M1 set without executing native code."""

    if backend_id != MP3D_BACKEND_ID:
        raise MP3DCandidateAdapterError(
            f"MP3D adapter requires backend_id={MP3D_BACKEND_ID!r}"
        )
    case_input = _one_of(
        _one_of(case_manifest_path, case_manifest, owner="case manifest"),
        case_path,
        owner="case manifest",
    )
    room_input = _one_of(
        _one_of(room_manifest_path, room_manifest, owner="room manifest"),
        room_path,
        owner="room manifest",
    )
    m1_input = _one_of(
        _one_of(m1_request_path, m1_request, owner="M1 request"),
        m1_path,
        owner="M1 request",
    )
    case, case_file = _document(case_input, owner="case manifest")
    room, room_file = _document(room_input, owner="room manifest")
    declared_m1 = case.get("m1_request_path")
    if m1_input is None and declared_m1 is not None:
        m1_input = _path_from_case(case_file, declared_m1, owner="case.m1_request_path")
    m1, m1_file = _optional_document(m1_input, owner="M1 request")
    if case.get("native_observed") is True:
        raise MP3DCandidateAdapterError("adapter expects a planned actor-track case")
    if room.get("room_kind") not in (None, MP3D_BACKEND_ID):
        raise MP3DCandidateAdapterError(
            f"MP3D room_kind must be {MP3D_BACKEND_ID!r}"
        )

    tracks, references = _track_inputs(
        case, case_path=case_file, explicit=track_paths
    )
    missing = {
        "fact": [
            "observed native emitter/root readbacks",
            "pixel visibility truth from observed semantic frames",
        ],
        "timeline": [],
        "selection": [],
        "capture_request": [],
        "audio_program": [],
    }
    actors, frames_by_slot = _actor_records(tracks, missing)
    timeline_input, timeline_file = _optional_document(
        planned_timeline_path, owner="planned timeline"
    )
    if timeline_input is None and case.get("planned_timeline_path") is not None:
        timeline_input, timeline_file = _optional_document(
            _path_from_case(
                case_file,
                case["planned_timeline_path"],
                owner="case.planned_timeline_path",
            ),
            owner="case planned timeline",
        )
    clock = _clock(case, timeline_input)
    timeline = timeline_input or _timeline(
        case=case,
        room=room,
        m1=m1,
        clock=clock,
        actors=actors,
        frames_by_slot=frames_by_slot,
    )
    if clock is None:
        clock = _clock(case, timeline)
    if clock is None:
        missing["timeline"].append("timeline.render clock")

    selection_input, selection_file = _optional_document(
        selection_path, owner="selection"
    )
    if selection_input is None:
        selection = {
            "schema": "avengine_qa_mp3d_candidate_selection_v1",
            "artifact_role": "planned_source_selection",
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "qualification_claim": False,
            "backend_id": MP3D_BACKEND_ID,
            "actors": deepcopy(actors),
        }
    else:
        selection = selection_input
        selection.setdefault("backend_id", MP3D_BACKEND_ID)

    program, program_file = _optional_document(
        audio_program_path, owner="audio program"
    )
    if program is None and case.get("audio_program_path") is not None:
        program, program_file = _optional_document(
            _path_from_case(
                case_file, case["audio_program_path"], owner="case.audio_program_path"
            ),
            owner="case audio program",
        )
    expected_ids = [actor["source_endpoint_id"] for actor in actors]
    if program is None:
        missing["audio_program"].append("audio_program")
    elif program.get("candidate_source_endpoint_ids") is None:
        missing["audio_program"].append(
            "audio_program.candidate_source_endpoint_ids"
        )
    elif program.get("candidate_source_endpoint_ids") != expected_ids:
        raise MP3DCandidateAdapterError(
            "audio program candidate endpoints differ from actor endpoint order"
        )

    if m1 is None:
        missing["capture_request"].append("m1_capture_request")
    else:
        if m1.get("room_id") is None:
            missing["capture_request"].append("capture_request.room_id")
        elif room.get("room_id") is not None and m1["room_id"] != room["room_id"]:
            raise MP3DCandidateAdapterError("M1 request room_id differs from room manifest")
        sources = m1.get("sources")
        if not isinstance(sources, list):
            missing["capture_request"].append("capture_request.sources")
        elif [item.get("source_id") for item in sources if isinstance(item, Mapping)] != expected_ids:
            raise MP3DCandidateAdapterError("M1 source order differs from actor endpoint order")
    if room.get("room_id") is None:
        missing["capture_request"].append("room_manifest.room_id")
    if not isinstance(room.get("scene"), Mapping):
        missing["capture_request"].append("room_manifest.scene")
    if not isinstance(room.get("room_kind"), str):
        missing["capture_request"].append("room_manifest.room_kind")

    fact = _fact(
        case=case,
        case_path=case_file,
        room=room,
        room_path=room_file,
        m1=m1,
        m1_path=m1_file,
        audio_program_path=program_file,
        clock=clock,
        actors=actors,
        frames_by_slot=frames_by_slot,
        audio_program=program,
        missing=missing,
    )
    missing_flat = sorted({item for values in missing.values() for item in values})
    return {
        "schema": MP3D_CANDIDATE_ADAPTER_SCHEMA,
        "status": "research_candidate",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "backend_id": MP3D_BACKEND_ID,
        "backend": {
            "backend_id": MP3D_BACKEND_ID,
            "description": "Native Habitat-Sim MP3D visual execution",
        },
        "case": {
            "path": None if case_file is None else str(case_file),
            "manifest": case,
            "track_references": references,
        },
        "room": {
            "path": None if room_file is None else str(room_file),
            "manifest": room,
        },
        "fact": fact,
        "timeline": timeline,
        "selection": selection,
        "capture_request": m1,
        "audio_program": program,
        "inputs": {
            "case_manifest": None if case_file is None else str(case_file),
            "room_manifest": None if room_file is None else str(room_file),
            "m1_request": None if m1_file is None else str(m1_file),
            "planned_timeline": None if timeline_file is None else str(timeline_file),
            "selection": None if selection_file is None else str(selection_file),
            "audio_program": None if program_file is None else str(program_file),
        },
        "missing": missing_flat,
        "missing_by_section": {key: sorted(set(value)) for key, value in missing.items()},
        "observed_requirements": {
            "frame_records": "native capture must write observed frame_records.json",
            "visual_receipt": "native capture research_receipt.json",
            "audio_receipt": "dynamic audio research_receipt.json",
        },
        "position_authority": {
            "planned": "track.frames[].planned_route_center_m",
            "observed": "native capture frame_records.frames[].source_positions_m",
            "rule": "planned centres remain expected/planned data and are never observed positions",
        },
    }


adapt_mp3d_candidate_from_paths = adapt_mp3d_candidate


__all__ = [
    "MP3D_BACKEND_ID",
    "MP3D_CANDIDATE_ADAPTER_SCHEMA",
    "MP3DCandidateAdapterError",
    "adapt_mp3d_candidate",
    "adapt_mp3d_candidate_from_paths",
]
