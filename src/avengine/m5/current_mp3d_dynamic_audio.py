"""Motion-following research audio for authored current routes.

The static ``render_current_m1_research_audio`` verb stays the frozen
baseline: its pair IRs carry one state per source, so mixes hold no motion.
This module renders the audio the visual capture actually shows instead:
captured per-frame emitter positions drive a strided keyframe grid through
the persistent-context M5.1 review renderer, and the dry buses come from an
M6 AudioProgram routing variant, so the probe sources take turns instead of
sharing one schedule. The core ``render_dynamic_research_audio`` is
room-agnostic; ``render_current_mp3d_dynamic_audio`` binds it to the current
MP3D capture record layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import sha256_file
from avengine.contracts.transforms import compose_transforms
from avengine.m1.contracts import validate_capture_request
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.m4.audio import write_float32_wav
from avengine.m4.current_m1_pair_ir import _load_simulation_request
from avengine.m5.timeline import frame_sample_interval
from avengine.m5_1.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
    render_research_review_binaural_rir_sequence,
)
from avengine.m6.audio_render import assemble_audio_program_dry_buses
from avengine.m6.sources import (
    endpoint_index,
    load_sound_asset_registry,
    load_source_endpoint_registry,
    sound_index,
)

CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA = "avengine_m5_current_mp3d_dynamic_audio_v1"
VISUAL_FRAME_RATE_HZ = 15
EPISODE_FRAME_COUNT = 75
AUDIO_SAMPLE_RATE_HZ = 16_000
EPISODE_SAMPLE_COUNT = 80_000
CLAIM_BOUNDARY = (
    "Research review audio only. Captured per-frame source positions and one "
    "AudioProgram routing variant drive strided per-state binaural RIRs; no "
    "dataset admission, qualification, or new gate is claimed."
)

# This is the explicit identity bridge for the retained current-MP3D route.
# The visual capture owns generic source/actor IDs while AudioProgram owns
# registered endpoint IDs. Keeping both sides here is intentional: deriving
# this relation from array position, numeric suffixes, or sorted IDs would make
# an axis swap look valid. The renderer verifies every field against the
# visual receipt and endpoint registry before using any trajectory.
CURRENT_MP3D_SOURCE_AXIS_BINDINGS: tuple[Mapping[str, Any], ...] = (
    {
        "axis_index": 0,
        "source_id": "source0",
        "actor_id": "actor0",
        "emitter_link_name": "beagle Xtra Mouth",
        "audio_endpoint_id": "beagle_0_muzzle",
        "endpoint_entity_instance_id": "beagle_0",
        "endpoint_emitter_anchor_id": "muzzle",
    },
    {
        "axis_index": 1,
        "source_id": "source1",
        "actor_id": "actor1",
        "emitter_link_name": "beagle Xtra Mouth",
        "audio_endpoint_id": "beagle_1_muzzle",
        "endpoint_entity_instance_id": "beagle_1",
        "endpoint_emitter_anchor_id": "muzzle",
    },
)


class CurrentMP3DDynamicAudioError(RuntimeError):
    """Raised when the dynamic research-audio contract is violated."""


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise CurrentMP3DDynamicAudioError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    return output


def _text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurrentMP3DDynamicAudioError(f"{owner} must be a non-empty string")
    return value


def _normalise_axis_bindings(
    source_axis_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(source_axis_bindings, Sequence) or isinstance(
        source_axis_bindings, (str, bytes)
    ):
        raise CurrentMP3DDynamicAudioError(
            "source_axis_bindings must be an explicit sequence"
        )
    result: list[dict[str, Any]] = []
    required_text_fields = (
        "source_id",
        "actor_id",
        "emitter_link_name",
        "audio_endpoint_id",
        "endpoint_entity_instance_id",
        "endpoint_emitter_anchor_id",
    )
    for item_index, raw in enumerate(source_axis_bindings):
        if not isinstance(raw, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"source_axis_bindings[{item_index}] must be an object"
            )
        axis_index = raw.get("axis_index")
        if isinstance(axis_index, bool) or not isinstance(axis_index, int):
            raise CurrentMP3DDynamicAudioError(
                f"source_axis_bindings[{item_index}].axis_index must be an integer"
            )
        record: dict[str, Any] = {"axis_index": axis_index}
        for field in required_text_fields:
            record[field] = _text(
                raw.get(field), owner=f"source_axis_bindings[{item_index}].{field}"
            )
        result.append(record)
    result.sort(key=lambda item: item["axis_index"])
    expected_axes = list(range(len(result)))
    axes = [item["axis_index"] for item in result]
    if axes != expected_axes:
        raise CurrentMP3DDynamicAudioError(
            "source axis indices must be one complete contiguous 0-based sequence"
        )
    for field in ("source_id", "actor_id", "audio_endpoint_id"):
        values = [str(item[field]) for item in result]
        if len(values) != len(set(values)):
            raise CurrentMP3DDynamicAudioError(f"source axis bindings repeat {field}")
    return result


def _capture_artifact(
    capture_root: Path, receipt: Mapping[str, Any], role: str
) -> Path:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CurrentMP3DDynamicAudioError(
            "visual receipt must declare its capture artifacts"
        )
    relative = _text(artifacts.get(role), owner=f"visual receipt artifacts.{role}")
    resolved = (capture_root / relative).resolve()
    if not resolved.is_relative_to(capture_root) or not resolved.is_file():
        raise CurrentMP3DDynamicAudioError(
            f"visual receipt artifact {role!r} is missing or escapes the capture"
        )
    return resolved


def _load_captured_source_evidence(
    visual_capture_dir: str | Path,
    source_axis_bindings: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[list[float]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Load explicitly named visual axes and cross-check JSON against the NPY.

    The visual receipt, not the AudioProgram candidate order, owns the array
    axis. Every axis is then joined to one explicit endpoint binding. This
    deliberately rejects swaps instead of guessing from list position.
    """

    capture_root = Path(visual_capture_dir).resolve()
    receipt_path = capture_root / "research_receipt.json"
    if not receipt_path.is_file():
        raise CurrentMP3DDynamicAudioError(
            f"visual capture is missing research_receipt.json: {receipt_path}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise CurrentMP3DDynamicAudioError("visual receipt must be an object")
    capture = receipt.get("capture")
    if (
        not isinstance(capture, Mapping)
        or capture.get("native_habitat_started") is not True
        or capture.get("frame_count") != EPISODE_FRAME_COUNT
    ):
        raise CurrentMP3DDynamicAudioError(
            "visual receipt must prove a native 75-frame Habitat capture"
        )
    actors = receipt.get("actors")
    if not isinstance(actors, Mapping):
        raise CurrentMP3DDynamicAudioError(
            "visual receipt must declare actor/source axis identity"
        )
    bindings = _normalise_axis_bindings(source_axis_bindings)
    expected_axis_values = {
        "source_ids": [item["source_id"] for item in bindings],
        "actor_ids": [item["actor_id"] for item in bindings],
        "emitter_link_names": [item["emitter_link_name"] for item in bindings],
    }
    for field, expected in expected_axis_values.items():
        observed = actors.get(field)
        if observed != expected:
            raise CurrentMP3DDynamicAudioError(
                f"visual receipt {field} axis order differs from the explicit "
                f"source binding: {observed!r} != {expected!r}"
            )
    records_path = _capture_artifact(capture_root, receipt, "frame_records")
    array_path = _capture_artifact(capture_root, receipt, "source_positions_m")
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    if not isinstance(frames, list) or len(frames) != EPISODE_FRAME_COUNT:
        raise CurrentMP3DDynamicAudioError(
            "frame_records must carry exactly the 75 episode frames"
        )
    frame_positions: list[list[list[float]]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
            raise CurrentMP3DDynamicAudioError(
                "frame_records indices must be contiguous from zero"
            )
        positions = frame.get("source_positions_m")
        if not isinstance(positions, list) or len(positions) != len(bindings):
            raise CurrentMP3DDynamicAudioError(
                "each frame must record one visual position per declared source axis"
            )
        converted: list[list[float]] = []
        for point in positions:
            if not isinstance(point, list):
                raise CurrentMP3DDynamicAudioError(
                    "source positions must be finite 3-vectors"
                )
            try:
                vector = [float(value) for value in point]
            except (TypeError, ValueError) as error:
                raise CurrentMP3DDynamicAudioError(
                    "source positions must be finite 3-vectors"
                ) from error
            if len(vector) != 3 or not all(np.isfinite(vector)):
                raise CurrentMP3DDynamicAudioError(
                    "source positions must be finite 3-vectors"
                )
            converted.append(vector)
        frame_positions.append(converted)
    json_array = np.asarray(frame_positions, dtype=np.float64)
    try:
        stored_array = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"unable to read visual source-position array: {error}"
        ) from error
    expected_shape = (EPISODE_FRAME_COUNT, len(bindings), 3)
    if stored_array.shape != expected_shape or not np.issubdtype(
        stored_array.dtype, np.number
    ):
        raise CurrentMP3DDynamicAudioError(
            f"visual source-position array must have shape {expected_shape}"
        )
    stored_float = np.asarray(stored_array, dtype=np.float64)
    if not np.all(np.isfinite(stored_float)) or not np.array_equal(
        stored_float, json_array
    ):
        raise CurrentMP3DDynamicAudioError(
            "visual source-position NPY and frame-record axes differ"
        )
    trajectories = {
        str(binding["audio_endpoint_id"]): json_array[:, axis_index, :].tolist()
        for axis_index, binding in enumerate(bindings)
    }
    evidence = {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "frame_records_path": records_path,
        "source_positions_array_path": array_path,
    }
    return trajectories, bindings, evidence


def load_captured_source_paths(
    visual_capture_dir: str | Path,
    source_axis_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, list[list[float]]]:
    """Return endpoint-keyed trajectories after explicit visual-axis checks."""

    trajectories, _, _ = _load_captured_source_evidence(
        visual_capture_dir, source_axis_bindings
    )
    return trajectories


def listener_pose_from_m1_request(
    m1_request: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Compose the static camera-colocated listener pose (wxyz orientation)."""

    errors = validate_capture_request(m1_request)
    if errors:
        raise CurrentMP3DDynamicAudioError("; ".join(errors))
    rig = m1_request["primary_camera_rig"]
    listener = m1_request["listener"]
    world_from_listener = compose_transforms(
        rig["world_from_rig"], listener["rig_from_listener"]
    )
    x, y, z, w = world_from_listener["rotation_xyzw"]
    return list(world_from_listener["translation_m"]), [w, x, y, z]


def _asset_bindings(
    sounds: Mapping[str, Any],
    *,
    repository_root: Path,
    external_sound_asset_paths: Mapping[str, Path],
    required_sound_ids: set[str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    records = sound_index(sounds)
    for sound_id in sorted(required_sound_ids):
        record = records.get(sound_id)
        if record is None:
            raise CurrentMP3DDynamicAudioError(
                f"program sound is not registered: {sound_id}"
            )
        uri = str(record["dry_audio"]["uri"])
        if uri.startswith("repo://"):
            resolved = (repository_root / uri.removeprefix("repo://")).resolve()
        else:
            path = external_sound_asset_paths.get(sound_id)
            if path is None:
                raise CurrentMP3DDynamicAudioError(
                    f"program sound {sound_id} requires an explicit external dry path"
                )
            resolved = Path(path).resolve()
        if not resolved.is_file():
            raise CurrentMP3DDynamicAudioError(f"dry audio is missing: {resolved}")
        expected = str(record["dry_audio"]["sha256"])
        if sha256_file(resolved) != expected:
            raise CurrentMP3DDynamicAudioError(
                f"dry audio differs from the registry digest for {sound_id}"
            )
        result[sound_id] = {"path": str(resolved), "sha256": expected}
    return result


def _input_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def build_source_identity_evidence(
    source_axis_bindings: Sequence[Mapping[str, Any]],
    *,
    candidate_source_endpoint_ids: Sequence[str],
    endpoint_registry: Mapping[str, Any],
    sound_registry: Mapping[str, Any],
    materialized_program: Mapping[str, Any],
    dry_asset_bindings: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Validate and materialize the visual-axis to audio-endpoint bijection."""

    bindings = _normalise_axis_bindings(source_axis_bindings)
    candidates = tuple(
        _text(value, owner="candidate_source_endpoint_ids item")
        for value in candidate_source_endpoint_ids
    )
    if len(candidates) != len(set(candidates)):
        raise CurrentMP3DDynamicAudioError(
            "AudioProgram candidate endpoints must be unique"
        )
    bound_endpoints = tuple(str(item["audio_endpoint_id"]) for item in bindings)
    if bound_endpoints != candidates:
        raise CurrentMP3DDynamicAudioError(
            "visual source axes and AudioProgram endpoints must form one complete "
            "ordered unique bijection"
        )
    endpoint_records = endpoint_index(endpoint_registry)
    sound_records = sound_index(sound_registry)
    raw_events = materialized_program.get("events")
    if not isinstance(raw_events, list):
        raise CurrentMP3DDynamicAudioError(
            "materialized AudioProgram events must be a list"
        )
    events: list[Mapping[str, Any]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"materialized AudioProgram events[{index}] must be an object"
            )
        events.append(event)

    physical_endpoint_owners: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for binding in bindings:
        endpoint_id = str(binding["audio_endpoint_id"])
        endpoint = endpoint_records.get(endpoint_id)
        if endpoint is None:
            raise CurrentMP3DDynamicAudioError(
                f"explicit audio endpoint is not registered: {endpoint_id}"
            )
        endpoint_binding = endpoint.get("binding")
        if not isinstance(endpoint_binding, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"audio endpoint {endpoint_id!r} has no binding object"
            )
        kind = endpoint_binding.get("kind")
        entity_instance_id = endpoint_binding.get("entity_instance_id")
        emitter_anchor_id = endpoint_binding.get("emitter_anchor_id")
        if (
            kind != "entity_anchor"
            or entity_instance_id != binding["endpoint_entity_instance_id"]
            or emitter_anchor_id != binding["endpoint_emitter_anchor_id"]
        ):
            raise CurrentMP3DDynamicAudioError(
                f"audio endpoint {endpoint_id!r} registry identity differs from "
                "the explicit visual-axis binding"
            )
        physical_owner = (str(kind), str(entity_instance_id), str(emitter_anchor_id))
        if physical_owner in physical_endpoint_owners:
            raise CurrentMP3DDynamicAudioError(
                "two audio endpoint IDs resolve to the same entity emitter anchor"
            )
        physical_endpoint_owners.add(physical_owner)

        endpoint_events = [
            event for event in events if event.get("source_endpoint_id") == endpoint_id
        ]
        event_records: list[dict[str, Any]] = []
        dry_asset_ids: list[str] = []
        for event_index, event in enumerate(endpoint_events):
            event_id = _text(
                event.get("event_id"),
                owner=f"endpoint {endpoint_id} event[{event_index}].event_id",
            )
            sound_asset_id = _text(
                event.get("sound_asset_id"),
                owner=f"endpoint {endpoint_id} event[{event_index}].sound_asset_id",
            )
            if sound_asset_id not in sound_records:
                raise CurrentMP3DDynamicAudioError(
                    f"event {event_id!r} references an unregistered sound asset"
                )
            if sound_asset_id not in dry_asset_bindings:
                raise CurrentMP3DDynamicAudioError(
                    f"event {event_id!r} has no resolved dry-audio asset"
                )
            dry_asset_ids.append(sound_asset_id)
            event_records.append(
                {
                    "event_id": event_id,
                    "audio_endpoint_id": endpoint_id,
                    "sound_asset_id": sound_asset_id,
                    "dry_audio_asset_id": sound_asset_id,
                    "start_sample": event.get("start_sample"),
                    "end_sample_exclusive": event.get("end_sample_exclusive"),
                    "start_tick": event.get("start_tick"),
                    "end_tick_exclusive": event.get("end_tick_exclusive"),
                }
            )
        dry_assets = []
        for sound_asset_id in sorted(set(dry_asset_ids)):
            sound = sound_records[sound_asset_id]
            dry = sound.get("dry_audio")
            local = dry_asset_bindings[sound_asset_id]
            if not isinstance(dry, Mapping) or not isinstance(local, Mapping):
                raise CurrentMP3DDynamicAudioError(
                    f"sound asset {sound_asset_id!r} has no dry-audio authority"
                )
            dry_assets.append(
                {
                    "dry_audio_asset_id": sound_asset_id,
                    "sound_asset_id": sound_asset_id,
                    "registry_uri": dry.get("uri"),
                    "registry_sha256": dry.get("sha256"),
                    "resolved_path": local.get("path"),
                    "resolved_sha256": local.get("sha256"),
                }
            )
        result.append(
            {
                "axis_index": binding["axis_index"],
                "source_id": binding["source_id"],
                "actor_id": binding["actor_id"],
                "emitter_link_name": binding["emitter_link_name"],
                "audio_endpoint_id": endpoint_id,
                "audio_endpoint_binding": {
                    "kind": kind,
                    "entity_instance_id": entity_instance_id,
                    "entity_asset_id": endpoint_binding.get("entity_asset_id"),
                    "entity_asset_revision": endpoint_binding.get(
                        "entity_asset_revision"
                    ),
                    "emitter_anchor_id": emitter_anchor_id,
                },
                "audio_event_ids": [item["event_id"] for item in event_records],
                "sound_asset_ids": sorted(set(dry_asset_ids)),
                "dry_audio_asset_ids": sorted(set(dry_asset_ids)),
                "event_bindings": event_records,
                "dry_audio_assets": dry_assets,
                "input_authority": {
                    "visual_identity": (
                        "visual research_receipt actors.source_ids/actor_ids/"
                        "emitter_link_names at axis_index"
                    ),
                    "visual_positions": (
                        "visual source_positions_m.npy cross-checked exactly against "
                        "frame_records source_positions_m"
                    ),
                    "audio_endpoint": (
                        "source endpoint registry record selected by explicit "
                        "audio_endpoint_id"
                    ),
                    "audio_events": "validated materialized AudioProgram variant",
                    "dry_audio": (
                        "sound asset registry dry_audio record plus resolved local "
                        "asset binding"
                    ),
                },
            }
        )
    return result


def build_audio_timing_evidence(
    materialized_program: Mapping[str, Any],
) -> dict[str, Any]:
    """Record exact scheduled intervals without claiming audible waveform onset."""

    timeline = materialized_program.get("timeline")
    if not isinstance(timeline, Mapping):
        raise CurrentMP3DDynamicAudioError(
            "materialized AudioProgram timeline must be an object"
        )
    expected_timeline = {
        "time_base_hz": 48_000,
        "ticks_per_frame": 3_200,
        "video_fps": VISUAL_FRAME_RATE_HZ,
        "frame_count": EPISODE_FRAME_COUNT,
        "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
        "ticks_per_sample": 3,
        "sample_count": EPISODE_SAMPLE_COUNT,
    }
    if any(timeline.get(key) != value for key, value in expected_timeline.items()):
        raise CurrentMP3DDynamicAudioError(
            "materialized AudioProgram timeline differs from the exact episode clock"
        )
    candidates = materialized_program.get("candidate_source_endpoint_ids")
    raw_events = materialized_program.get("events")
    if not isinstance(candidates, list) or not isinstance(raw_events, list):
        raise CurrentMP3DDynamicAudioError(
            "materialized AudioProgram candidates/events must be lists"
        )
    if candidates != sorted(set(candidates)):
        raise CurrentMP3DDynamicAudioError(
            "materialized AudioProgram candidates must be unique and canonical"
        )
    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"materialized AudioProgram events[{index}] must be an object"
            )
        event_id = _text(raw.get("event_id"), owner=f"events[{index}].event_id")
        endpoint_id = _text(
            raw.get("source_endpoint_id"),
            owner=f"events[{index}].source_endpoint_id",
        )
        sound_asset_id = _text(
            raw.get("sound_asset_id"), owner=f"events[{index}].sound_asset_id"
        )
        if event_id in seen_event_ids:
            raise CurrentMP3DDynamicAudioError("AudioProgram event IDs must be unique")
        seen_event_ids.add(event_id)
        if endpoint_id not in candidates:
            raise CurrentMP3DDynamicAudioError(
                f"event {event_id!r} endpoint is not a program candidate"
            )
        start_sample = raw.get("start_sample")
        end_sample = raw.get("end_sample_exclusive")
        start_tick = raw.get("start_tick")
        end_tick = raw.get("end_tick_exclusive")
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or isinstance(end_sample, bool)
            or not isinstance(end_sample, int)
            or not 0 <= start_sample < end_sample <= EPISODE_SAMPLE_COUNT
            or start_tick != start_sample * 3
            or end_tick != end_sample * 3
        ):
            raise CurrentMP3DDynamicAudioError(
                f"event {event_id!r} sample/tick interval is invalid"
            )
        events.append(
            {
                "event_id": event_id,
                "audio_endpoint_id": endpoint_id,
                "sound_asset_id": sound_asset_id,
                "dry_audio_asset_id": sound_asset_id,
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample,
                "start_tick": start_tick,
                "end_tick_exclusive": end_tick,
            }
        )
    frames = []
    for frame_index in range(EPISODE_FRAME_COUNT):
        start_sample, end_sample = frame_sample_interval(frame_index)
        active = [
            event
            for event in events
            if event["start_sample"] < end_sample
            and start_sample < event["end_sample_exclusive"]
        ]
        active_endpoint_set = {event["audio_endpoint_id"] for event in active}
        frames.append(
            {
                "frame_index": frame_index,
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample,
                "start_tick": frame_index * 3_200,
                "end_tick_exclusive": (frame_index + 1) * 3_200,
                "active_event_ids": [event["event_id"] for event in active],
                "active_audio_endpoint_ids": [
                    endpoint_id
                    for endpoint_id in candidates
                    if endpoint_id in active_endpoint_set
                ],
            }
        )
    return {
        "authority": "validated materialized AudioProgram scheduled intervals",
        "event_count": len(events),
        "events": events,
        "frames": frames,
        "waveform_activity_verification": {
            "status": "pending",
            "actual_waveform_activity_verified": None,
            "audible_onset_claim": False,
            "reason": (
                "this render records scheduled intervals but does not run an "
                "independent decoded-waveform activity/onset detector"
            ),
        },
    }


def _listener_view_identity(
    *,
    m1_request: Mapping[str, Any],
    m1_request_path: str | Path,
    visual_evidence: Mapping[str, Any],
    source_axis_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = visual_evidence.get("receipt")
    if not isinstance(receipt, Mapping):
        raise CurrentMP3DDynamicAudioError("visual evidence has no receipt")
    receipt_inputs = receipt.get("inputs")
    capture_m1 = (
        receipt_inputs.get("m1_request")
        if isinstance(receipt_inputs, Mapping)
        else None
    )
    if isinstance(capture_m1, Mapping):
        capture_m1 = capture_m1.get("path")
    supplied_m1 = Path(m1_request_path).resolve()
    if not isinstance(capture_m1, str) or Path(capture_m1).resolve() != supplied_m1:
        raise CurrentMP3DDynamicAudioError(
            "visual capture and audio render must use the same M1 listener/view request"
        )
    bindings = _normalise_axis_bindings(source_axis_bindings)
    sources = m1_request.get("sources")
    m1_source_order = (
        [
            item.get("source_id") if isinstance(item, Mapping) else None
            for item in sources
        ]
        if isinstance(sources, list)
        else None
    )
    expected_source_order = [item["source_id"] for item in bindings]
    if m1_source_order != expected_source_order:
        raise CurrentMP3DDynamicAudioError(
            "M1 source order differs from the explicit visual source-axis identity"
        )
    selected_room = receipt.get("selected_room")
    selected_room_id = (
        selected_room.get("room_id") if isinstance(selected_room, Mapping) else None
    )
    if selected_room_id != m1_request.get("room_id"):
        raise CurrentMP3DDynamicAudioError(
            "visual capture room differs from the M1 listener/view room"
        )
    rig = m1_request.get("primary_camera_rig")
    listener = m1_request.get("listener")
    if not isinstance(rig, Mapping) or not isinstance(listener, Mapping):
        raise CurrentMP3DDynamicAudioError("M1 request has no listener/camera identity")
    return {
        "m1_request_id": _text(m1_request.get("request_id"), owner="M1 request_id"),
        "room_id": _text(m1_request.get("room_id"), owner="M1 room_id"),
        "camera_rig_id": _text(rig.get("rig_id"), owner="M1 camera rig_id"),
        "view_id": _text(rig.get("view_id"), owner="M1 view_id"),
        "listener_id": _text(listener.get("listener_id"), owner="M1 listener_id"),
        "visual_capture_dir": str(
            Path(visual_evidence["receipt_path"]).resolve().parent
        ),
        "visual_receipt_path": str(Path(visual_evidence["receipt_path"]).resolve()),
        "m1_request_path": str(supplied_m1),
        "visual_source_axis_order": expected_source_order,
    }


def render_dynamic_research_audio(
    *,
    source_trajectories_m: Mapping[str, Sequence[Sequence[float]]],
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    external_sound_asset_paths: Mapping[str, str | Path],
    hrtf_file_path: str | Path,
    output_path: str | Path,
    position_authority: str,
    listener_authority: str,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    hrtf_license_path: str | Path | None = None,
    extra_inputs: Mapping[str, Any] | None = None,
    source_identity_bindings: Sequence[Mapping[str, Any]] | None = None,
    listener_view_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Room-agnostic core: trajectories + program variant -> binaural episode."""

    output = _fresh_output(output_path)
    program_path = Path(audio_program_path).resolve()
    program = json.loads(program_path.read_text(encoding="utf-8"))
    endpoint_registry_path = Path(source_endpoint_registry_path).resolve()
    sound_registry_path = Path(sound_asset_registry_path).resolve()
    endpoints = load_source_endpoint_registry(endpoint_registry_path)
    sounds = load_sound_asset_registry(sound_registry_path)

    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    if len(source_ids) < 2:
        raise CurrentMP3DDynamicAudioError(
            "the program must carry at least two candidate source endpoints"
        )
    if set(source_ids) != set(source_trajectories_m):
        raise CurrentMP3DDynamicAudioError(
            "trajectory source IDs must equal the program candidates: "
            f"{sorted(source_trajectories_m)} != {sorted(source_ids)}"
        )
    timeline = program.get("timeline") or {}
    if (
        timeline.get("frame_count") != EPISODE_FRAME_COUNT
        or timeline.get("sample_rate_hz") != AUDIO_SAMPLE_RATE_HZ
        or timeline.get("sample_count") != EPISODE_SAMPLE_COUNT
    ):
        raise CurrentMP3DDynamicAudioError(
            "the program timeline must match the 75-frame research episode"
        )
    if listener_view_identity is not None and not isinstance(
        listener_view_identity, Mapping
    ):
        raise CurrentMP3DDynamicAudioError(
            "listener_view_identity must be an object when provided"
        )

    raw_events = program.get("events")
    if not isinstance(raw_events, list) or any(
        not isinstance(event, Mapping)
        or not isinstance(event.get("sound_asset_id"), str)
        or not event.get("sound_asset_id")
        for event in raw_events
    ):
        raise CurrentMP3DDynamicAudioError(
            "AudioProgram events must explicitly identify their sound assets"
        )
    required_sounds = {str(event["sound_asset_id"]) for event in raw_events}
    repository_root = Path(__file__).resolve().parents[3]
    bindings = _asset_bindings(
        sounds,
        repository_root=repository_root,
        external_sound_asset_paths={
            key: Path(value) for key, value in external_sound_asset_paths.items()
        },
        required_sound_ids=required_sounds,
    )
    assembly = assemble_audio_program_dry_buses(
        program,
        variant_id,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
        asset_bindings=bindings,
    )
    materialized = assembly.materialized_program
    timing_evidence = build_audio_timing_evidence(materialized)
    if source_identity_bindings is None:
        source_identity_status = "pending"
        source_identity_records: list[dict[str, Any]] = []
        source_identity_reason: str | None = (
            "the room-agnostic caller supplied no explicit visual-axis identity bridge"
        )
    else:
        source_identity_records = build_source_identity_evidence(
            source_identity_bindings,
            candidate_source_endpoint_ids=source_ids,
            endpoint_registry=endpoints,
            sound_registry=sounds,
            materialized_program=materialized,
            dry_asset_bindings=bindings,
        )
        source_identity_status = "verified_complete_unique_bijection"
        source_identity_reason = None
    dry_buses = assembly.dry_audio.buses

    ordered_trajectories = {
        source_id: [
            list(map(float, point)) for point in source_trajectories_m[source_id]
        ]
        for source_id in source_ids
    }
    grid = build_strided_review_keyframes(
        ordered_trajectories,
        visual_frame_rate_hz=VISUAL_FRAME_RATE_HZ,
        rir_stride_frames=rir_stride_frames,
        listener_position_m=list(listener_position_m),
        listener_orientation_wxyz=list(listener_orientation_wxyz),
    )
    _, simulation = _load_simulation_request(Path(simulation_request_path).resolve())
    scene = load_compiled_acoustic_scene(
        package_manifest_path, allow_nonpassing_research_qa=True
    )
    hrtf = Path(hrtf_file_path).resolve()
    sequence = render_research_review_binaural_rir_sequence(
        scene, simulation, grid=grid, hrtf_file_path=str(hrtf)
    )

    stems, mixture = render_research_review_binaural_audio(
        dry_buses, sequence, grid=grid
    )

    audio_root = output / "audio"
    outputs: dict[str, str] = {}

    def _write(path: Path, samples: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_float32_wav(path, samples, AUDIO_SAMPLE_RATE_HZ)
        outputs[str(path.relative_to(output))] = sha256_file(path)

    for source_id in source_ids:
        dry = np.asarray(dry_buses[source_id], dtype=np.float64)[None, :]
        _write(audio_root / "dry" / f"{source_id}.wav", dry)
        _write(
            audio_root / "binaural" / f"{source_id}_stem.wav",
            stems[source_id].episode,
        )
    _write(audio_root / "binaural" / "mixture.wav", np.asarray(mixture))

    inputs: dict[str, Any] = {
        "simulation_request": _input_record(simulation_request_path),
        "package_manifest": _input_record(package_manifest_path),
        "source_endpoint_registry": _input_record(endpoint_registry_path),
        "sound_asset_registry": _input_record(sound_registry_path),
        "hrtf": {
            **_input_record(hrtf),
            "license_path": (
                str(Path(hrtf_license_path).resolve())
                if hrtf_license_path is not None
                else None
            ),
        },
        "dry_assets": bindings,
    }
    if extra_inputs:
        inputs.update({key: value for key, value in extra_inputs.items()})

    receipt: dict[str, Any] = {
        "schema": CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA,
        "status": "pass",
        "claim_boundary": CLAIM_BOUNDARY,
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "audio": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "sample_count": EPISODE_SAMPLE_COUNT,
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
        },
        "audio_program": {
            "path": str(program_path),
            "program_id": program.get("program_id"),
            "revision": program.get("revision"),
            "mode": program.get("mode"),
            "variant_id": variant_id,
            "program_content_sha256": materialized.get("program_content_sha256"),
            "event_count": len(program.get("events") or ()),
        },
        "audio_timing": timing_evidence,
        "sources": {
            "source_ids": list(source_ids),
            "audio_endpoint_ids": list(source_ids),
            "visual_source_ids": [
                item["source_id"] for item in source_identity_records
            ],
            "frame_count": EPISODE_FRAME_COUNT,
            "position_authority": position_authority,
            "identity_status": source_identity_status,
            "identity_reason": source_identity_reason,
            "identity_bindings": source_identity_records,
        },
        "listener": {
            "position_m": list(listener_position_m),
            "orientation_wxyz": list(listener_orientation_wxyz),
            "authority": listener_authority,
            "view_identity": (
                dict(listener_view_identity)
                if listener_view_identity is not None
                else None
            ),
        },
        "rir": {
            "stride_frames": rir_stride_frames,
            "keyframe_count": len(sequence.keyframe_samples),
            "keyframe_samples": list(sequence.keyframe_samples),
            "trajectory_sha256": sequence.trajectory_sha256,
            "listener_view_render_policy": "fresh_per_output_invocation",
            "prior_listener_rir_reused": False,
            "listener_view_identity": (
                dict(listener_view_identity)
                if listener_view_identity is not None
                else None
            ),
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def render_current_mp3d_dynamic_audio(
    *,
    visual_capture_dir: str | Path,
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    external_sound_asset_paths: Mapping[str, str | Path],
    hrtf_file_path: str | Path,
    output_path: str | Path,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    hrtf_license_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render one motion-following binaural research episode for MP3D."""

    trajectories, source_axis_bindings, visual_evidence = (
        _load_captured_source_evidence(
            visual_capture_dir, CURRENT_MP3D_SOURCE_AXIS_BINDINGS
        )
    )
    m1_request = json.loads(Path(m1_request_path).resolve().read_text(encoding="utf-8"))
    listener_position, listener_wxyz = listener_pose_from_m1_request(m1_request)
    view_identity = _listener_view_identity(
        m1_request=m1_request,
        m1_request_path=m1_request_path,
        visual_evidence=visual_evidence,
        source_axis_bindings=source_axis_bindings,
    )
    return render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=listener_position,
        listener_orientation_wxyz=listener_wxyz,
        simulation_request_path=simulation_request_path,
        package_manifest_path=package_manifest_path,
        audio_program_path=audio_program_path,
        source_endpoint_registry_path=source_endpoint_registry_path,
        sound_asset_registry_path=sound_asset_registry_path,
        external_sound_asset_paths=external_sound_asset_paths,
        hrtf_file_path=hrtf_file_path,
        output_path=output_path,
        position_authority=(
            "current-visual receipt-declared source axis; source_positions_m.npy "
            "cross-checked exactly against frame_records per-frame positions"
        ),
        listener_authority=(
            "research M1 request primary_camera_rig composed with rig_from_listener"
        ),
        rir_stride_frames=rir_stride_frames,
        variant_id=variant_id,
        hrtf_license_path=hrtf_license_path,
        extra_inputs={
            "visual_capture_frame_records": _input_record(
                visual_evidence["frame_records_path"]
            ),
            "visual_capture_source_positions": _input_record(
                visual_evidence["source_positions_array_path"]
            ),
            "visual_capture_receipt": _input_record(visual_evidence["receipt_path"]),
            "m1_request": _input_record(m1_request_path),
        },
        source_identity_bindings=source_axis_bindings,
        listener_view_identity=view_identity,
    )
