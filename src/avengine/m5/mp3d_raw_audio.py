"""Optional per-view MP3D native-audio raw-clue joins.

The consumer validates schedule, identity, and native RIR registration facts
already present in one current dynamic-audio receipt. It never decodes a WAV,
infers perceptual onset, ranks a camera, or changes candidate retention.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.contracts.transforms import compose_transforms
from avengine.m5.timeline import frame_sample_interval


FRAME_COUNT = 75
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
TICKS_PER_SAMPLE = 3
TICKS_PER_FRAME = 3_200
EVENT_COUNT = 6
RIR_STRIDE_FRAMES = 3
RIR_KEYFRAME_COUNT = 25
SCHEMA = "avengine_m5_current_mp3d_dynamic_audio_v1"


class MP3DRawAudioError(ValueError):
    """A native-audio receipt cannot support an honest per-view join."""


@dataclass(frozen=True)
class LoadedMP3DRawAudio:
    root: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    m1_request_id: str
    room_id: str
    view_id: str
    listener_id: str
    raw_clues: Mapping[str, Any]


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DRawAudioError(f"{owner} must be an object")
    return value


def _text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise MP3DRawAudioError(f"{owner} must be a nonempty string")
    return value


def _integer(value: Any, *, owner: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MP3DRawAudioError(f"{owner} must be an integer >= {minimum}")
    return value


def _two_unique_strings(value: Any, *, owner: str) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != 2
    ):
        raise MP3DRawAudioError(f"{owner} must contain two unique IDs")
    return value[0], value[1]


def _root_and_receipt(audio_root: Path) -> tuple[Path, Path, Mapping[str, Any]]:
    try:
        root = audio_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise MP3DRawAudioError(
            f"dynamic audio directory cannot be resolved: {audio_root}"
        ) from exc
    if not root.is_dir():
        raise MP3DRawAudioError(f"dynamic audio input is not a directory: {root}")
    receipt_path = root / "research_receipt.json"
    try:
        receipt = _mapping(load_json(receipt_path), owner="dynamic audio receipt")
    except OSError as exc:
        raise MP3DRawAudioError(
            f"dynamic audio receipt is unavailable: {receipt_path}"
        ) from exc
    return root, receipt_path, receipt


def dynamic_audio_request_id(audio_root: Path) -> str:
    """Return the capture-time M1 request identity used for CLI indexing."""

    _root, _path, receipt = _root_and_receipt(audio_root)
    listener = _mapping(receipt.get("listener"), owner="audio listener")
    view = _mapping(listener.get("view_identity"), owner="listener view identity")
    return _text(view.get("m1_request_id"), owner="listener M1 request ID")


def _expected_capture_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    sources = request.get("sources")
    if not isinstance(sources, list):
        raise MP3DRawAudioError("joined M1 request lacks source records")
    return {
        "request_id": request.get("request_id"),
        "room_id": request.get("room_id"),
        "primary_camera_rig": request.get("primary_camera_rig"),
        "listener": request.get("listener"),
        "source_axis_order": [
            item.get("source_id") if isinstance(item, Mapping) else None
            for item in sources
        ],
        "qa_views": request.get("qa_views", []),
    }


def _view_identity(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    request_path: Path,
    visual_receipt_path: Path,
) -> tuple[dict[str, Any], str, str, str, tuple[str, str]]:
    listener = _mapping(receipt.get("listener"), owner="audio listener")
    view = _mapping(listener.get("view_identity"), owner="listener view identity")
    rig = _mapping(request.get("primary_camera_rig"), owner="M1 camera rig")
    m1_listener = _mapping(request.get("listener"), owner="M1 listener")
    expected_identity = _expected_capture_identity(request)
    expected_source_ids = _two_unique_strings(
        expected_identity["source_axis_order"], owner="M1 source order"
    )
    request_id = _text(request.get("request_id"), owner="M1 request_id")
    room_id = _text(request.get("room_id"), owner="M1 room_id")
    view_id = _text(rig.get("view_id"), owner="M1 view_id")
    listener_id = _text(m1_listener.get("listener_id"), owner="M1 listener_id")
    expected = {
        "m1_request_id": request_id,
        "room_id": room_id,
        "camera_rig_id": _text(rig.get("rig_id"), owner="M1 rig_id"),
        "view_id": view_id,
        "listener_id": listener_id,
    }
    for field, value in expected.items():
        if view.get(field) != value:
            raise MP3DRawAudioError(
                f"audio listener view identity {field} differs from joined M1"
            )
    if view.get("capture_m1_identity") != expected_identity:
        raise MP3DRawAudioError(
            "audio listener capture-time M1 identity differs from joined M1"
        )
    if view.get("visual_source_axis_order") != list(expected_source_ids):
        raise MP3DRawAudioError(
            "audio listener visual source axis differs from joined M1"
        )
    expected_request_path = request_path.resolve(strict=True)
    for field in ("m1_request_path", "capture_declared_m1_request_path"):
        raw = view.get(field)
        if (
            not isinstance(raw, str)
            or Path(raw).resolve(strict=True) != expected_request_path
        ):
            raise MP3DRawAudioError(
                f"audio listener view identity {field} differs from joined M1 path"
            )
    raw_visual_receipt = view.get("visual_receipt_path")
    if not isinstance(raw_visual_receipt, str) or Path(raw_visual_receipt).resolve(
        strict=True
    ) != visual_receipt_path.resolve(strict=True):
        raise MP3DRawAudioError(
            "audio listener view identity points to a different visual receipt"
        )
    top_listener_id = _text(listener.get("listener_id"), owner="audio listener_id")
    if top_listener_id != listener_id:
        raise MP3DRawAudioError("audio listener identity differs from joined M1")
    world_from_listener = compose_transforms(
        dict(_mapping(rig.get("world_from_rig"), owner="M1 world_from_rig")),
        dict(
            _mapping(m1_listener.get("rig_from_listener"), owner="M1 rig_from_listener")
        ),
    )
    expected_position = np.asarray(
        world_from_listener["translation_m"], dtype=np.float64
    )
    x, y, z, w = world_from_listener["rotation_xyzw"]
    expected_orientation = np.asarray([w, x, y, z], dtype=np.float64)
    position = np.asarray(listener.get("position_m"), dtype=np.float64)
    orientation = np.asarray(listener.get("orientation_wxyz"), dtype=np.float64)
    if (
        position.shape != (3,)
        or orientation.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(orientation))
        or not np.allclose(position, expected_position, rtol=0.0, atol=2.0e-6)
        or not np.allclose(orientation, expected_orientation, rtol=0.0, atol=2.0e-6)
    ):
        raise MP3DRawAudioError("audio listener pose differs from joined M1 listener")
    return deepcopy(dict(view)), request_id, room_id, view_id, expected_source_ids


def _source_bindings(
    receipt: Mapping[str, Any],
    *,
    visual_raw_clues: Mapping[str, Any],
    expected_visual_source_ids: tuple[str, str],
) -> tuple[list[dict[str, Any]], tuple[str, str]]:
    sources = _mapping(receipt.get("sources"), owner="audio sources")
    if (
        sources.get("identity_status") != "verified_complete_unique_bijection"
        or sources.get("identity_reason") is not None
        or sources.get("frame_count") != FRAME_COUNT
        or sources.get("visual_source_ids") != list(expected_visual_source_ids)
    ):
        raise MP3DRawAudioError(
            "audio source identity/source axis is not one verified 75-frame bijection"
        )
    endpoint_ids = _two_unique_strings(
        sources.get("audio_endpoint_ids"), owner="audio endpoint IDs"
    )
    if sources.get("source_ids") != list(endpoint_ids):
        raise MP3DRawAudioError("audio source IDs differ from endpoint IDs")
    identity_axes = _mapping(
        visual_raw_clues.get("identity_axes"), owner="visual raw identity axes"
    )
    visual_actor_ids = _two_unique_strings(
        identity_axes.get("actor_axis_order"), owner="visual actor axis"
    )
    visual_source_ids = _two_unique_strings(
        identity_axes.get("source_axis_order"), owner="visual source axis"
    )
    if visual_source_ids != expected_visual_source_ids:
        raise MP3DRawAudioError("visual raw source axis differs from joined M1")
    visual_axis_bindings = identity_axes.get("axis_bindings")
    raw_bindings = sources.get("identity_bindings")
    if (
        not isinstance(visual_axis_bindings, list)
        or len(visual_axis_bindings) != 2
        or not isinstance(raw_bindings, list)
        or len(raw_bindings) != 2
    ):
        raise MP3DRawAudioError(
            "visual/audio source axes must each contain two bindings"
        )
    checked: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for axis, raw in enumerate(raw_bindings):
        binding = _mapping(raw, owner=f"audio identity binding[{axis}]")
        visual = _mapping(
            visual_axis_bindings[axis], owner=f"visual identity binding[{axis}]"
        )
        if (
            binding.get("axis_index") != axis
            or binding.get("source_id") != visual_source_ids[axis]
            or binding.get("actor_id") != visual_actor_ids[axis]
            or binding.get("emitter_link_name") != visual.get("emitter_link_name")
            or binding.get("audio_endpoint_id") != endpoint_ids[axis]
        ):
            raise MP3DRawAudioError(
                f"audio identity binding axis {axis} differs from visual identity axes"
            )
        endpoint = _mapping(
            binding.get("audio_endpoint_binding"),
            owner=f"audio endpoint binding[{axis}]",
        )
        for field in (
            "kind",
            "entity_instance_id",
            "entity_asset_id",
            "entity_asset_revision",
            "emitter_anchor_id",
        ):
            _text(endpoint.get(field), owner=f"audio endpoint binding[{axis}].{field}")
        event_ids = binding.get("audio_event_ids")
        event_bindings = binding.get("event_bindings")
        if (
            not isinstance(event_ids, list)
            or len(event_ids) != 3
            or any(not isinstance(item, str) or not item for item in event_ids)
            or not isinstance(event_bindings, list)
            or len(event_bindings) != 3
            or [
                item.get("event_id") if isinstance(item, Mapping) else None
                for item in event_bindings
            ]
            != event_ids
            or seen_events.intersection(event_ids)
        ):
            raise MP3DRawAudioError(
                f"audio identity binding axis {axis} has damaged event identities"
            )
        seen_events.update(event_ids)
        checked.append(deepcopy(dict(binding)))
    if len(seen_events) != EVENT_COUNT:
        raise MP3DRawAudioError("audio identity bindings must cover exactly six events")
    return checked, endpoint_ids


def _event(value: Any, *, owner: str, endpoint_ids: tuple[str, str]) -> dict[str, Any]:
    record = _mapping(value, owner=owner)
    event_id = _text(record.get("event_id"), owner=f"{owner}.event_id")
    endpoint_id = _text(
        record.get("audio_endpoint_id"), owner=f"{owner}.audio_endpoint_id"
    )
    if endpoint_id not in endpoint_ids:
        raise MP3DRawAudioError(f"{owner} names an unknown endpoint")
    start = _integer(record.get("start_sample"), owner=f"{owner}.start_sample")
    end = _integer(
        record.get("end_sample_exclusive"), owner=f"{owner}.end_sample_exclusive"
    )
    start_tick = _integer(record.get("start_tick"), owner=f"{owner}.start_tick")
    end_tick = _integer(
        record.get("end_tick_exclusive"), owner=f"{owner}.end_tick_exclusive"
    )
    if (
        not start < end <= SAMPLE_COUNT
        or start_tick != start * 3
        or end_tick != end * 3
    ):
        raise MP3DRawAudioError(f"{owner} has an invalid exact sample/tick interval")
    return {
        "event_id": event_id,
        "audio_endpoint_id": endpoint_id,
        "sound_asset_id": _text(
            record.get("sound_asset_id"), owner=f"{owner}.sound_asset_id"
        ),
        "dry_audio_asset_id": _text(
            record.get("dry_audio_asset_id"), owner=f"{owner}.dry_audio_asset_id"
        ),
        "start_sample": start,
        "end_sample_exclusive": end,
        "start_tick": start_tick,
        "end_tick_exclusive": end_tick,
    }


def _timing(
    receipt: Mapping[str, Any], *, endpoint_ids: tuple[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    timing = _mapping(receipt.get("audio_timing"), owner="audio timing")
    raw_events = timing.get("events")
    raw_frames = timing.get("frames")
    if (
        timing.get("event_count") != EVENT_COUNT
        or not isinstance(raw_events, list)
        or len(raw_events) != EVENT_COUNT
        or not isinstance(raw_frames, list)
        or len(raw_frames) != FRAME_COUNT
    ):
        raise MP3DRawAudioError("audio timing must contain six events and 75 frames")
    events = [
        _event(value, owner=f"audio events[{index}]", endpoint_ids=endpoint_ids)
        for index, value in enumerate(raw_events)
    ]
    event_ids = [item["event_id"] for item in events]
    if len(set(event_ids)) != EVENT_COUNT:
        raise MP3DRawAudioError("audio event IDs must be unique")
    frames: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_frames):
        frame = _mapping(raw, owner=f"audio frames[{index}]")
        start_sample, end_sample = frame_sample_interval(index)
        active = [
            event
            for event in events
            if event["start_sample"] < end_sample
            and start_sample < event["end_sample_exclusive"]
        ]
        expected_event_ids = [event["event_id"] for event in active]
        active_endpoint_set = {event["audio_endpoint_id"] for event in active}
        expected_endpoint_ids = [
            endpoint for endpoint in endpoint_ids if endpoint in active_endpoint_set
        ]
        expected = {
            "frame_index": index,
            "start_sample": start_sample,
            "end_sample_exclusive": end_sample,
            "start_tick": index * TICKS_PER_FRAME,
            "end_tick_exclusive": (index + 1) * TICKS_PER_FRAME,
            "active_event_ids": expected_event_ids,
            "active_audio_endpoint_ids": expected_endpoint_ids,
        }
        if dict(frame) != expected:
            raise MP3DRawAudioError(
                f"audio frame {index} differs from exact event/sample/tick overlap"
            )
        frames.append(expected)
    waveform = _mapping(
        timing.get("waveform_activity_verification"),
        owner="waveform activity verification",
    )
    if (
        waveform.get("status") != "pending"
        or waveform.get("actual_waveform_activity_verified") is not None
        or waveform.get("audible_onset_claim") is not False
    ):
        raise MP3DRawAudioError(
            "audio receipt must leave waveform activity and audible onset pending"
        )
    boundary = {
        "status": "pending_receipt_does_not_verify_waveform_activity",
        "actual_waveform_activity_verified": None,
        "perceptual_onset": None,
        "audible_onset_claim": False,
        "receipt_record": deepcopy(dict(waveform)),
        "wav_paths_or_output_digests_effect": "none_without_structured_waveform_verifier",
    }
    return events, frames, boundary


def _crosscheck_binding_events(
    bindings: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> None:
    by_id = {str(item["event_id"]): dict(item) for item in events}
    for axis, binding in enumerate(bindings):
        endpoint = str(binding["audio_endpoint_id"])
        event_ids = binding["audio_event_ids"]
        expected = [by_id[event_id] for event_id in event_ids]
        if any(item["audio_endpoint_id"] != endpoint for item in expected):
            raise MP3DRawAudioError(
                f"audio identity axis {axis} binds an event owned by another endpoint"
            )
        if binding["event_bindings"] != expected:
            raise MP3DRawAudioError(
                f"audio identity axis {axis} event intervals differ from audio timing"
            )


def _rir(
    receipt: Mapping[str, Any],
    *,
    view_identity: Mapping[str, Any],
    listener_id: str,
    endpoint_ids: tuple[str, str],
    bindings: Sequence[Mapping[str, Any]],
    visual_raw_clues: Mapping[str, Any],
) -> dict[str, Any]:
    rir = _mapping(receipt.get("rir"), owner="audio RIR")
    expected_samples = [
        frame_sample_interval(index * RIR_STRIDE_FRAMES)[0]
        for index in range(RIR_KEYFRAME_COUNT)
    ]
    if (
        rir.get("stride_frames") != RIR_STRIDE_FRAMES
        or rir.get("keyframe_count") != RIR_KEYFRAME_COUNT
        or rir.get("keyframe_samples") != expected_samples
        or rir.get("prior_listener_rir_reused") is not False
        or rir.get("listener_view_render_policy") != "fresh_per_output_invocation"
        or rir.get("listener_view_identity") != view_identity
    ):
        raise MP3DRawAudioError("audio RIR keyframe/view/fresh identity is damaged")
    trajectory = _text(rir.get("trajectory_sha256"), owner="RIR trajectory identity")
    sequence = _mapping(
        rir.get("sequence_metadata_registration"), owner="RIR sequence registration"
    )
    endpoint_receipts = sequence.get("endpoint_receipts")
    ir_identities = sequence.get("ir_sha256_by_keyframe_source")
    if (
        sequence.get("listener_id") != listener_id
        or sequence.get("source_ids") != list(endpoint_ids)
        or sequence.get("trajectory_sha256") != trajectory
        or not isinstance(endpoint_receipts, list)
        or len(endpoint_receipts) != RIR_KEYFRAME_COUNT
        or not isinstance(ir_identities, list)
        or len(ir_identities) != RIR_KEYFRAME_COUNT
    ):
        raise MP3DRawAudioError("RIR sequence identities do not cover 25 keyframes")
    visual_frames = visual_raw_clues.get("frames")
    if not isinstance(visual_frames, list) or len(visual_frames) != FRAME_COUNT:
        raise MP3DRawAudioError("current visual raw clues must contain 75 frames")
    checked_receipts: list[dict[str, Any]] = []
    checked_ir: list[dict[str, str]] = []
    listener = _mapping(receipt.get("listener"), owner="audio listener")
    listener_position = np.asarray(listener.get("position_m"), dtype=np.float64)
    listener_orientation = np.asarray(
        listener.get("orientation_wxyz"), dtype=np.float64
    )
    for index in range(RIR_KEYFRAME_COUNT):
        hashes = _mapping(ir_identities[index], owner=f"RIR IR identities[{index}]")
        if set(hashes) != set(endpoint_ids) or any(
            not isinstance(value, str) or not value for value in hashes.values()
        ):
            raise MP3DRawAudioError(
                f"RIR keyframe {index} lacks one per-source IR identity"
            )
        checked_ir.append({key: str(value) for key, value in hashes.items()})
        registration = _mapping(
            endpoint_receipts[index], owner=f"RIR endpoint receipt[{index}]"
        )
        visual_index = index * RIR_STRIDE_FRAMES
        if (
            registration.get("authority") != "native_registration_readback"
            or registration.get("keyframe_index") != index
            or registration.get("sample_index") != expected_samples[index]
            or registration.get("tick") != expected_samples[index] * TICKS_PER_SAMPLE
            or registration.get("visual_frame_index") != visual_index
        ):
            raise MP3DRawAudioError(
                f"RIR keyframe {index} native registration interval is damaged"
            )
        native_listener = _mapping(
            registration.get("listener"), owner=f"RIR listener[{index}]"
        )
        native_position = np.asarray(
            native_listener.get("position_m"), dtype=np.float64
        )
        native_orientation = np.asarray(
            native_listener.get("orientation_wxyz"), dtype=np.float64
        )
        if (
            native_listener.get("listener_id") != listener_id
            or native_listener.get("native_realized") is not True
            or native_listener.get("canonical_native_index") != 0
            or native_position.shape != (3,)
            or native_orientation.shape != (4,)
            or not np.allclose(
                native_position, listener_position, rtol=0.0, atol=2.0e-6
            )
            or not np.allclose(
                native_orientation, listener_orientation, rtol=0.0, atol=2.0e-6
            )
        ):
            raise MP3DRawAudioError(
                f"RIR keyframe {index} native listener identity is damaged"
            )
        native_sources = registration.get("sources")
        if not isinstance(native_sources, list) or len(native_sources) != 2:
            raise MP3DRawAudioError(
                f"RIR keyframe {index} lacks two native source registrations"
            )
        for axis, native_value in enumerate(native_sources):
            native = _mapping(native_value, owner=f"RIR source[{index},{axis}]")
            visual_source_id = str(bindings[axis]["source_id"])
            expected_position = np.asarray(
                visual_frames[visual_index]["sources"][visual_source_id][
                    "actual_emitter_position_m"
                ],
                dtype=np.float64,
            )
            native_source_position = np.asarray(
                native.get("position_m"), dtype=np.float64
            )
            if (
                native.get("source_id") != endpoint_ids[axis]
                or native.get("native_realized") is not True
                or native.get("canonical_native_index") != axis
                or native_source_position.shape != (3,)
                or not np.allclose(
                    native_source_position, expected_position, rtol=0.0, atol=2.0e-6
                )
            ):
                raise MP3DRawAudioError(
                    f"RIR keyframe {index} source axis {axis} identity is damaged"
                )
        checked_receipts.append(deepcopy(dict(registration)))
    return {
        "status": "native_registration_and_ir_identities_verified",
        "stride_frames": RIR_STRIDE_FRAMES,
        "keyframe_count": RIR_KEYFRAME_COUNT,
        "keyframe_samples": expected_samples,
        "trajectory_identity": trajectory,
        "prior_rir_reused": False,
        "listener_view_render_policy": "fresh_per_output_invocation",
        "native_registration_receipts": checked_receipts,
        "per_keyframe_source_ir_identities": checked_ir,
    }


def load_dynamic_audio_raw_clues(
    audio_root: Path,
    *,
    request: Mapping[str, Any],
    request_path: Path,
    visual_receipt_path: Path,
    visual_raw_clues: Mapping[str, Any],
) -> LoadedMP3DRawAudio:
    """Validate and describe one current per-view native-audio receipt."""

    root, receipt_path, receipt = _root_and_receipt(audio_root)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("research_only") is not True
        or receipt.get("episode_counted") is not False
        or receipt.get("formal_dataset_count") != 0
        or receipt.get("qualification_claim") is not False
    ):
        raise MP3DRawAudioError(
            "dynamic audio receipt must be current, passing, research-only, and uncounted"
        )
    audio = _mapping(receipt.get("audio"), owner="audio format")
    if (
        audio.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or audio.get("sample_count") != SAMPLE_COUNT
        or audio.get("layout_type") != "binaural"
        or audio.get("channel_labels") != ["left", "right"]
    ):
        raise MP3DRawAudioError(
            "dynamic audio format is not 80,000-sample 16 kHz binaural"
        )
    if visual_raw_clues.get("has_current_post_render_pose_authority") is not True:
        raise MP3DRawAudioError(
            "native audio join requires current visual post-render raw clues"
        )
    view_identity, request_id, room_id, view_id, visual_source_ids = _view_identity(
        receipt,
        request=request,
        request_path=request_path,
        visual_receipt_path=visual_receipt_path,
    )
    listener_id = str(view_identity["listener_id"])
    bindings, endpoint_ids = _source_bindings(
        receipt,
        visual_raw_clues=visual_raw_clues,
        expected_visual_source_ids=visual_source_ids,
    )
    events, frames, waveform_boundary = _timing(receipt, endpoint_ids=endpoint_ids)
    _crosscheck_binding_events(bindings, events)
    rir = _rir(
        receipt,
        view_identity=view_identity,
        listener_id=listener_id,
        endpoint_ids=endpoint_ids,
        bindings=bindings,
        visual_raw_clues=visual_raw_clues,
    )
    program = _mapping(receipt.get("audio_program"), owner="audio program")
    if program.get("event_count") != EVENT_COUNT:
        raise MP3DRawAudioError("audio program must declare exactly six events")
    raw_clues = {
        "status": "native_audio_schedule_and_rir_identity_joined_waveform_truth_pending",
        "research_only": True,
        "episode_counted": False,
        "listener_view_identity": view_identity,
        "listener_id": listener_id,
        "visual_source_to_audio_endpoint_bindings": bindings,
        "planned_audio_timing": {
            "status": "exact_scheduled_intervals_joined_waveform_truth_pending",
            "authority": timing_authority(receipt),
            "source": "dynamic_audio_receipt.audio_timing",
            "event_count": EVENT_COUNT,
            "events": events,
            "frames": frames,
        },
        "rir_identity": rir,
        "waveform_truth": waveform_boundary,
        "declared_outputs_uninspected": deepcopy(
            dict(_mapping(receipt.get("outputs"), owner="audio outputs"))
        ),
        "external_waveform_verifier": {
            "path": None,
            "reported_status": None,
            "binding_status": "unbound_no_structured_view_identity",
            "raw_clue_effect": "none",
            "perceptual_onset_claim": None,
        },
        "legality_effect": "none",
        "candidate_retained": True,
    }
    return LoadedMP3DRawAudio(
        root=root,
        receipt_path=receipt_path,
        receipt=deepcopy(dict(receipt)),
        m1_request_id=request_id,
        room_id=room_id,
        view_id=view_id,
        listener_id=listener_id,
        raw_clues=raw_clues,
    )


def timing_authority(receipt: Mapping[str, Any]) -> str:
    timing = _mapping(receipt.get("audio_timing"), owner="audio timing")
    return _text(timing.get("authority"), owner="audio timing authority")


def apply_dynamic_audio_raw_clues(
    visual_raw_clues: Mapping[str, Any], audio_raw_clues: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach exact scheduled endpoint/event facts to all 75 visual frames."""

    result = deepcopy(dict(visual_raw_clues))
    frames = result.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise MP3DRawAudioError("visual raw clues must contain 75 frames")
    bindings = audio_raw_clues.get("visual_source_to_audio_endpoint_bindings")
    timing = _mapping(
        audio_raw_clues.get("planned_audio_timing"), owner="planned audio timing"
    )
    timing_frames = timing.get("frames")
    if (
        not isinstance(bindings, list)
        or len(bindings) != 2
        or not isinstance(timing_frames, list)
    ):
        raise MP3DRawAudioError("audio raw clues lack bindings or frame timing")
    binding_by_source = {str(item["source_id"]): item for item in bindings}
    for frame_index, frame in enumerate(frames):
        timing_frame = timing_frames[frame_index]
        frame["planned_audio_timing"] = deepcopy(timing_frame)
        sources = _mapping(
            frame.get("sources"), owner=f"visual frame {frame_index} sources"
        )
        if set(sources) != set(binding_by_source):
            raise MP3DRawAudioError(
                f"visual frame {frame_index} source IDs differ from audio bindings"
            )
        for source_id, source in sources.items():
            binding = binding_by_source[source_id]
            endpoint = str(binding["audio_endpoint_id"])
            active_events = [
                event_id
                for event_id in timing_frame["active_event_ids"]
                if event_id in binding["audio_event_ids"]
            ]
            source["native_audio"] = {
                "status": "scheduled_endpoint_and_event_identity_joined",
                "audio_endpoint_id": endpoint,
                "active_event_ids": active_events,
                "scheduled_active": bool(active_events),
                "frame_sample_interval": [
                    timing_frame["start_sample"],
                    timing_frame["end_sample_exclusive"],
                ],
                "frame_tick_interval": [
                    timing_frame["start_tick"],
                    timing_frame["end_tick_exclusive"],
                ],
                "timing_authority": "dynamic_audio_receipt.audio_timing",
                "actual_waveform_activity_verified": None,
                "perceptual_onset": None,
            }
    result["audio_timing"] = deepcopy(dict(timing))
    result["native_audio_join"] = deepcopy(dict(audio_raw_clues))
    return result


def mark_dynamic_audio_pending(visual_raw_clues: Mapping[str, Any]) -> dict[str, Any]:
    """Keep one input view and all of its source frames explicitly pending."""

    result = deepcopy(dict(visual_raw_clues))
    pending = {
        "status": "pending_no_dynamic_audio_receipt_for_this_view",
        "receipt": None,
        "visual_source_to_audio_endpoint_bindings": None,
        "planned_audio_timing": None,
        "rir_identity": None,
        "waveform_truth": None,
        "legality_effect": "none",
        "candidate_retained": True,
    }
    frames = result.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            sources = frame.get("sources") if isinstance(frame, Mapping) else None
            if isinstance(sources, Mapping):
                for source in sources.values():
                    if isinstance(source, dict):
                        source["native_audio"] = {
                            "status": "pending_no_dynamic_audio_receipt_for_this_view",
                            "audio_endpoint_id": None,
                            "active_event_ids": None,
                            "scheduled_active": None,
                            "actual_waveform_activity_verified": None,
                            "perceptual_onset": None,
                        }
    result["native_audio_join"] = pending
    return result


def unbound_external_verifier(path: Path) -> dict[str, Any]:
    """Preserve a verifier path/status without binding it to any view or claim."""

    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise MP3DRawAudioError(
            f"external audio verifier is unavailable: {path}"
        ) from exc
    if not resolved.is_file():
        raise MP3DRawAudioError(f"external audio verifier is not a file: {resolved}")
    try:
        first_line = resolved.open(encoding="utf-8").readline().strip()
    except (OSError, UnicodeError) as exc:
        raise MP3DRawAudioError(
            f"external audio verifier cannot be read: {resolved}"
        ) from exc
    reported_status = None
    for token in first_line.split():
        if token.startswith("status=") and len(token) > len("status="):
            reported_status = token.removeprefix("status=")
            break
    return {
        "path": str(resolved),
        "reported_status": reported_status,
        "binding_status": "unbound_no_structured_view_identity",
        "waveform_truth_effect": "none",
        "perceptual_onset_claim": None,
    }


__all__ = [
    "LoadedMP3DRawAudio",
    "MP3DRawAudioError",
    "apply_dynamic_audio_raw_clues",
    "dynamic_audio_request_id",
    "load_dynamic_audio_raw_clues",
    "mark_dynamic_audio_pending",
    "unbound_external_verifier",
]
