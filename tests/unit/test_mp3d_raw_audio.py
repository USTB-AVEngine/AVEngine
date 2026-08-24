from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.transforms import compose_transforms
from avengine.m5.mp3d_raw_audio import (
    MP3DRawAudioError,
    apply_dynamic_audio_raw_clues,
    load_dynamic_audio_raw_clues,
    mark_dynamic_audio_pending,
    unbound_external_verifier,
)
from avengine.m5.timeline import frame_sample_interval
from test_mp3d_raw_clues import _current_inputs, _load, _write_json
from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor


ENDPOINT_IDS = ("beagle_0_muzzle", "beagle_1_muzzle")
EVENT_INTERVALS = (
    ("beagle0_bark_1", 0, 6_000, 10_800),
    ("beagle1_bark_1", 1, 16_000, 20_800),
    ("beagle0_bark_2", 0, 28_000, 32_800),
    ("beagle1_bark_2", 1, 40_000, 44_800),
    ("beagle0_bark_3", 0, 52_000, 56_800),
    ("beagle1_bark_3", 1, 64_000, 68_800),
)


def _events() -> list[dict]:
    return [
        {
            "event_id": event_id,
            "audio_endpoint_id": ENDPOINT_IDS[axis],
            "sound_asset_id": "dog_beagle_v2_scheduled_dry",
            "dry_audio_asset_id": "dog_beagle_v2_scheduled_dry",
            "start_sample": start,
            "end_sample_exclusive": end,
            "start_tick": start * 3,
            "end_tick_exclusive": end * 3,
        }
        for event_id, axis, start, end in EVENT_INTERVALS
    ]


def _audio_frames(events: list[dict]) -> list[dict]:
    result = []
    for frame_index in range(75):
        start, end = frame_sample_interval(frame_index)
        active = [
            event
            for event in events
            if event["start_sample"] < end and start < event["end_sample_exclusive"]
        ]
        active_endpoints = {event["audio_endpoint_id"] for event in active}
        result.append(
            {
                "frame_index": frame_index,
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_tick": frame_index * 3_200,
                "end_tick_exclusive": (frame_index + 1) * 3_200,
                "active_event_ids": [event["event_id"] for event in active],
                "active_audio_endpoint_ids": [
                    endpoint
                    for endpoint in ENDPOINT_IDS
                    if endpoint in active_endpoints
                ],
            }
        )
    return result


def _audio_input(tmp_path: Path, *, name: str = "audio"):
    visual_root, request, request_path = _current_inputs(
        tmp_path, name=f"{name}-visual"
    )
    visual = _load(visual_root, request, request_path)
    visual_raw = visual.raw_clues
    identity = {
        "request_id": request["request_id"],
        "room_id": request["room_id"],
        "primary_camera_rig": request["primary_camera_rig"],
        "listener": request["listener"],
        "source_axis_order": [item["source_id"] for item in request["sources"]],
        "qa_views": request["qa_views"],
    }
    view_identity = {
        "m1_request_id": request["request_id"],
        "room_id": request["room_id"],
        "camera_rig_id": request["primary_camera_rig"]["rig_id"],
        "view_id": request["primary_camera_rig"]["view_id"],
        "listener_id": request["listener"]["listener_id"],
        "visual_capture_dir": str(visual_root.resolve()),
        "visual_receipt_path": str(visual.receipt_path.resolve()),
        "m1_request_path": str(request_path.resolve()),
        "capture_declared_m1_request_path": str(request_path.resolve()),
        "capture_m1_identity": identity,
        "visual_source_axis_order": ["source0", "source1"],
    }
    world_from_listener = compose_transforms(
        request["primary_camera_rig"]["world_from_rig"],
        request["listener"]["rig_from_listener"],
    )
    x, y, z, w = world_from_listener["rotation_xyzw"]
    listener_position = world_from_listener["translation_m"]
    listener_orientation = [w, x, y, z]
    events = _events()
    visual_axes = visual_raw["identity_axes"]["axis_bindings"]
    bindings = []
    for axis in range(2):
        endpoint = ENDPOINT_IDS[axis]
        endpoint_events = [
            event for event in events if event["audio_endpoint_id"] == endpoint
        ]
        bindings.append(
            {
                "axis_index": axis,
                "source_id": f"source{axis}",
                "actor_id": f"actor{axis}",
                "emitter_link_name": visual_axes[axis]["emitter_link_name"],
                "audio_endpoint_id": endpoint,
                "audio_endpoint_binding": {
                    "kind": "entity_anchor",
                    "entity_instance_id": f"beagle_{axis}",
                    "entity_asset_id": "beagle_asset",
                    "entity_asset_revision": "v1",
                    "emitter_anchor_id": "muzzle",
                },
                "audio_event_ids": [item["event_id"] for item in endpoint_events],
                "event_bindings": deepcopy(endpoint_events),
                "dry_audio_asset_ids": ["dog_beagle_v2_scheduled_dry"],
                "sound_asset_ids": ["dog_beagle_v2_scheduled_dry"],
                "dry_audio_assets": [],
                "input_authority": {},
            }
        )
    keyframe_samples = [frame_sample_interval(index * 3)[0] for index in range(25)]
    registrations = []
    ir_identities = []
    for keyframe_index, sample_index in enumerate(keyframe_samples):
        visual_frame_index = keyframe_index * 3
        registrations.append(
            {
                "authority": "native_registration_readback",
                "keyframe_index": keyframe_index,
                "sample_index": sample_index,
                "tick": sample_index * 3,
                "visual_frame_index": visual_frame_index,
                "listener": {
                    "listener_id": "listener0",
                    "native_realized": True,
                    "canonical_native_index": 0,
                    "position_m": listener_position,
                    "orientation_wxyz": listener_orientation,
                },
                "sources": [
                    {
                        "source_id": ENDPOINT_IDS[axis],
                        "native_realized": True,
                        "canonical_native_index": axis,
                        "position_m": visual_raw["frames"][visual_frame_index][
                            "sources"
                        ][f"source{axis}"]["actual_emitter_position_m"],
                    }
                    for axis in range(2)
                ],
            }
        )
        ir_identities.append(
            {
                ENDPOINT_IDS[0]: f"ir-0-{keyframe_index}",
                ENDPOINT_IDS[1]: f"ir-1-{keyframe_index}",
            }
        )
    receipt = {
        "schema": "avengine_m5_current_mp3d_dynamic_audio_v1",
        "status": "pass",
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "audio": {
            "sample_rate_hz": 16_000,
            "sample_count": 80_000,
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
        },
        "audio_program": {"event_count": 6, "program_id": "turns"},
        "audio_timing": {
            "authority": "validated materialized AudioProgram scheduled intervals",
            "event_count": 6,
            "events": events,
            "frames": _audio_frames(events),
            "waveform_activity_verification": {
                "status": "pending",
                "actual_waveform_activity_verified": None,
                "audible_onset_claim": False,
                "reason": "no decoded waveform verifier",
            },
        },
        "sources": {
            "source_ids": list(ENDPOINT_IDS),
            "audio_endpoint_ids": list(ENDPOINT_IDS),
            "visual_source_ids": ["source0", "source1"],
            "frame_count": 75,
            "identity_status": "verified_complete_unique_bijection",
            "identity_reason": None,
            "identity_bindings": bindings,
        },
        "listener": {
            "listener_id": "listener0",
            "position_m": listener_position,
            "orientation_wxyz": listener_orientation,
            "view_identity": view_identity,
        },
        "rir": {
            "stride_frames": 3,
            "keyframe_count": 25,
            "keyframe_samples": keyframe_samples,
            "trajectory_sha256": "trajectory-id",
            "listener_view_render_policy": "fresh_per_output_invocation",
            "prior_listener_rir_reused": False,
            "listener_view_identity": view_identity,
            "sequence_metadata_registration": {
                "listener_id": "listener0",
                "source_ids": list(ENDPOINT_IDS),
                "trajectory_sha256": "trajectory-id",
                "endpoint_receipts": registrations,
                "ir_sha256_by_keyframe_source": ir_identities,
            },
        },
        "outputs": {
            "audio/binaural/mixture.wav": "declared-output-id",
        },
    }
    audio_root = tmp_path / name
    audio_root.mkdir()
    _write_json(audio_root / "research_receipt.json", receipt)
    return audio_root, request, request_path, visual, receipt


def _join(audio_root: Path, request: dict, request_path: Path, visual):
    return load_dynamic_audio_raw_clues(
        audio_root,
        request=request,
        request_path=request_path,
        visual_receipt_path=visual.receipt_path,
        visual_raw_clues=visual.raw_clues,
    )


def test_joins_exact_schedule_endpoint_and_rir_identities(tmp_path: Path) -> None:
    audio_root, request, request_path, visual, _receipt = _audio_input(tmp_path)

    loaded = _join(audio_root, request, request_path, visual)
    attached = apply_dynamic_audio_raw_clues(visual.raw_clues, loaded.raw_clues)

    assert loaded.raw_clues["status"] == (
        "native_audio_schedule_and_rir_identity_joined_waveform_truth_pending"
    )
    assert len(loaded.raw_clues["planned_audio_timing"]["frames"]) == 75
    assert len(loaded.raw_clues["planned_audio_timing"]["events"]) == 6
    assert loaded.raw_clues["rir_identity"]["keyframe_count"] == 25
    assert (
        len(loaded.raw_clues["rir_identity"]["per_keyframe_source_ir_identities"]) == 25
    )
    assert loaded.raw_clues["rir_identity"]["prior_rir_reused"] is False
    assert (
        loaded.raw_clues["waveform_truth"]["actual_waveform_activity_verified"] is None
    )
    assert loaded.raw_clues["declared_outputs_uninspected"] == {
        "audio/binaural/mixture.wav": "declared-output-id"
    }
    source0 = attached["frames"][5]["sources"]["source0"]["native_audio"]
    source1 = attached["frames"][5]["sources"]["source1"]["native_audio"]
    assert source0["audio_endpoint_id"] == "beagle_0_muzzle"
    assert source0["active_event_ids"] == ["beagle0_bark_1"]
    assert source0["scheduled_active"] is True
    assert source0["actual_waveform_activity_verified"] is None
    assert source0["perceptual_onset"] is None
    assert source1["active_event_ids"] == []


def test_partial_one_of_twelve_keeps_other_views_pending_and_retained(
    tmp_path: Path,
) -> None:
    audio_root, request, request_path, visual, _receipt = _audio_input(tmp_path)
    joined_audio = _join(audio_root, request, request_path, visual).raw_clues
    results = [apply_dynamic_audio_raw_clues(visual.raw_clues, joined_audio)]
    results.extend(mark_dynamic_audio_pending(visual.raw_clues) for _ in range(11))

    assert len(results) == 12
    assert results[0]["native_audio_join"]["status"].startswith("native_audio")
    assert all(
        item["native_audio_join"]["status"]
        == "pending_no_dynamic_audio_receipt_for_this_view"
        for item in results[1:]
    )
    assert all(item["candidate_retained"] for item in results)
    assert all(
        item["frames"][0]["sources"]["source0"]["native_audio"]["status"]
        == "pending_no_dynamic_audio_receipt_for_this_view"
        for item in results[1:]
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("m1", "m1_request_id differs"),
        ("room", "room_id differs"),
        ("view", "view_id differs"),
        ("listener", "listener_id differs"),
        ("source_axis", "source axis"),
        ("ir", "lacks one per-source IR identity"),
    ),
)
def test_rejects_wrong_view_listener_source_axes_and_missing_ir(
    tmp_path: Path, damage: str, message: str
) -> None:
    audio_root, request, request_path, visual, receipt = _audio_input(
        tmp_path, name=damage
    )
    if damage in {"m1", "room", "view", "listener"}:
        field = {
            "m1": "m1_request_id",
            "room": "room_id",
            "view": "view_id",
            "listener": "listener_id",
        }[damage]
        receipt["listener"]["view_identity"][field] = "wrong"
    elif damage == "source_axis":
        receipt["sources"]["visual_source_ids"] = ["source1", "source0"]
    else:
        receipt["rir"]["sequence_metadata_registration"][
            "ir_sha256_by_keyframe_source"
        ][4].pop("beagle_1_muzzle")
    _write_json(audio_root / "research_receipt.json", receipt)

    with pytest.raises(MP3DRawAudioError, match=message):
        _join(audio_root, request, request_path, visual)


def test_rejects_duplicate_audio_request_identity(tmp_path: Path) -> None:
    audio_root, *_unused = _audio_input(tmp_path)

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="repeat one M1 request"
    ):
        descriptor._dynamic_audio_roots([audio_root, audio_root])


def test_cli_accepts_repeated_audio_dirs_and_external_verifier_is_unbound(
    tmp_path: Path,
) -> None:
    verifier = tmp_path / "verification.log"
    verifier.write_text("status=pass waveform_window_activity_status=verified\n")
    record = unbound_external_verifier(verifier)
    args = descriptor.parse_args(
        [
            "--viewpoint-bank",
            "bank.json",
            "--probe-root",
            "probes",
            "--dynamic-audio-dir",
            "audio-a",
            "--dynamic-audio-dir",
            "audio-b",
            "--external-audio-verifier",
            str(verifier),
            "--output",
            "/tmp/output",
        ]
    )

    assert args.dynamic_audio_dir == [Path("audio-a"), Path("audio-b")]
    assert record == {
        "path": str(verifier.resolve()),
        "reported_status": "pass",
        "binding_status": "unbound_no_structured_view_identity",
        "waveform_truth_effect": "none",
        "perceptual_onset_claim": None,
    }
