from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import avengine.m5.current_mp3d_dynamic_audio as dynamic_audio
from avengine.contracts.json_io import canonical_json_sha256
from avengine.m4.audio import read_float32_wav
from avengine.m4.spatial import rlr_foa_wav_metadata
from avengine.m5.acoustics import DynamicRIRSequence
from avengine.m5.current_mp3d_dynamic_audio import (
    CURRENT_MP3D_REGION_SOURCE_AXIS_BINDINGS,
    CURRENT_MP3D_SOURCE_AXIS_BINDINGS,
    CurrentMP3DDynamicAudioError,
    _listener_view_identity,
    _source_axis_bindings_for_visual_capture,
    build_audio_timing_evidence,
    build_source_identity_evidence,
    listener_pose_from_m1_request,
    load_captured_source_paths,
)
from avengine.m5_1.acoustics import research_review_trajectory_record
from avengine.m5.timeline import frame_sample_interval
from avengine.m6.audio_program import validate_audio_program
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM_PATH = (
    REPOSITORY
    / "examples/m5/current_mp3d/audio_programs"
    / "current_mp3d_two_beagle_turn_taking_v1.json"
)


def test_apartment_dynamic_audio_tool_description_declares_dual_output() -> None:
    tool_path = (
        REPOSITORY
        / "tools/m7/render_current_apartment_dynamic_audio.py"
    )
    description = ast.get_docstring(
        ast.parse(tool_path.read_text(encoding="utf-8"))
    )
    assert description is not None
    assert description.startswith(
        "Render motion-following binaural plus native FOA audio"
    )

def _axis_bindings() -> list[dict[str, object]]:
    return [dict(item) for item in CURRENT_MP3D_SOURCE_AXIS_BINDINGS]


def _region_axis_bindings() -> list[dict[str, object]]:
    return [dict(item) for item in CURRENT_MP3D_REGION_SOURCE_AXIS_BINDINGS]


def _valid_m1_request(*, listener_id: str = "listener0") -> dict[str, object]:
    request = json.loads(
        (REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json").read_text(
            encoding="utf-8"
        )
    )
    request["listener"]["listener_id"] = listener_id
    return request


def _capture_identity(request: dict[str, object]) -> dict[str, object]:
    return {
        "request_id": request["request_id"],
        "room_id": request["room_id"],
        "primary_camera_rig": request["primary_camera_rig"],
        "listener": request["listener"],
        "source_axis_order": [item["source_id"] for item in request["sources"]],
        "qa_views": request.get("qa_views", []),
    }


def _write_visual_capture(
    root: Path,
    *,
    receipt_source_ids: list[str] | None = None,
    stored_positions: np.ndarray | None = None,
    m1_request: dict[str, object] | None = None,
    declared_m1_path: Path | None = None,
    source_axis_bindings: list[dict[str, object]] | None = None,
    camera_route_decoupled: bool = False,
) -> tuple[list[dict[str, object]], np.ndarray]:
    bindings = (
        _axis_bindings() if source_axis_bindings is None else source_axis_bindings
    )
    request = _valid_m1_request() if m1_request is None else m1_request
    m1_path = root / "m1_request.json"
    m1_path.write_text(json.dumps(request), encoding="utf-8")
    positions = np.asarray(
        [
            [
                [float(index), 0.5, 0.0],
                [float(index), 0.5, 1.0],
            ]
            for index in range(75)
        ],
        dtype=np.float64,
    )
    frames = [
        {
            "frame_index": index,
            "pts_ticks": index * 3_200,
            "pose_alignment": {
                "array_frame_index": index,
                "actor_axis_order": [str(item["actor_id"]) for item in bindings],
                "source_axis_order": [str(item["source_id"]) for item in bindings],
            },
            "source_positions_m": positions[index].tolist(),
        }
        for index in range(75)
    ]
    arrays = root / "arrays"
    arrays.mkdir()
    np.save(
        arrays / "source_positions_m.npy",
        positions if stored_positions is None else stored_positions,
    )
    (root / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    (root / "research_receipt.json").write_text(
        json.dumps(
            {
                "status": "research_only",
                "research_only": True,
                "episode_counted": False,
                "capture": {
                    "native_habitat_started": True,
                    "m1_request_id": request["request_id"],
                    "room_id": request["room_id"],
                    "frame_count": 75,
                },
                "inputs": {
                    "m1_request": str(
                        m1_path if declared_m1_path is None else declared_m1_path
                    )
                },
                "selected_room": {
                    "room_id": request["room_id"],
                    "room_kind": "habitat_native",
                },
                "m1_capture_identity": _capture_identity(request),
                **(
                    {
                        "case": {
                            "camera_binding_source": "explicit_posthoc_rebind",
                            "camera_participation_in_route_generation": False,
                        }
                    }
                    if camera_route_decoupled
                    else {}
                ),
                "actors": {
                    "source_ids": receipt_source_ids
                    or [str(item["source_id"]) for item in bindings],
                    "actor_ids": [str(item["actor_id"]) for item in bindings],
                    "semantic_ids": [int(item["semantic_id"]) for item in bindings],
                    "emitter_link_names": [
                        str(item["emitter_link_name"]) for item in bindings
                    ],
                    "axis_bindings": [
                        {
                            "axis_index": index,
                            "actor_id": item["actor_id"],
                            "source_id": item["source_id"],
                            "semantic_id": item["semantic_id"],
                            "emitter_link_name": item["emitter_link_name"],
                            # Habitat link IDs are local to each articulated actor.
                            "runtime_emitter_link_id": 100,
                            "audio_endpoint_status": "pending",
                            "audio_endpoint_id": None,
                        }
                        for index, item in enumerate(bindings)
                    ],
                },
                "artifacts": {
                    "frame_records": "frame_records.json",
                    "source_positions_m": "arrays/source_positions_m.npy",
                },
            }
        ),
        encoding="utf-8",
    )
    return bindings, positions


def test_region_visual_selects_explicit_source1_source2_bridge(
    tmp_path: Path,
) -> None:
    request = _valid_m1_request()
    bindings, _ = _write_visual_capture(
        tmp_path,
        m1_request=request,
        source_axis_bindings=_region_axis_bindings(),
        camera_route_decoupled=True,
    )
    selected = _source_axis_bindings_for_visual_capture(tmp_path)
    assert [item["source_id"] for item in selected] == ["source1", "source2"]
    trajectories, axes, evidence = dynamic_audio._load_captured_source_evidence(
        tmp_path, selected
    )
    assert list(trajectories) == ["beagle_0_muzzle", "beagle_1_muzzle"]
    identity = _listener_view_identity(
        m1_request=request,
        m1_request_path=tmp_path / "m1_request.json",
        visual_evidence=evidence,
        source_axis_bindings=axes,
    )
    assert [item["source_id"] for item in bindings] == ["source1", "source2"]
    assert identity["visual_source_axis_order"] == ["source1", "source2"]
    assert identity["m1_request_source_axis_order"] == ["source0", "source1"]
    assert identity["camera_route_decoupled"] is True


def _resolved_dry_assets() -> dict[str, dict[str, str]]:
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    sound = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == "dog_beagle_v2_scheduled_dry"
    )
    return {
        sound["sound_asset_id"]: {
            "path": "/external/research/beagle.wav",
            "sha256": sound["dry_audio"]["sha256"],
        }
    }


def _timing_program(events: list[dict[str, object]]) -> dict[str, object]:
    return {
        "timeline": {
            "time_base_hz": 48_000,
            "ticks_per_frame": 3_200,
            "video_fps": 15,
            "frame_count": 75,
            "sample_rate_hz": 16_000,
            "ticks_per_sample": 3,
            "sample_count": 80_000,
        },
        "candidate_source_endpoint_ids": ["endpoint0", "endpoint1"],
        "events": events,
    }


def _event(
    event_id: str, endpoint_id: str, start_sample: int, end_sample: int
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "source_endpoint_id": endpoint_id,
        "sound_asset_id": "dry0",
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_tick": start_sample * 3,
        "end_tick_exclusive": end_sample * 3,
    }


def test_turn_taking_program_validates_against_repository_registries() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert errors == []
    assert program["mode"] == "sequential_sources"
    assert program["candidate_source_endpoint_ids"] == [
        "beagle_0_muzzle",
        "beagle_1_muzzle",
    ]


def test_turn_taking_program_alternates_without_overlap() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    events = program["events"]
    spans = [(event["start_sample"], event["end_sample_exclusive"]) for event in events]
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    sources = [event["source_endpoint_id"] for event in events]
    assert all(sources[i] != sources[i + 1] for i in range(len(sources) - 1))
    assert set(sources) == {"beagle_0_muzzle", "beagle_1_muzzle"}


def test_load_captured_source_paths_reads_all_frames(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    trajectories = load_captured_source_paths(tmp_path, bindings)
    assert len(trajectories["beagle_0_muzzle"]) == 75
    assert trajectories["beagle_1_muzzle"][74] == [74.0, 0.5, 1.0]


def test_load_captured_source_paths_rejects_wrong_slot_count(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    value = json.loads((tmp_path / "frame_records.json").read_text())
    value["frames"][0]["source_positions_m"] = [[0.0, 0.0, 0.0]]
    (tmp_path / "frame_records.json").write_text(json.dumps(value))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="one visual position"):
        load_captured_source_paths(tmp_path, bindings)


def test_load_captured_source_paths_rejects_short_captures(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    value = json.loads((tmp_path / "frame_records.json").read_text())
    value["frames"] = value["frames"][:10]
    (tmp_path / "frame_records.json").write_text(json.dumps(value))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="75 episode frames"):
        load_captured_source_paths(tmp_path, bindings)


def test_visual_receipt_axis_swap_is_identity_corruption(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(
        tmp_path, receipt_source_ids=["source1", "source0"]
    )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="axis order differs"):
        load_captured_source_paths(tmp_path, bindings)


def test_visual_receipt_axis_binding_drift_is_rejected(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    receipt_path = tmp_path / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["actors"]["axis_bindings"][0]["source_id"] = "source1"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(CurrentMP3DDynamicAudioError, match="explicit binding"):
        load_captured_source_paths(tmp_path, bindings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "pass"),
        ("research_only", False),
        ("episode_counted", True),
    ],
)
def test_visual_receipt_research_boundary_is_required(
    tmp_path: Path, field: str, value: object
) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    receipt_path = tmp_path / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(CurrentMP3DDynamicAudioError, match="research_only"):
        load_captured_source_paths(tmp_path, bindings)


@pytest.mark.parametrize("mutation", ["pts_ticks", "pose_axis"])
def test_visual_frame_clock_and_pose_axis_are_required(
    tmp_path: Path, mutation: str
) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    records_path = tmp_path / "frame_records.json"
    value = json.loads(records_path.read_text(encoding="utf-8"))
    if mutation == "pts_ticks":
        value["frames"][12]["pts_ticks"] += 1
        error = "pts_ticks"
    else:
        value["frames"][12]["pose_alignment"]["source_axis_order"] = [
            "source1",
            "source0",
        ]
        error = "pose_alignment"
    records_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CurrentMP3DDynamicAudioError, match=error):
        load_captured_source_paths(tmp_path, bindings)


def test_visual_npy_axis_swap_against_frame_records_is_rejected(
    tmp_path: Path,
) -> None:
    positions = np.asarray(
        [[[float(index), 0.5, 0.0], [float(index), 0.5, 1.0]] for index in range(75)],
        dtype=np.float64,
    )
    bindings, _ = _write_visual_capture(
        tmp_path, stored_positions=positions[:, ::-1, :]
    )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="NPY and frame-record"):
        load_captured_source_paths(tmp_path, bindings)


def test_visual_complex_source_position_array_is_rejected(tmp_path: Path) -> None:
    positions = np.asarray(
        [[[float(index), 0.5, 0.0], [float(index), 0.5, 1.0]] for index in range(75)],
        dtype=np.complex128,
    )
    bindings, _ = _write_visual_capture(tmp_path, stored_positions=positions)
    with pytest.raises(CurrentMP3DDynamicAudioError, match="complex dtype"):
        load_captured_source_paths(tmp_path, bindings)


def test_spatial_source_exchange_preserves_identity_axes(tmp_path: Path) -> None:
    bindings, positions = _write_visual_capture(tmp_path)
    exchanged = positions.copy()
    exchanged[:, 0, 0] = np.arange(75, dtype=np.float64)
    exchanged[:, 1, 0] = np.arange(74, -1, -1, dtype=np.float64)
    np.save(tmp_path / "arrays/source_positions_m.npy", exchanged)
    frames = json.loads((tmp_path / "frame_records.json").read_text())["frames"]
    for index, frame in enumerate(frames):
        frame["source_positions_m"] = exchanged[index].tolist()
    (tmp_path / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    trajectories = load_captured_source_paths(tmp_path, bindings)
    assert trajectories["beagle_0_muzzle"][74][0] == 74.0
    assert trajectories["beagle_1_muzzle"][74][0] == 0.0


def test_source_identity_is_complete_and_carries_per_entity_audio_ids() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    records = build_source_identity_evidence(
        _axis_bindings(),
        candidate_source_endpoint_ids=program["candidate_source_endpoint_ids"],
        endpoint_registry=endpoints,
        sound_registry=sounds,
        materialized_program=program,
        dry_asset_bindings=_resolved_dry_assets(),
    )
    assert [item["axis_index"] for item in records] == [0, 1]
    assert [item["source_id"] for item in records] == ["source0", "source1"]
    assert [item["actor_id"] for item in records] == ["actor0", "actor1"]
    assert [item["audio_endpoint_id"] for item in records] == [
        "beagle_0_muzzle",
        "beagle_1_muzzle",
    ]
    assert all(item["emitter_link_name"] == "beagle Xtra Mouth" for item in records)
    assert records[0]["audio_event_ids"] == [
        "beagle0_bark_1",
        "beagle0_bark_2",
        "beagle0_bark_3",
    ]
    assert records[1]["dry_audio_asset_ids"] == ["dog_beagle_v2_scheduled_dry"]
    assert records[0]["event_bindings"][0]["start_sample"] == 6000
    assert records[0]["event_bindings"][0]["start_tick"] == 18000


def test_source_identity_rejects_swapped_audio_endpoints() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    bindings = _axis_bindings()
    bindings[0]["audio_endpoint_id"], bindings[1]["audio_endpoint_id"] = (
        bindings[1]["audio_endpoint_id"],
        bindings[0]["audio_endpoint_id"],
    )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="ordered unique bijection"):
        build_source_identity_evidence(
            bindings,
            candidate_source_endpoint_ids=program["candidate_source_endpoint_ids"],
            endpoint_registry=endpoints,
            sound_registry=sounds,
            materialized_program=program,
            dry_asset_bindings=_resolved_dry_assets(),
        )


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_source_identity_rejects_duplicate_or_missing_endpoint(mutation: str) -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    bindings = _axis_bindings()
    if mutation == "duplicate":
        bindings[1]["audio_endpoint_id"] = bindings[0]["audio_endpoint_id"]
    else:
        bindings.pop()
    with pytest.raises(CurrentMP3DDynamicAudioError):
        build_source_identity_evidence(
            bindings,
            candidate_source_endpoint_ids=program["candidate_source_endpoint_ids"],
            endpoint_registry=endpoints,
            sound_registry=sounds,
            materialized_program=program,
            dry_asset_bindings=_resolved_dry_assets(),
        )


def test_audio_timing_handles_no_events_without_audible_onset_claim() -> None:
    evidence = build_audio_timing_evidence(_timing_program([]))
    assert evidence["event_count"] == 0
    assert len(evidence["frames"]) == 75
    assert all(not item["active_event_ids"] for item in evidence["frames"])
    assert evidence["frames"][-1]["end_sample_exclusive"] == 80_000
    assert evidence["frames"][-1]["end_tick_exclusive"] == 240_000
    waveform = evidence["waveform_activity_verification"]
    assert waveform["status"] == "pending"
    assert waveform["actual_waveform_activity_verified"] is None
    assert waveform["audible_onset_claim"] is False


def test_audio_timing_preserves_cross_source_overlap_and_exact_boundaries() -> None:
    boundary = frame_sample_interval(1)[0]
    evidence = build_audio_timing_evidence(
        _timing_program(
            [
                _event("event0", "endpoint0", boundary - 100, boundary + 100),
                _event("event1", "endpoint1", boundary, boundary + 200),
            ]
        )
    )
    assert evidence["frames"][0]["active_event_ids"] == ["event0"]
    assert evidence["frames"][1]["active_event_ids"] == ["event0", "event1"]
    assert evidence["frames"][1]["active_audio_endpoint_ids"] == [
        "endpoint0",
        "endpoint1",
    ]
    assert evidence["events"][1]["start_sample"] == boundary
    assert evidence["events"][1]["start_tick"] == boundary * 3


def test_listener_view_identity_uses_content_not_declared_path(tmp_path: Path) -> None:
    request = _valid_m1_request()
    declared = tmp_path / "capture-time-location.json"
    declared.write_text(json.dumps(request), encoding="utf-8")
    bindings, _ = _write_visual_capture(
        tmp_path, m1_request=request, declared_m1_path=declared
    )
    receipt_path = tmp_path / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["m1_capture_identity"]["future_optional_note"] = "retained"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    _, axes, evidence = dynamic_audio._load_captured_source_evidence(tmp_path, bindings)
    supplied = tmp_path / "supplied-copy.json"
    supplied.write_text(json.dumps(request), encoding="utf-8")
    identity = _listener_view_identity(
        m1_request=request,
        m1_request_path=supplied,
        visual_evidence=evidence,
        source_axis_bindings=axes,
    )
    assert identity["m1_request_path"] == str(supplied)
    assert identity["capture_declared_m1_request_path"] == str(declared)
    assert identity["m1_request_id"] == request["request_id"]
    assert identity["capture_m1_identity"]["future_optional_note"] == "retained"


def test_listener_view_identity_rejects_same_path_m1_rewrite(tmp_path: Path) -> None:
    original = _valid_m1_request()
    bindings, _ = _write_visual_capture(tmp_path, m1_request=original)
    _, axes, evidence = dynamic_audio._load_captured_source_evidence(tmp_path, bindings)
    same_path = tmp_path / "m1_request.json"
    rewritten = json.loads(json.dumps(original))
    rewritten["primary_camera_rig"]["shared_calibration"]["hfov_degrees"] = 73.0
    same_path.write_text(json.dumps(rewritten), encoding="utf-8")
    with pytest.raises(CurrentMP3DDynamicAudioError, match="primary_camera_rig"):
        _listener_view_identity(
            m1_request=rewritten,
            m1_request_path=same_path,
            visual_evidence=evidence,
            source_axis_bindings=axes,
        )


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("request_id", "request_id"),
        ("room_id", "room_id"),
        ("world_from_rig", "primary_camera_rig"),
        ("rig_from_listener", "listener"),
        ("source_axis_order", "source order"),
        ("qa_views", "qa_views"),
    ],
)
def test_listener_view_identity_compares_every_capture_field(
    tmp_path: Path, field: str, expected_error: str
) -> None:
    original = _valid_m1_request()
    bindings, _ = _write_visual_capture(tmp_path, m1_request=original)
    _, axes, evidence = dynamic_audio._load_captured_source_evidence(tmp_path, bindings)
    supplied = json.loads(json.dumps(original))
    if field == "request_id":
        supplied["request_id"] = "rewritten_request"
    elif field == "room_id":
        supplied["room_id"] = "rewritten_room"
    elif field == "world_from_rig":
        supplied["primary_camera_rig"]["world_from_rig"]["translation_m"][0] += 1.0
    elif field == "rig_from_listener":
        supplied["listener"]["rig_from_listener"]["translation_m"][0] += 0.1
    elif field == "source_axis_order":
        supplied["sources"] = list(reversed(supplied["sources"]))
    else:
        supplied["qa_views"] = []
    supplied_path = tmp_path / f"supplied-{field}.json"
    supplied_path.write_text(json.dumps(supplied), encoding="utf-8")
    with pytest.raises(CurrentMP3DDynamicAudioError, match=expected_error):
        _listener_view_identity(
            m1_request=supplied,
            m1_request_path=supplied_path,
            visual_evidence=evidence,
            source_axis_bindings=axes,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing_keyframe", "every keyframe"),
        ("missing_source", "every source"),
        ("trajectory", "trajectory_sha256"),
    ],
)
def test_sequence_registration_requires_existing_hash_metadata_coverage(
    mutation: str, expected_error: str
) -> None:
    source_ids = ("beagle_0_muzzle", "beagle_1_muzzle")
    endpoint_receipts = [
        {
            "authority": "native_registration_readback",
            "sources": [
                {"source_id": source_id, "native_realized": True}
                for source_id in source_ids
            ],
            "listener": {"listener_id": "listener_x", "native_realized": True},
        }
        for _ in range(2)
    ]
    hashes = [
        {source_id: f"rir-{index}-{source_id}" for source_id in source_ids}
        for index in range(2)
    ]
    sequence = SimpleNamespace(
        keyframe_samples=(0, 3200),
        trajectory_sha256="trajectory",
        metadata={
            "listener_id": "listener_x",
            "source_ids": list(source_ids),
            "endpoint_receipts": endpoint_receipts,
            "trajectory_sha256": "trajectory",
            "ir_sha256_by_keyframe_source": hashes,
        },
    )
    if mutation == "missing_keyframe":
        sequence.metadata["ir_sha256_by_keyframe_source"] = hashes[:1]
    elif mutation == "missing_source":
        sequence.metadata["ir_sha256_by_keyframe_source"][1].pop(source_ids[1])
    else:
        sequence.metadata["trajectory_sha256"] = "different"
    with pytest.raises(CurrentMP3DDynamicAudioError, match=expected_error):
        dynamic_audio._sequence_registration_evidence(
            sequence,
            expected_listener_id="listener_x",
            expected_source_ids=source_ids,
        )


def test_render_core_forwards_listener_id_and_uses_materialized_event_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    materialized = json.loads(json.dumps(program))
    materialized["events"] = materialized["events"][:1]
    source_ids = tuple(program["candidate_source_endpoint_ids"])
    buses = {source_id: np.zeros(80_000, dtype=np.float64) for source_id in source_ids}
    captured_listener_ids: list[str] = []

    monkeypatch.setattr(
        dynamic_audio,
        "_asset_bindings",
        lambda *_args, **_kwargs: {
            "dog_beagle_v2_scheduled_dry": {
                "path": str(tmp_path / "dry.wav"),
                "sha256": "0" * 64,
            }
        },
    )
    monkeypatch.setattr(
        dynamic_audio,
        "assemble_audio_program_dry_buses",
        lambda *_args, **_kwargs: SimpleNamespace(
            materialized_program=materialized,
            dry_audio=SimpleNamespace(buses=buses),
        ),
    )
    grid = object()
    monkeypatch.setattr(
        dynamic_audio, "build_strided_review_keyframes", lambda *_args, **_kwargs: grid
    )
    monkeypatch.setattr(
        dynamic_audio, "_load_simulation_request", lambda _path: (object(), object())
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_compiled_acoustic_scene",
        lambda *_args, **_kwargs: object(),
    )

    def fake_rir_sequence(
        *_args: object,
        grid: object,
        hrtf_file_path: str,
        listener_id: str,
    ) -> SimpleNamespace:
        assert grid is not None
        assert hrtf_file_path == str((tmp_path / "hrtf.sofa").resolve())
        captured_listener_ids.append(listener_id)
        endpoint_receipt = {
            "keyframe_index": 0,
            "visual_frame_index": 0,
            "tick": 0,
            "sample_index": 0,
            "authority": "native_registration_readback",
            "sources": [
                {"source_id": source_id, "native_realized": True}
                for source_id in source_ids
            ],
            "listener": {
                "listener_id": listener_id,
                "native_realized": True,
            },
        }
        return SimpleNamespace(
            source_ids=source_ids,
            keyframe_ticks=(0,),
            keyframe_samples=(0,),
            trajectory_sha256="mock-trajectory",
            layout_type="binaural",
            layout_id="rlr_binaural_lr_v1",
            channel_labels=("left", "right"),
            metadata={
                "listener_id": listener_id,
                "source_ids": list(source_ids),
                "endpoint_receipts": [endpoint_receipt],
                "trajectory_sha256": "mock-trajectory",
                "ir_sha256_by_keyframe_source": [
                    {source_id: f"rir-{source_id}" for source_id in source_ids}
                ],
            },
        )

    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_binaural_rir_sequence",
        fake_rir_sequence,
    )

    def fake_foa_sequence(
        *_args: object,
        grid: object,
        listener_id: str,
    ) -> SimpleNamespace:
        sequence = fake_rir_sequence(
            *_args,
            grid=grid,
            hrtf_file_path=str((tmp_path / "hrtf.sofa").resolve()),
            listener_id=listener_id,
        )
        sequence.layout_type = "ambisonics"
        sequence.layout_id = "rlr_foa_acn_n3d_world_v1"
        sequence.channel_labels = ("W", "Y", "Z", "X")
        return sequence

    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_foa_rir_sequence",
        fake_foa_sequence,
    )
    stems = {
        source_id: SimpleNamespace(episode=np.zeros((2, 80_000), dtype=np.float64))
        for source_id in source_ids
    }
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_binaural_audio",
        lambda *_args, **_kwargs: (
            stems,
            np.zeros((2, 80_000), dtype=np.float64),
        ),
    )
    foa_stems = {
        source_id: SimpleNamespace(episode=np.zeros((4, 80_000), dtype=np.float64))
        for source_id in source_ids
    }
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_foa_audio",
        lambda *_args, **_kwargs: (
            foa_stems,
            np.zeros((4, 80_000), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "write_float32_wav",
        lambda path, _samples, _rate, **_kwargs: Path(path).write_bytes(b"mock-wav"),
    )
    for name in ("simulation.json", "package.json", "hrtf.sofa"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    custom_listener_id = "listener_custom"
    receipt = dynamic_audio.render_dynamic_research_audio(
        source_trajectories_m={
            source_id: [[float(index), 0.5, float(axis)] for index in range(75)]
            for axis, source_id in enumerate(source_ids)
        },
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        simulation_request_path=tmp_path / "simulation.json",
        package_manifest_path=tmp_path / "package.json",
        audio_program_path=PROGRAM_PATH,
        source_endpoint_registry_path=(
            REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
        ),
        sound_asset_registry_path=(
            REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
        ),
        external_sound_asset_paths={},
        hrtf_file_path=tmp_path / "hrtf.sofa",
        output_path=tmp_path / "output",
        position_authority="unit-test visual axes",
        listener_authority="unit-test M1 identity",
        listener_id=custom_listener_id,
        listener_view_identity={"listener_id": custom_listener_id},
    )
    assert captured_listener_ids == [custom_listener_id, custom_listener_id]
    assert receipt["listener"]["listener_id"] == custom_listener_id
    registration = receipt["rir"]["sequence_metadata_registration"]
    assert registration["listener_id"] == custom_listener_id
    assert registration["source_ids"] == list(source_ids)
    assert registration["trajectory_sha256"] == "mock-trajectory"
    assert registration["ir_sha256_by_keyframe_source"] == [
        {source_id: f"rir-{source_id}" for source_id in source_ids}
    ]
    assert (
        registration["endpoint_receipts"][0]["listener"]["listener_id"]
        == custom_listener_id
    )
    assert receipt["audio_program"]["event_count"] == 1
    assert receipt["audio_timing"]["event_count"] == 1
    assert receipt["audio_timing"]["waveform_activity_verification"] == {
        "status": "pending",
        "actual_waveform_activity_verified": None,
        "audible_onset_claim": False,
        "reason": (
            "this render records scheduled intervals but does not run an "
            "independent decoded-waveform activity/onset detector"
        ),
    }
    assert receipt["rir"]["prior_listener_rir_reused"] is False


def test_listener_pose_composes_camera_colocated_listener() -> None:
    request = json.loads(
        (REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json").read_text(
            encoding="utf-8"
        )
    )
    position, orientation_wxyz = listener_pose_from_m1_request(request)
    assert len(position) == 3
    assert len(orientation_wxyz) == 4
    assert abs(float(np.linalg.norm(np.asarray(orientation_wxyz))) - 1.0) < 1.0e-6


def _fake_dynamic_sequence(
    grid, *, layout: str, listener_id: str = "listener0"
) -> DynamicRIRSequence:
    if layout == "binaural":
        channel_count = 2
        layout_type = "binaural"
        layout_id = "rlr_binaural_lr_v1"
        labels = ("left", "right")
        metadata = {
            "normalization": "not_applicable",
            "coordinate_frame": "listener_local",
        }
    else:
        channel_count = 4
        layout_type = "ambisonics"
        layout_id = "rlr_foa_acn_n3d_world_v1"
        labels = ("W", "Y", "Z", "X")
        metadata = {
            "normalization": "N3D",
            "coordinate_frame": "avengine_world",
            "ambisonics": {
                "order": 1,
                "channel_ordering": "ACN",
                "acn_indices": [0, 1, 2, 3],
            },
        }
    samples = np.ones(
        (len(grid.keyframes), len(grid.source_ids), channel_count, 1),
        dtype=np.float32,
    )
    trajectory = research_review_trajectory_record(grid)
    trajectory_sha256 = canonical_json_sha256(trajectory)
    metadata.update(
        {
            "listener_id": listener_id,
            "source_ids": list(grid.source_ids),
            "trajectory_sha256": trajectory_sha256,
            "endpoint_receipts": [
                {
                    "authority": "native_registration_readback",
                    "sources": [
                        {"source_id": source_id, "native_realized": True}
                        for source_id in grid.source_ids
                    ],
                    "listener": {
                        "listener_id": listener_id,
                        "native_realized": True,
                    },
                }
                for _ in grid.keyframes
            ],
            "ir_sha256_by_keyframe_source": [
                {source_id: f"rir-{source_id}" for source_id in grid.source_ids}
                for _ in grid.keyframes
            ],
        }
    )
    return DynamicRIRSequence(
        samples=samples,
        lengths=np.ones(
            (len(grid.keyframes), len(grid.source_ids)), dtype=np.uint32
        ),
        source_ids=grid.source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=grid.sample_rate_hz,
        layout_type=layout_type,
        layout_id=layout_id,
        channel_labels=labels,
        trajectory_sha256=trajectory_sha256,
        metadata=metadata,
    )


def _install_dual_layout_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift_foa_identity: bool = False,
) -> list[tuple[str, object, str | None]]:
    calls: list[tuple[str, object, str | None]] = []

    def binaural(scene, simulation, *, grid, hrtf_file_path, listener_id):
        calls.append(("binaural", grid, hrtf_file_path))
        return _fake_dynamic_sequence(
            grid, layout="binaural", listener_id=listener_id
        )

    def foa(scene, simulation, *, grid, listener_id):
        calls.append(("foa", grid, None))
        sequence = _fake_dynamic_sequence(grid, layout="foa", listener_id=listener_id)
        if drift_foa_identity:
            sequence = replace(
                sequence,
                trajectory_sha256="0" * 64,
                metadata={**sequence.metadata, "trajectory_sha256": "0" * 64},
            )
        return sequence

    monkeypatch.setattr(
        dynamic_audio,
        "_load_simulation_request",
        lambda path: ({}, object()),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_compiled_acoustic_scene",
        lambda path, **kwargs: object(),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_source_endpoint_registry",
        lambda path: {},
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_sound_asset_registry",
        lambda path: {},
    )
    monkeypatch.setattr(
        dynamic_audio,
        "_asset_bindings",
        lambda *args, **kwargs: {
            "sound0": {"path": "/fake/sound0.wav", "sha256": "1" * 64}
        },
    )
    def fake_assembly(program, *args, **kwargs):
        materialized = json.loads(json.dumps(program))
        materialized["program_content_sha256"] = "2" * 64
        return SimpleNamespace(
            dry_audio=SimpleNamespace(
                buses={
                    "a": np.ones(80_000, dtype=np.float64),
                    "b": np.full(80_000, 0.25, dtype=np.float64),
                }
            ),
            materialized_program=materialized,
        )

    monkeypatch.setattr(
        dynamic_audio,
        "assemble_audio_program_dry_buses",
        fake_assembly,
    )
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_binaural_rir_sequence",
        binaural,
    )
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_foa_rir_sequence",
        foa,
    )
    return calls


def _dynamic_render_kwargs(tmp_path: Path) -> dict:
    program_path = tmp_path / "program.json"
    program_path.write_text(
        json.dumps(
            {
                "program_id": "dual_layout_test",
                "revision": "v1",
                "mode": "sequential_sources",
                "candidate_source_endpoint_ids": ["a", "b"],
                "timeline": {
                    "time_base_hz": 48_000,
                    "ticks_per_frame": 3_200,
                    "video_fps": 15,
                    "frame_count": 75,
                    "sample_rate_hz": 16_000,
                    "ticks_per_sample": 3,
                    "sample_count": 80_000,
                },
                "events": [
                    {
                        "event_id": "event0",
                        "source_endpoint_id": "a",
                        "sound_asset_id": "sound0",
                        "start_sample": 0,
                        "end_sample_exclusive": 1_000,
                        "start_tick": 0,
                        "end_tick_exclusive": 3_000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    placeholders = {}
    for name in ("simulation", "package", "endpoints", "sounds"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        placeholders[name] = path
    hrtf = tmp_path / "listener.sofa"
    hrtf.write_bytes(b"fixture-hrtf")
    trajectories = {
        source_id: [[float(index), 0.5, slot] for index in range(75)]
        for slot, source_id in enumerate(("a", "b"))
    }
    return {
        "source_trajectories_m": trajectories,
        "listener_position_m": [0.0, 1.5, 0.0],
        "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "simulation_request_path": placeholders["simulation"],
        "package_manifest_path": placeholders["package"],
        "audio_program_path": program_path,
        "source_endpoint_registry_path": placeholders["endpoints"],
        "sound_asset_registry_path": placeholders["sounds"],
        "external_sound_asset_paths": {},
        "hrtf_file_path": hrtf,
        "output_path": tmp_path / "output",
        "position_authority": "fixture trajectories",
        "listener_authority": "fixture listener",
    }


def test_dynamic_core_dual_writes_same_trajectory_foa_and_binaural(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_dual_layout_fakes(monkeypatch)
    kwargs = _dynamic_render_kwargs(tmp_path)

    receipt = dynamic_audio.render_dynamic_research_audio(**kwargs)

    assert [call[0] for call in calls] == ["binaural", "foa"]
    assert calls[0][1] is calls[1][1]
    assert calls[0][2] == str(Path(kwargs["hrtf_file_path"]).resolve())
    assert calls[1][2] is None
    assert receipt["audio"] == {
        "sample_rate_hz": 16_000,
        "sample_count": 80_000,
        "layout_type": "binaural",
        "channel_labels": ["left", "right"],
    }
    assert "native FOA" in receipt["claim_boundary"]
    assert receipt["audio_layouts"]["foa"]["channel_labels"] == [
        "W",
        "Y",
        "Z",
        "X",
    ]
    assert receipt["audio_layouts"]["foa"]["channel_ordering"] == "ACN"
    assert receipt["audio_layouts"]["foa"]["normalization"] == "N3D"
    assert receipt["audio_layouts"]["foa"]["hrtf_used"] is False
    assert receipt["dual_layout_identity"]["same_source_and_trajectory_identity"] is True
    assert (
        receipt["rir_layouts"]["foa"]["trajectory_sha256"]
        == receipt["rir_layouts"]["binaural"]["trajectory_sha256"]
    )
    output = Path(kwargs["output_path"])
    assert len(list(output.rglob("*.wav"))) == 8
    foa_mix = read_float32_wav(output / "audio/foa/mixture.wav")
    assert foa_mix.samples.shape == (
        4,
        80_000,
    )
    assert read_float32_wav(
        output / "audio/binaural/mixture.wav"
    ).samples.shape == (2, 80_000)
    assert set(receipt["outputs"]) == {
        "audio/dry/a.wav",
        "audio/dry/b.wav",
        "audio/binaural/a_stem.wav",
        "audio/binaural/b_stem.wav",
        "audio/binaural/mixture.wav",
        "audio/foa/a_stem.wav",
        "audio/foa/b_stem.wav",
        "audio/foa/mixture.wav",
    }
    expected_spatial = rlr_foa_wav_metadata()["spatial_format"]
    assert expected_spatial["raw_channel_order"] == ["W", "Y", "Z", "X"]
    assert expected_spatial["acn_indices"] == [0, 1, 2, 3]
    assert expected_spatial["normalization"] == "N3D"
    assert expected_spatial["coordinate_frame"] == "avengine_world"
    for source_id in ("a", "b"):
        stem = read_float32_wav(output / f"audio/foa/{source_id}_stem.wav")
        assert stem.sidecar is not None
        metadata = stem.sidecar["metadata"]
        assert metadata["spatial_format"] == expected_spatial
        assert metadata["audio_role"] == "dynamic_wet_stem"
        assert metadata["source_id"] == source_id
        assert metadata["hrtf_used"] is False
        assert metadata["resampling"] == "not_applied"
        assert metadata["amplitude_normalization"] == "not_applied"
        assert metadata["limiter"] == "not_applied"
    assert foa_mix.sidecar is not None
    mix_metadata = foa_mix.sidecar["metadata"]
    assert mix_metadata["spatial_format"] == expected_spatial
    assert mix_metadata["audio_role"] == "dynamic_wet_mixture"
    assert mix_metadata["source_ids"] == ["a", "b"]
    assert mix_metadata["summation_order"] == ["a", "b"]
    assert mix_metadata["hrtf_used"] is False
    assert mix_metadata["resampling"] == "not_applied"
    assert mix_metadata["amplitude_normalization"] == "not_applied"
    assert mix_metadata["limiter"] == "not_applied"
    for relative in (
        "audio/dry/a.wav",
        "audio/binaural/a_stem.wav",
        "audio/binaural/mixture.wav",
    ):
        legacy = read_float32_wav(output / relative)
        assert legacy.sidecar is not None
        assert legacy.sidecar["metadata"] == {}


def test_dynamic_core_rejects_dual_layout_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dual_layout_fakes(monkeypatch, drift_foa_identity=True)
    kwargs = _dynamic_render_kwargs(tmp_path)

    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="share source and trajectory identity",
    ):
        dynamic_audio.render_dynamic_research_audio(**kwargs)


def test_dynamic_core_preserves_fresh_no_clobber(
    tmp_path: Path,
) -> None:
    kwargs = _dynamic_render_kwargs(tmp_path)
    Path(kwargs["output_path"]).mkdir()

    with pytest.raises(CurrentMP3DDynamicAudioError, match="refusing to replace"):
        dynamic_audio.render_dynamic_research_audio(**kwargs)
