from __future__ import annotations

import wave
from pathlib import Path
import struct

import numpy as np
import pytest

from avengine.dataset import binding_group_identity as identity


def _request() -> dict:
    return {
        "binding_identity": {
            "sound_identity_field": "speaker_id",
            "event_start_s": 0.25,
            "minimum_silence_after_wet_tail_s": 0.1,
            "post_motion_silence_s": 0.0,
            "walk_speed_range_mps": [0.65, 0.8],
            "minimum_motion_s": 0.5,
            "end_hold_s": 3.0,
            "minimum_entity_separation_m": 0.95,
            "same_floor_tolerance_m": 0.35,
            "path_length_range_m": [0.8, 2.0],
            "route_retry_budget": 32,
            "visibility_margin_deg": 5.0,
            "minimum_bearing_change_deg": 8.0,
            "angle_tolerance_deg": 10.0,
        }
    }


def _clock() -> dict:
    return {
        "frame_count": 10,
        "frame_rate_hz": 10.0,
        "sample_rate_hz": 16000,
        "sample_count": 64000,
        "time_base_hz": 48000,
        "ticks_per_frame": 4800,
    }


def _actor() -> dict:
    return {
        "actor_id": "source1",
        "emitter_binding": {"emitter_offset_m": [0.0, 1.0, 0.0]},
        "timeline": {
            "walking_action_id": "walk",
            "idle_action_id": "idle",
            "walk_phase_period_frames": 4,
            "local_anatomical_forward_axis": [0.0, 0.0, 1.0],
        },
    }


def _initial() -> dict:
    return {
        "actor_id": "source1",
        "root_transform": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "planned_emitter_m": [0.0, 1.0, 0.0],
    }


def test_config_requires_explicit_identity_settings() -> None:
    with pytest.raises(identity.IdentityNativeError, match="binding_identity"):
        identity._config({})
    config = identity._config(_request())
    assert config["walk_speed_range_mps"] == [0.65, 0.8]


def test_actor_states_keep_native_idle_walk_period_and_continuous_root() -> None:
    actor = _actor()
    states = identity._actor_states(
        actor,
        _initial(),
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]] + [[1.0, 0.0, 0.0]] * 7),
        0,
        2,
        _clock(),
    )
    assert [state["action_id"] for state in states[:3]] == ["walk", "walk", "idle"]
    assert all(state["action_id"] == "idle" for state in states[2:])
    assert states[1]["moving"] is True
    assert states[2]["moving"] is False
    assert states[2]["root_transform"]["translation_m"] == [1.0, 0.0, 0.0]
    assert states[2]["planned_emitter_m"] == pytest.approx([1.0, 1.0, 0.0])


def test_pose_alignment_leaves_pre_motion_idle_unchanged() -> None:
    states = [
        {"frame_index": 0, "moving": False, "root_transform": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        {"frame_index": 2, "moving": True, "root_transform": {"rotation_xyzw": [0.0, 0.2, 0.0, 0.98]}},
        {"frame_index": 3, "moving": False, "root_transform": {"rotation_xyzw": [0.0, 0.2, 0.0, 0.98]}},
    ]
    target = [0.0, 0.97, 0.0, -0.23]
    motion_end = 3
    for state in states:
        if (int(state["frame_index"]) >= motion_end and not state["moving"]):
            state["root_transform"]["rotation_xyzw"] = list(target)
    assert states[0]["root_transform"]["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert states[1]["root_transform"]["rotation_xyzw"] == [0.0, 0.2, 0.0, 0.98]
    assert states[2]["root_transform"]["rotation_xyzw"] == target


def test_resume_result_normalizes_qa_delivery_audio_alias(tmp_path: Path) -> None:
    wav = tmp_path / "mixture.wav"
    result = {"lossless_stereo_wav": str(wav), "facts": "facts.json"}
    normalized = identity._normalize_resume_result(result)
    assert normalized["audio"] == str(wav.resolve())
    assert normalized["facts"] == "facts.json"


def test_variant_targets_encode_physical_identity_truth_table() -> None:
    assert identity._variant_targets("v0", "a0") == {"event_001": "source1", "event_002": "source1"}
    assert identity._variant_targets("v0", "a1") == {"event_001": "source2", "event_002": "source1"}
    assert identity._variant_targets("v1", "a0") == {"event_001": "source1", "event_002": "source2"}
    assert identity._variant_targets("v1", "a1") == {"event_001": "source2", "event_002": "source2"}


def test_event_preserves_complete_clip_and_integer_sample_tick_clock() -> None:
    sound = {
        "sound_asset_id": "clip_a",
        "sample_count": 32000,
        "sample_rate_hz": 16000,
        "audible_start_sample": 480,
        "audible_end_sample_exclusive": 30000,
        "compatible_asset_ids": ["blue", "green"],
    }
    event = identity._event(sound, event_id="event_001", actor_id="source1", start_s=0.25, clock=_clock())
    assert event["start_sample"] == 4000
    assert event["end_sample_exclusive"] == 36000
    assert event["source_start_sample"] == 0
    assert event["source_end_sample_exclusive"] == 32000
    assert event["planned_audible_interval_samples"] == [4480, 34000]
    assert event["start_tick"] == 12000


def _wave(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(4)
        stream.setframerate(16000)
        stream.writeframes(payload)


def test_event2_alignment_uses_first_settled_rir_key() -> None:
    event = {
        "event_id": "event_002",
        "start_sample": 72533,
        "end_sample": 109333,
        "end_sample_exclusive": 109333,
        "start_tick": 217599,
        "end_tick": 327999,
        "end_tick_exclusive": 327999,
        "start_s": 4.5333125,
        "end_s": 6.8333125,
        "sample_count": 36800,
        "audible_start_sample": 480,
        "audible_end_sample_exclusive": 36320,
        "planned_audible_interval_samples": [73013, 108853],
    }
    plan = {
        "clock": {
            "frame_rate_hz": 15.0,
            "sample_rate_hz": 16000,
            "time_base_hz": 48000,
            "sample_count": 160000,
        },
        "audio_events": [event],
        "voice_bindings": [{**event}],
    }
    alignment = identity._align_event2_to_rir_boundary(
        plan, settled_frame=67, rir_stride_frames=5
    )
    assert alignment == {
        "settled_frame": 67,
        "rir_key_frame": 70,
        "start_sample": 74667,
        "end_sample_exclusive": 111467,
    }
    assert event["start_sample"] == 74667
    assert event["end_sample_exclusive"] == 111467
    assert event["planned_audible_interval_samples"] == [75147, 110987]
    assert plan["voice_bindings"][0]["start_sample"] == 74667


def test_pcm_equality_reads_ieee_float_wave_payload(tmp_path: Path) -> None:
    payload = np.asarray([0.1, -0.2, 0.3, -0.4], dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, 16000, 128000, 8, 32)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    wav = tmp_path / "float.wav"
    wav.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    assert identity._pcm_signature(wav)["payload"] == payload
    assert identity._pcm_signature(wav)["sample_width"] == 4


def test_pcm_equality_uses_actual_wave_payload(tmp_path: Path) -> None:
    payload = np.asarray([0.1, -0.2, 0.3, -0.4], dtype="<f4").tobytes()
    left, right, changed = tmp_path / "left.wav", tmp_path / "right.wav", tmp_path / "changed.wav"
    _wave(left, payload)
    _wave(right, payload)
    _wave(changed, payload[:-4] + np.asarray([0.9], dtype="<f4").tobytes())
    assert identity._pcm_equal(left, right)["same"] is True
    assert identity._pcm_equal(left, changed)["same"] is False


def test_variant_targets_support_source2_first_event_anchor() -> None:
    assert identity._variant_targets(
        "v0", "a0", first_event_actor_id="source2"
    ) == {"event_001": "source2", "event_002": "source1"}
    assert identity._variant_targets(
        "v0", "a1", first_event_actor_id="source2"
    ) == {"event_001": "source1", "event_002": "source1"}
    assert identity._variant_targets(
        "v1", "a0", first_event_actor_id="source2"
    ) == {"event_001": "source2", "event_002": "source2"}
    assert identity._variant_targets(
        "v1", "a1", first_event_actor_id="source2"
    ) == {"event_001": "source1", "event_002": "source2"}


def test_first_event_actor_is_explicit_or_seeded_once_per_group() -> None:
    seeded = _request()
    seeded["seed"] = 12345
    actor_a, metadata_a = identity._first_event_actor(seeded)
    actor_b, metadata_b = identity._first_event_actor(seeded)
    assert actor_a == actor_b
    assert actor_a in {"source1", "source2"}
    assert seeded["binding_identity"]["first_event_actor_id"] == actor_a
    assert metadata_a["selection"] == "seeded_uniform_two_compatible_actors"
    assert metadata_a["uniform_seed"] == 15446
    assert metadata_b["actor_id"] == actor_a

    explicit = _request()
    explicit["binding_identity"]["first_event_actor_id"] = "source2"
    actor, metadata = identity._first_event_actor(explicit)
    assert actor == "source2"
    assert metadata == {"actor_id": "source2", "selection": "explicit_request"}


def test_navigation_authority_is_read_from_room_evidence() -> None:
    plan = {
        "room_capabilities": {
            "evidence_refs": {
                "navigation": {
                    "authority": "retained_walkable_grid_with_raster_astar"
                }
            }
        }
    }
    assert identity._navigation_authority(plan) == "retained_walkable_grid_with_raster_astar"


def test_with_audio_preserves_ue_navigation_authority_and_actor_endpoint() -> None:
    event = {
        "event_id": "event_001",
        "actor_id": "source2",
        "start_sample": 4000,
        "end_sample_exclusive": 36000,
    }
    plan = {
        "room_capabilities": {
            "evidence_refs": {
                "navigation": {"authority": "retained_walkable_grid_with_raster_astar"}
            }
        },
        "visual_plan": {"camera": {"motion": "static"}},
    }
    value = identity._with_audio(plan, {"episode_id": "test"}, [event])
    assert value["planned_conditions"]["authority"] == (
        "retained_walkable_grid_with_raster_astar_and_explicit_identity_intervention"
    )
    assert value["activity_plan"]["authority"] == "retained_walkable_grid_with_raster_astar"
    assert value["voice_bindings"][0]["actor_id"] == "source2"
    assert value["voice_bindings"][0]["source_endpoint_id"] == "source2_mouth"


def test_materialize_visual_accepts_ue_room_package_without_habitat_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    (base / "plan").mkdir(parents=True)
    (base / "plan" / "room_package.json").write_text(
        '{"family": "kujiale", "renderer": "ue_spear"}\n', encoding="utf-8"
    )
    plan = {
        "resources": {"room_package": {"family": "kujiale"}},
        "visual_plan": {"camera": {"motion": "static"}},
    }
    output = identity._materialize_visual(base, tmp_path / "output", plan, {"episode_id": "test"})
    assert output == (tmp_path / "output").resolve()
    assert (output / "plan/room_package.json").is_file()
    assert not (output / "plan/habitat_execution").exists()


def test_raw_recast_turn_checks_adjacent_frame_speeds() -> None:
    # The sparse native polyline has a turn; fixed-FPS sampling keeps its
    # shape and validates the actual displacement of every adjacent frame.
    polyline = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.7]],
        dtype=float,
    )
    sampled, summary = identity._sample_native_polyline(
        polyline, 16, 15.0, [0.65, 0.8], owner="turn-test"
    )
    assert sampled.shape == (16, 3)
    assert np.allclose(sampled[0], polyline[0])
    assert np.allclose(sampled[-1], polyline[-1])
    assert summary["native_timing_preserved"] is False
    assert summary["minimum_step_speed_mps"] >= 0.65 - 1e-5
    assert summary["maximum_step_speed_mps"] <= 0.8 + 1e-5
    assert np.any(np.abs(np.diff(sampled[:, 0])) > 0.0)
    assert np.any(np.abs(np.diff(sampled[:, 2])) > 0.0)

    malformed = sampled.copy()
    malformed[8, 0] += 0.1
    with pytest.raises(identity.IdentityNativeError, match="adjacent-frame speed"):
        identity._validate_polyline_frame_speeds(
            malformed, 15.0, [0.65, 0.8], owner="turn-test"
        )
