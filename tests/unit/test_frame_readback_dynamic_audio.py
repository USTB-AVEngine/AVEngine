from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np

from tools.acoustics import render_frame_readback_sequential_speech as renderer
from avengine.acoustics.dynamic_cache import (
    DYNAMIC_RIR_CACHE_SCHEMA,
    _array_digest,
    _request_metadata,
    load_dynamic_rir_cache,
)
from avengine.contracts.json_io import canonical_json_sha256


def _write_wav(path: Path, values: list[int]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())


def _clock() -> dict[str, int | float]:
    return {
        "frame_count": 5,
        "frame_rate_hz": 15.0,
        "ticks_per_frame": 3_200,
        "time_base_hz": 48_000,
        "sample_rate_hz": 16_000,
        "sample_count": 5_333,
    }


def test_dynamic_cache_round_trip_and_request_rejection(tmp_path: Path) -> None:
    samples = np.ones((2, 2, 2, 3), dtype="<f4")
    lengths = np.full((2, 2), 3, dtype="<u4")
    metadata = {
        "source_ids": ["a", "b"],
        "keyframe_samples": [0, 2_133],
        "sample_rate_hz": 16_000,
        "layout_type": "binaural",
    }
    cache = tmp_path / "legacy_sequence_fixture"
    cache.mkdir()
    metadata = {
        **metadata,
        "schema": DYNAMIC_RIR_CACHE_SCHEMA,
        "array_shape": list(samples.shape),
        "array_dtype": samples.dtype.str,
        "lengths_shape": list(lengths.shape),
        "lengths_dtype": lengths.dtype.str,
    }
    metadata["content_sha256"] = _array_digest(samples, lengths)
    metadata["cache_identity_sha256"] = canonical_json_sha256(
        _request_metadata(metadata)
    )
    np.savez_compressed(cache / "sequence.npz", samples=samples, lengths=lengths)
    (cache / "manifest.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    loaded = load_dynamic_rir_cache(
        cache,
        expected_cache_identity_sha256=metadata["cache_identity_sha256"],
        expected_source_ids=["a", "b"],
        expected_keyframe_samples=[0, 2_133],
    )
    assert np.array_equal(loaded.samples, samples)
    assert loaded.samples.flags.writeable is False


def test_plan_events_use_plan_timing_and_detect_leading_silence(tmp_path: Path) -> None:
    clip = tmp_path / "voice.wav"
    _write_wav(clip, [0, 0, 0, 10_000, 20_000, 10_000, 0])
    plan = {"clock": _clock(), "candidate_source_endpoint_ids": ["actor0_mouth", "actor1_mouth"], "audio_events": [{
        "event_id": "e0",
        "actor_id": "actor0",
        "sound_asset_id": "voice0",
        "start_sample": 11,
        "end_sample_exclusive": 18,
        "linear_gain": 0.25,
    }]}
    events, _, _ = renderer._normalize_plan_events(
        plan,
        [{"actor_id": "actor0", "path": str(clip), "sound_asset_id": "voice0"}],
        clock=_clock(),
    )
    assert events[0]["start_sample"] == 11
    assert events[0]["source_activity_interval"]["start_sample"] == 3
    assert events[0]["source_activity_interval"]["end_sample_exclusive"] == 6
    program = renderer._program_from_plan_events(plan, events, _clock())
    assert program["events"][0]["start_tick"] == 33


def test_standard_audio_program_events_are_accepted_with_external_binding(tmp_path: Path) -> None:
    clip = tmp_path / "voice.wav"
    _write_wav(clip, [0, 10_000, 20_000, 10_000])
    standard = {
        "timeline": _clock(),
        "events": [
            {
                "event_id": "e0",
                "source_endpoint_id": "actor0_mouth",
                "sound_asset_id": "voice0",
                "start_sample": 10,
                "end_sample_exclusive": 14,
                "source_start_sample": 0,
                "source_end_sample_exclusive": 4,
                "linear_gain": 0.2,
                "fade_samples": 1,
            }
        ],
    }
    events, _, _ = renderer._normalize_plan_events(
        standard,
        [{"actor_id": "actor0", "path": str(clip), "sound_asset_id": "voice0"}],
        clock=_clock(),
    )
    assert events[0]["actor_id"] == "actor0"
    assert events[0]["path"] == str(clip.resolve())


def test_readback_keyframes_keep_moving_source_and_listener_poses() -> None:
    readback = {
        "camera": [
            {"frame_index": i, "location_cm": [100 + i, 200, 300], "rotation_deg": [0, 0, i]}
            for i in range(5)
        ],
        "emitters": {
            "actor0": [
                {"frame_index": i, "location_cm": [100 + i, 0, 300], "rotation_deg": [0, 0, 0]}
                for i in range(5)
            ],
            "actor1": [
                {"frame_index": i, "location_cm": [200, 0, 300 + i], "rotation_deg": [0, 0, 0]}
                for i in range(5)
            ],
        },
    }
    keyframes, trajectories = renderer._readback_keyframes(
        readback,
        actor_by_endpoint={
            "actor0_mouth": {"actor_id": "actor0"},
            "actor1_mouth": {"actor_id": "actor1"},
        },
        frame_count=5,
        frame_rate_hz=15.0,
        ticks_per_frame=3_200,
        time_base_hz=48_000,
        sample_rate_hz=16_000,
        rir_stride_frames=2,
    )
    assert [item["visual_frame_index"] for item in keyframes] == [0, 2, 4]
    assert keyframes[1]["source_positions_m"]["actor0_mouth"] == [1.02, 3.0, 0.0]
    assert keyframes[1]["listener_position_m"] != keyframes[0]["listener_position_m"]
    assert len(trajectories["actor1_mouth"]) == 5


def test_existing_cache_plan_binds_each_pose_and_two_source_slots() -> None:
    keyframes = [
        {
            "visual_frame_index": frame,
            "source_positions_m": {
                "a": [float(frame), 1.0, 2.0],
                "b": [float(frame + 1), 1.0, 2.0],
            },
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        for frame in (0, 2)
    ]
    plan = renderer._existing_rir_plan(source_ids=["a", "b"], keyframes=keyframes)
    assert plan["listener_pose_mode"] == "per_episode_frame"
    assert plan["cache_key_fields"] == [
        "source_position_m",
        "listener_position_m",
        "listener_orientation_wxyz",
    ]
    assert len(plan["jobs"]) == 4
    assert plan["jobs"][0]["uses"][0]["source_slot_id"] == "source1"
    assert plan["jobs"][1]["uses"][0]["source_slot_id"] == "source2"


def test_existing_cache_plan_merges_repeated_static_states_for_reuse() -> None:
    keyframes = [
        {
            "visual_frame_index": frame,
            "source_positions_m": {"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 2.0]},
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        for frame in (0, 2, 4)
    ]
    plan = renderer._existing_rir_plan(source_ids=["a", "b"], keyframes=keyframes)
    assert len(plan["jobs"]) == 2
    assert len(plan["jobs"][0]["uses"]) == 3


def test_four_sources_partition_over_existing_two_slot_cache(tmp_path: Path, monkeypatch) -> None:
    keyframes = [
        {
            "visual_frame_index": 0,
            "sample_index": 0,
            "source_positions_m": {
                "a": [0.0, 1.0, 0.0],
                "b": [1.0, 1.0, 0.0],
                "c": [2.0, 1.0, 0.0],
                "d": [3.0, 1.0, 0.0],
            },
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    ]

    def fake_pair(**kwargs):
        source_ids = kwargs["source_ids"]
        return (
            np.ones((1, 2, 2, 2), dtype="<f4") * len(source_ids),
            np.full((1, 2), 2, dtype="<u4"),
            {"status": "pass"},
            {
                "status": "hit_existing_rir_cache",
                "request_identity_sha256": "id",
            },
        )

    monkeypatch.setattr(renderer, "_existing_rir_cache_sequence", fake_pair)
    samples, lengths, evidence, record = renderer._existing_rir_cache_pair_sequence(
        cache_path=tmp_path / "cache",
        source_ids=["a", "b", "c", "d"],
        keyframes=keyframes,
        frame_count=1,
        frame_rate_hz=15.0,
        episode_id="episode",
        scene=object(),
        simulation=renderer._simulation(),
        package_path=tmp_path / "package.json",
        hrtf_path=tmp_path / "hrtf.sofa",
        runtime_prefix=tmp_path,
        rlr_sdk_root=tmp_path,
        magnum_python_site=tmp_path,
    )
    assert samples.shape == (1, 4, 2, 2)
    assert lengths.shape == (1, 4)
    assert record["status"] == "hit_existing_rir_cache_pairs"
    assert json.loads((tmp_path / "cache" / "pair_sequence_index.json").read_text())["pairs"] == [
        {"source_ids": ["a", "b"], "cache_path": "pair_00"},
        {"source_ids": ["c", "d"], "cache_path": "pair_01"},
    ]


def test_three_sources_reuses_first_endpoint_as_discarded_pair_companion(tmp_path: Path, monkeypatch) -> None:
    keyframes = [
        {
            "visual_frame_index": 0,
            "sample_index": 0,
            "source_positions_m": {
                "a": [0.0, 1.0, 0.0],
                "b": [1.0, 1.0, 0.0],
                "c": [2.0, 1.0, 0.0],
            },
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    ]

    def fake_pair(**kwargs):
        return (
            np.ones((1, 2, 2, 2), dtype="<f4"),
            np.full((1, 2), 2, dtype="<u4"),
            {"status": "pass"},
            {"status": "hit_existing_rir_cache", "request_identity_sha256": "id"},
        )

    monkeypatch.setattr(renderer, "_existing_rir_cache_sequence", fake_pair)
    samples, lengths, _, record = renderer._existing_rir_cache_pair_sequence(
        cache_path=tmp_path / "cache",
        source_ids=["a", "b", "c"],
        keyframes=keyframes,
        frame_count=1,
        frame_rate_hz=15.0,
        episode_id="episode",
        scene=object(),
        simulation=renderer._simulation(),
        package_path=tmp_path / "package.json",
        hrtf_path=tmp_path / "hrtf.sofa",
        runtime_prefix=tmp_path,
        rlr_sdk_root=tmp_path,
        magnum_python_site=tmp_path,
    )
    assert samples.shape == (1, 3, 2, 2)
    assert lengths.shape == (1, 3)
    assert record["status"] == "hit_existing_rir_cache_pairs"
    assert json.loads((tmp_path / "cache" / "pair_sequence_index.json").read_text())["pairs"][-1] == {
        "source_ids": ["c", "a"],
        "cache_path": "pair_01",
    }


def test_plan_renderer_routes_multi_source_through_pair_cache_adapter(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "voice.wav"
    _write_wav(clip, [0, 12_000, 18_000, 12_000])
    readback = {
        "clock": _clock(),
        "camera": [
            {"frame_index": i, "location_cm": [0, 0, 100], "rotation_deg": [0, 0, 0]}
            for i in range(5)
        ],
        "emitters": {
            actor: [
                {"frame_index": i, "location_cm": [100 * (index + 1) + i, 0, 100], "rotation_deg": [0, 0, 0]}
                for i in range(5)
            ]
            for index, actor in enumerate(("actor0", "actor1", "actor2"))
        },
        "animations": {},
    }
    readback_path = tmp_path / "frame_readbacks.json"
    readback_path.write_text(json.dumps(readback), encoding="utf-8")
    plan = {
        "clock": _clock(),
        "audio_events": [
            {"event_id": "e0", "actor_id": "actor0", "sound_asset_id": "voice0", "path": str(clip), "start_sample": 10, "end_sample_exclusive": 14, "linear_gain": 0.2},
            {"event_id": "e1", "actor_id": "actor1", "sound_asset_id": "voice0", "path": str(clip), "start_sample": 30, "end_sample_exclusive": 34, "linear_gain": 0.2},
            {"event_id": "e2", "actor_id": "actor2", "sound_asset_id": "voice0", "path": str(clip), "start_sample": 50, "end_sample_exclusive": 54, "linear_gain": 0.2},
        ],
    }
    plan_path = tmp_path / "audio_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    binding_path = tmp_path / "bindings.json"
    binding_path.write_text(json.dumps({"bindings": [
        {"actor_id": "actor0", "path": str(clip), "sound_asset_id": "voice0"},
        {"actor_id": "actor1", "path": str(clip), "sound_asset_id": "voice0"},
        {"actor_id": "actor2", "path": str(clip), "sound_asset_id": "voice0"},
    ]}), encoding="utf-8")
    package = tmp_path / "package.json"
    package.write_text("{}", encoding="utf-8")
    hrtf = tmp_path / "hrtf.sofa"
    hrtf.write_bytes(b"fixture")
    cache = tmp_path / "persistent_cache"

    monkeypatch.setattr(renderer, "load_compiled_acoustic_scene", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        renderer,
        "_existing_rir_cache_pair_sequence",
        lambda **kwargs: (
            np.ones((len(kwargs["keyframes"]), len(kwargs["source_ids"]), 2, 3), dtype="<f4"),
            np.full((len(kwargs["keyframes"]), len(kwargs["source_ids"])), 3, dtype="<u4"),
            {"status": "pass"},
            {"status": "hit_existing_rir_cache_pairs"},
        ),
    )
    first = renderer.render(
        frame_readbacks=readback_path,
        package_manifest=package,
        voice_binding=binding_path,
        audio_plan=plan_path,
        output=tmp_path / "out1",
        runtime_prefix=tmp_path,
        rlr_sdk_root=tmp_path,
        magnum_python_site=tmp_path,
        hrtf_file=hrtf,
        rir_cache=cache,
        rir_stride_frames=2,
        direct_ray_count=2,
        indirect_ray_count=2,
        source_ray_count=2,
        indirect_ray_depth=2,
        source_ray_depth=2,
    )
    assert first["dynamic_rir"]["cache"]["status"] == "hit_existing_rir_cache_pairs"
    assert first["events"][0]["wet_tail_end_sample"] > 10

    second = renderer.render(
        frame_readbacks=readback_path,
        package_manifest=package,
        voice_binding=binding_path,
        audio_plan=plan_path,
        output=tmp_path / "out2",
        runtime_prefix=tmp_path,
        rlr_sdk_root=tmp_path,
        magnum_python_site=tmp_path,
        hrtf_file=hrtf,
        rir_cache=cache,
        rir_stride_frames=2,
        direct_ray_count=2,
        indirect_ray_count=2,
        source_ray_count=2,
        indirect_ray_depth=2,
        source_ray_depth=2,
    )
    assert second["dynamic_rir"]["cache"]["status"] == "hit_existing_rir_cache_pairs"
