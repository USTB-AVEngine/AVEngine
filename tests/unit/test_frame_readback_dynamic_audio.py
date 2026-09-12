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


def test_repeated_actor_bindings_select_each_events_complete_clip(tmp_path: Path) -> None:
    clips = [tmp_path / "first.wav", tmp_path / "second.wav"]
    _write_wav(clips[0], [0, 10000, 20000, 0])
    _write_wav(clips[1], [0, 4000, 8000, 0])
    rows = [
        {"event_id": f"e{index}", "actor_id": "actor0",
         "source_endpoint_id": "actor0_mouth", "path": str(clip),
         "sound_asset_id": f"voice{index}", "transcript": f"utterance {index}"}
        for index, clip in enumerate(clips)
    ]
    binding_path = tmp_path / "bindings.json"
    # Reversing records must not exchange the two clips or their metadata.
    binding_path.write_text(json.dumps(rows[::-1]))
    bindings = renderer._load_voice_binding_records(binding_path)
    plan = {"visual_plan": {"actors": [{"actor_id": "actor0"}, {"actor_id": "actor1"}]},
            "audio_events": [
                {"event_id": f"e{index}", "actor_id": "actor0",
                 "start_sample": 10 + index * 20, "end_sample_exclusive": 14 + index * 20}
                for index in range(2)]}
    events, _, _ = renderer._normalize_plan_events(plan, bindings, clock=_clock())
    assert [event["path"] for event in events] == [str(path.resolve()) for path in clips]
    assert [event["transcript"] for event in events] == ["utterance 0", "utterance 1"]
    assert {event["actor_id"] for event in events} == {"actor0"}
    program = renderer._program_from_plan_events(plan, events, _clock())
    assert program["mode"] == "one_active_of_n"
    assert len(program["events"]) == 2


def test_repeated_actor_binding_requires_unambiguous_event_ids(tmp_path: Path) -> None:
    import pytest
    path = tmp_path / "bindings.json"
    rows = [{"actor_id": "actor0", "path": "/first.wav"},
            {"actor_id": "actor0", "path": "/second.wav"}]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique event IDs"):
        renderer._load_voice_binding_records(path)
    for row in rows:
        row["event_id"] = "same_event"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique event IDs"):
        renderer._load_voice_binding_records(path)


def test_event_binding_cannot_change_the_planned_actor(tmp_path: Path) -> None:
    import pytest
    plan = {"audio_events": [{"event_id": "e0", "actor_id": "actor0",
                             "start_sample": 0, "end_sample_exclusive": 4}]}
    bindings = [{"event_id": "e0", "actor_id": "actor1", "path": "/voice.wav"}]
    with pytest.raises(ValueError, match="different actor"):
        renderer._normalize_plan_events(plan, bindings, clock=_clock())


def test_layout_request_parsing_accepts_cli_and_sequence_forms() -> None:
    import pytest

    assert renderer.normalize_requested_layouts(None) == ("binaural",)
    assert renderer.normalize_requested_layouts("binaural") == ("binaural",)
    assert renderer.normalize_requested_layouts("binaural,ambisonics") == (
        "binaural",
        "ambisonics",
    )
    assert renderer.normalize_requested_layouts(
        ["ambisonics", "binaural"]
    ) == ("ambisonics", "binaural")
    with pytest.raises(ValueError, match="unsupported layouts"):
        renderer.normalize_requested_layouts("binaural,stereo")
    with pytest.raises(ValueError, match="must not repeat"):
        renderer.normalize_requested_layouts("binaural,binaural")
    with pytest.raises(ValueError, match="at least one"):
        renderer.normalize_requested_layouts("")


def test_layout_cache_root_keeps_binaural_at_the_historical_path(tmp_path: Path) -> None:
    # An already-written binaural cache must stay readable at its own root,
    # and a second layout must never share that directory.
    assert renderer.layout_cache_root(tmp_path, "binaural") == tmp_path
    assert renderer.layout_cache_root(tmp_path, "ambisonics") == tmp_path / "foa"


# Shaped like the real avengine_rlr_rir_cache_request_v1 files on disk: all
# 35 retained cache requests carry these scene digests and this runtime block.
def _cache_request(**overrides) -> dict:
    request = {
        "acoustic_scene": {
            "package_id": "pkg_a",
            "manifest_path": "/pkg/a.json",
            "manifest_sha256": "a" * 64,
            "package_content_sha256": "b" * 64,
        },
        "output": {"layout_type": "binaural", "hrtf_path": "/hrtf/kemar.sofa"},
        "runtime_policy": {
            "coordinate_translation_m": [0.0, 0.0, 0.0],
            "source_radius_m": 0.0,
            "listener_radius_m": 0.0,
        },
    }
    for key, value in overrides.items():
        if value is None:
            request.pop(key, None)
        else:
            request[key] = value
    return request


def _cache_scene(**overrides):
    values = {
        "package_id": "pkg_a",
        "manifest_path": "/pkg/a.json",
        "manifest_sha256": "a" * 64,
        "package_content_sha256": "b" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cache_reuse_requires_every_recorded_execution_condition(tmp_path: Path) -> None:
    scene = _cache_scene()
    hrtf = Path("/hrtf/kemar.sofa")

    assert renderer.existing_rir_cache_execution_reason(
        _cache_request(), layout_type="binaural", scene=scene, hrtf_path=hrtf
    ) is None

    # A binaural payload can never answer an ambisonics request.
    assert "needs 'ambisonics'" in renderer.existing_rir_cache_execution_reason(
        _cache_request(), layout_type="ambisonics", scene=scene, hrtf_path=None
    )

    # Same poses and same simulation in a different room is still a different
    # acoustic result, so the package identity has to match.
    assert "package_id" in renderer.existing_rir_cache_execution_reason(
        _cache_request(),
        layout_type="binaural",
        scene=_cache_scene(package_id="pkg_b"),
        hrtf_path=hrtf,
    )

    # package_id alone is not scene identity. Two different valid packages can
    # carry one id, and the pose check would still line up, so a differing
    # manifest or package content must block reuse.
    assert "manifest_sha256" in renderer.existing_rir_cache_execution_reason(
        _cache_request(),
        layout_type="binaural",
        scene=_cache_scene(manifest_sha256="d" * 64, package_content_sha256="e" * 64),
        hrtf_path=hrtf,
    )
    assert (
        "package_content_sha256"
        in renderer.existing_rir_cache_execution_reason(
            _cache_request(),
            layout_type="binaural",
            scene=_cache_scene(package_content_sha256="e" * 64),
            hrtf_path=hrtf,
        )
    )

    # A cache rendered under a different coordinate translation or source
    # radius describes different geometry.
    assert "coordinate translation" in renderer.existing_rir_cache_execution_reason(
        _cache_request(),
        layout_type="binaural",
        scene=scene,
        hrtf_path=hrtf,
        coordinate_translation_m=(0.0, 0.0, 1.5),
    )
    assert "source_radius_m" in renderer.existing_rir_cache_execution_reason(
        _cache_request(),
        layout_type="binaural",
        scene=scene,
        hrtf_path=hrtf,
        source_radius_m=0.1,
    )

    # A different HRTF changes binaural samples.
    assert "HRTF" in renderer.existing_rir_cache_execution_reason(
        _cache_request(),
        layout_type="binaural",
        scene=scene,
        hrtf_path=Path("/hrtf/other.sofa"),
    )

    # Metadata that was never recorded cannot be matched, so it is not a hit.
    for missing in (
        {"output": {"hrtf_path": "/hrtf/kemar.sofa"}},
        {"output": {"layout_type": "binaural"}},
        {"acoustic_scene": {"manifest_path": "/pkg/a.json"}},
        {"acoustic_scene": {"package_id": "pkg_a", "manifest_sha256": "a" * 64}},
        {"runtime_policy": {"source_radius_m": 0.0, "listener_radius_m": 0.0}},
        {"runtime_policy": None},
    ):
        assert renderer.existing_rir_cache_execution_reason(
            _cache_request(**missing),
            layout_type="binaural",
            scene=scene,
            hrtf_path=hrtf,
        ) is not None
    assert renderer.existing_rir_cache_execution_reason(
        _cache_request(output=None),
        layout_type="binaural",
        scene=scene,
        hrtf_path=hrtf,
    ) == "cache request records no output block"


def test_plan_renderer_gives_each_layout_its_own_cache_and_layout_id(
    tmp_path: Path, monkeypatch
) -> None:
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
                {
                    "frame_index": i,
                    "location_cm": [100 * (index + 1) + i, 0, 100],
                    "rotation_deg": [0, 0, 0],
                }
                for i in range(5)
            ]
            for index, actor in enumerate(("actor0", "actor1"))
        },
        "animations": {},
    }
    readback_path = tmp_path / "frame_readbacks.json"
    readback_path.write_text(json.dumps(readback), encoding="utf-8")
    plan = {
        "clock": _clock(),
        "audio_events": [
            {"event_id": "e0", "actor_id": "actor0", "sound_asset_id": "voice0",
             "path": str(clip), "start_sample": 10, "end_sample_exclusive": 14,
             "linear_gain": 0.2},
            {"event_id": "e1", "actor_id": "actor1", "sound_asset_id": "voice0",
             "path": str(clip), "start_sample": 30, "end_sample_exclusive": 34,
             "linear_gain": 0.2},
        ],
    }
    plan_path = tmp_path / "audio_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    binding_path = tmp_path / "bindings.json"
    binding_path.write_text(json.dumps({"bindings": [
        {"actor_id": "actor0", "path": str(clip), "sound_asset_id": "voice0"},
        {"actor_id": "actor1", "path": str(clip), "sound_asset_id": "voice0"},
    ]}), encoding="utf-8")
    package = tmp_path / "package.json"
    package.write_text("{}", encoding="utf-8")
    hrtf = tmp_path / "hrtf.sofa"
    hrtf.write_bytes(b"fixture")
    cache = tmp_path / "persistent_cache"

    seen: list[dict] = []

    def fake_sequence(**kwargs):
        layout = kwargs["layout_type"]
        channels = 2 if layout == "binaural" else 4
        seen.append({"layout": layout, "cache_path": kwargs["cache_path"]})
        samples = np.zeros(
            (len(kwargs["keyframes"]), len(kwargs["source_ids"]), channels, 3),
            dtype="<f4",
        )
        # Give each channel distinct content so a real FOA buffer is not
        # mistaken for a tiled binaural pair downstream.
        for channel in range(channels):
            samples[:, :, channel, :] = 0.1 * (channel + 1)
        return (
            samples,
            np.full(
                (len(kwargs["keyframes"]), len(kwargs["source_ids"])), 3, dtype="<u4"
            ),
            {"status": "pass"},
            {"status": "hit_existing_rir_cache", "layout_type": layout},
        )

    monkeypatch.setattr(
        renderer, "load_compiled_acoustic_scene", lambda *a, **k: object()
    )
    monkeypatch.setattr(renderer, "_dynamic_rir_sequence", fake_sequence)

    report = renderer.render(
        frame_readbacks=readback_path,
        package_manifest=package,
        voice_binding=binding_path,
        audio_plan=plan_path,
        output=tmp_path / "out",
        runtime_prefix=tmp_path,
        rlr_sdk_root=tmp_path,
        magnum_python_site=tmp_path,
        hrtf_file=hrtf,
        rir_cache=cache,
        rir_stride_frames=2,
        layouts="binaural,ambisonics",
        direct_ray_count=2,
        indirect_ray_count=2,
        source_ray_count=2,
        indirect_ray_depth=2,
        source_ray_depth=2,
    )

    assert report["layouts"] == ["binaural", "ambisonics"]
    assert [item["layout"] for item in seen] == ["binaural", "ambisonics"]
    # Binaural keeps the historical root; FOA gets its own directory.
    assert seen[0]["cache_path"] == cache.resolve()
    assert seen[1]["cache_path"] == cache.resolve() / "foa"

    delivery = report["audio"]["layout_delivery"]
    assert delivery["ambisonics"]["layout_id"] == "rlr_foa_acn_n3d_world_v1"
    assert delivery["ambisonics"]["normalization"] == "N3D"
    assert delivery["ambisonics"]["coordinate_frame"] == "avengine_world"
    assert delivery["ambisonics"]["rir_source"] == "reused_override"
    assert delivery["binaural"]["layout_id"] == "rlr_binaural_lr_v1"

    foa_mixture = Path(delivery["ambisonics"]["mixture"]["path"])
    assert foa_mixture.is_file()
    assert foa_mixture.parent.name == "foa"


def test_non_default_layouts_require_an_explicit_audio_plan(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="requires an explicit audio plan"):
        renderer.render(
            frame_readbacks=tmp_path / "missing.json",
            package_manifest=tmp_path / "package.json",
            voice_binding=tmp_path / "bindings.json",
            output=tmp_path / "out",
            runtime_prefix=tmp_path,
            rlr_sdk_root=tmp_path,
            magnum_python_site=tmp_path,
            layouts="ambisonics",
        )


def test_cache_simulation_match_is_layout_aware() -> None:
    """An ambisonics cache must be re-hittable.

    ``render_rir_cache`` stamps the selected layout into the simulation it
    records, so a FOA cache records channel_layout ambisonics/4. Comparing
    every layout against the binaural form made a written FOA cache
    impossible to hit: the layout was the only difference, and it always
    differed. Found by running a real write-then-hit, not by unit tests.
    """
    from avengine.spatial_audio.runtime import simulation_with_layout

    simulation = renderer._simulation()
    binaural = simulation_with_layout(
        simulation, layout_type="binaural", channel_count=2
    )
    ambisonics = simulation_with_layout(
        simulation, layout_type="ambisonics", channel_count=4
    )
    assert binaural.to_dict() != ambisonics.to_dict()

    def request_for(effective):
        return {"simulation": {"effective": effective}}

    foa_cache = request_for(ambisonics.to_dict())
    binaural_cache = request_for(binaural.to_dict())

    # Each layout matches its own recorded simulation...
    assert renderer.existing_rir_cache_simulation_matches(foa_cache, ambisonics)
    assert renderer.existing_rir_cache_simulation_matches(binaural_cache, binaural)
    # ...and never the other layout's.
    assert not renderer.existing_rir_cache_simulation_matches(foa_cache, binaural)
    assert not renderer.existing_rir_cache_simulation_matches(
        binaural_cache, ambisonics
    )

    # The layouts differ only in channel_layout, which is exactly why the
    # comparison has to apply the layout before comparing.
    left, right = binaural.to_dict(), ambisonics.to_dict()
    assert {k for k in set(left) | set(right) if left.get(k) != right.get(k)} == {
        "channel_layout"
    }
