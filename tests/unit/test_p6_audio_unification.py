from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

import avengine.cli as cli
import avengine.timeline.current_mp3d_dynamic_audio as dynamic_audio
import tools.acoustics.render_frame_readback_sequential_speech as speech
from avengine.contracts.json_io import sha256_file
from avengine.rooms.qa_delivery import build_audio_command
from avengine.timeline.audio_program import bind_audio_program_hash
from avengine.spatial_audio.audio import write_float32_wav
from avengine.timeline.unified_audio_receipt import (
    UnifiedAudioReceiptError,
    validate_unified_audio_receipt,
)
from tools.acoustics.render_frame_readback_sequential_speech import (
    _normalize_plan_events,
    _simulation,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _write_pcm16(path: Path, samples: np.ndarray) -> None:
    values = np.asarray(samples, dtype=np.float64)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(
            np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        )


def _neutral(frame_count: int = 15) -> dict:
    clock = {
        "frame_count": frame_count,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16_000,
        "sample_count": 16_000,
        "time_base_hz": 48_000,
        "ticks_per_frame": 3_200,
    }
    camera = [
        {
            "frame_index": index,
            "pts_ticks": index * 3_200,
            "position_m": [0.0, 1.5, 0.0],
            "basis": {
                "forward": [0.0, 0.0, -1.0],
                "right": [1.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            },
        }
        for index in range(frame_count)
    ]
    entities = {
        actor: [
            {
                "frame_index": index,
                "pts_ticks": index * 3_200,
                "root": [float(offset), 0.0, -2.0],
                "emitter": [float(offset), 1.0, -2.0],
                "moving": False,
            }
            for index in range(frame_count)
        ]
        for actor, offset in (("source1", -1.0), ("source2", 1.0))
    }
    return {
        "schema": "avengine_neutral_readback_v1",
        "clock": clock,
        "coordinate_frame": {
            "linear_unit": "meter",
            "up_axis": "+Y",
            "handedness": "right",
        },
        "camera": camera,
        "entities": entities,
        "producer": {
            "module": "test_p6",
            "renderer": "test",
            "source_readbacks": ["fixture/frame_readbacks.json"],
        },
    }


def _program(clip: Path) -> dict:
    return bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "p6_fixture",
            "revision": "v1",
            "mode": "one_active_of_n",
            "timeline": {
                "time_base_hz": 48_000,
                "ticks_per_frame": 3_200,
                "video_fps": 15,
                "frame_count": 15,
                "sample_rate_hz": 16_000,
                "ticks_per_sample": 3,
                "sample_count": 16_000,
            },
            "candidate_source_endpoint_ids": ["source1_mouth", "source2_mouth"],
            "events": [
                {
                    "event_id": "e1",
                    "source_endpoint_id": "source1_mouth",
                    "sound_asset_id": "tone",
                    "start_tick": 3_000,
                    "end_tick_exclusive": 6_000,
                    "start_sample": 1_000,
                    "end_sample_exclusive": 2_000,
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": 1_000,
                    "linear_gain": 0.15,
                    "fade_samples": 20,
                    "render_source_stem": True,
                    "normalization_policy": "use_sound_asset_policy",
                }
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )


def _simulation_mapping() -> dict:
    return _simulation(
        direct_ray_count=2,
        indirect_ray_count=2,
        source_ray_count=2,
        indirect_ray_depth=2,
        source_ray_depth=2,
    ).to_dict()


def test_shared_neutral_renderer_emits_common_receipt_and_source_activity(
    tmp_path: Path, monkeypatch
) -> None:
    clip = tmp_path / "tone.wav"
    samples = np.zeros(1_000, dtype=np.float64)
    samples[100:900] = 0.5
    _write_pcm16(clip, samples)
    package = tmp_path / "package.json"
    package.write_text("{}", encoding="utf-8")
    hrtf = tmp_path / "hrtf.sofa"
    hrtf.write_bytes(b"fixture")
    program = _program(clip)

    def fake_rir(_scene, _simulation, *, grid, layout_type, hrtf_file_path=None):
        labels = ("left", "right")
        return SimpleNamespace(
            layout_type=layout_type,
            layout_id="test-binaural",
            channel_labels=labels,
            keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
            samples=np.ones((len(grid.keyframes), 2, 2, 2), dtype="<f4"),
            lengths=np.full((len(grid.keyframes), 2), 2, dtype="<u4"),
            trajectory_sha256="test-trajectory",
        )

    def fake_audio(dry_buses, sequence, *, grid):
        count = int(grid.episode_sample_count)
        stems = {
            source_id: SimpleNamespace(
                episode=np.zeros((2, count), dtype=np.float64)
            )
            for source_id in dry_buses
        }
        return stems, np.zeros((2, count), dtype=np.float64)

    monkeypatch.setattr(dynamic_audio, "load_compiled_acoustic_scene", lambda *args, **kwargs: object())
    monkeypatch.setattr(dynamic_audio, "render_research_review_rir_sequence", fake_rir)
    monkeypatch.setattr(dynamic_audio, "render_research_review_audio", fake_audio)
    receipt = dynamic_audio.render_neutral_readback_audio(
        _neutral(),
        audio_program=program,
        source_endpoint_by_entity={
            "source1": "source1_mouth",
            "source2": "source2_mouth",
        },
        simulation_mapping=_simulation_mapping(),
        package_manifest_path=package,
        event_asset_bindings={"tone": clip},
        hrtf_file_path=hrtf,
        output_path=tmp_path / "output",
    )

    assert receipt["schema"] == "avengine_unified_audio_receipt_v1"
    assert receipt["audio_program"]["variant_id"] == "A"
    assert receipt["source_clip_preserved"] is True
    assert receipt["source_clip_preservation"]["status"] == "pass"
    assert receipt["sentence_preservation"]["status"] == "not_assessed"
    assert receipt["complete_sentences_preserved"] is None
    assert receipt["gain_application"]["applied_once_per_event"] is True
    assert receipt["propagation"] == {
        "diffraction": False,
        "max_diffraction_order": 0,
    }
    assert receipt["input_neutral_readback"]["path"].endswith(
        "/output/neutral_readback.json"
    )
    event = receipt["events"][0]
    assert event["gain_application"]["application_count"] == 1
    assert event["source_activity_intervals_samples"] == [
        {"start_sample": 1_100, "end_sample_exclusive": 1_900}
    ]
    assert event["wet_tail_interval"][0] >= 1_000
    assert event["wet_tail_interval"][1] > event["wet_tail_interval"][0]
    assert len(receipt["wet_tail_intervals"]) == 1
    assert receipt["hrtf"]["id"] == "hrtf.sofa"
    assert Path(receipt["mixture_path"]).is_file()
    assert validate_unified_audio_receipt(receipt)["status"] == "pass"

    legacy_receipt = dynamic_audio.render_dynamic_research_audio(
        source_trajectories_m={
            "source1_mouth": [[-1.0, 0.0, -2.0] for _ in range(15)],
            "source2_mouth": [[1.0, 0.0, -2.0] for _ in range(15)],
        },
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        simulation_mapping=_simulation_mapping(),
        package_manifest_path=package,
        audio_program=program,
        event_asset_bindings={"tone": clip},
        hrtf_file_path=hrtf,
        output_path=tmp_path / "legacy-direct-output",
        position_authority="legacy-test",
        listener_authority="legacy-test",
    )
    assert legacy_receipt["complete_sentences_preserved"] is True
    assert "sentence_preservation" not in legacy_receipt

    bad_clock = deepcopy(receipt)
    bad_clock["clock"]["sample_count"] += 1
    with pytest.raises(UnifiedAudioReceiptError, match="clock validation failed"):
        validate_unified_audio_receipt(bad_clock)

    foa_only = deepcopy(receipt)
    foa_only["audio"]["layouts"] = ["ambisonics"]
    with pytest.raises(UnifiedAudioReceiptError, match="actual binaural"):
        validate_unified_audio_receipt(foa_only)

    mono_path = tmp_path / "mono-fake-binaural.wav"
    write_float32_wav(mono_path, np.zeros((1, 16_000), dtype=np.float64), 16_000)
    mono_output = deepcopy(receipt)
    mono_output["audio"]["mixture_path"] = str(mono_path)
    mono_output["outputs_by_layout"]["binaural"]["mixture"] = str(mono_path)
    with pytest.raises(UnifiedAudioReceiptError, match="two-channel"):
        validate_unified_audio_receipt(mono_output)




def test_event_level_voice_binding_overrides_actor_binding(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_pcm16(first, np.ones(8, dtype=np.float64) * 0.1)
    _write_pcm16(second, np.ones(8, dtype=np.float64) * 0.2)
    plan = {
        "clock": {
            "frame_count": 15,
            "frame_rate_hz": 15,
            "sample_rate_hz": 16_000,
            "sample_count": 16_000,
            "time_base_hz": 48_000,
            "ticks_per_frame": 3_200,
        },
        "audio_events": [
            {
                "event_id": "e1",
                "actor_id": "actor0",
                "source_endpoint_id": "actor0_mouth",
                "sound_asset_id": "first",
                "path": str(first),
                "start_sample": 10,
                "end_sample_exclusive": 18,
                "source_start_sample": 0,
                "source_end_sample_exclusive": 8,
            },
            {
                "event_id": "e2",
                "actor_id": "actor0",
                "source_endpoint_id": "actor0_mouth",
                "start_sample": 30,
                "end_sample_exclusive": 38,
                "source_start_sample": 0,
                "source_end_sample_exclusive": 8,
                "voice_binding": {
                    "actor_id": "actor0",
                    "sound_asset_id": "second",
                    "path": str(second),
                    "linear_gain": 0.15,
                },
            },
        ],
    }
    actor_binding = [{"actor_id": "actor0", "sound_asset_id": "first", "path": str(first)}]
    normalized, _, _ = _normalize_plan_events(
        plan,
        actor_binding,
        clock=plan["clock"],
    )
    by_id = {row["event_id"]: row for row in normalized}
    assert by_id["e1"]["path"] == str(first.resolve())
    assert by_id["e2"]["path"] == str(second.resolve())
    assert by_id["e2"]["sound_asset_id"] == "second"
    assert by_id["e2"]["linear_gain"] == 0.15


def test_prepared_source_activity_subtracts_crop_start(tmp_path: Path) -> None:
    clip = tmp_path / "prepared.wav"
    _write_pcm16(clip, np.ones(400, dtype=np.float64) * 0.2)
    manifest = tmp_path / "prepared_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "prepared_audio_id": "prepared",
                        "prepared": str(clip),
                        "facts": {"source_crop_start_sample": 100},
                        "source_activity": {
                            "intervals": [
                                {"start_sample": 120, "end_sample_exclusive": 160},
                                {"start_sample": 220, "end_sample_exclusive": 260},
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    activity = dynamic_audio._event_source_activity(
        {
            "event_id": "prepared-event",
            "sound_asset_id": "prepared",
            "start_sample": 1_000,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 400,
        },
        event_metadata=None,
        prepared_index=dynamic_audio._prepared_activity_index(manifest),
        asset_path=clip,
        episode_sample_count=2_000,
    )
    assert activity["source_activity_intervals_samples"] == [
        {"start_sample": 1_020, "end_sample_exclusive": 1_060},
        {"start_sample": 1_120, "end_sample_exclusive": 1_160},
    ]
    assert (
        activity["source_activity_provenance"]
        == "prepared_manifest_original_source_samples_minus_crop_start"
    )


def test_p6_cli_and_delivery_command_expose_neutral_and_diffraction_flags() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "m5",
            "render-current-mp3d-dynamic-audio",
            "--visual-capture-dir",
            "capture",
            "--m1-request",
            "m1.json",
            "--simulation-request",
            "simulation.json",
            "--package-manifest",
            "package.json",
            "--audio-program",
            "program.json",
            "--beagle-audio",
            "beagle.wav",
            "--hrtf",
            "hrtf.sofa",
            "--runtime-prefix",
            "runtime",
            "--rlr-sdk-root",
            "rlr",
            "--neutral-readback",
            "neutral.json",
            "--no-diffraction",
            "--max-diffraction-order",
            "2",
            "--output",
            "out",
        ]
    )
    assert args.neutral_readback == "neutral.json"
    assert args.diffraction is False
    assert args.max_diffraction_order == 2

    command = build_audio_command(
        {
            "runtime": {
                "runtime_prefix": "runtime",
                "rlr_sdk_root": "rlr",
                "magnum_python_site": "magnum",
                "hrtf": "hrtf.sofa",
                "neutral_readback": "neutral.json",
                "diffraction": True,
                "max_diffraction_order": 3,
            },
            "rir_stride": 3,
        },
        {
            "resources": {"acoustic_package": "package.json"},
            "neutral_readback": "plan-neutral.json",
        },
        Path("episode"),
        Path("audio"),
        repository=REPOSITORY,
    )
    assert "--neutral-readback" in command
    assert command[command.index("--neutral-readback") + 1] == "neutral.json"
    assert "--diffraction" in command
    assert command[command.index("--max-diffraction-order") + 1] == "3"



def _moving_legacy_plan_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Return a two-source old-format plan with a genuinely moving listener."""
    frame_count = 3
    clock = {
        "frame_count": frame_count,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16_000,
        "sample_count": 3_200,
        "time_base_hz": 48_000,
        "ticks_per_frame": 3_200,
    }
    camera = [
        {
            "frame_index": index,
            "location_cm": [float(index * 20), 0.0, 150.0],
            "rotation_deg": [0.0, 0.0, 0.0],
        }
        for index in range(frame_count)
    ]
    emitters = {
        actor: [
            {
                "frame_index": index,
                "location_cm": [float(offset), 0.0, 100.0],
            }
            for index in range(frame_count)
        ]
        for actor, offset in (("actor0", -100.0), ("actor1", 100.0))
    }
    readback_path = tmp_path / "frame-readbacks.json"
    readback_path.write_text(
        json.dumps({"clock": clock, "camera": camera, "emitters": emitters}),
        encoding="utf-8",
    )
    clips = []
    bindings = []
    events = []
    for index, actor in enumerate(("actor0", "actor1")):
        clip = tmp_path / f"{actor}.wav"
        _write_pcm16(clip, np.full(200, 0.1, dtype=np.float64))
        clips.append(clip)
        bindings.append(
            {
                "actor_id": actor,
                "source_endpoint_id": f"{actor}_mouth",
                "sound_asset_id": actor,
                "path": str(clip),
            }
        )
        start = index * 500
        events.append(
            {
                "event_id": f"event-{actor}",
                "actor_id": actor,
                "source_endpoint_id": f"{actor}_mouth",
                "sound_asset_id": actor,
                "path": str(clip),
                "start_sample": start,
                "end_sample_exclusive": start + 200,
                "source_start_sample": 0,
                "source_end_sample_exclusive": 200,
                "linear_gain": 0.15,
                "fade_samples": 0,
            }
        )
    plan_path = tmp_path / "audio-plan.json"
    plan_path.write_text(json.dumps({"clock": clock, "audio_events": events}), encoding="utf-8")
    binding_path = tmp_path / "voice-binding.json"
    binding_path.write_text(json.dumps(bindings), encoding="utf-8")
    package_path = tmp_path / "package.json"
    package_path.write_text("{}", encoding="utf-8")
    return readback_path, plan_path, binding_path, package_path


def test_old_moving_listener_dispatches_to_real_dynamic_renderer_and_preserves_keyframes(
    tmp_path: Path, monkeypatch
) -> None:
    readback, plan, binding, package = _moving_legacy_plan_fixture(tmp_path)
    hrtf = tmp_path / "hrtf.sofa"
    hrtf.write_bytes(b"fixture")
    seen: dict[str, object] = {}

    def fake_rir(**kwargs):
        keyframes = kwargs["keyframes"]
        seen["keyframes"] = keyframes
        count = len(keyframes)
        return (
            np.ones((count, 2, 2, 1), dtype="<f4"),
            np.ones((count, 2), dtype="<u4"),
            {"source": "moving-listener-test"},
            {"status": "hit", "path": "fixture-cache"},
        )

    monkeypatch.setattr(speech, "load_compiled_acoustic_scene", lambda *args, **kwargs: object())
    monkeypatch.setattr(speech, "_dynamic_rir_sequence", fake_rir)
    report = speech.render(
        frame_readbacks=readback,
        package_manifest=package,
        voice_binding=binding,
        output=tmp_path / "legacy-output",
        runtime_prefix="runtime",
        rlr_sdk_root="rlr",
        magnum_python_site="magnum",
        audio_plan=plan,
        hrtf_file=hrtf,
        rir_stride_frames=1,
    )

    keyframes = seen["keyframes"]
    assert isinstance(keyframes, list)
    assert keyframes[0]["listener_position_m"] != keyframes[1]["listener_position_m"]
    assert report["dynamic_rir"]["keyframes"][0]["listener_position_m"] != report["dynamic_rir"]["keyframes"][1]["listener_position_m"]
    assert Path(report["mixture_path"]).is_file()

    conditioned_plan = json.loads(plan.read_text(encoding="utf-8"))
    conditioned_plan["request"] = {"sampling_policy": "conditioned_static_v2"}
    conditioned_path = tmp_path / "conditioned-moving-plan.json"
    conditioned_path.write_text(json.dumps(conditioned_plan), encoding="utf-8")

    def unexpected_legacy_call(**kwargs):
        raise AssertionError("conditioned static plan must not use legacy dynamic renderer")

    monkeypatch.setattr(speech, "_render_plan_audio_legacy_dynamic", unexpected_legacy_call)
    with pytest.raises(ValueError, match="conditioned_static_v2.*static listener"):
        speech.render(
            frame_readbacks=readback,
            package_manifest=package,
            voice_binding=binding,
            output=tmp_path / "conditioned-output",
            runtime_prefix="runtime",
            rlr_sdk_root="rlr",
            magnum_python_site="magnum",
            audio_plan=conditioned_path,
            hrtf_file=hrtf,
            rir_stride_frames=1,
        )


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_explicit_neutral_input_needs_no_legacy_readback_placeholder(tmp_path, monkeypatch, cache_enabled):
    from avengine.capture.ue_neutral_readback import neutral_from_ue_readbacks
    readback, plan_path, binding, package = _moving_legacy_plan_fixture(tmp_path)
    raw = json.loads(readback.read_text())
    for row in raw["camera"]:
        row["location_cm"] = [0., 0., 150.]
    raw["actors"] = raw["emitters"]
    plan = json.loads(plan_path.read_text())
    neutral = neutral_from_ue_readbacks(raw, plan, source_readbacks=str(readback))
    path = tmp_path / "actual-neutral.json"
    path.write_text(json.dumps(neutral))
    seen = {}
    def fake_shared(actual, **kwargs):
        seen["actual"] = actual
        seen.update(kwargs)
        Path(kwargs["output_path"]).mkdir()
        return {"schema": "test-receipt", "input_neutral_readback": {"path": str(actual)}}
    monkeypatch.setattr(speech, "render_neutral_readback_audio", fake_shared)
    if cache_enabled:
        def fake_cache(**kwargs):
            seen["cache_keyframes"] = kwargs["keyframes"]
            return np.zeros((len(kwargs["keyframes"]), 2, 2, 1)), np.ones((len(kwargs["keyframes"]), 2), dtype=int), {}, {"status": "test_cache"}
        monkeypatch.setattr(speech, "load_compiled_acoustic_scene", lambda *args, **kwargs: object())
        monkeypatch.setattr(speech, "_dynamic_rir_sequence", fake_cache)
    report = speech.render(
        neutral_readback=path, audio_plan=plan_path,
        rir_cache=tmp_path / "cache" if cache_enabled else None,
        package_manifest=package, voice_binding=binding, output=tmp_path / "output",
        runtime_prefix="runtime", rlr_sdk_root="rlr", magnum_python_site="magnum")
    assert Path(seen["actual"]) == path
    assert "frame_readbacks" not in seen["extra_inputs"]
    assert "frame_readbacks" not in report
    assert seen["listener_authority"] == "P1 NeutralReadback.camera[0]"
    assert seen["position_authority"] == "P1 NeutralReadback entities[].emitter"

    if cache_enabled:
        first = seen["cache_keyframes"][0]
        assert first["sample_index"] == first["visual_frame_index"] == 0
        assert first["listener_position_m"] == neutral["camera"][0]["position_m"]
        assert sorted(first["source_positions_m"].values()) == sorted(rows[0]["emitter"] for rows in neutral["entities"].values())
        assert seen["rir_sequence_override"]["binaural"]["cache"]["status"] == "test_cache"

def test_unified_receipt_rejects_wet_tail_past_clock() -> None:
    from avengine.timeline.unified_audio_receipt import (
        UNIFIED_AUDIO_RECEIPT_SCHEMA,
        UnifiedAudioReceiptError,
        validate_unified_audio_receipt,
    )

    receipt = {
        "schema": UNIFIED_AUDIO_RECEIPT_SCHEMA,
        "clock": {
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
            "frame_rate_hz": 15,
            "frame_count": 15,
            "sample_rate_hz": 16000,
            "sample_count": 16000,
            "duration_seconds": 1.0,
            "ticks_per_sample": 3,
        },
        "audio": {
            "sample_rate_hz": 16000,
            "sample_count": 16000,
            "layouts": ["binaural"],
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
            "mixture_path": "mixture.wav",
            "stems": {"source1_mouth": "stem.wav"},
            "by_layout": {
                "binaural": {
                    "channel_count": 2,
                    "channel_labels": ["left", "right"],
                    "sample_rate_hz": 16000,
                    "sample_count": 16000,
                }
            },
        },
        "outputs_by_layout": {
            "binaural": {"mixture": "mixture.wav", "stems": {"source1_mouth": "stem.wav"}}
        },
        "events": [
            {
                "event_id": "e1",
                "source_activity_intervals_samples": [{"start_sample": 1000, "end_sample_exclusive": 2000}],
                "wet_tail_intervals": [{"start_sample": 1000, "end_sample_exclusive": 20000}],
                "gain_application": {"application_count": 1, "post_assembly_convolution_gain": 1.0},
            }
        ],
        "wet_tail_intervals": [
            {"event_id": "e1", "start_sample": 1000, "end_sample_exclusive": 20000}
        ],
        "peak_dbfs": {"mixture": -12.0},
        "gain_application": {"applied_once_per_event": True, "normalization": False},
        "propagation": {"diffraction": False, "max_diffraction_order": 0},
        "hrtf": {"id": "test"},
        "input_neutral_readback": {"path": "neutral.json"},
    }
    with pytest.raises(UnifiedAudioReceiptError, match="escapes the episode clock"):
        validate_unified_audio_receipt(receipt, require_files=False)

