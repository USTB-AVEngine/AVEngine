from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file, write_json
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m5.audio import (
    M5_AUDIO_SAMPLE_RATE_HZ,
    extract_faded_clip,
    place_simultaneous_events,
    read_pcm16_mono_wav,
    render_dynamic_stems_and_mix,
)
import avengine.m5.canary as canary
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory


_TEST_CLIP_START = 0
_TEST_CLIP_END = 256
_TEST_FADE_SAMPLES = 80
_TEST_LINEAR_GAIN = 0.18
_TEST_WINDOWS = ((256, 512), (768, 1_024), (1_280, 1_536))


def test_current_dynamic_pair_forwards_one_runtime_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_render(
        _scene: object, _simulation: object, **kwargs: object
    ) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(trajectory_sha256="same-trajectory")

    monkeypatch.setattr(canary, "render_dynamic_rir_sequence", fake_render)
    prefix = Path("/external/habitat-prefix")
    magnum_site = Path("/external/magnum-site")
    sdk_root = Path("/external/rlr-sdk")
    hrtf = tmp_path / "kemar-16k.sofa"
    installed_runtime = SimpleNamespace(
        prefix=prefix,
        magnum_python_site=magnum_site,
    )

    foa, binaural = canary._render_current_dynamic_rir_pair(
        scene=object(),
        simulation=SimpleNamespace(),
        keyframes=(),
        hrtf_path=hrtf,
        installed_runtime=installed_runtime,
        rlr_sdk_root=sdk_root,
    )

    assert foa.trajectory_sha256 == binaural.trajectory_sha256
    assert len(calls) == 2
    runtime_keys = (
        "runtime_mode",
        "runtime_prefix",
        "rlr_sdk_root",
        "magnum_python_site",
    )
    assert {key: calls[0][key] for key in runtime_keys} == {
        key: calls[1][key] for key in runtime_keys
    }
    assert calls[0]["runtime_mode"] == "current-installed"
    assert calls[0]["runtime_prefix"] == prefix
    assert calls[0]["rlr_sdk_root"] == sdk_root
    assert calls[0]["magnum_python_site"] == magnum_site
    assert calls[0]["layout_type"] == "ambisonics"
    assert calls[0]["channel_count"] == 4
    assert "hrtf_file_path" not in calls[0]
    assert calls[1]["layout_type"] == "binaural"
    assert calls[1]["channel_count"] == 2
    assert calls[1]["hrtf_file_path"] == str(hrtf.resolve())


def test_current_runtime_root_is_only_an_installed_prefix_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    selected = SimpleNamespace(prefix=Path("/external/prefix"))

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return selected

    monkeypatch.setattr(canary, "prepare_installed_habitat_runtime", fake_prepare)
    result = canary._prepare_m5_installed_runtime(
        runtime_prefix=None,
        runtime_root="/external/prefix",
        mp3d_root=None,
        magnum_python_site="/external/magnum",
        rlr_sdk_root="/external/rlr",
    )

    assert result is selected
    assert calls == [
        {
            "runtime_prefix": None,
            "runtime_root": "/external/prefix",
            "mp3d_root": None,
            "magnum_python_site": "/external/magnum",
            "rlr_sdk_root": "/external/rlr",
            "allow_mp3d_environment": False,
        }
    ]


def test_current_native_runtime_unavailable_preserves_blocked_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient = "/ambient/mp3d"
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", ambient)

    def unavailable(**kwargs: object) -> None:
        calls.append(dict(kwargs))
        raise FileNotFoundError("installed prefix missing")

    monkeypatch.setattr(canary, "prepare_installed_habitat_runtime", unavailable)

    with pytest.raises(
        canary.RuntimeUnavailableError,
        match="current installed Habitat/RLR runtime is unavailable",
    ):
        canary._prepare_m5_installed_runtime(
            runtime_prefix="/external/missing-prefix",
            runtime_root=None,
            mp3d_root=None,
            magnum_python_site="/external/magnum",
            rlr_sdk_root="/external/rlr",
        )
    assert calls[0]["mp3d_root"] is None
    assert calls[0]["allow_mp3d_environment"] is False
    assert canary.os.environ["AVENGINE_MP3D_ROOT"] == ambient


def test_current_runtime_never_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", "/ambient/prefix")
    monkeypatch.setenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", "/ambient/magnum")
    monkeypatch.setenv("AVENGINE_RLR_SDK_ROOT", "/ambient/rlr")

    with pytest.raises(canary.M5CanaryError, match="requires explicit"):
        canary._prepare_m5_installed_runtime(
            runtime_prefix=None,
            runtime_root=None,
            mp3d_root=None,
            magnum_python_site=None,
            rlr_sdk_root=None,
        )


def test_current_runtime_rejects_prefix_plus_compatibility_alias() -> None:
    with pytest.raises(canary.M5CanaryError, match="specify only"):
        canary._prepare_m5_installed_runtime(
            runtime_prefix="/external/prefix-a",
            runtime_root="/external/prefix-b",
            mp3d_root=None,
            magnum_python_site="/external/magnum",
            rlr_sdk_root="/external/rlr",
        )


def test_acoustic_keyframes_follow_the_sensor_rig_listener_pose() -> None:
    rig = materialize_sensor_rig_trajectory(
        trajectory_id="m5_rotate_listener",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [0.0, 1.55, 0.0],
            "start_yaw_deg": 0.0,
            "end_yaw_deg": 90.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )
    visual = SimpleNamespace(
        source_ids=("source0", "source1"),
        source_positions_m=np.repeat(
            np.asarray([[[0.0, 1.55, -2.0], [2.0, 1.55, 0.0]]]),
            75,
            axis=0,
        ),
        listener_position_m=(0.0, 1.55, 0.0),
        listener_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        sensor_rig_trajectory=rig,
    )
    keyframes = canary._acoustic_keyframes(visual)
    assert len(keyframes) == 75
    assert keyframes[0].listener_position_m == (0.0, 1.55, 0.0)
    assert keyframes[0].listener_orientation_wxyz == pytest.approx(
        (1.0, 0.0, 0.0, 0.0)
    )
    assert keyframes[-1].listener_orientation_wxyz != pytest.approx(
        keyframes[0].listener_orientation_wxyz
    )


def test_sensor_rig_evidence_binds_visual_timeline_and_acoustics(
    tmp_path: Path,
) -> None:
    rig = materialize_sensor_rig_trajectory(
        trajectory_id="m5_evidence_rotate_listener",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [0.0, 1.55, 0.0],
            "start_yaw_deg": 0.0,
            "end_yaw_deg": 90.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )
    (tmp_path / "trajectory").mkdir()
    (tmp_path / "visual" / "arrays").mkdir(parents=True)
    (tmp_path / "inputs").mkdir()
    for variant in ("A", "B"):
        (tmp_path / "episodes" / variant).mkdir(parents=True)

    write_json(tmp_path / "trajectory" / "sensor_rig_trajectory.json", rig)
    write_json(tmp_path / "inputs" / "sensor_rig_trajectory.json", rig)
    positions = np.asarray(
        [frame["world_from_rig"]["translation_m"] for frame in rig["frames"]],
        dtype=np.float64,
    )
    orientations = np.asarray(
        [
            [
                frame["world_from_rig"]["rotation_xyzw"][3],
                *frame["world_from_rig"]["rotation_xyzw"][:3],
            ]
            for frame in rig["frames"]
        ],
        dtype=np.float64,
    )
    np.save(tmp_path / "visual" / "arrays" / "listener_positions_m.npy", positions)
    np.save(
        tmp_path / "visual" / "arrays" / "listener_orientations_wxyz.npy",
        orientations,
    )
    write_json(
        tmp_path / "visual" / "frame_records.json",
        {
            "frames": [
                {
                    "frame_index": frame["frame_index"],
                    "pts_ticks": frame["pts_ticks"],
                    "world_from_rig": frame["world_from_rig"],
                    "view_pose_hash": frame["pose_hash"],
                }
                for frame in rig["frames"]
            ]
        },
    )
    write_json(
        tmp_path / "trajectory" / "emitter_path.json",
        {
            "keyframes": [
                {
                    "tick": frame["pts_ticks"],
                    "listener_position_m": positions[index].tolist(),
                    "listener_orientation_wxyz": orientations[index].tolist(),
                }
                for index, frame in enumerate(rig["frames"])
            ]
        },
    )
    timeline = {
        "frames": [
            {"view_pose_hashes": {"view0": frame["pose_hash"]}}
            for frame in rig["frames"]
        ]
    }
    for variant in ("A", "B"):
        write_json(tmp_path / "episodes" / variant / "timeline.json", timeline)
    evidence = {
        "inputs": {
            "sensor_rig_trajectory": {
                "path": "inputs/sensor_rig_trajectory.json"
            }
        },
        "visual": {
            "metadata": {
                "sensor_rig_trajectory": {
                    "trajectory_id": rig["trajectory_id"],
                    "schema": rig["schema"],
                    "content_sha256": canonical_json_sha256(rig),
                    "moving": True,
                }
            }
        }
    }

    assert canary._sensor_rig_evidence_errors(tmp_path, evidence) == []

    declaration = evidence["visual"]["metadata"]["sensor_rig_trajectory"]
    evidence["visual"]["metadata"]["sensor_rig_trajectory"] = None
    assert canary._sensor_rig_evidence_errors(tmp_path, evidence) == [
        "visual sensor-rig trajectory declaration is absent"
    ]
    evidence["visual"]["metadata"]["sensor_rig_trajectory"] = declaration

    timeline["frames"][12]["view_pose_hashes"]["view0"] = "0" * 64
    write_json(tmp_path / "episodes" / "B" / "timeline.json", timeline)
    assert canary._sensor_rig_evidence_errors(tmp_path, evidence) == [
        "episode B Timeline view poses differ from sensor rig"
    ]


def _write_pcm16(path: Path, frequency_hz: float) -> None:
    indices = np.arange(_TEST_CLIP_END, dtype=np.float64)
    samples = np.asarray(
        np.sin(2.0 * np.pi * frequency_hz * indices / M5_AUDIO_SAMPLE_RATE_HZ) * 12_000,
        dtype="<i2",
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(M5_AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(samples.tobytes())


def _audio_record(path: Path, samples: np.ndarray) -> dict[str, object]:
    artifact = write_float32_wav(
        path,
        samples,
        M5_AUDIO_SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata={"test_authority": True},
    )
    return {
        "audio_path": path.as_posix(),
        "sidecar_path": artifact.sidecar_path.as_posix(),
        "audio_sha256": artifact.audio_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
    }


def _reconstruction_fixture(
    root: Path,
) -> tuple[
    dict[str, object],
    dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]],
    dict[str, object],
]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    beagle = inputs / "beagle_dry.wav"
    golden = inputs / "golden_dry.wav"
    _write_pcm16(beagle, 320.0)
    _write_pcm16(golden, 510.0)

    raw_assets: dict[str, tuple[str, np.ndarray]] = {}
    for asset_id, path in (("beagle_call", beagle), ("golden_call", golden)):
        samples, _ = read_pcm16_mono_wav(path)
        raw_assets[sha256_file(path)] = (
            asset_id,
            extract_faded_clip(
                samples,
                start_sample=_TEST_CLIP_START,
                end_sample=_TEST_CLIP_END,
                fade_samples=_TEST_FADE_SAMPLES,
            ),
        )
    beagle_hash = sha256_file(beagle)
    golden_hash = sha256_file(golden)
    audio_program = {
        "program_id": "test_program",
        "clip_source_interval": {
            "start_sample": _TEST_CLIP_START,
            "end_sample": _TEST_CLIP_END,
        },
        "fade_samples": _TEST_FADE_SAMPLES,
        "linear_gain": _TEST_LINEAR_GAIN,
        "simultaneous_windows": [
            {
                "window_id": f"window{index}",
                "start_sample": start,
                "end_sample": end,
            }
            for index, (start, end) in enumerate(_TEST_WINDOWS)
        ],
    }
    requests = {
        "A": {
            "audio_program": audio_program,
            "events": [
                {"source_id": "source0", "dry_audio_asset_sha256": beagle_hash},
                {"source_id": "source1", "dry_audio_asset_sha256": golden_hash},
            ],
        },
        "B": {
            "audio_program": audio_program,
            "events": [
                {"source_id": "source0", "dry_audio_asset_sha256": golden_hash},
                {"source_id": "source1", "dry_audio_asset_sha256": beagle_hash},
            ],
        },
    }
    write_json(
        root / "episodes" / "counterfactual_pair.json",
        {
            "episodes": {
                variant: {"request": request} for variant, request in requests.items()
            }
        },
    )

    keyframe_samples = (0, canary.M5_AUDIO_SAMPLE_COUNT // 2)
    trajectory: dict[str, object] = {
        "source_ids": ["source0", "source1"],
        "keyframes": [{"sample_index": value} for value in keyframe_samples],
    }
    rir: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for layout, channels in (("foa", 4), ("binaural", 2)):
        samples = np.zeros((len(keyframe_samples), 2, channels, 1), dtype="<f4")
        samples[:, 0, :, 0] = np.arange(1, channels + 1, dtype=np.float32) / 20.0
        samples[:, 1, :, 0] = np.arange(channels, 0, -1, dtype=np.float32) / 24.0
        lengths = np.ones((len(keyframe_samples), 2), dtype="<u4")
        rir[layout] = (samples, lengths, {})

    audio: dict[str, object] = {}
    for variant, request in requests.items():
        route = {
            event["source_id"]: raw_assets[event["dry_audio_asset_sha256"]][0]
            for event in request["events"]
        }
        buses, _ = place_simultaneous_events(
            {asset_id: clip for asset_id, clip in raw_assets.values()},
            route,
            start_samples=tuple(start for start, _end in _TEST_WINDOWS),
            output_sample_count=canary.M5_AUDIO_SAMPLE_COUNT,
            linear_gain=_TEST_LINEAR_GAIN,
        )
        records: dict[str, object] = {"dry_buses": {}, "foa": {}, "binaural": {}}
        for source_id in ("source0", "source1"):
            relative = Path("episodes") / variant / "audio" / "dry" / f"{source_id}.wav"
            record = _audio_record(root / relative, buses[source_id][None, :])
            record["audio_path"] = relative.as_posix()
            record["sidecar_path"] = relative.with_suffix(".wav.json").as_posix()
            records["dry_buses"][source_id] = record  # type: ignore[index]
        for layout in ("foa", "binaural"):
            rir_samples, rir_lengths, _ = rir[layout]
            stems, mixture = render_dynamic_stems_and_mix(
                buses,
                rir_samples,
                rir_lengths,
                source_ids=("source0", "source1"),
                keyframe_samples=keyframe_samples,
                output_sample_count=canary.M5_AUDIO_SAMPLE_COUNT,
            )
            for source_id in ("source0", "source1"):
                relative = (
                    Path("episodes")
                    / variant
                    / "audio"
                    / layout
                    / f"{source_id}_stem.wav"
                )
                record = _audio_record(root / relative, stems[source_id].episode)
                record["audio_path"] = relative.as_posix()
                record["sidecar_path"] = relative.with_suffix(".wav.json").as_posix()
                records[layout][source_id] = record  # type: ignore[index]
            relative = Path("episodes") / variant / "audio" / layout / "mixture.wav"
            record = _audio_record(root / relative, mixture)
            record["audio_path"] = relative.as_posix()
            record["sidecar_path"] = relative.with_suffix(".wav.json").as_posix()
            records[layout]["mixture"] = record  # type: ignore[index]
        audio[variant] = records

    evidence: dict[str, object] = {
        "inputs": {
            "beagle_dry": {"path": "inputs/beagle_dry.wav", "sha256": beagle_hash},
            "golden_dry": {"path": "inputs/golden_dry.wav", "sha256": golden_hash},
        },
        "audio": audio,
    }
    return evidence, rir, trajectory


def test_rehashed_stem_tamper_fails_independent_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(canary, "M5_AUDIO_SAMPLE_COUNT", 2_048)
    evidence, rir, trajectory = _reconstruction_fixture(tmp_path)
    assert (
        canary._audio_reconstruction_errors(tmp_path, evidence, rir, trajectory) == []
    )

    record = evidence["audio"]["A"]["binaural"]["source0"]  # type: ignore[index]
    audio_path = tmp_path / record["audio_path"]  # type: ignore[index]
    sidecar_path = tmp_path / record["sidecar_path"]  # type: ignore[index]
    decoded = read_float32_wav(audio_path, verify_sidecar=True)
    tampered = decoded.samples.copy()
    tampered[0, 100] += np.float32(0.25)
    audio_path.unlink()
    sidecar_path.unlink()
    artifact = write_float32_wav(
        audio_path,
        tampered,
        M5_AUDIO_SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata={"test_authority": True},
    )
    record["audio_sha256"] = artifact.audio_sha256  # type: ignore[index]
    record["sidecar_sha256"] = artifact.sidecar_sha256  # type: ignore[index]
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)

    errors = canary._audio_reconstruction_errors(tmp_path, evidence, rir, trajectory)
    assert any("A/binaural/source0 stem cannot be rebuilt" in item for item in errors)
    assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == record["audio_sha256"]  # type: ignore[index]
