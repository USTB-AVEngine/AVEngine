from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import avengine.optional_backends.spear_apartment as apartment


_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_spear_apartment_canary",
    Path(__file__).resolve().parents[2] / "tools/m6y/run_spear_apartment_canary.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)


def _plan(scenario_id: str = "S3") -> dict:
    actors = [
        {
            "actor_id": "dog0",
            "asset_id": apartment.BEAGLE_ASSET_ID,
            "blueprint_class_path": "dog_bp",
            "idle_animation": "dog_idle",
            "walking_animation": "dog_walk",
        },
        {
            "actor_id": "human0",
            "asset_id": apartment.HUMAN_ASSET_ID,
            "blueprint_class_path": "human_bp",
            "idle_animation": "human_idle",
            "walking_animation": "human_walk",
        },
    ]
    frames = []
    for frame_index in range(apartment.FRAME_COUNT):
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3_200,
                "actor_states": [
                    {
                        "actor_id": "dog0",
                        "translation_ue_cm": [1.0, 2.0, 27.1],
                        "actor_yaw_ue_deg": -90.0,
                        "anatomical_forward_ue_world": [0.0, -1.0, 0.0],
                    },
                    {
                        "actor_id": "human0",
                        "translation_ue_cm": [3.0 + frame_index, 4.0, 27.1],
                        "actor_yaw_ue_deg": -145.0,
                        "anatomical_forward_ue_world": [
                            0.573576436,
                            -0.819152044,
                            0.0,
                        ],
                    },
                ],
            }
        )
    return {
        "schema": apartment.PLAN_SCHEMA,
        "backend_role": "comparison_visual",
        "authority": {"backend_may_replan": False},
        "room": {
            "source_scene_provenance": {
                "provider": "SPEAR_Unreal",
                "scene_id": "apartment_0000",
            }
        },
        "camera": {
            "ue_position_cm": [-70.0, 65.0, 147.1],
            "ue_yaw_deg": -145.0,
            "horizontal_fov_deg": 105.0,
        },
        "render": {
            "frame_count": 75,
            "fps_num": 15,
            "fps_den": 1,
            "ticks_per_frame": 3_200,
        },
        "actors": actors,
        "source_logic": {"scenario_id": scenario_id},
        "frames": frames,
    }


def _make_input_tree(root: Path, scenario_id: str = "S3") -> dict[str, Path]:
    scenario_directory, variant_id = apartment.SCENARIO_DIRECTORIES[scenario_id]
    metadata = (
        root / "scenarios" / scenario_directory / "variants" / variant_id / "metadata"
    )
    videos = metadata.parent / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    values = {
        "timeline": metadata / "timeline.json",
        "source_manifest": metadata / "source_manifest.json",
        "flags": metadata / "flags.json",
        "room_capsule": root / "inputs/fixed_apartment_config/room_capsule.json",
        "qualification": root / "room/qualification.json",
        "authoritative_clean_binaural": videos / "clean_binaural.mp4",
        "authoritative_diagnostic_topdown": videos / "diagnostic_topdown_binaural.mp4",
    }
    for path in values.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    return values


def _make_motion_pilot_input_tree(
    root: Path, scenario_id: str = "P0"
) -> dict[str, Path]:
    episode_directory = apartment.MOTION_PILOT_DIRECTORIES[scenario_id]
    metadata = root / "episodes" / episode_directory / "metadata"
    videos = metadata.parent / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    values = {
        "timeline": metadata / "timeline.json",
        "source_manifest": metadata / "source_manifest.json",
        "flags": metadata / "flags.json",
        "room_capsule": root / "room/room_capsule.json",
        "qualification": root / "room/qualification.json",
        "authoritative_clean_binaural": videos / "clean_binaural.mp4",
        "authoritative_diagnostic_topdown": videos / "diagnostic_topdown_binaural.mp4",
    }
    for path in values.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    return values


def test_scenario_path_discovery_is_bounded_to_s0_s3_s4(tmp_path: Path) -> None:
    expected = _make_input_tree(tmp_path, "S3")
    observed = apartment.scenario_input_paths(tmp_path, "S3")
    assert observed == {key: value.resolve() for key, value in expected.items()}
    with pytest.raises(apartment.SpearApartmentError, match="unsupported"):
        apartment.scenario_input_paths(tmp_path, "S2")


def test_motion_pilot_path_discovery_and_suite_use_p0_to_p3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _make_motion_pilot_input_tree(tmp_path, "P0")
    observed = apartment.motion_pilot_input_paths(tmp_path, "P0")
    assert observed == {key: value.resolve() for key, value in expected.items()}
    with pytest.raises(apartment.SpearApartmentError, match="unsupported"):
        apartment.motion_pilot_input_paths(tmp_path, "S0")

    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: _plan("P0"),
    )
    suite = apartment.build_native_apartment_motion_pilot_suite(
        tmp_path, scenario_ids=("P0",)
    )
    scenario = suite["scenarios"][0]
    assert scenario["scenario_id"] == "P0"
    assert scenario["scenario_directory"] == "00_static_static"
    assert scenario["variant_id"] == "A"
    assert suite["authority"]["spear_unreal"] == ["final RGB pixels"]


def test_scenario_execution_keeps_native_map_and_habitat_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_input_tree(tmp_path, "S3")
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: _plan("S3"),
    )
    record = apartment.build_native_apartment_scenario(tmp_path, "S3")
    assert record["native_scene"] == {
        "map": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
        "layout": "native_map_unchanged",
        "lighting": "native_map_unchanged_no_added_lights",
        "lighting_profile": dict(apartment.NATIVE_LIGHTING_PROFILE),
        "outdoor_view": "native_map_assets_and_postprocess",
    }
    assert record["render"] == {
        "width": 1280,
        "height": 720,
        "frame_count": 75,
        "frame_rate_hz": 15,
        "horizontal_fov_deg": 105.0,
        "streaming_warmup_frames": 120,
        "camera_warmup_frames": 40,
    }
    assert record["reuse_contract"]["audio_camera_fov_cutoff"] is False
    assert record["plan"]["authority"]["backend_may_replan"] is False
    dog = next(
        value for value in record["plan"]["actors"] if value["actor_id"] == "dog0"
    )
    assert dog["ue_component_frame_delta"] == {
        "schema": "avengine_spear_component_frame_delta_v1",
        "rotation_deg": [0.0, 90.0, 0.0],
        "translation_cm": [0.0, 0.0, 33.64],
        "composition": "add_relative_preserving_blueprint_transform",
        "reason": "exact_M2_GLTF_to_UE_asset_local_axis_and_floor_calibration",
    }
    assert record["authoritative_inputs"] == {
        key: value.relative_to(tmp_path).as_posix() for key, value in paths.items()
    }


def test_apartment_lighting_profiles_keep_native_map_and_validate_photometry() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/m6y/spear_apartment_lighting_profiles.json"
    )
    profile = apartment.load_apartment_lighting_profile(path, "warm_indoor_fill")
    assert profile["profile_id"] == "warm_indoor_fill"
    assert len(profile["generated_lights"]) == 2
    assert all(light["cast_shadows"] for light in profile["generated_lights"])

    document = {
        "schema": apartment.LIGHTING_PROFILE_SCHEMA,
        "default_profile_id": "bad",
        "profiles": [
            {
                "profile_id": "bad",
                "label": "bad",
                "claim_boundary": "test",
                "generated_lights": [
                    {
                        "light_id": "x",
                        "position_ue_cm": [0, 0, 250],
                        "intensity_lumens": -1,
                        "attenuation_radius_cm": 400,
                        "temperature_kelvin": 4000,
                    }
                ],
            }
        ],
    }
    with pytest.raises(apartment.SpearApartmentError, match="not physical"):
        apartment.resolve_apartment_lighting_profile(document)

    document["profiles"][0]["generated_lights"][0]["intensity_lumens"] = 100
    document["profiles"][0]["generated_lights"][0]["cast_shadows"] = "false"
    with pytest.raises(apartment.SpearApartmentError, match="must be boolean"):
        apartment.resolve_apartment_lighting_profile(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("map", "not the native"),
        ("scenario", "disagrees"),
        ("replan", "must not replan"),
        ("hfov", "105 degree"),
        ("actors", "actor closure"),
    ],
)
def test_native_plan_validation_fails_closed(mutation: str, message: str) -> None:
    plan = _plan("S3")
    if mutation == "map":
        plan["room"]["source_scene_provenance"]["scene_id"] = "proxy"
    elif mutation == "scenario":
        plan["source_logic"]["scenario_id"] = "S4"
    elif mutation == "replan":
        plan["authority"]["backend_may_replan"] = True
    elif mutation == "hfov":
        plan["camera"]["horizontal_fov_deg"] = 90.0
    elif mutation == "actors":
        plan["actors"].reverse()
    with pytest.raises(apartment.SpearApartmentError, match=message):
        apartment._validate_native_plan(plan, scenario_id="S3")


def test_animation_position_uses_normalized_timeline_phase() -> None:
    assert apartment.animation_position_seconds(0.625, 1.6) == pytest.approx(1.0)
    with pytest.raises(apartment.SpearApartmentError, match=r"\[0,1\)"):
        apartment.animation_position_seconds(1.0, 1.6)
    with pytest.raises(apartment.SpearApartmentError, match="positive"):
        apartment.animation_position_seconds(0.5, 0.0)


def test_root_readback_gate_covers_every_actor_and_camera_frame() -> None:
    plan = _plan()
    actor_readbacks = {"dog0": [], "human0": []}
    camera_readbacks = []
    for frame_index, frame in enumerate(plan["frames"]):
        for state in frame["actor_states"]:
            actor_readbacks[state["actor_id"]].append(
                {
                    "frame_index": frame_index,
                    "location_cm": list(state["translation_ue_cm"]),
                    "rotation_deg": [0.0, 0.0, state["actor_yaw_ue_deg"]],
                }
            )
        camera_readbacks.append(
            {
                "frame_index": frame_index,
                "location_cm": [-70.0, 65.0, 147.1],
                "rotation_deg": [0.0, 0.0, -145.0],
            }
        )
    summary = apartment.summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
        camera_position_cm=[-70.0, 65.0, 147.1],
        camera_yaw_deg=-145.0,
    )
    assert set(summary) == {"dog0", "human0", "camera"}
    assert all(value["status"] == "pass" for value in summary.values())

    drifted = deepcopy(actor_readbacks)
    drifted["human0"][11]["location_cm"][0] += 1.0
    with pytest.raises(apartment.SpearApartmentError, match="human0.*drifted"):
        apartment.summarize_root_readbacks(
            expected_frames=plan["frames"],
            actor_readbacks=drifted,
            camera_readbacks=camera_readbacks,
            camera_position_cm=[-70.0, 65.0, 147.1],
            camera_yaw_deg=-145.0,
        )


def test_media_commands_copy_audio_and_reuse_only_topdown_right_panel() -> None:
    clean = apartment.build_clean_binaural_mux_command(
        ue_video_path="ue.mp4",
        authoritative_clean_path="habitat_clean.mp4",
        output_path="clean.mp4",
    )
    assert clean[clean.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in clean
    assert clean[clean.index("-c:a") + 1] == "copy"
    assert "-shortest" not in clean

    topdown = apartment.build_topdown_visual_command(
        ue_video_path="ue.mp4",
        authoritative_diagnostic_path="habitat_diag.mp4",
        output_path="topdown.mp4",
    )
    graph = topdown[topdown.index("-filter_complex") + 1]
    assert "crop=640:480:iw-640:0[topdown]" in graph
    assert "[ue][topdown]hstack" in graph
    assert "-an" in topdown
    assert "-shortest" not in topdown

    nvenc = apartment.build_topdown_visual_command(
        ue_video_path="ue.mp4",
        authoritative_diagnostic_path="habitat_diag.mp4",
        output_path="topdown.mp4",
        video_encoder="h264_nvenc",
        encoder_gpu=3,
    )
    assert nvenc[nvenc.index("-c:v") + 1] == "h264_nvenc"
    assert nvenc[nvenc.index("-gpu") + 1] == "3"
    assert nvenc[nvenc.index("-preset") + 1] == "p5"

    raw = apartment.build_rawvideo_encode_command(
        output_path="raw.mp4",
        video_encoder="h264_nvenc",
        encoder_gpu=3,
    )
    assert raw[raw.index("-f") + 1] == "rawvideo"
    assert raw[raw.index("-pixel_format") + 1] == "bgr24"
    assert raw[raw.index("-video_size") + 1] == "1280x720"
    assert raw[raw.index("-framerate") + 1] == "15"
    assert raw[raw.index("-i") + 1] == "pipe:0"
    assert raw[raw.index("-frames:v") + 1] == "75"
    assert raw[raw.index("-gpu") + 1] == "3"

    with pytest.raises(apartment.SpearApartmentError, match="pixel_format"):
        apartment.build_rawvideo_encode_command(
            output_path="bad.mp4", pixel_format="rgba"
        )

    with pytest.raises(apartment.SpearApartmentError, match="unsupported"):
        apartment.build_png_encode_command(
            frames_pattern="frame_%04d.png",
            output_path="bad.mp4",
            video_encoder="unknown",
        )


def test_resume_reopens_complete_scenario_and_discards_only_incomplete_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = {"scenario_id": "episode_0001"}
    scenario_root = tmp_path / "episode_0001"
    scenario_root.mkdir()
    media = {
        media_id: {"status": "pass", "path": f"{media_id}.mp4"}
        for media_id in _RUNNER.MEDIA_EXPECTATIONS
    }
    (scenario_root / "evidence.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scenario_id": "episode_0001",
                "timing": {"video_encoder": "h264_nvenc"},
                "media": media,
            }
        ),
        encoding="utf-8",
    )

    def fake_probe(path: Path, **_kwargs: object) -> dict:
        return media[path.stem]

    monkeypatch.setattr(_RUNNER, "_probe_media", fake_probe)
    reopened = _RUNNER._load_resumable_scenario_record(
        output_root=tmp_path,
        scenario=scenario,
        video_encoder="h264_nvenc",
    )
    assert reopened is not None
    assert reopened["scenario_id"] == "episode_0001"

    incomplete = tmp_path / "episode_0002"
    incomplete.mkdir()
    (incomplete / "partial.mp4").touch()
    assert (
        _RUNNER._load_resumable_scenario_record(
            output_root=tmp_path,
            scenario={"scenario_id": "episode_0002"},
            video_encoder="h264_nvenc",
        )
        is None
    )
    assert not incomplete.exists()


def test_resume_requires_the_same_retained_execution_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    bundle.mkdir()
    output.mkdir()
    retained = {"scenarios": [{"scenario_id": "old"}]}
    (output / "suite_execution_plan.json").write_text(
        json.dumps(retained), encoding="utf-8"
    )
    current = {"scenarios": [{"scenario_id": "new"}]}
    monkeypatch.setattr(
        _RUNNER, "load_apartment_lighting_profile", lambda *_args: {}
    )
    monkeypatch.setattr(
        _RUNNER, "build_native_apartment_asset_bound_suite", lambda *_args, **_kwargs: current
    )
    monkeypatch.setattr(
        _RUNNER, "_assert_suite_actor_binding_closure", lambda _suite: None
    )
    args = _RUNNER.parse_args(
        [
            "--bundle-root",
            str(bundle),
            "--input-layout",
            "asset-bound-batch",
            "--output-dir",
            str(output),
            "--resume",
        ]
    )
    with pytest.raises(
        RuntimeError, match="differs from the retained execution plan"
    ):
        _RUNNER.run(args)
    assert json.loads(
        (output / "suite_execution_plan.json").read_text(encoding="utf-8")
    ) == retained


def test_exact_episode_shards_are_balanced_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    episode_ids = tuple(f"episode_{index:04d}" for index in range(7))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "episode_count": len(episode_ids),
                "episode_ids": list(episode_ids),
            }
        ),
        encoding="utf-8",
    )

    declared = apartment.asset_bound_bundle_episode_ids(tmp_path)
    shards = [
        apartment.contiguous_episode_shard(
            declared, shard_count=3, shard_index=index
        )
        for index in range(3)
    ]

    assert [len(shard) for shard in shards] == [3, 2, 2]
    assert tuple(value for shard in shards for value in shard) == episode_ids
    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]).isdisjoint(shards[2])
    assert set(shards[1]).isdisjoint(shards[2])


def test_runner_dry_run_records_only_its_exact_manifest_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    bundle.mkdir()
    episode_ids = [f"episode_{index:04d}" for index in range(5)]
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "episode_count": len(episode_ids),
                "episode_ids": episode_ids,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _RUNNER, "load_apartment_lighting_profile", lambda *_args: {}
    )

    def fake_suite(
        _bundle: Path,
        *,
        scenario_ids: tuple[str, ...],
        lighting_profile: dict,
    ) -> dict:
        assert lighting_profile == {}
        return {
            "scenarios": [
                {"scenario_id": scenario_id} for scenario_id in scenario_ids
            ]
        }

    monkeypatch.setattr(
        _RUNNER, "build_native_apartment_asset_bound_suite", fake_suite
    )
    monkeypatch.setattr(
        _RUNNER, "_assert_suite_actor_binding_closure", lambda _suite: None
    )
    args = _RUNNER.parse_args(
        [
            "--bundle-root",
            str(bundle),
            "--input-layout",
            "asset-bound-batch",
            "--output-dir",
            str(output),
            "--shard-count",
            "2",
            "--shard-index",
            "1",
            "--dry-run",
        ]
    )

    plan_path = _RUNNER.run(args)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert [value["scenario_id"] for value in plan["scenarios"]] == episode_ids[3:]
    assert plan["execution_partition"] == {
        "kind": "contiguous_manifest_episode_ids",
        "shard_count": 2,
        "shard_index": 1,
        "total_episode_count": 5,
        "selected_episode_count": 2,
        "first_episode_id": "episode_0003",
        "last_episode_id": "episode_0004",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--shard-count", "2"],
        ["--shard-index", "0"],
        ["--shard-count", "2", "--shard-index", "2"],
        ["--shard-count", "2", "--shard-index", "0", "--scenario", "x"],
    ],
)
def test_runner_rejects_incomplete_or_overlapping_shard_selection(
    tmp_path: Path, argv: list[str]
) -> None:
    with pytest.raises(SystemExit):
        _RUNNER.parse_args(
            [
                "--input-layout",
                "asset-bound-batch",
                "--output-dir",
                str(tmp_path / "output"),
                *argv,
            ]
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are unavailable",
)
def test_media_probe_requires_full_packet_identical_binaural_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    visual = tmp_path / "visual.mp4"
    copied = tmp_path / "copied.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x48:r=15:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=5",
            "-frames:v",
            "75",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "2",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x48:r=15:d=5",
            "-frames:v",
            "75",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(visual),
        ],
        check=True,
    )
    subprocess.run(
        apartment.build_clean_binaural_mux_command(
            ue_video_path=visual,
            authoritative_clean_path=source,
            output_path=copied,
        ),
        check=True,
    )

    probe = _RUNNER._probe_media(
        copied,
        expected_width=64,
        expected_height=48,
        expect_audio=True,
    )
    assert probe["size_bytes"] == copied.stat().st_size
    assert probe["audio_packet_sha256"] == _RUNNER._audio_packet_sha256(source)


def test_runtime_timing_contract_requires_rgb_and_topdown_outputs() -> None:
    assert _RUNNER.TIMING_SCHEMA == "avengine_apartment_runtime_timing_v1"
    assert _RUNNER.REQUIRED_SAMPLE_OUTPUTS == (
        "ue_visual_only.mp4",
        "ue_topdown_visual_only.mp4",
        "ue_clean_binaural.mp4",
        "ue_topdown_binaural.mp4",
    )
    started = _RUNNER.time.perf_counter()
    assert _RUNNER._elapsed_seconds(started) >= 0.0


def test_default_asset_forward_bindings_are_explicit() -> None:
    assert (
        apartment.DEFAULT_ACTOR_BINDINGS[apartment.BEAGLE_ASSET_ID][
            "ue_anatomical_forward_yaw_deg"
        ]
        == 180.0
    )
    assert (
        apartment.DEFAULT_ACTOR_BINDINGS[apartment.HUMAN_ASSET_ID][
            "ue_anatomical_forward_yaw_deg"
        ]
        == 90.0
    )
    assert (
        "Standing_Idle"
        in apartment.DEFAULT_ACTOR_BINDINGS[apartment.HUMAN_ASSET_ID]["idle_animation"]
    )
    assert apartment.DEFAULT_ACTOR_BINDINGS[apartment.BEAGLE_ASSET_ID][
        "ue_component_frame_delta"
    ]["rotation_deg"] == [0.0, 90.0, 0.0]
    assert apartment.DEFAULT_ACTOR_BINDINGS[apartment.BEAGLE_ASSET_ID][
        "ue_component_frame_delta"
    ]["translation_cm"] == [0.0, 0.0, 33.64]
    assert apartment.DEFAULT_ACTOR_BINDINGS[apartment.HUMAN_ASSET_ID][
        "ue_component_frame_delta"
    ]["translation_cm"] == [0.0, 0.0, 0.0]
    border_collie = apartment.DEFAULT_ACTOR_BINDINGS[
        apartment.BORDER_COLLIE_ASSET_ID
    ]
    assert border_collie["ue_anatomical_forward_yaw_deg"] == 0.0
    assert border_collie["ue_component_frame_delta"]["rotation_deg"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert border_collie["ue_component_frame_delta"]["translation_cm"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert border_collie["ue_anatomical_basis_bones"] == {
        "rear": "bone_0",
        "front": "bone_4",
        "body": "bone_0",
        "left_foot": "bone_67",
        "right_foot": "bone_56",
    }
    assert apartment.anatomical_basis_bones_for_asset(
        apartment.BORDER_COLLIE_ASSET_ID
    ) == border_collie["ue_anatomical_basis_bones"]
    cat = apartment.DEFAULT_ACTOR_BINDINGS[apartment.CAT_ASSET_ID]
    assert cat["ue_anatomical_forward_yaw_deg"] == 0.0
    assert cat["ue_anatomical_basis_bones"] == {
        "rear": "bone_0",
        "front": "bone_4",
        "body": "bone_0",
        "left_foot": "bone_9",
        "right_foot": "bone_14",
    }
    assert cat["ue_component_frame_delta"]["translation_cm"] == [
        0.0,
        0.0,
        42.25,
    ]
    assert apartment.anatomical_basis_bones_for_asset(
        apartment.CAT_ASSET_ID
    ) == cat["ue_anatomical_basis_bones"]
    assert (
        apartment.anatomical_basis_bones_for_asset(apartment.HUMAN_ASSET_ID)
        is None
    )


def test_generated_anatomical_basis_mapping_is_exact_and_asset_local() -> None:
    bindings = deepcopy(apartment.DEFAULT_ACTOR_BINDINGS)
    bindings[apartment.BORDER_COLLIE_ASSET_ID]["ue_anatomical_basis_bones"].pop(
        "front"
    )
    with pytest.raises(apartment.SpearApartmentError, match="define exactly"):
        apartment.anatomical_basis_bones_for_asset(
            apartment.BORDER_COLLIE_ASSET_ID, actor_bindings=bindings
        )


def test_generated_anatomical_basis_mapping_reaches_runtime_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_input_tree(tmp_path, "S3")
    plan = _plan("S3")
    plan["actors"][0]["asset_id"] = apartment.BORDER_COLLIE_ASSET_ID
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: deepcopy(plan),
    )
    record = apartment.build_native_apartment_scenario(tmp_path, "S3")
    dog = next(
        value
        for value in record["plan"]["actors"]
        if value["actor_id"] == "dog0"
    )
    assert dog["ue_anatomical_basis_bones"]["front"] == "bone_4"


def test_component_frame_delta_must_preserve_blueprint_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_input_tree(tmp_path, "S3")
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: _plan("S3"),
    )
    bindings = deepcopy(apartment.DEFAULT_ACTOR_BINDINGS)
    bindings[apartment.BEAGLE_ASSET_ID]["ue_component_frame_delta"]["composition"] = (
        "replace_blueprint_transform"
    )
    with pytest.raises(apartment.SpearApartmentError, match="may not replace"):
        apartment.build_native_apartment_scenario(
            tmp_path, "S3", actor_bindings=bindings
        )


def test_runtime_component_delta_is_added_to_authored_blueprint_transform() -> None:
    class FakeComponent:
        def __init__(self) -> None:
            self.location = {"X": 1.0, "Y": -2.0, "Z": 3.0}
            self.rotation = {"Roll": 4.0, "Pitch": 5.0, "Yaw": 6.0}

        def get_property_value(self, *, property_name: str):
            if property_name == "RelativeLocation":
                return dict(self.location)
            if property_name == "RelativeRotation":
                return dict(self.rotation)
            raise AssertionError(property_name)

        def K2_AddRelativeLocation(self, *, DeltaLocation, **_):
            for axis in ("X", "Y", "Z"):
                self.location[axis] += DeltaLocation[axis]

        def K2_AddRelativeRotation(self, *, DeltaRotation, **_):
            for axis in ("Roll", "Pitch", "Yaw"):
                self.rotation[axis] += DeltaRotation[axis]

    declaration = {
        "actor_id": "dog0",
        "asset_id": apartment.BEAGLE_ASSET_ID,
        "ue_component_frame_delta": apartment.DEFAULT_ACTOR_BINDINGS[
            apartment.BEAGLE_ASSET_ID
        ]["ue_component_frame_delta"],
    }
    result = apartment.apply_ue_component_frame_delta(FakeComponent(), declaration)
    assert result["blueprint_relative_before"] == {
        "translation_cm": [1.0, -2.0, 3.0],
        "rotation_deg": [4.0, 5.0, 6.0],
    }
    assert result["blueprint_relative_after"] == {
        "translation_cm": [1.0, -2.0, 36.64],
        "rotation_deg": [4.0, 95.0, 6.0],
    }
    assert result["timeline_anchor_mutated"] is False
    assert result["target"] == "attached_visual_actor_root_component"


def test_runtime_component_delta_accepts_equivalent_gimbal_rotator_readback() -> None:
    class GimbalComponent:
        def __init__(self) -> None:
            self.location = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            self.rotation = {"Roll": 0.0, "Pitch": 0.0, "Yaw": 0.0}

        def get_property_value(self, *, property_name: str):
            if property_name == "RelativeLocation":
                return dict(self.location)
            if property_name == "RelativeRotation":
                return dict(self.rotation)
            raise AssertionError(property_name)

        def K2_AddRelativeLocation(self, *, DeltaLocation, **_):
            for axis in ("X", "Y", "Z"):
                self.location[axis] += DeltaLocation[axis]

        def K2_AddRelativeRotation(self, *, DeltaRotation, **_):
            assert DeltaRotation == {"Roll": 0.0, "Pitch": 90.0, "Yaw": 0.0}
            # UE may report the same +90 degree pitch as this Euler triplet.
            self.rotation = {"Roll": 180.0, "Pitch": 90.0, "Yaw": 180.0}

    declaration = {
        "actor_id": "dog0",
        "asset_id": apartment.BEAGLE_ASSET_ID,
        "ue_component_frame_delta": apartment.DEFAULT_ACTOR_BINDINGS[
            apartment.BEAGLE_ASSET_ID
        ]["ue_component_frame_delta"],
    }
    result = apartment.apply_ue_component_frame_delta(GimbalComponent(), declaration)
    assert result["euler_component_rotation_delta_error_deg"] == pytest.approx(180.0)
    assert result["quaternion_equivalence_rotation_error_deg"] == pytest.approx(0.0)
    assert result["maximum_rotation_delta_error_deg"] == pytest.approx(0.0)


def test_visual_bounds_gate_proves_beagle_floor_contact_and_horizontal_frame() -> None:
    plan = _plan()
    records = {"dog0": [], "human0": []}
    for frame_index, frame in enumerate(plan["frames"]):
        dog_root = frame["actor_states"][0]["translation_ue_cm"]
        human_root = frame["actor_states"][1]["translation_ue_cm"]
        records["dog0"].append(
            {
                "frame_index": frame_index,
                "minimum_cm": [dog_root[0] - 35.0, dog_root[1] - 25.0, dog_root[2]],
                "maximum_cm": [
                    dog_root[0] + 35.0,
                    dog_root[1] + 25.0,
                    dog_root[2] + 50.0,
                ],
            }
        )
        records["human0"].append(
            {
                "frame_index": frame_index,
                "minimum_cm": [
                    human_root[0] - 20.0,
                    human_root[1] - 20.0,
                    human_root[2],
                ],
                "maximum_cm": [
                    human_root[0] + 20.0,
                    human_root[1] + 20.0,
                    human_root[2] + 175.0,
                ],
            }
        )
    summary = apartment.summarize_actor_bounds(
        expected_frames=plan["frames"],
        actor_declarations=plan["actors"],
        actor_bounds=records,
    )
    assert summary["dog0"]["status"] == "pass"
    assert summary["dog0"]["maximum_floor_error_cm"] == 0.0
    assert summary["human0"]["status"] == "observed"

    border_collie_plan = deepcopy(plan)
    border_collie_plan["actors"][0]["asset_id"] = apartment.BORDER_COLLIE_ASSET_ID
    border_collie_summary = apartment.summarize_actor_bounds(
        expected_frames=border_collie_plan["frames"],
        actor_declarations=border_collie_plan["actors"],
        actor_bounds=records,
    )
    assert border_collie_summary["dog0"]["status"] == "pass"

    drifted = deepcopy(records)
    drifted["dog0"][4]["minimum_cm"][2] -= 6.0
    with pytest.raises(apartment.SpearApartmentError, match="actor-root floor"):
        apartment.summarize_actor_bounds(
            expected_frames=plan["frames"],
            actor_declarations=plan["actors"],
            actor_bounds=drifted,
        )


def test_anatomical_forward_gate_rejects_a_visually_reversed_skeleton() -> None:
    plan = _plan()
    readbacks = {
        "dog0": [
            {
                "frame_index": frame_index,
                "basis_kind": "prefixed_bip_quadruped_longitudinal_v1",
                "forward_vector_ue": [0.0, -1.0, 0.0],
                "bone_names": {"rear": "beagle Pelvis", "front": "beagle Spine2"},
            }
            for frame_index in (0, 37, 74)
        ],
        "human0": [
            {
                "frame_index": frame_index,
                "basis_kind": "humanoid_semantic_v1",
                "forward_vector_ue": [0.573576436, -0.819152044, 0.0],
                "bone_names": {"pelvis": "Bip01 Pelvis", "spine": "Bip01 Spine2"},
            }
            for frame_index in (0, 37, 74)
        ],
    }
    summary = apartment.summarize_anatomical_forward_readbacks(
        expected_frames=plan["frames"],
        visual_forward_readbacks=readbacks,
    )
    assert summary["dog0"]["status"] == "pass"
    assert summary["dog0"]["maximum_angular_error_deg"] == pytest.approx(0.0)

    reversed_readbacks = deepcopy(readbacks)
    reversed_readbacks["dog0"][1]["forward_vector_ue"] = [0.0, 1.0, 0.0]
    with pytest.raises(apartment.SpearApartmentError, match="faces away"):
        apartment.summarize_anatomical_forward_readbacks(
            expected_frames=plan["frames"],
            visual_forward_readbacks=reversed_readbacks,
        )

    tilted_readbacks = deepcopy(readbacks)
    tilted_readbacks["dog0"][1]["forward_vector_ue"] = [
        0.0,
        -0.1,
        0.994987437,
    ]
    with pytest.raises(apartment.SpearApartmentError, match="not horizontal"):
        apartment.summarize_anatomical_forward_readbacks(
            expected_frames=plan["frames"],
            visual_forward_readbacks=tilted_readbacks,
        )
