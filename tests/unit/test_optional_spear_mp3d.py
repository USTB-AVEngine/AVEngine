from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from avengine.optional_backends.spear_mp3d import (
    DOG_BP_CLASS_PATH,
    EXECUTION_SCHEMA,
    HUMAN_BP_CLASS_PATH,
    M5_1_CAMERA_HABITAT_M,
    M5_1_CAPTURE_SCHEMA,
    M5_1_CAPTURE_INSTALLED_SCHEMA_V2,
    M5_1_EMITTER_SCHEMA,
    M5_1_EXECUTION_SCHEMA,
    M5_1_FRAME_COUNT,
    M5_1_GATE_SCHEMA,
    M5_1_ROUTE_ID,
    M5_1_ROUTE_SCHEMA,
    M5_1_SOURCE_PROGRAM_SCHEMA,
    MATERIAL_COLOR_SCHEMA,
    MP3DExecutionError,
    MP3D_ROOM_ID,
    build_m5_1_mp3d_execution_plan,
    build_mp3d_execution_plan,
    fixed_exposure_profile,
    luminance_exposure_qa,
    render_color_fidelity_qa,
)


_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_spear_mp3d_canary",
    Path(__file__).resolve().parents[2] / "tools/rooms/run_spear_mp3d_canary.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)
_LuminanceAccumulator = _RUNNER._LuminanceAccumulator
_audio_packet_sha256 = _RUNNER._audio_packet_sha256
_probe_media = _RUNNER._probe_media
_root_gate = _RUNNER._root_gate
parse_args = _RUNNER.parse_args
run = _RUNNER.run


def _visual_plan() -> dict:
    return {
        "schema": "avengine_optional_spear_visual_plan_v1",
        "backend_role": "comparison_visual",
        "authority": {
            "actor_state": "Timeline_v2",
            "room_identity_and_layout": "RoomCapsule",
            "source_logic": "source_manifest_and_flags",
            "source_center_placement": "room_qualification",
            "backend_may_replan": False,
        },
        "room": {"room_id": MP3D_ROOM_ID},
        "render": {
            "frame_count": 75,
            "fps_num": 15,
            "fps_den": 1,
            "ticks_per_frame": 3200,
        },
        "camera": {
            "ue_position_cm": [0, 0, 150],
            "ue_yaw_deg": 0,
            "horizontal_fov_deg": 90,
        },
        "actors": [
            {
                "actor_id": "dog0",
                "asset_id": "beagle",
                "blueprint_class_path": "/Game/AVEngine/BP_Beagle.BP_Beagle_C",
            }
        ],
        "frames": [
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "actor_states": [
                    {
                        "actor_id": "dog0",
                        "action_id": "walk",
                        "action_phase": (index % 25) / 25,
                        "translation_ue_cm": [index, 0, 0],
                        "actor_yaw_ue_deg": 0,
                        "ue_animation": "/Game/AVEngine/Walking.Walking",
                    }
                ],
            }
            for index in range(75)
        ],
        "source_logic": {"sources": []},
        "qualification": {"status": "pass", "claim_boundary": "source_center_only"},
    }


def _import_manifest() -> dict:
    return {
        "schema": "avengine_mp3d_ue_import_result_v1",
        "status": "passed",
        "reload_verification": {"status": "passed"},
        "scene_content": {
            "static_meshes": [
                f"/Game/AVEngine/MP3D/mesh_{index:03d}" for index in range(71)
            ],
            "material_count": 23,
            "texture_count": 23,
        },
    }


def _material_color_result() -> dict:
    base_color_textures = [
        {
            "source_texture_index": index,
            "texture_path": f"/Game/AVEngine/MP3D/texture_{index:03d}_basecolor_srgb",
            "srgb": True,
            "semantic": "base_color_srgb",
        }
        for index in range(23)
    ]
    occlusion_textures = [
        {
            "source_texture_index": index,
            "texture_path": f"/Game/AVEngine/MP3D/texture_{index:03d}",
            "srgb": False,
            "semantic": "occlusion_linear_red_channel",
        }
        for index in range(23)
    ]
    return {
        "schema": MATERIAL_COLOR_SCHEMA,
        "status": "pass",
        "operation": "verify_only",
        "fresh_editor_reload": True,
        "content_root": "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy",
        "source_gltf_contract": {
            "material_count": 23,
            "base_color_reference_count": 23,
            "occlusion_reference_count": 23,
            "shared_base_color_and_occlusion_texture_count": 23,
            "other_texture_reference_count": 0,
        },
        "counts": {
            "source_texture_count": 23,
            "material_count": 23,
            "base_color_texture_count": 23,
            "base_color_binding_count": 23,
            "base_color_srgb_true_count": 23,
            "occlusion_texture_count": 23,
            "occlusion_binding_count": 23,
            "occlusion_srgb_false_count": 23,
            "unexpected_texture_binding_count": 0,
        },
        "base_color_textures": base_color_textures,
        "occlusion_textures": occlusion_textures,
        "material_bindings": [
            {
                "source_material_index": index,
                "material_path": f"/Game/AVEngine/MP3D/material_{index:03d}",
                "base_color_parameter_name": "BaseColorTexture",
                "base_color_texture_path": base_color_textures[index]["texture_path"],
                "occlusion_parameter_name": "OcclusionTexture",
                "occlusion_texture_path": occlusion_textures[index]["texture_path"],
                "unexpected_bound_texture_parameters": [],
            }
            for index in range(23)
        ],
    }


def _m5_1_inputs() -> dict:
    starts = {
        "human0": [-4.6, 0.072447, -2.7],
        "dog0": [-3.7, 0.072447, -2.7],
    }
    ends = {
        "human0": [-4.6, 0.072447, -3.8],
        "dog0": [-3.7, 0.072447, -3.8],
    }

    def point(actor_id: str, frame_index: int) -> list[float]:
        return [
            starts[actor_id][axis]
            + (ends[actor_id][axis] - starts[actor_id][axis])
            * frame_index
            / (M5_1_FRAME_COUNT - 1)
            for axis in range(3)
        ]

    route_manifest = {
        "schema": M5_1_ROUTE_SCHEMA,
        "room_id": MP3D_ROOM_ID,
        "route_id": M5_1_ROUTE_ID,
        "frame_count": M5_1_FRAME_COUNT,
        "frame_rate_hz": 15,
        "center_navigation_semantics": "actor_root_center_only",
        "path_generation": "linear_endpoint_interpolation_v1",
        "routes": {
            actor_id: {"start_m": starts[actor_id], "end_m": ends[actor_id]}
            for actor_id in ("human0", "dog0")
        },
    }
    navmesh_gate = {
        "schema": M5_1_GATE_SCHEMA,
        "status": "pass",
        "route_id": M5_1_ROUTE_ID,
        "frame_count": M5_1_FRAME_COUNT,
        "frame_rate_hz": 15,
        "pathfinder": {
            "declared_navmesh_loaded": True,
            "center_navigation_semantics": "actor_root_center_only",
            "routes": {
                actor_id: {
                    "all_frames_navigable": True,
                    "navigable_frame_count": M5_1_FRAME_COUNT,
                    "frame_count": M5_1_FRAME_COUNT,
                    "no_sliding_passed_segment_count": M5_1_FRAME_COUNT - 1,
                    "start_m": starts[actor_id],
                    "end_m": ends[actor_id],
                }
                for actor_id in ("human0", "dog0")
            },
        },
        "gates": [{"status": "pass"} for _ in range(14)],
    }
    frame_readback = [
        {
            "frame_index": frame_index,
            "pts_ticks": frame_index * 3200,
            "human": {
                "actor_root_position_m": point("human0", frame_index),
                "action_sample_index": frame_index % 16,
            },
            "beagle": {
                "actor_root_position_m": point("dog0", frame_index),
                "action_sample_index": (frame_index % 45) % 25,
            },
        }
        for frame_index in range(M5_1_FRAME_COUNT)
    ]
    source_program = {
        "schema": M5_1_SOURCE_PROGRAM_SCHEMA,
        "applicability": "taxonomy_event_timing_and_audio_program_only",
        "legacy_spatial_trajectory_applicable": False,
        "clip_time_and_audio_contract": {
            "frame_count": M5_1_FRAME_COUNT,
            "fps_num": 15,
            "fps_den": 1,
            "sample_rate_hz": 16_000,
            "sample_count": 288_000,
        },
        "sources": [
            {
                "source_id": source_id,
                "event_windows": [
                    {
                        "start_frame": 0,
                        "end_frame_exclusive": M5_1_FRAME_COUNT,
                        "start_sample": 0,
                        "end_sample_exclusive": 288_000,
                    }
                ],
            }
            for source_id in ("source0", "source1")
        ],
    }
    emitter_trajectories = {
        "schema": M5_1_EMITTER_SCHEMA,
        "source_ids": ["source0", "source1"],
        "sources": {
            source_id: {
                "frame_count": M5_1_FRAME_COUNT,
                "position_authority": (
                    "animated_articulated_link_world_transform_readback"
                ),
                "link_name": f"{source_id}_mouth",
                "positions_m": [
                    point(actor_id, frame_index)
                    for frame_index in range(M5_1_FRAME_COUNT)
                ],
            }
            for source_id, actor_id in (
                ("source0", "human0"),
                ("source1", "dog0"),
            )
        },
    }
    imported = _import_manifest()
    imported["m2_beagle"] = {
        "content": {
            "blueprint_class_path": DOG_BP_CLASS_PATH,
            "animations": {"Walking": "/Game/AVEngine/BeagleWalking"},
        }
    }
    return {
        "route_manifest": route_manifest,
        "capture_evidence": {
            "schema": M5_1_CAPTURE_SCHEMA,
            "status": "pass",
            "frame_count": M5_1_FRAME_COUNT,
            "frame_rate_hz": 15,
            "time_base_hz": 48_000,
            "qualification_claim": False,
            "research_only": True,
            "camera": {
                "position_m": list(M5_1_CAMERA_HABITAT_M),
                "rotation_xyzw": [0, 0, 0, 1],
                "horizontal_fov_deg": 90.0,
            },
        },
        "frame_readback": frame_readback,
        "navmesh_gate": navmesh_gate,
        "source_program": source_program,
        "emitter_trajectories": emitter_trajectories,
        "room_registry": {
            "records": [
                {
                    "room_id": MP3D_ROOM_ID,
                    "revision": "fixture",
                    "provider_id": "mp3d",
                    "tier": "visual_research_only",
                    "admission_state": "not_admitted",
                    "lineage": {"episode_layout_id": "m5_1_human_beagle_route_v1"},
                }
            ]
        },
        "raw_room_qualification": {
            "schema": "avengine_m6_room_qualification_report_v1",
            "subject": {"room_id": MP3D_ROOM_ID},
            "dataset_admission": False,
            "dimensions": {
                "visual_runtime_status": {"status": "pass"},
                "navigation_status": {"status": "pass"},
            },
        },
        "ue_import_manifest": imported,
        "ue_material_color_result": _material_color_result(),
        "human_ue_manifest": {
            "content": {
                "blueprint": HUMAN_BP_CLASS_PATH.rsplit(".", 1)[0],
                "animations": {"Walking": "/Game/AVEngine/HumanWalking"},
            }
        },
    }


def test_builds_timeline_v2_mp3d_execution_plan_with_fixed_exposure() -> None:
    plan = build_mp3d_execution_plan(
        visual_plan=_visual_plan(),
        ue_import_manifest=_import_manifest(),
        ue_material_color_result=_material_color_result(),
    )

    assert plan["schema"] == EXECUTION_SCHEMA
    assert plan["authority"]["actor_state"] == "Timeline_v2"
    assert plan["authority"]["backend_may_replan"] is False
    assert plan["scene"]["spawned_scene_mesh_actor_count"] == 71
    assert plan["render"] == {
        "width": 1280,
        "height": 720,
        "frame_count": 75,
        "fps_num": 15,
        "fps_den": 1,
        "streaming_warmup_frames": 120,
    }
    exposure = plan["exposure_and_lighting"]
    assert exposure["eye_adaptation"] == "disabled"
    assert "r.EyeAdaptationQuality 0" in exposure["console_commands"]
    assert exposure["directional_key"]["intensity_lux"] < 10.0
    assert exposure["fixed_output_gain"] == 1.0
    assert plan["scene"]["material_color_contract"]["base_color_srgb_true_count"] == 23
    assert plan["scene"]["material_color_contract"]["occlusion_srgb_false_count"] == 23
    assert plan["frames"][7]["actor_states"][0]["translation_ue_cm"] == [7, 0, 0]


def test_fixed_gain_highlight_qa_rejects_no_temporal_or_per_frame_adaptation() -> None:
    bright = np.full((3, 32, 32, 3), 230, dtype=np.uint8)
    profile = fixed_exposure_profile(output_gain=0.72)
    qa = luminance_exposure_qa(bright, profile)

    assert qa["status"] == "pass"
    assert qa["saturated_fraction"] == 0.0
    assert qa["mean_luminance"] == pytest.approx((230 / 255) * 0.72)

    uncorrected = luminance_exposure_qa(bright, fixed_exposure_profile(output_gain=1.0))
    assert uncorrected["status"] == "fail"


def test_fixed_exposure_rejects_black_frames_in_array_and_streaming_qa() -> None:
    black = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    profile = fixed_exposure_profile()

    array_qa = luminance_exposure_qa(black, profile)
    assert array_qa["status"] == "fail"
    assert array_qa["nonblack_fraction"] == 0.0

    streaming = _LuminanceAccumulator(profile)
    streaming.add_bgr(black[0])
    streaming_qa = streaming.result(profile)
    assert streaming_qa["status"] == "fail"
    assert streaming_qa["mean_luminance"] == 0.0


def test_render_color_fidelity_qa_rejects_desaturated_ue_and_accepts_retention() -> (
    None
):
    habitat = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    habitat[..., 0] = 150
    habitat[..., 1] = 110
    habitat[..., 2] = 80
    desaturated = np.full_like(habitat, 115)
    retained = habitat.copy()
    profile = fixed_exposure_profile()

    failed = render_color_fidelity_qa(desaturated, habitat, profile)
    passed = render_color_fidelity_qa(retained, habitat, profile)

    assert failed["status"] == "fail"
    assert failed["mean_chroma_ratio_ue_to_habitat"] == 0.0
    assert passed["status"] == "pass"
    assert passed["mean_chroma_ratio_ue_to_habitat"] == pytest.approx(1.0)


def test_capture_reader_preserves_v1_capture_schema() -> None:
    plan = build_m5_1_mp3d_execution_plan(**_m5_1_inputs())

    assert plan["schema"] == M5_1_EXECUTION_SCHEMA
    assert plan["backend_role"] == "comparison_visual"
    assert plan["clock"] == {
        "timeline_v2_applicable": False,
        "timeline_v2_non_applicability_reason": (
            "avengine_authoritative_timeline_v2 freezes a 75-frame/5-second "
            "episode; this retained compatibility route is 270 frames/18 seconds"
        ),
        "compatibility_authority_schema": M5_1_ROUTE_SCHEMA,
        "time_base_hz": 48_000,
        "ticks_per_frame": 3_200,
        "duration_ticks": 864_000,
        "frame_count": 270,
        "fps_num": 15,
        "fps_den": 1,
        "sample_rate_hz": 16_000,
        "sample_count": 288_000,
    }
    assert plan["scene"]["spawned_scene_mesh_actor_count"] == 71
    assert len(plan["frames"]) == 270
    assert plan["frames"][44]["actor_states"][1]["action_sample_index"] == 19
    assert plan["frames"][45]["actor_states"][1]["action_sample_index"] == 0
    assert plan["frames"][269]["actor_states"][1]["action_sample_index"] == 19
    assert plan["frames"][269]["pts_ticks"] == 269 * 3_200
    assert plan["authority"]["backend_may_replan"] is False
    route = plan["route_characterization"]
    assert route["retained_compatibility_route"] is True
    assert route["normal_speed_requirement_resolved"] is False
    assert route["duration_seconds"] == 18.0
    assert route["distance_m_by_actor"] == pytest.approx({"human0": 1.1, "dog0": 1.1})
    assert route["average_speed_m_s_by_actor"] == pytest.approx(
        {"human0": 1.1 / 18.0, "dog0": 1.1 / 18.0}
    )
    assert "retained slow route" in plan["claim_boundary"]
    assert plan["source_logic"]["audio_visibility_policy"] == (
        "360_degree_no_camera_fov_cutoff"
    )
    assert [
        record["source_id"] for record in plan["source_logic"]["animated_emitters"]
    ] == ["source0", "source1"]
    assert plan["room"]["tier"] == "visual_research_only"
    assert plan["qualification"]["dataset_admission"] is False


def test_capture_reader_accepts_installed_prefix_v2_shared_projection() -> None:
    v1_inputs = _m5_1_inputs()
    v1_plan = build_m5_1_mp3d_execution_plan(**v1_inputs)

    v2_inputs = _m5_1_inputs()
    v2_capture = v2_inputs["capture_evidence"]
    v2_capture["schema"] = M5_1_CAPTURE_INSTALLED_SCHEMA_V2
    v2_capture["runtime"] = {
        "installed_habitat_runtime": {
            "kind": "installed_prefix",
            "prefix": "/opt/avengine/habitat",
            "mp3d_root": "/data/avengine/mp3d",
            "magnum_python_site": "/opt/magnum/site-packages",
            "physics_config_path": (
                "/opt/avengine/habitat/config/default.physics_config.json"
            ),
        }
    }

    assert build_m5_1_mp3d_execution_plan(**v2_inputs) == v1_plan


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nav_gate", "14/14"),
        ("emitter", "trajectory 'source1'.*incomplete"),
        ("room", "claim boundary"),
        ("dog_wrap", "dog0 action sample differs"),
    ],
)
def test_capture_compatibility_plan_fails_closed_on_authority_drift(
    mutation: str, message: str
) -> None:
    inputs = _m5_1_inputs()
    if mutation == "nav_gate":
        inputs["navmesh_gate"]["gates"][5]["status"] = "fail"
    elif mutation == "emitter":
        inputs["emitter_trajectories"]["sources"]["source1"]["positions_m"].pop()
    elif mutation == "room":
        inputs["room_registry"]["records"][0]["admission_state"] = "admitted"
    elif mutation == "dog_wrap":
        inputs["frame_readback"][45]["beagle"]["action_sample_index"] = 20
    with pytest.raises(MP3DExecutionError, match=message):
        build_m5_1_mp3d_execution_plan(**inputs)


def test_root_gate_uses_each_frame_yaw_and_rejects_readback_reordering() -> None:
    expected_frames = [
        {
            "frame_index": frame_index,
            "actor_states": [
                {
                    "actor_id": "human0",
                    "translation_ue_cm": [float(frame_index), 0.0, 0.0],
                    "actor_yaw_ue_deg": 10.0 + frame_index * 15.0,
                },
                {
                    "actor_id": "dog0",
                    "translation_ue_cm": [0.0, float(frame_index), 0.0],
                    "actor_yaw_ue_deg": -20.0 - frame_index * 15.0,
                },
            ],
        }
        for frame_index in range(2)
    ]
    actor_readbacks = {
        actor_id: [
            {
                "frame_index": frame["frame_index"],
                "location_cm": next(
                    state["translation_ue_cm"]
                    for state in frame["actor_states"]
                    if state["actor_id"] == actor_id
                ),
                "rotation_deg": [
                    0.0,
                    0.0,
                    next(
                        state["actor_yaw_ue_deg"]
                        for state in frame["actor_states"]
                        if state["actor_id"] == actor_id
                    ),
                ],
            }
            for frame in expected_frames
        ]
        for actor_id in ("human0", "dog0")
    }
    camera_readbacks = [
        {
            "frame_index": frame_index,
            "location_cm": [1.0, 2.0, 3.0],
            "rotation_deg": [0.0, 0.0, 90.0],
        }
        for frame_index in range(2)
    ]
    plan = {"camera": {"ue_position_cm": [1.0, 2.0, 3.0], "ue_yaw_deg": 90.0}}

    result = _root_gate(expected_frames, actor_readbacks, camera_readbacks, plan)
    assert result["human0"]["maximum_absolute_yaw_error_deg"] == 0.0

    reordered = deepcopy(actor_readbacks)
    reordered["dog0"][1]["frame_index"] = 0
    with pytest.raises(RuntimeError, match="frame order"):
        _root_gate(expected_frames, reordered, camera_readbacks, plan)

    pitched_camera = deepcopy(camera_readbacks)
    pitched_camera[1]["rotation_deg"][1] = 0.1
    with pytest.raises(RuntimeError, match="camera readback gate"):
        _root_gate(expected_frames, actor_readbacks, pitched_camera, plan)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_runner_dry_run_writes_validated_compatibility_plan(tmp_path: Path) -> None:
    inputs = _m5_1_inputs()
    paths = {}
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    output_dir = tmp_path / "dry_run"
    args = parse_args(
        [
            "--spear-root",
            str(tmp_path / "read_only_spear"),
            "--route-manifest",
            str(paths["route_manifest"]),
            "--capture-evidence",
            str(paths["capture_evidence"]),
            "--frame-readback",
            str(paths["frame_readback"]),
            "--navmesh-gate",
            str(paths["navmesh_gate"]),
            "--source-program",
            str(paths["source_program"]),
            "--emitter-trajectories",
            str(paths["emitter_trajectories"]),
            "--room-registry",
            str(paths["room_registry"]),
            "--room-qualification",
            str(paths["raw_room_qualification"]),
            "--ue-import-manifest",
            str(paths["ue_import_manifest"]),
            "--ue-material-color-result",
            str(paths["ue_material_color_result"]),
            "--human-ue-manifest",
            str(paths["human_ue_manifest"]),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    plan_path = run(args)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_path == output_dir / "execution_plan.json"
    assert plan["clock"]["timeline_v2_applicable"] is False
    assert len(plan["frames"]) == M5_1_FRAME_COUNT


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are unavailable",
)
def test_media_probe_and_audio_packet_hash_survive_stream_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
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
            "color=c=gray:s=64x48:r=15:d=18",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=18",
            "-frames:v",
            "270",
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
            "-i",
            str(source),
            "-c",
            "copy",
            str(copied),
        ],
        check=True,
    )

    probe = _probe_media(copied, width=64, height=48, expect_audio=True)
    assert probe["frame_count"] == 270
    assert probe["audio_packet_sha256"] == _audio_packet_sha256(source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("role", "backend role"),
        ("room", "room_id"),
        ("frames", "75-frame"),
        ("replan", "replanning"),
        ("mesh_count", "reload/PBR"),
    ],
)
def test_mp3d_execution_fails_closed_on_authority_or_import_drift(
    mutation: str, message: str
) -> None:
    visual = _visual_plan()
    imported = _import_manifest()
    if mutation == "role":
        visual["backend_role"] = "production_visual"
    elif mutation == "room":
        visual["room"]["room_id"] = "another_room"
    elif mutation == "frames":
        visual["frames"].pop()
    elif mutation == "replan":
        visual["authority"]["backend_may_replan"] = True
    elif mutation == "mesh_count":
        imported["scene_content"]["static_meshes"].pop()
    with pytest.raises(MP3DExecutionError, match=message):
        build_mp3d_execution_plan(
            visual_plan=deepcopy(visual),
            ue_import_manifest=deepcopy(imported),
            ue_material_color_result=_material_color_result(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("base_srgb", "base-color Texture2D is not sRGB"),
        ("occlusion_srgb", "occlusion Texture2D is not linear"),
        ("binding", "material color-slot closure differs"),
        ("fresh_reload", "fresh 23/23 sRGB readback"),
    ],
)
def test_mp3d_material_color_contract_fails_closed(mutation: str, message: str) -> None:
    color = _material_color_result()
    if mutation == "base_srgb":
        color["base_color_textures"][0]["srgb"] = False
    elif mutation == "occlusion_srgb":
        color["occlusion_textures"][0]["srgb"] = True
    elif mutation == "binding":
        color["material_bindings"][0]["occlusion_texture_path"] = color[
            "occlusion_textures"
        ][1]["texture_path"]
    elif mutation == "fresh_reload":
        color["fresh_editor_reload"] = False
    with pytest.raises(MP3DExecutionError, match=message):
        build_mp3d_execution_plan(
            visual_plan=_visual_plan(),
            ue_import_manifest=_import_manifest(),
            ue_material_color_result=color,
        )
