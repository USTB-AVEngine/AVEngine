from __future__ import annotations

import importlib.util
import json
import struct
import wave
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPOSITORY / "tools/qa/build_strict_two_human_full_episode_batch.py"
AUDIT_PATH = REPOSITORY / "tools/qa/audit_strict_two_human_room_expansion.py"
DEBUG_BUILDER_PATH = (
    REPOSITORY / "tools/qa/build_strict_two_human_debug_room_preflight.py"
)
FINALIZER_PATH = REPOSITORY / "tools/qa/finalize_strict_two_human_full75_canary.py"
PUBLISHER_PATH = (
    REPOSITORY / "tools/qa/publish_strict_two_human_full75_canary_summary.py"
)
REQUEST_PATH = (
    REPOSITORY / "examples/qa/native_strict_two_human_full_episode_batch_v1.json"
)
DEBUG_PLAN_PATH = (
    REPOSITORY / "examples/qa/native_strict_two_human_debug_room_canary_plan_v1.json"
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module("strict2h_full75_builder", BUILDER_PATH)
AUDIT = _module("strict2h_room_audit", AUDIT_PATH)
DEBUG_BUILDER = _module("strict2h_debug_room_builder", DEBUG_BUILDER_PATH)
FINALIZER = _module("strict2h_full75_finalizer", FINALIZER_PATH)
PUBLISHER = _module("strict2h_full75_publisher", PUBLISHER_PATH)


def test_full75_finalizer_accepts_ieee_float32_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "binaural_float32.wav"
    channel_count = 2
    sample_rate_hz = 16000
    sample_count = 80000
    bits_per_sample = 32
    block_align = channel_count * bits_per_sample // 8
    fmt = struct.pack(
        "<HHIIHH",
        3,
        channel_count,
        sample_rate_hz,
        sample_rate_hz * block_align,
        block_align,
        bits_per_sample,
    )
    data = bytes(sample_count * block_align)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    wav_path.write_bytes(
        b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks
    )

    assert FINALIZER._wav_contract(wav_path) == {
        "format_tag": 3,
        "channel_count": 2,
        "sample_rate_hz": 16000,
        "sample_count": 80000,
        "sample_width_bytes": 4,
    }


def test_full75_finalizer_accepts_canonical_pixel_truth_status(
    tmp_path: Path,
) -> None:
    target_frames = [
        {
            "frame_index": index,
            "visible_fraction": 1.0,
            "visible_pixels": 40000,
            "target_centroid_xy_px": [800.0, 400.0],
            "target_bbox_xyxy_px": [710, 180, 870, 665],
        }
        for index in range(75)
    ]
    distractor_frames = [
        {
            "frame_index": index,
            "visible_fraction": 0.66,
            "visible_pixels": 9700,
            "target_centroid_xy_px": [365.0, 395.0],
            "target_bbox_xyxy_px": [320, 255, 415, 545],
        }
        for index in range(75)
    ]
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "frame_indices": list(range(75)),
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source1": {"frames": target_frames},
            "source2": {"frames": distractor_frames},
        },
    }
    (tmp_path / "pixel_visibility_truth.json").write_text(
        json.dumps(truth), encoding="utf-8"
    )

    result = FINALIZER._validate_pixels(
        {"speech_frame_window_inclusive": [7, 31], "target_side": "right"},
        tmp_path,
    )

    assert result["status"] == "pass"
    assert result["minimum_target_visible_fraction_during_speech"] == 1.0
    assert result["minimum_distractor_visible_fraction"] == 0.66


def test_full75_canary_publisher_requires_four_passed_balanced_rows(
    tmp_path: Path,
) -> None:
    plan_rows = []
    finalization_root = tmp_path / "final"
    for index, (target, distractor, side) in enumerate(
        (
            ("M", "F", "right"),
            ("M", "F", "left"),
            ("F", "M", "right"),
            ("F", "M", "left"),
        ),
        start=1,
    ):
        episode_id = f"episode_{index}"
        capture_root = tmp_path / f"capture_{index}"
        rgb_root = capture_root / "rgb_frames"
        rgb_root.mkdir(parents=True)
        for frame_index in PUBLISHER.REVIEW_FRAME_INDICES:
            Image.new("RGB", (1280, 720), (index * 40, frame_index, 80)).save(
                rgb_root / f"frame_{frame_index:06d}.png"
            )
        video = capture_root / "native_rgb_binaural.mp4"
        video.write_bytes(b"test")
        plan_rows.append(
            {
                "canary_index": index,
                "episode_id": episode_id,
                "target_identity_key": target,
                "distractor_identity_key": distractor,
                "target_side": side,
                "output_root": str(capture_root),
            }
        )
        finalization = {
            "schema": "avengine_native_strict_two_human_full75_canary_finalization_v1",
            "status": "pass",
            "full75_canary_pass": True,
            "canary_index": index,
            "episode_id": episode_id,
            "captured_frame_count": 75,
            "duration_seconds": 5,
            "native_arrays": {
                "normal_rgb_frame_count": 75,
                "metric_depth_frame_count": 75,
                "source1_target_only_frame_count": 75,
                "source2_target_only_frame_count": 75,
            },
            "pixels": {
                "minimum_target_visible_fraction_during_speech": 1.0,
                "minimum_distractor_visible_fraction": 0.75,
                "minimum_target_visible_pixels_during_speech": 10000,
                "minimum_distractor_visible_pixels": 8000,
            },
            "gpu": {"physical_index": 1},
            "artifacts": {
                "binaural_video": str(video),
                "binaural_wav": str(capture_root / "audio.wav"),
                "pixel_visibility_truth": str(capture_root / "pixels.json"),
                "runtime_asset_readbacks": str(capture_root / "runtime.json"),
            },
        }
        final_path = finalization_root / f"canary_{index:02d}" / "finalization.json"
        final_path.parent.mkdir(parents=True)
        final_path.write_text(json.dumps(finalization), encoding="utf-8")
    plan = tmp_path / "canary_plan.json"
    plan.write_text(json.dumps({"canaries": plan_rows}), encoding="utf-8")

    summary_path = PUBLISHER.publish(plan, finalization_root, tmp_path / "review")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["machine_pass_count"] == 4
    assert summary["normal_rgb_frame_count"] == 300
    assert summary["target_only_frame_count"] == 600
    assert summary["native_render_frame_count"] == 900
    assert summary["static_full75_pipeline_gate_pass"] is True
    assert summary["dynamic_mechanism_full75_gate_pass_count"] == 0
    assert summary["dynamic_mechanism_full75_gate_required_count"] == 4
    assert summary["single_room_mechanism_pilot_authorized"] is False
    assert summary["final_multi_room_100_authorized"] is False
    assert summary["formal_episode_count"] == 0
    assert Path(summary["artifacts"]["contact_sheet"]).is_file()
    with Image.open(summary["artifacts"]["contact_sheet"]) as sheet:
        assert sheet.size == (960, 816)


def test_request_is_interim_single_room_and_gpu1_only() -> None:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    assert request["output_contract"]["episode_count"] == 100
    assert request["output_contract"]["first_phase_ready_pilot_target"] == 20
    assert request["output_contract"]["final_multi_room_minimum_ready_room_count"] == 3
    assert request["native_room_scope"]["ready_room_count"] == 1
    assert request["native_room_scope"]["interim_single_room_candidate_bank"] is True
    assert request["formal_episode_count"] == 0
    assert request["qualification_claim"] is False
    assert request["gpu_policy"] == {
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "forbidden_physical_gpu_indices": [0, 3],
        "cpu_builder_must_not_launch_gpu": True,
        "first_20_blocked_until_canaries_pass": True,
        "rows_21_to_100_blocked_until_three_real_rooms_are_ready": True,
    }


def test_projection_and_answer_order_balance() -> None:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    target = [[-0.8, 0.4, -3.0]] * 75
    distractor = [[0.8, 0.4, -3.0]] * 75
    metrics = BUILDER._geometry_metrics(
        request=request,
        camera=[0.0, 1.471, 0.0],
        yaw_deg=0.0,
        target_path=target,
        distractor_path=distractor,
        target_side="left",
        camera_pan=False,
    )
    assert metrics is not None
    assert metrics["minimum_projected_x_separation_fraction"] >= 0.15
    questions = [
        BUILDER._question(index, "left" if index % 2 == 0 else "right", f"e{index}")
        for index in range(100)
    ]
    assert Counter(item["correct_index"] for item in questions) == {0: 50, 1: 50}
    assert Counter(item["option_order_id"] for item in questions) == {
        "left_right": 50,
        "right_left": 50,
    }


def test_debug_maps_are_not_implicitly_promoted_to_rooms() -> None:
    assert (
        AUDIT._package_fragment(
            "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000.debug_0000"
        )
        == "spearsim/content/spear/scenes/debug_0000/maps/debug_0000"
    )
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    excluded = {
        item["room_id"]
        for item in request["native_room_scope"]["not_counted_as_ready_rooms"]
    }
    assert excluded == {"replicacad_apt_0", "habitat_mp3d_example_17DRP5sb8fy"}


def test_debug_room_target_allocation_requires_both_room_gates() -> None:
    plan = json.loads(DEBUG_PLAN_PATH.read_text(encoding="utf-8"))
    assert [item["scene_id"] for item in plan["candidates"]] == [
        "debug_0000",
        "debug_0001",
    ]
    assert plan["final_multi_room_target_allocation_if_both_pass"] == {
        "legacy_ue_apartment_0000_v1": 40,
        "spear_debug_0000_research_room_v1": 30,
        "spear_debug_0001_research_room_v1": 30,
    }
    assert plan["visual_probe_contract"]["audio_is_acoustic_evidence"] is False
    assert (
        plan["visual_probe_contract"]["provisional_placement_is_floor_evidence"]
        is False
    )


@pytest.mark.skipif(
    not Path(
        "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
        "lead_b_siamese_post_approval_v1/packaged_runtime_v1/"
        "Standalone-Development/Linux/Manifest_UFSFiles_Linux.txt"
    ).is_file()
    or not (
        REPOSITORY
        / "tmp/lead_d_strict_two_human_canary_v1/final_gate_v1/suite_execution_plan.json"
    ).is_file(),
    reason="current cooked package or strict base suite is not mounted",
)
def test_real_debug_room_preflight_is_visual_only_and_fail_closed(
    tmp_path: Path,
) -> None:
    result_path = DEBUG_BUILDER.build(DEBUG_PLAN_PATH, tmp_path / "debug_rooms")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["candidate_count"] == 2
    assert result["cooked_map_count"] == 2
    assert result["native_visual_probe_pass_count"] == 0
    assert result["exact_acoustic_closure_count"] == 0
    assert result["additional_ready_room_count"] == 0
    assert result["final_multi_room_100_authorized"] is False
    expected_maps = {
        "spear_debug_0000": "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000",
        "spear_debug_0001": "/Game/SPEAR/Scenes/debug_0001/Maps/debug_0001",
    }
    for row in result["rows"]:
        suite = json.loads(Path(row["suite"]).read_text(encoding="utf-8"))
        assert suite["native_map"] == expected_maps[row["candidate_id"]]
        assert suite["debug_room_probe_boundary"]["audio_is_acoustic_evidence"] is False
        acoustic = json.loads(Path(row["acoustic_plan"]).read_text(encoding="utf-8"))
        assert acoustic["executable"] is False
        assert acoustic["counts_as_exact_rir_evidence"] is False
    silence = tmp_path / "debug_rooms/transport_silence_5s_16k_stereo.wav"
    with wave.open(str(silence), "rb") as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getnframes()) == (
            2,
            16000,
            80000,
        )


@pytest.mark.skipif(
    not Path(
        "/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/"
        "apartment_asset_bound_ue_unique1000_full_20260723_01/"
        "suite_execution_plan.json"
    ).is_file(),
    reason="retained native Apartment 1000-Episode authority is not mounted",
)
def test_real_cpu_builder_emits_100_independent_rows(tmp_path: Path) -> None:
    paths = BUILDER.build(REQUEST_PATH, tmp_path / "plan")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    assert len(episodes) == 100
    assert len({item["camera_cluster_id"] for item in episodes}) == 100
    assert len({item["native_source_scenario_id"] for item in episodes}) == 100
    assert len({item["dedup_key_text"] for item in episodes}) == 100
    assert Counter(item["target"]["side"] for item in episodes) == {
        "left": 50,
        "right": 50,
    }
    assert Counter(item["mechanism"] for item in episodes) == {
        "both_static": 20,
        "target_moves": 20,
        "distractor_moves": 20,
        "both_move": 20,
        "camera_pan_both_static": 20,
    }
    assert all(item["formal"] is False for item in episodes)
    assert summary["single_room_mechanism_pilot_count"] == 20
    assert summary["final_multi_room_episode_count"] == 0
    assert summary["batch_launch_authorized"] is False
