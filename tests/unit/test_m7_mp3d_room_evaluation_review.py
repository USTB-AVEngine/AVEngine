from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.spatial_audio.audio import write_float32_wav
from avengine.m7.room_evaluation import (
    build_room_evaluation_plan,
    build_static_source_trajectory_bank,
)
from avengine.m7.sensor_rig import m7_sensor_rig_binding
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory
from tools.m7 import build_mp3d_room_evaluation_review as review_tool


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    plan_root = tmp_path / "plan"
    audio_root = tmp_path / "audio"
    visual_root = tmp_path / "visual"
    plan_root.mkdir()
    audio_root.mkdir()
    visual_root.mkdir()

    sensor_rig = materialize_sensor_rig_trajectory(
        trajectory_id="mp3d_17DRP5sb8fy_review_fixture",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [-8.9, 1.5, -1.3],
                    "yaw_deg": 90.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [-4.2, 1.5, -1.2],
                    "yaw_deg": 90.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    bank = build_static_source_trajectory_bank(
        {
            "source1": [-7.5, 1.2, -3.0],
            "source2": [-10.0, 1.2, -4.0],
        },
        frame_count=75,
        frame_rate_hz=15,
        episode_id="mp3d_17DRP5sb8fy_static_sources_000",
        seed=17,
    )
    plan = build_room_evaluation_plan(
        bank,
        listener_position_m=sensor_rig["frames"][0]["world_from_rig"][
            "translation_m"
        ],
        listener_orientation_wxyz=[
            sensor_rig["frames"][0]["world_from_rig"]["rotation_xyzw"][3],
            *sensor_rig["frames"][0]["world_from_rig"]["rotation_xyzw"][:3],
        ],
        stride_frames=1,
        episode_count=1,
        sound_classes=("dog barking", "human speech"),
        sensor_rig_trajectory=sensor_rig,
    )
    plan_documents = {
        "delivery.json": plan.summary,
        "trajectory_bank.json": plan.trajectory_bank,
        "sound_assignments.json": plan.sound_assignments,
        "rir_job_plan.json": plan.rir_job_plan,
        "sensor_rig_trajectory.json": sensor_rig,
    }
    for name, value in plan_documents.items():
        write_json(plan_root / name, value)

    binding = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "registry",
        "registry_selection_applied": True,
        "room_ref": dict(review_tool.EXPECTED_ROOM_REF),
        "profile_ref": {
            "profile_id": review_tool.EXPECTED_PROFILE_ID,
            "revision": "soundspaces_fixture",
        },
        "binding_id": review_tool.EXPECTED_BINDING_ID,
        "registry_selection_content_sha256": "11" * 32,
        "effective_selection_content_sha256": "22" * 32,
        "acoustic_package_manifest_sha256": "33" * 32,
        "simulation_request_sha256": "44" * 32,
        "input_receipt_sha256": "55" * 32,
    }
    binding["binding_content_sha256"] = canonical_json_sha256(binding)
    binding_sha256 = binding["binding_content_sha256"]
    sensor_binding = m7_sensor_rig_binding(sensor_rig)
    input_closure = {
        "schema": "avengine_room_evaluation_binaural_input_closure_v1",
        "status": "pass",
        "files": {
            name: {"path": name, "sha256": sha256_file(plan_root / name)}
            for name in plan_documents
        },
        "sensor_rig_trajectory": sensor_binding,
        "acoustic_selection_binding": binding,
        "acoustic_selection_binding_sha256": binding_sha256,
    }
    sample_id = "mp3d_17DRP5sb8fy_static_sources_000__v00"
    mixture_path = audio_root / "audio" / "binaural" / f"{sample_id}.wav"
    artifact = write_float32_wav(
        mixture_path,
        np.full((2, 80_000), 0.025, dtype=np.float32),
        16_000,
        metadata={
            "role": "room_evaluation_binaural_mixture",
            "sample_id": sample_id,
            "episode_id": "mp3d_17DRP5sb8fy_static_sources_000",
            "source_classes": {
                "source1": "dog barking",
                "source2": "human speech",
            },
            "mixture": "exact_persisted_source1_plus_source2_stem_sum",
            "normalization": False,
            "limiting": False,
            "acoustic_selection_binding_sha256": binding_sha256,
        },
    )
    sample = {
        "sample_id": sample_id,
        "episode_id": "mp3d_17DRP5sb8fy_static_sources_000",
        "ordinal": 0,
        "split": "test",
        "both_sources_active": True,
        "source_classes": {
            "source1": "dog barking",
            "source2": "human speech",
        },
        "audio_path": str(artifact.audio_path.relative_to(audio_root)),
        "audio_sidecar_path": str(
            artifact.sidecar_path.relative_to(audio_root)
        ),
        "audio_sha256": artifact.audio_sha256,
        "audio_sidecar_sha256": artifact.sidecar_sha256,
        "audio_sample_rate_hz": 16_000,
        "audio_sample_count": 80_000,
        "audio_channel_count": 2,
        "mixture_is_exact_persisted_source_stem_sum": True,
        "acoustic_selection_binding_sha256": binding_sha256,
    }
    write_json(
        audio_root / "samples.json",
        {
            "schema": "avengine_room_evaluation_binaural_samples_v1",
            "status": "pass",
            "sample_count": 1,
            "acoustic_selection_binding": binding,
            "acoustic_selection_binding_sha256": binding_sha256,
            "input_closure": input_closure,
            "samples": [sample],
        },
    )
    write_json(
        audio_root / "delivery.json",
        {
            "schema": "avengine_room_evaluation_binaural_batch_v1",
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "sample_count": 1,
            "both_sources_active": True,
            "source_slots": ["source1", "source2"],
            "acoustic_selection_binding": binding,
            "acoustic_selection_binding_sha256": binding_sha256,
            "input_closure": input_closure,
            "mixture_is_exact_persisted_source_stem_sum": True,
            "layout": "native_RLR_HRTF_binaural_left_right",
            "output_closure": {
                "status": "pass",
                "files": {
                    "samples.json": {
                        "path": "samples.json",
                        "sha256": sha256_file(audio_root / "samples.json"),
                    }
                },
            },
        },
    )

    write_json(visual_root / "sensor_rig_trajectory.json", sensor_rig)
    rgb = np.zeros((75, 4, 6, 3), dtype=np.uint8)
    rgb[:, :, :, 1] = np.arange(75, dtype=np.uint8)[:, None, None]
    topdown = np.zeros((75, 5, 5, 3), dtype=np.uint8)
    topdown[:, :, :, :] = 96
    np.save(visual_root / "rgb.npy", rgb, allow_pickle=False)
    np.save(visual_root / "topdown_rgb.npy", topdown, allow_pickle=False)
    write_json(
        visual_root / "summary.json",
        {
            "status": "visual_pass_audio_blocked",
            "scene": {
                "room_id": review_tool.EXPECTED_ROOM_REF["room_id"],
                "scene_id": review_tool.EXPECTED_SCENE_ID,
            },
            "trajectory": {
                "trajectory_id": sensor_rig["trajectory_id"],
                "frame_count": 75,
            },
            "visual": {
                "rgb_shape": list(rgb.shape),
                "topdown_shape": list(topdown.shape),
                "topdown_is_qa_only": True,
            },
            "artifacts": {
                "sensor_rig_trajectory": {
                    "path": "sensor_rig_trajectory.json",
                    "sha256": sha256_file(
                        visual_root / "sensor_rig_trajectory.json"
                    ),
                }
            },
        },
    )
    return audio_root, visual_root, plan_root


def test_mp3d_review_closes_registry_audio_plan_visual_and_geometry(
    tmp_path: Path,
) -> None:
    audio_root, visual_root, plan_root = _fixture_roots(tmp_path)

    result = review_tool.load_review_inputs(
        audio_root=audio_root,
        visual_capture_root=visual_root,
        plan_root=plan_root,
    )

    assert result.episode_id == "mp3d_17DRP5sb8fy_static_sources_000"
    assert result.acoustic_selection_binding["selection_mode"] == "registry"
    assert result.sensor_rig_binding["dynamic"] is True
    assert len(result.frame_geometry) == 75
    assert result.alignment["checked_source_frame_count"] == 150
    assert (
        result.frame_geometry[0]["sources"]["source1"]["position_m"]
        == [-7.5, 1.2, -3.0]
    )
    assert (
        result.frame_geometry[-1]["pose_hash"]
        == result.sensor_rig_binding["last_pose_hash"]
    )


def test_mp3d_review_rejects_nonregistry_or_wrong_room_audio(
    tmp_path: Path,
) -> None:
    audio_root, visual_root, plan_root = _fixture_roots(tmp_path)
    delivery = load_json(audio_root / "delivery.json")
    samples = load_json(audio_root / "samples.json")
    for document in (delivery, samples):
        document["acoustic_selection_binding"]["room_ref"]["room_id"] = (
            "different_room"
        )
    write_json(audio_root / "delivery.json", delivery)
    write_json(audio_root / "samples.json", samples)

    with pytest.raises(
        review_tool.MP3DRoomEvaluationReviewError,
        match="not registry-bound",
    ):
        review_tool.load_review_inputs(
            audio_root=audio_root,
            visual_capture_root=visual_root,
            plan_root=plan_root,
        )


def test_mp3d_review_rejects_visual_sensor_rig_content_drift(
    tmp_path: Path,
) -> None:
    audio_root, visual_root, plan_root = _fixture_roots(tmp_path)
    visual_sensor = load_json(visual_root / "sensor_rig_trajectory.json")
    changed = deepcopy(visual_sensor)
    changed["trajectory_id"] = "different_visual_trajectory"
    write_json(visual_root / "sensor_rig_trajectory.json", changed)
    summary = load_json(visual_root / "summary.json")
    summary["artifacts"]["sensor_rig_trajectory"]["sha256"] = sha256_file(
        visual_root / "sensor_rig_trajectory.json"
    )
    summary["trajectory"]["trajectory_id"] = changed["trajectory_id"]
    write_json(visual_root / "summary.json", summary)

    with pytest.raises(
        review_tool.MP3DRoomEvaluationReviewError,
        match="content differs",
    ):
        review_tool.load_review_inputs(
            audio_root=audio_root,
            visual_capture_root=visual_root,
            plan_root=plan_root,
        )


def test_mp3d_review_rejects_source_position_drift_between_bank_and_rir_plan(
    tmp_path: Path,
) -> None:
    audio_root, visual_root, plan_root = _fixture_roots(tmp_path)
    bank = load_json(plan_root / "trajectory_bank.json")
    bank["episodes"][0]["source_center_paths_m"]["source1"][37][0] += 0.1
    write_json(plan_root / "trajectory_bank.json", bank)
    delivery = load_json(audio_root / "delivery.json")
    samples = load_json(audio_root / "samples.json")
    new_hash = sha256_file(plan_root / "trajectory_bank.json")
    delivery["input_closure"]["files"]["trajectory_bank.json"]["sha256"] = (
        new_hash
    )
    samples["input_closure"]["files"]["trajectory_bank.json"]["sha256"] = (
        new_hash
    )
    write_json(audio_root / "samples.json", samples)
    delivery["output_closure"]["files"]["samples.json"]["sha256"] = sha256_file(
        audio_root / "samples.json"
    )
    write_json(audio_root / "delivery.json", delivery)

    with pytest.raises(
        review_tool.MP3DRoomEvaluationReviewError,
        match="position differs at frame 37",
    ):
        review_tool.load_review_inputs(
            audio_root=audio_root,
            visual_capture_root=visual_root,
            plan_root=plan_root,
        )


def test_mp3d_review_builds_evidence_and_uses_existing_media_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_root, visual_root, plan_root = _fixture_roots(tmp_path)
    output = tmp_path / "review"
    observed: dict[str, object] = {}

    def fake_compose(**kwargs):
        observed["tracks"] = kwargs["tracks"]
        observed["diagnostics"] = kwargs["audio_diagnostic_by_frame"]
        return np.zeros((75, 480, 1280, 3), dtype=np.uint8)

    def fake_encode(frames, path, *, fps, audio_path):
        assert np.asarray(frames).shape == (75, 480, 1280, 3)
        path = Path(path)
        path.write_bytes(b"fixture-mp4")
        return {
            "schema": "avengine_m5_1_annotated_review_v1",
            "path": str(path),
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
            "frame_count": 75,
            "frame_rate_hz": fps,
            "duration_seconds": 5.0,
            "width": 1280,
            "height": 480,
            "topdown_is_qa_only": True,
            "audio_muxed": audio_path is not None,
            "ffprobe": {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264"},
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "16000",
                        "channels": 2,
                    },
                ]
            },
        }

    monkeypatch.setattr(review_tool, "compose_annotated_frames", fake_compose)
    monkeypatch.setattr(review_tool, "encode_annotated_review", fake_encode)

    assert (
        review_tool.build_review(
            audio_root=audio_root,
            visual_capture_root=visual_root,
            plan_root=plan_root,
            output=output,
        )
        == output
    )
    evidence = load_json(output / "evidence.json")
    assert evidence["status"] == "pass"
    assert evidence["cross_modal_alignment"]["status"] == "pass"
    assert evidence["media"]["path"] == review_tool.REVIEW_FILENAME
    assert len(evidence["frame_geometry"]) == 75
    assert len(observed["tracks"]) == 2
    assert len(observed["diagnostics"]) == 75
