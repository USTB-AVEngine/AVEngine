from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import avengine.cli as cli
from avengine.m5.current_visual_review import (
    CurrentVisualReviewError,
    generate_current_visual_review,
)


def _write_capture_output(
    root: Path,
    *,
    status: str = "research_only",
    include_semantic: bool = True,
    action_id: str = "walk",
    pose_readback: bool = False,
) -> Path:
    root.mkdir()
    arrays = root / "arrays"
    arrays.mkdir()
    frame_count = 75 if pose_readback else 3
    rgb = np.arange(frame_count * 2 * 4 * 3, dtype=np.uint8).reshape(
        frame_count, 2, 4, 3
    )
    depth = (
        np.arange(1, frame_count * 8 + 1, dtype=np.float32).reshape(frame_count, 2, 4)
        / 2.0
    )
    semantic_pattern = np.asarray(
        [
            [[0, 210, 210, 0], [211, 211, 0, 0]],
            [[0, 210, 0, 0], [211, 0, 211, 0]],
            [[210, 210, 0, 0], [0, 211, 211, 0]],
        ],
        dtype=np.int32,
    )
    semantic = np.stack(
        [semantic_pattern[index % 3] for index in range(frame_count)], axis=0
    )
    actor = np.tile(np.eye(4, dtype=np.float64), (frame_count, 2, 1, 1))
    actor[:, 0, 0, 3] = np.arange(frame_count, dtype=np.float64) + 1.0
    actor[:, 1, 2, 3] = np.arange(frame_count, dtype=np.float64) + 4.0
    sources = np.empty((frame_count, 2, 3), dtype=np.float64)
    for index in range(frame_count):
        sources[index] = (
            (1.0 + index / 10.0, 2.0 + index / 10.0, 3.0 + index / 10.0),
            (4.0 + index / 10.0, 5.0 + index / 10.0, 6.0 + index / 10.0),
        )
    visibility_pattern = np.asarray([[3, 2], [1, 2], [2, 2]], dtype=np.int64)
    visibility = np.stack(
        [visibility_pattern[index % 3] for index in range(frame_count)], axis=0
    )
    np.save(arrays / "rgb.npy", rgb)
    np.save(arrays / "depth.npy", depth)
    if include_semantic:
        np.save(arrays / "semantic.npy", semantic)
    np.save(arrays / "actor_world_matrices.npy", actor)
    np.save(arrays / "source_positions_m.npy", sources)
    np.save(arrays / "semantic_visibility_pixels.npy", visibility)
    if pose_readback:
        planned_skin_roots = actor.copy()
        actual_skin_roots = actor.copy()
        actual_skin_roots[:, :, 0, 3] += 1.0e-7
        actual_agent_poses = np.tile(
            np.asarray([9.0, 8.0, 7.0, 0.0, 2**-0.5, 0.0, 2**-0.5]),
            (frame_count, 1),
        )
        actual_sensor_poses = np.tile(actual_agent_poses[:, None, :], (1, 3, 1))
        actual_sensor_poses[:, 0, 0] += 0.1
        actual_sensor_poses[:, 1, 0] += 0.2
        actual_sensor_poses[:, 2, 0] += 0.3
        actual_listener_poses = actual_agent_poses.copy()
        np.save(
            arrays / "planned_applied_skin_root_matrices.npy",
            planned_skin_roots,
        )
        np.save(
            arrays / "actual_skin_root_readback_matrices.npy",
            actual_skin_roots,
        )
        np.save(
            arrays / "actual_agent_camera_poses_txyz_qxyzw.npy",
            actual_agent_poses,
        )
        np.save(
            arrays / "actual_sensor_poses_txyz_qxyzw.npy",
            actual_sensor_poses,
        )
        np.save(
            arrays / "actual_logical_listener_poses_txyz_qxyzw.npy",
            actual_listener_poses,
        )
    frames = [
        {
            "frame_index": index,
            "pts_ticks": index * 3200,
            "action_id": action_id,
            "action_sample_index": index,
            **(
                {
                    "actor_world_positions_m": actor[index, :, :3, 3].tolist(),
                    "source_positions_m": sources[index].tolist(),
                    "semantic_visibility_pixels": visibility[index].tolist(),
                    "snapshot_world_time_seconds": 0.0,
                    "pose_alignment": {
                        "array_frame_index": index,
                        "actor_axis_order": ["actor0", "actor1"],
                        "source_axis_order": ["source0", "source1"],
                        "sensor_axis_order": [
                            "rig_depth",
                            "rig_rgb",
                            "rig_semantic",
                        ],
                        "pose_component_order": [
                            "translation_x_m",
                            "translation_y_m",
                            "translation_z_m",
                            "quaternion_x",
                            "quaternion_y",
                            "quaternion_z",
                            "quaternion_w",
                        ],
                    },
                }
                if pose_readback
                else {}
            ),
        }
        for index in range(frame_count)
    ]
    (root / "frame_records.json").write_text(
        json.dumps({"frames": frames}, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "rgb": "arrays/rgb.npy",
        "depth": "arrays/depth.npy",
        "semantic": "arrays/semantic.npy",
        "actor_world_matrices": "arrays/actor_world_matrices.npy",
        "source_positions_m": "arrays/source_positions_m.npy",
        "semantic_visibility_pixels": "arrays/semantic_visibility_pixels.npy",
        "frame_records": "frame_records.json",
    }
    if pose_readback:
        artifacts.update(
            {
                "planned_applied_skin_root_matrices": (
                    "arrays/planned_applied_skin_root_matrices.npy"
                ),
                "actual_skin_root_readback_matrices": (
                    "arrays/actual_skin_root_readback_matrices.npy"
                ),
                "actual_agent_camera_poses": (
                    "arrays/actual_agent_camera_poses_txyz_qxyzw.npy"
                ),
                "actual_sensor_poses": ("arrays/actual_sensor_poses_txyz_qxyzw.npy"),
                "actual_logical_listener_poses": (
                    "arrays/actual_logical_listener_poses_txyz_qxyzw.npy"
                ),
            }
        )
    if not include_semantic:
        artifacts.pop("semantic")
    receipt = {
        "status": status,
        "research_only": status == "research_only",
        "episode_counted": False,
        "capture": {"native_habitat_started": True, "frame_count": frame_count},
        "artifacts": artifacts,
    }
    if pose_readback:
        receipt.update(
            {
                "actors": {
                    "actor_ids": ["actor0", "actor1"],
                    "source_ids": ["source0", "source1"],
                    "semantic_ids": [210, 211],
                    "emitter_link_names": [
                        "beagle Xtra Mouth",
                        "beagle Xtra Mouth",
                    ],
                    "axis_bindings": [
                        {
                            "axis_index": 0,
                            "actor_id": "actor0",
                            "source_id": "source0",
                            "semantic_id": 210,
                            "emitter_link_name": "beagle Xtra Mouth",
                            "runtime_emitter_link_id": 100,
                            "audio_endpoint_status": "pending",
                            "audio_endpoint_id": None,
                        },
                        {
                            "axis_index": 1,
                            "actor_id": "actor1",
                            "source_id": "source1",
                            "semantic_id": 211,
                            "emitter_link_name": "beagle Xtra Mouth",
                            "runtime_emitter_link_id": 101,
                            "audio_endpoint_status": "pending",
                            "audio_endpoint_id": None,
                        },
                    ],
                },
                "pose_readback": {
                    "status": "native_habitat_pose_readback_recorded",
                    "frame_count": frame_count,
                    "pose_component_order": [
                        "translation_x_m",
                        "translation_y_m",
                        "translation_z_m",
                        "quaternion_x",
                        "quaternion_y",
                        "quaternion_z",
                        "quaternion_w",
                    ],
                    "actor_axis_order": ["actor0", "actor1"],
                    "source_axis_order": ["source0", "source1"],
                    "sensor_order": [
                        {
                            "axis_index": 0,
                            "sensor_uuid": "rig_depth",
                            "modality": "depth",
                        },
                        {
                            "axis_index": 1,
                            "sensor_uuid": "rig_rgb",
                            "modality": "rgb",
                        },
                        {
                            "axis_index": 2,
                            "sensor_uuid": "rig_semantic",
                            "modality": "semantic",
                        },
                    ],
                    "planned_actor_world_matrices": {
                        "artifact_key": "actor_world_matrices",
                        "pose_kind": "planned_actor_transform_with_instance_offset",
                        "actual_readback": False,
                    },
                    "planned_applied_skin_root_matrices": {
                        "artifact_key": "planned_applied_skin_root_matrices",
                        "pose_kind": "planned_transform_applied_to_skin_root",
                        "actual_readback": False,
                    },
                    "actual_skin_root_readback_matrices": {
                        "artifact_key": "actual_skin_root_readback_matrices",
                        "pose_kind": "articulated_root_scene_node_after_render",
                        "actual_readback": True,
                    },
                    "actual_agent_camera_poses": {
                        "artifact_key": "actual_agent_camera_poses",
                        "pose_kind": "agent_camera_from_state_snapshot",
                        "actual_readback": True,
                    },
                    "actual_sensor_poses": {
                        "artifact_key": "actual_sensor_poses",
                        "pose_kind": "sensor_states_from_state_snapshot",
                        "actual_readback": True,
                    },
                    "actual_logical_listener_poses": {
                        "artifact_key": "actual_logical_listener_poses",
                        "pose_kind": (
                            "logical_listener_composed_from_actual_agent_readback"
                        ),
                        "actual_readback": True,
                    },
                },
            }
        )
    (root / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_generates_offline_synchronized_research_only_review(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture")
    receipt_path = output / "research_receipt.json"
    receipt_before = receipt_path.read_bytes()

    review = generate_current_visual_review(output)

    assert review.frame_count == 3
    assert review.html_path == output / "review" / "index.html"
    assert review.html_path.is_file()
    assert receipt_path.read_bytes() == receipt_before
    html = review.html_path.read_text(encoding="utf-8")
    assert 'type="range"' in html
    assert "RGB" in html
    assert "Depth" in html
    assert "Semantic" in html
    assert "localStorage" in html
    assert "Research-only visual review" in html
    assert "planned_actor_world_positions_m" in html
    assert "runtime_source_readback_positions_m" in html
    assert "semantic_visibility_pixels" in html
    assert "approve" not in html.lower()
    assert "formal/admission evidence" in html
    for modality in ("rgb", "depth", "semantic"):
        paths = sorted((review.review_directory / "frames" / modality).glob("*.png"))
        assert len(paths) == 3
        assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_validates_and_displays_complete_native_pose_readback(tmp_path: Path) -> None:
    output = _write_capture_output(
        tmp_path / "capture-with-readback", pose_readback=True
    )

    review = generate_current_visual_review(output)

    assert review.frame_count == 75
    html = review.html_path.read_text(encoding="utf-8")
    assert "actual_agent_camera_pose_txyz_qxyzw" in html
    assert "actual_sensor_poses_txyz_qxyzw" in html
    assert "actual_logical_listener_pose_txyz_qxyzw" in html
    assert "actual_skin_root_readback_positions_m" in html
    assert "actor_source_axis_bindings" in html


def test_rejects_swapped_source_axis_in_pose_frame_alignment(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture", pose_readback=True)
    records_path = output / "frame_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["frames"][7]["pose_alignment"]["source_axis_order"] = [
        "source1",
        "source0",
    ]
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CurrentVisualReviewError, match="identity axes are damaged"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_rejects_damaged_actor_source_identity_binding(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture", pose_readback=True)
    receipt_path = output / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["actors"]["axis_bindings"][0]["source_id"] = "source1"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(CurrentVisualReviewError, match="identity mapping is damaged"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_rejects_partial_pose_artifacts_without_upgrading_legacy(
    tmp_path: Path,
) -> None:
    output = _write_capture_output(tmp_path / "legacy-capture")
    receipt_path = output / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["actual_agent_camera_poses"] = (
        "arrays/actor_world_matrices.npy"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(CurrentVisualReviewError, match="evidence is partial"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


@pytest.mark.parametrize("damage", ("shape", "nonfinite"))
def test_rejects_invalid_actual_pose_array(tmp_path: Path, damage: str) -> None:
    output = _write_capture_output(tmp_path / damage, pose_readback=True)
    path = output / "arrays/actual_agent_camera_poses_txyz_qxyzw.npy"
    value = np.load(path)
    if damage == "shape":
        value = value[:-1]
    else:
        value[12, 0] = np.nan
    np.save(path, value)

    with pytest.raises(CurrentVisualReviewError, match="actual agent/camera poses"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_rejects_pose_artifact_path_escape(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture", pose_readback=True)
    receipt_path = output / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["actual_sensor_poses"] = "../outside.npy"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(CurrentVisualReviewError, match="normalized relative path"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_rejects_pose_pts_misaligned_with_array_frame(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture", pose_readback=True)
    records_path = output / "frame_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["frames"][12]["pts_ticks"] += 1
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CurrentVisualReviewError, match="15 fps timeline"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_escapes_case_insensitive_script_end_marker_in_frame_metadata(
    tmp_path: Path,
) -> None:
    action_id = "</SCRIPT><script>injected()</script>"
    output = _write_capture_output(tmp_path / "capture", action_id=action_id)

    review = generate_current_visual_review(output)

    html = review.html_path.read_text(encoding="utf-8")
    assert action_id not in html
    assert "\\u003c/SCRIPT\\u003e\\u003cscript\\u003e" in html
    assert html.lower().count("</script") == 1


def test_rejects_frame_record_index_misaligned_with_arrays(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture")
    records_path = output / "frame_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["frames"][1]["frame_index"] = 0
    records_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CurrentVisualReviewError, match="must equal its array index"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_cli_generates_external_offline_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _write_capture_output(tmp_path / "capture")
    external_review = tmp_path / "external-review"

    assert (
        cli.main(
            [
                "m5",
                "review-current-visual",
                "--research-output",
                str(output),
                "--review-output",
                str(external_review),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "research_only"
    assert payload["episode_counted"] is False
    assert payload["notes_storage"] == "browser localStorage only"
    assert Path(payload["review"]) == external_review / "index.html"
    assert (external_review / "index.html").is_file()


@pytest.mark.parametrize("status", ("not_run", "fail"))
def test_rejects_missing_or_noncompleted_research_output(
    tmp_path: Path, status: str
) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(CurrentVisualReviewError, match="research receipt"):
        generate_current_visual_review(missing)

    noncompleted = _write_capture_output(tmp_path / status, status=status)
    with pytest.raises(CurrentVisualReviewError, match="completed research_only"):
        generate_current_visual_review(noncompleted)


def test_rejects_missing_arrays_and_never_replaces_review(tmp_path: Path) -> None:
    missing_array = _write_capture_output(
        tmp_path / "missing-array",
        include_semantic=False,
    )
    with pytest.raises(
        CurrentVisualReviewError, match="missing required review artifacts"
    ):
        generate_current_visual_review(missing_array)

    output = _write_capture_output(tmp_path / "capture")
    generate_current_visual_review(output)
    with pytest.raises(CurrentVisualReviewError, match="refusing to replace"):
        generate_current_visual_review(output)
