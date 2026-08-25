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
    object_id_readback: bool = False,
) -> Path:
    if object_id_readback and not pose_readback:
        raise ValueError("object-ID fixture requires pose_readback=True")
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
    object_id = np.zeros((frame_count, 2, 4), dtype=np.int32)
    object_id[:, 0, 0:2] = 500
    object_id[:, 1, 0] = 501
    object_id[-1, object_id[-1] == 501] = 0
    np.save(arrays / "rgb.npy", rgb)
    np.save(arrays / "depth.npy", depth)
    if include_semantic:
        np.save(arrays / "semantic.npy", semantic)
    np.save(arrays / "actor_world_matrices.npy", actor)
    np.save(arrays / "source_positions_m.npy", sources)
    np.save(arrays / "semantic_visibility_pixels.npy", visibility)
    if object_id_readback:
        np.save(arrays / "object_id.npy", object_id)
    if pose_readback:
        planned_skin_roots = actor.copy()
        actual_skin_roots = actor.copy()
        actual_skin_roots[:, :, 0, 3] += 1.0e-7
        actual_agent_poses = np.tile(
            np.asarray([9.0, 8.0, 7.0, 0.0, 2**-0.5, 0.0, 2**-0.5]),
            (frame_count, 1),
        )
        sensor_count = 4 if object_id_readback else 3
        actual_sensor_poses = np.tile(
            actual_agent_poses[:, None, :], (1, sensor_count, 1)
        )
        actual_sensor_poses[:, 0, 0] += 1.0e-7
        actual_sensor_poses[:, 1, 0] += 2.0e-7
        actual_sensor_poses[:, 2, 0] += 3.0e-7
        if object_id_readback:
            actual_sensor_poses[:, 3, 0] += 4.0e-7
        actual_listener_poses = actual_agent_poses.copy()
        actual_listener_poses[:, 1] += 0.2
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
                    **(
                        {
                            "object_id_pixel_counts": [
                                int(np.count_nonzero(object_id[index] == 500)),
                                int(np.count_nonzero(object_id[index] == 501)),
                            ],
                            "object_id_bboxes_xyxy": [
                                {
                                    "x_min": 0,
                                    "y_min": 0,
                                    "x_max": 1,
                                    "y_max": 0,
                                    "width_pixels": 2,
                                    "height_pixels": 1,
                                },
                                (
                                    {
                                        "x_min": 0,
                                        "y_min": 1,
                                        "x_max": 0,
                                        "y_max": 1,
                                        "width_pixels": 1,
                                        "height_pixels": 1,
                                    }
                                    if index < frame_count - 1
                                    else None
                                ),
                            ],
                        }
                        if object_id_readback
                        else {}
                    ),
                    "snapshot_world_time_seconds": 0.0,
                    "pre_render_snapshot_world_time_seconds": 0.0,
                    "post_render_snapshot_world_time_seconds": 0.0,
                    "pose_alignment": {
                        "array_frame_index": index,
                        "actor_axis_order": ["actor0", "actor1"],
                        "source_axis_order": ["source0", "source1"],
                        "sensor_axis_order": [
                            "rig_depth",
                            *(["rig_object_id"] if object_id_readback else []),
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
    if object_id_readback:
        artifacts["object_id"] = "arrays/object_id.npy"
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
        receipt["capture"].update(
            {
                "m1_request_id": "m1_capture_fixture_v1",
                "room_id": "habitat_mp3d_example_17DRP5sb8fy",
                "modalities": [
                    "rgb",
                    "depth",
                    "semantic",
                    *(["object_id"] if object_id_readback else []),
                ],
            }
        )
        receipt.update(
            {
                "selected_room": {
                    "room_id": "habitat_mp3d_example_17DRP5sb8fy",
                    "room_kind": "habitat_native",
                },
                "m1_capture_identity": {
                    "request_id": "m1_capture_fixture_v1",
                    "room_id": "habitat_mp3d_example_17DRP5sb8fy",
                    "primary_camera_rig": {
                        "rig_id": "camera_rig_0",
                        "view_id": "view0",
                        "world_from_rig": {
                            "translation_m": [9.0, 8.0, 7.0],
                            "rotation_xyzw": [
                                0.0,
                                2**-0.5,
                                0.0,
                                2**-0.5,
                            ],
                        },
                        "shared_calibration": {
                            "projection": "pinhole",
                            "resolution_hw": [2, 4],
                            "hfov_degrees": 90.0,
                            "near_m": 0.05,
                            "far_m": 100.0,
                            "rig_from_sensor": {
                                "translation_m": [0.0, 0.0, 0.0],
                                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        "modalities": [
                            {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                            {
                                "modality": "depth",
                                "sensor_uuid": "rig_depth",
                            },
                            {
                                "modality": "semantic",
                                "sensor_uuid": "rig_semantic",
                            },
                        ],
                    },
                    "listener": {
                        "listener_id": "listener0",
                        "attached_to": "camera_rig_0",
                        "rig_from_listener": {
                            "translation_m": [0.0, 0.2, 0.0],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    "source_axis_order": ["source0", "source1"],
                    "qa_views": [
                        {
                            "qa_id": "navmesh_topdown",
                            "kind": "topdown",
                        }
                    ],
                },
                "actors": {
                    "actor_ids": ["actor0", "actor1"],
                    "source_ids": ["source0", "source1"],
                    "semantic_ids": [210, 211],
                    "emitter_link_names": [
                        "beagle Xtra Mouth",
                        "beagle Xtra Mouth",
                    ],
                    **(
                        {"runtime_object_ids": [500, 501]} if object_id_readback else {}
                    ),
                    "axis_bindings": [
                        {
                            "axis_index": 0,
                            "actor_id": "actor0",
                            "source_id": "source0",
                            "semantic_id": 210,
                            **(
                                {"runtime_object_id": 500} if object_id_readback else {}
                            ),
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
                            **(
                                {"runtime_object_id": 501} if object_id_readback else {}
                            ),
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
                    "coordinate_convention": {
                        "frame": "habitat_world",
                        "axis_order": ["X", "Y", "Z"],
                        "up_axis": "+Y",
                    },
                    "matrix_convention": {
                        "transform_direction": "world_from_local",
                        "translation_location": "last_column",
                        "homogeneous_last_row": [0.0, 0.0, 0.0, 1.0],
                    },
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
                            "sensor_uuid": (
                                "rig_object_id" if object_id_readback else "rig_rgb"
                            ),
                            "modality": ("object_id" if object_id_readback else "rgb"),
                        },
                        *(
                            [
                                {
                                    "axis_index": 2,
                                    "sensor_uuid": "rig_rgb",
                                    "modality": "rgb",
                                }
                            ]
                            if object_id_readback
                            else []
                        ),
                        {
                            "axis_index": 3 if object_id_readback else 2,
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
                        "pose_kind": ("post_render_agent_camera_from_state_snapshot"),
                        "actual_readback": True,
                    },
                    "actual_sensor_poses": {
                        "artifact_key": "actual_sensor_poses",
                        "pose_kind": ("post_render_sensor_states_from_state_snapshot"),
                        "actual_readback": True,
                    },
                    "actual_logical_listener_poses": {
                        "artifact_key": "actual_logical_listener_poses",
                        "pose_kind": (
                            "logical_listener_composed_from_post_render_agent_readback"
                        ),
                        "actual_readback": False,
                        "derived_from_actual_agent_readback": True,
                        "native_listener_sensor_readback": False,
                    },
                },
                **(
                    {
                        "object_id_visibility": {
                            "status": "native_object_id_raster_recorded",
                            "sensor_uuid": "rig_object_id",
                            "semantic_sensor_target": "OBJECT_ID",
                            "frame_count": 75,
                            "runtime_id_scope": (
                                "one_simulator_invocation_not_stable_across_runs"
                            ),
                            "runtime_id_mapping_authority": (
                                "articulated_object.object_id_readback"
                            ),
                            "candidate_filtering": "disabled",
                            "candidate_legality_effect": "descriptor_only",
                            "zero_pixels_affect_legality": False,
                            "actor_modal_visibility_authority_status": (
                                "pending_fresh_runtime_a_b_and_visual_review"
                            ),
                            "per_actor": {
                                "actor0": {
                                    "runtime_object_id": 500,
                                    "pixel_counts_by_frame": [2] * 75,
                                    "bboxes_xyxy_by_frame": [
                                        {
                                            "x_min": 0,
                                            "y_min": 0,
                                            "x_max": 1,
                                            "y_max": 0,
                                            "width_pixels": 2,
                                            "height_pixels": 1,
                                        }
                                    ]
                                    * 75,
                                    "positive_pixel_frame_count": 75,
                                    "zero_pixel_frame_indices": [],
                                },
                                "actor1": {
                                    "runtime_object_id": 501,
                                    "pixel_counts_by_frame": [1] * 74 + [0],
                                    "bboxes_xyxy_by_frame": [
                                        {
                                            "x_min": 0,
                                            "y_min": 1,
                                            "x_max": 0,
                                            "y_max": 1,
                                            "width_pixels": 1,
                                            "height_pixels": 1,
                                        }
                                    ]
                                    * 74
                                    + [None],
                                    "positive_pixel_frame_count": 74,
                                    "zero_pixel_frame_indices": [74],
                                },
                            },
                        }
                    }
                    if object_id_readback
                    else {}
                ),
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
    assert "pending_legacy_capture_without_object_id_raster" in html


def test_validates_object_id_mapping_counts_bboxes_and_zero_pixels(
    tmp_path: Path,
) -> None:
    output = _write_capture_output(
        tmp_path / "capture-object-id",
        pose_readback=True,
        object_id_readback=True,
    )

    review = generate_current_visual_review(output)

    assert review.frame_count == 75
    html = review.html_path.read_text(encoding="utf-8")
    assert "OBJECT_ID (false color; per-run IDs)" in html
    assert "pending_fresh_runtime_a_b_and_visual_review" in html
    assert '"runtime_object_ids":[500,501]' in html
    assert '"pixel_count":0' in html
    object_frames = sorted(
        (review.review_directory / "frames" / "object_id").glob("*.png")
    )
    assert len(object_frames) == 75
    assert object_frames[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("count", "counts/bboxes differ"),
        ("runtime_axis", "runtime object IDs"),
        ("mapping", "mapping is damaged"),
        ("missing_artifact", "evidence is partial"),
    ),
)
def test_rejects_damaged_or_partial_object_id_evidence(
    tmp_path: Path, damage: str, message: str
) -> None:
    output = _write_capture_output(
        tmp_path / damage,
        pose_readback=True,
        object_id_readback=True,
    )
    receipt_path = output / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if damage == "count":
        records_path = output / "frame_records.json"
        records = json.loads(records_path.read_text())
        records["frames"][12]["object_id_pixel_counts"][0] += 1
        records_path.write_text(json.dumps(records, indent=2) + "\n")
    elif damage == "runtime_axis":
        receipt["actors"]["runtime_object_ids"] = [500, 500]
    elif damage == "mapping":
        receipt["actors"]["axis_bindings"][1]["runtime_object_id"] = 500
    else:
        receipt["artifacts"].pop("object_id")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    with pytest.raises(CurrentVisualReviewError, match=message):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


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


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("room", "M1 request or room identity is damaged"),
        ("source", "M1 source identities do not match"),
        ("rig", "actual agent/camera differs from capture-time M1 rig"),
    ),
)
def test_rejects_damaged_capture_time_m1_identity(
    tmp_path: Path, damage: str, message: str
) -> None:
    output = _write_capture_output(tmp_path / damage, pose_readback=True)
    receipt_path = output / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt["m1_capture_identity"]
    if damage == "room":
        identity["room_id"] = "later_rewritten_room"
    elif damage == "source":
        identity["source_axis_order"] = ["source1", "source0"]
    else:
        identity["primary_camera_rig"]["world_from_rig"]["translation_m"][0] += 0.1
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(CurrentVisualReviewError, match=message):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_rejects_actual_listener_that_cannot_be_recomputed_from_m1_identity(
    tmp_path: Path,
) -> None:
    output = _write_capture_output(tmp_path / "capture", pose_readback=True)
    path = output / "arrays/actual_logical_listener_poses_txyz_qxyzw.npy"
    value = np.load(path)
    value[18, 0] += 1.0e-3
    np.save(path, value)

    with pytest.raises(
        CurrentVisualReviewError,
        match="logical listener does not match actual agent",
    ):
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


@pytest.mark.parametrize(
    "quaternion",
    (
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
    ),
)
def test_rejects_zero_or_nonunit_pose_quaternion(
    tmp_path: Path, quaternion: list[float]
) -> None:
    output = _write_capture_output(tmp_path / "capture", pose_readback=True)
    path = output / "arrays/actual_agent_camera_poses_txyz_qxyzw.npy"
    value = np.load(path)
    value[8, 3:7] = quaternion
    np.save(path, value)

    with pytest.raises(CurrentVisualReviewError, match="finite unit quaternions"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("reflection", "determinants must be \\+1"),
        ("bad_last_row", "homogeneous last row"),
        ("nonorthogonal", "rotation blocks must be orthogonal"),
    ),
)
def test_rejects_nonrigid_planned_or_actual_matrix(
    tmp_path: Path, damage: str, message: str
) -> None:
    output = _write_capture_output(tmp_path / damage, pose_readback=True)
    path = output / "arrays/actual_skin_root_readback_matrices.npy"
    value = np.load(path)
    if damage == "reflection":
        value[4, 0, 0, 0] = -1.0
    elif damage == "bad_last_row":
        value[4, 0, 3, 0] = 0.25
    else:
        value[4, 0, 0, 1] = 0.25
    np.save(path, value)

    with pytest.raises(CurrentVisualReviewError, match=message):
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
