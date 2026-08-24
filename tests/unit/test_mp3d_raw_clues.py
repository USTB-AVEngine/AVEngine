from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import avengine.m5.mp3d_raw_clues as raw_clue_module
from avengine.m5.mp3d_raw_clues import (
    MP3DRawClueError,
    load_dynamic_capture_raw_clues,
)
from test_m5_current_visual_review import _write_capture_output


ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _current_inputs(
    tmp_path: Path, *, name: str, object_id_readback: bool = False
) -> tuple[Path, dict, Path]:
    root = _write_capture_output(
        tmp_path / name,
        pose_readback=True,
        object_id_readback=object_id_readback,
    )
    receipt_path = root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt["m1_capture_identity"]
    request = {
        "schema": "avengine_m1_capture_request_v1",
        "request_id": identity["request_id"],
        "room_id": identity["room_id"],
        "primary_camera_rig": identity["primary_camera_rig"],
        "listener": identity["listener"],
        "sources": [
            {"source_id": "source0"},
            {"source_id": "source1"},
        ],
        "qa_views": identity["qa_views"],
    }
    request_path = tmp_path / f"{name}_m1.json"
    _write_json(request_path, request)
    frame_path = root / "frame_records.json"
    frame_document = json.loads(frame_path.read_text())
    frame_records = frame_document["frames"]
    semantic = np.load(root / "arrays/semantic.npy")
    visibility = np.stack(
        [
            np.count_nonzero(semantic == 210, axis=(1, 2)),
            np.count_nonzero(semantic == 211, axis=(1, 2)),
        ],
        axis=1,
    ).astype(np.int64)
    np.save(root / "arrays/semantic_visibility_pixels.npy", visibility)
    for index, frame in enumerate(frame_records):
        frame["semantic_visibility_pixels"] = visibility[index].tolist()
    _write_json(frame_path, frame_document)
    m2 = {
        "schema": "avengine_m2_articulated_capture_request_v1",
        "request_id": f"{name}_m2",
        "states": [
            {
                "frame_index": index,
                "pts_ticks": frame["pts_ticks"],
                "action_id": frame["action_id"],
            }
            for index, frame in enumerate(frame_records)
        ],
    }
    m2_path = tmp_path / f"{name}_m2.json"
    _write_json(m2_path, m2)
    receipt["inputs"] = {
        "m1_request": str(request_path.resolve()),
        "m2_request": str(m2_path.resolve()),
    }
    receipt["visibility"] = {
        "status": "native_semantic_pixels_recorded",
        "candidate_filtering": "disabled",
        "per_actor": {
            "actor0": {
                "per_frame_visible_pixels": visibility[:, 0].tolist(),
            },
            "actor1": {
                "per_frame_visible_pixels": visibility[:, 1].tolist(),
            },
        },
    }
    _write_json(receipt_path, receipt)
    return root, request, request_path


def _load(root: Path, request: dict, request_path: Path):
    return load_dynamic_capture_raw_clues(
        root,
        request=request,
        request_path=request_path,
        expected_room_id=ROOM_ID,
    )


def _set_yaw(
    root: Path,
    request: dict,
    request_path: Path,
    *,
    rotation_xyzw: list[float],
) -> None:
    receipt_path = root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["m1_capture_identity"]["primary_camera_rig"]["world_from_rig"][
        "rotation_xyzw"
    ] = rotation_xyzw
    _write_json(receipt_path, receipt)
    request["primary_camera_rig"]["world_from_rig"]["rotation_xyzw"] = rotation_xyzw
    _write_json(request_path, request)
    for filename in (
        "actual_agent_camera_poses_txyz_qxyzw.npy",
        "actual_logical_listener_poses_txyz_qxyzw.npy",
    ):
        path = root / "arrays" / filename
        values = np.load(path)
        values[:, 3:7] = rotation_xyzw
        np.save(path, values)
    sensor_path = root / "arrays/actual_sensor_poses_txyz_qxyzw.npy"
    sensors = np.load(sensor_path)
    sensors[:, :, 3:7] = rotation_xyzw
    np.save(sensor_path, sensors)


def test_rgb_clue_with_zero_semantic_ids_retains_close_exchange_and_pending_visibility(
    tmp_path: Path,
) -> None:
    root, request, request_path = _current_inputs(tmp_path, name="exchange")
    fractions = np.linspace(0.0, 1.0, 75)
    source0 = np.stack(
        (-0.2 + 0.4 * fractions, np.zeros(75), np.full(75, -4.0)), axis=1
    )
    source1 = np.stack((0.2 - 0.4 * fractions, np.zeros(75), np.full(75, -4.0)), axis=1)
    sources = np.stack((source0, source1), axis=1)
    np.save(root / "arrays/source_positions_m.npy", sources)
    actual_skin = np.load(root / "arrays/actual_skin_root_readback_matrices.npy")
    actual_skin[:, 0, :3, 3] = source0
    actual_skin[:, 1, :3, 3] = source1
    np.save(root / "arrays/actual_skin_root_readback_matrices.npy", actual_skin)
    semantic = np.load(root / "arrays/semantic.npy")
    semantic[semantic == 210] = 0
    semantic[semantic == 211] = 0
    np.save(root / "arrays/semantic.npy", semantic)
    rgb = np.load(root / "arrays/rgb.npy")
    rgb[0, 0, 0] = [255, 17, 201]
    np.save(root / "arrays/rgb.npy", rgb)
    assert rgb[0, 0, 0].tolist() == [255, 17, 201]
    visibility = np.zeros((75, 2), dtype=np.int64)
    np.save(root / "arrays/semantic_visibility_pixels.npy", visibility)
    frame_path = root / "frame_records.json"
    frame_document = json.loads(frame_path.read_text())
    for index, frame in enumerate(frame_document["frames"]):
        frame["source_positions_m"] = sources[index].tolist()
        frame["semantic_visibility_pixels"] = [0, 0]
    _write_json(frame_path, frame_document)
    receipt_path = root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for actor_id in ("actor0", "actor1"):
        receipt["visibility"]["per_actor"][actor_id]["per_frame_visible_pixels"] = [
            0
        ] * 75
    _write_json(receipt_path, receipt)

    loaded = _load(root, request, request_path)
    raw = loaded.raw_clues
    pair = raw["pair_tracks"]["actual_emitter_relative_to_actual_listener"]

    assert loaded.has_current_pose_authority is True
    assert raw["candidate_filtering"] == "none"
    assert raw["candidate_retained"] is True
    assert pair["difficulty"]["minimum_pair_distance_horizontal"] < 0.6
    assert pair["observed_crossing_and_swap"]["endpoint_exchange_horizontal"] is True
    assert pair["observed_crossing_and_swap"]["between_frame_crossing_status"] == (
        "not_inferred"
    )
    assert (
        raw["planned_continuous_polyline"]["between_frame_interpolation_performed"]
        is False
    )
    for frame in raw["frames"]:
        for source in frame["sources"].values():
            raster = source["native_semantic_id_raster"]
            assert raster["semantic_id_pixel_count"] == 0
            assert raster["semantic_id_bbox_xyxy"] is None
            assert raster["semantic_id_pixel_presence"] is False
            assert raster["semantic_id_coverage_completeness_status"] == (
                "semantic_id_coverage_completeness_not_certified"
            )
            assert raster["actor_modal_visibility"] is None
            assert raster["occlusion"] is None
            assert raster["off_screen"] is None
    assert raw["semantic_id_raster_boundary"] == {
        "status": "semantic_id_coverage_completeness_not_certified",
        "positive_count_interpretation": "semantic_id_appears_in_raster",
        "zero_count_interpretation": "does_not_prove_actor_absence",
        "actor_modal_visibility": None,
        "occlusion": None,
        "off_screen": None,
    }
    assert pair["frames"][0]["source1"]["visible_pixels"] is None
    assert pair["frames"][0]["source1"]["semantic_id_pixel_count"] == 0
    assert raw["audio_timing"] == {
        "status": "pending_no_audio_program_or_native_audio_receipt",
        "audio_program": None,
        "audio_receipt": None,
        "source_event_timing_by_frame": None,
        "p1_or_rir_identity": None,
    }
    assert raw["p1_amodal_visibility"]["status"] == ("pending_no_p1_or_amodal_receipt")
    assert raw["motion_timing"]["planned"]["action_sample_index_status"] == (
        "retained_not_independently_rederived"
    )


def test_actual_yaw_changes_listener_doa_and_pinhole_projection(tmp_path: Path) -> None:
    yaw90_root, yaw90_request, yaw90_path = _current_inputs(tmp_path, name="yaw90")
    yaw0_root, yaw0_request, yaw0_path = _current_inputs(tmp_path, name="yaw0")
    _set_yaw(
        yaw0_root,
        yaw0_request,
        yaw0_path,
        rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    yaw90 = _load(yaw90_root, yaw90_request, yaw90_path).raw_clues["frames"][0]
    yaw0 = _load(yaw0_root, yaw0_request, yaw0_path).raw_clues["frames"][0]
    yaw90_source = yaw90["sources"]["source0"]
    yaw0_source = yaw0["sources"]["source0"]

    assert (
        yaw90_source["actual_listener_local_geometry"]["azimuth_deg"]
        != (yaw0_source["actual_listener_local_geometry"]["azimuth_deg"])
    )
    assert (
        yaw90_source["analytic_pinhole_center"]["normalized_pinhole_x_right"]
        != yaw0_source["analytic_pinhole_center"]["normalized_pinhole_x_right"]
    )
    assert (
        yaw90["camera"]["actual_post_render_agent_camera"]["rotation_xyzw"]
        != yaw0["camera"]["actual_post_render_agent_camera"]["rotation_xyzw"]
    )
    raster = yaw90_source["native_semantic_id_raster"]
    assert raster["semantic_id_pixel_count"] > 0
    assert raster["semantic_id_pixel_presence"] is True
    assert raster["actor_modal_visibility"] is None
    assert raster["semantic_id_pixel_presence_interpretation"].startswith(
        "positive_count_proves_only"
    )


def test_object_id_modal_raw_clues_are_joined_without_semantic_authority(
    tmp_path: Path,
) -> None:
    root, request, request_path = _current_inputs(
        tmp_path, name="object-id", object_id_readback=True
    )

    raw = _load(root, request, request_path).raw_clues
    frame0_source0 = raw["frames"][0]["sources"]["source0"]
    frame74_source1 = raw["frames"][74]["sources"]["source1"]
    modal0 = frame0_source0["native_object_id_raster"]
    modal74 = frame74_source1["native_object_id_raster"]

    assert raw["identity_axes"]["runtime_object_id_axis_order"] == [500, 501]
    assert raw["object_id_raster_boundary"]["status"] == (
        "native_object_id_raster_modal_descriptors_recorded"
    )
    assert raw["object_id_raster_boundary"]["occlusion_by_frame"] is None
    assert raw["object_id_raster_boundary"]["off_screen_by_frame"] is None
    assert modal0["runtime_object_id"] == 500
    assert modal0["object_id_pixel_count"] == 2
    assert modal0["object_id_bbox_xyxy"] == {
        "x_min": 0,
        "y_min": 0,
        "x_max": 1,
        "y_max": 0,
        "width_pixels": 2,
        "height_pixels": 1,
    }
    assert modal0["actor_modal_visibility"] is True
    assert modal0["in_frame"] is True
    assert modal0["occlusion"] is None
    assert modal0["off_screen"] is None
    assert raw["pair_tracks"]["actual_emitter_relative_to_actual_listener"][
        "frames"
    ][0]["source1"]["visible_pixels"] == 2
    assert raw["pair_tracks"]["actual_emitter_relative_to_actual_listener"][
        "frames"
    ][0]["source1"]["in_frame"] is True
    assert modal74["object_id_pixel_count"] == 0
    assert modal74["object_id_bbox_xyxy"] is None
    assert modal74["actor_modal_visibility"] is False
    assert modal74["in_frame"] is False
    assert modal74["occlusion"] is None
    assert modal74["off_screen"] is None
    assert frame74_source1["native_semantic_id_raster"]["actor_modal_visibility"] is None


@pytest.mark.parametrize("damage", ("runtime_id", "frame_count"))
def test_object_id_mapping_and_75_frames_fail_fast(
    tmp_path: Path, damage: str
) -> None:
    root, request, request_path = _current_inputs(
        tmp_path, name=f"object-id-{damage}", object_id_readback=True
    )
    receipt_path = root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if damage == "runtime_id":
        receipt["actors"]["runtime_object_ids"] = [500, 500]
    else:
        receipt["object_id_visibility"]["frame_count"] = 74
    _write_json(receipt_path, receipt)

    with pytest.raises(
        MP3DRawClueError, match="object-ID|runtime object IDs|frame_count"
    ):
        _load(root, request, request_path)


def test_pinhole_continuous_pixel_center_edge_and_sign_convention() -> None:
    poses = np.tile(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (75, 1))
    positions = np.zeros((75, 2, 3), dtype=np.float64)
    positions[:, 0] = [0.0, 0.0, -1.0]
    positions[:, 1] = [1.0, 0.0, -1.0]
    positions[1, 0] = [-1.0, 0.0, -1.0]
    positions[2, 0] = [0.0, 2.0 / 3.0, -1.0]

    tracks = raw_clue_module._pinhole_tracks(
        positions,
        poses,
        hfov_degrees=90.0,
        height=4,
        width=6,
    )

    center = tracks[0][0]
    assert center["normalized_pinhole_x_right"] == pytest.approx(0.0)
    assert center["normalized_pinhole_y_up"] == pytest.approx(0.0)
    assert center["pixel_x_right"] == pytest.approx(2.5)
    assert center["pixel_y_down"] == pytest.approx(1.5)
    assert center["frustum_margin_normalized"] == pytest.approx(1.0)
    assert center["pixel_center_convention"] == (
        "continuous_zero_based_pixel_centers_x_0_to_width_minus_1_y_0_to_height_minus_1"
    )
    right_edge = tracks[1][0]
    assert right_edge["normalized_pinhole_x_right"] == pytest.approx(1.0)
    assert right_edge["pixel_x_right"] == pytest.approx(5.0)
    assert right_edge["frustum_margin_normalized"] == pytest.approx(0.0)
    assert right_edge["center_in_frustum"] is True
    left_edge = tracks[0][1]
    assert left_edge["normalized_pinhole_x_right"] == pytest.approx(-1.0)
    assert left_edge["pixel_x_right"] == pytest.approx(0.0)
    upper_edge = tracks[0][2]
    assert upper_edge["normalized_pinhole_y_up"] == pytest.approx(1.0)
    assert upper_edge["pixel_y_down"] == pytest.approx(0.0)


def test_projection_uses_semantic_sensor_pose_not_agent_pose(tmp_path: Path) -> None:
    root, request, request_path = _current_inputs(tmp_path, name="sensor-extrinsic")
    _set_yaw(
        root,
        request,
        request_path,
        rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    half_angle = np.deg2rad(22.5)
    sensor_yaw = [0.0, float(np.sin(half_angle)), 0.0, float(np.cos(half_angle))]
    receipt_path = root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["m1_capture_identity"]["primary_camera_rig"]["shared_calibration"][
        "rig_from_sensor"
    ]["rotation_xyzw"] = sensor_yaw
    _write_json(receipt_path, receipt)
    request["primary_camera_rig"]["shared_calibration"]["rig_from_sensor"][
        "rotation_xyzw"
    ] = sensor_yaw
    _write_json(request_path, request)
    sensor_path = root / "arrays/actual_sensor_poses_txyz_qxyzw.npy"
    sensor_poses = np.load(sensor_path)
    sensor_poses[:, :, 3:7] = sensor_yaw
    np.save(sensor_path, sensor_poses)
    source_positions = np.zeros((75, 2, 3), dtype=np.float64)
    source_positions[:, 0] = [9.0, 8.0, 3.0]
    source_positions[:, 1] = [9.5, 8.0, 3.0]
    np.save(root / "arrays/source_positions_m.npy", source_positions)
    frame_path = root / "frame_records.json"
    frame_document = json.loads(frame_path.read_text())
    for index, frame in enumerate(frame_document["frames"]):
        frame["source_positions_m"] = source_positions[index].tolist()
    _write_json(frame_path, frame_document)

    raw = _load(root, request, request_path).raw_clues
    frame = raw["frames"][0]
    projected = frame["sources"]["source0"]["analytic_pinhole_center"]
    agent_pose = np.asarray(
        [
            [
                *frame["camera"]["actual_post_render_agent_camera"]["translation_m"],
                *frame["camera"]["actual_post_render_agent_camera"]["rotation_xyzw"],
            ]
        ]
        * 75,
        dtype=np.float64,
    )
    agent_projection = raw_clue_module._pinhole_tracks(
        source_positions,
        agent_pose,
        hfov_degrees=90.0,
        height=2,
        width=4,
    )[0][0]

    assert raw["calibration"]["pose_authority"] == (
        "actual_post_render_semantic_sensor_pose"
    )
    assert projected["authority"] == (
        "actual_post_render_semantic_sensor_pose_plus_capture_time_m1_intrinsics"
    )
    assert abs(projected["normalized_pinhole_x_right"]) == pytest.approx(
        1.0, abs=2.0e-6
    )
    assert agent_projection["normalized_pinhole_x_right"] == pytest.approx(
        0.0, abs=2.0e-6
    )
    assert frame["sources"]["source0"]["actual_listener_local_geometry"][
        "azimuth_deg"
    ] == pytest.approx(0.0, abs=2.0e-6)


def test_current_capture_rejects_damaged_identity_axis(tmp_path: Path) -> None:
    root, request, request_path = _current_inputs(tmp_path, name="axis")
    receipt_path = root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["pose_readback"]["source_axis_order"] = ["source1", "source0"]
    _write_json(receipt_path, receipt)

    with pytest.raises(MP3DRawClueError, match="source identity"):
        _load(root, request, request_path)


def test_current_capture_rejects_mutable_m1_path_identity_drift(tmp_path: Path) -> None:
    root, request, request_path = _current_inputs(tmp_path, name="identity")
    request["request_id"] = "later_rewritten_request_at_same_path"
    _write_json(request_path, request)

    with pytest.raises(MP3DRawClueError, match="m1_capture_identity differs"):
        _load(root, request, request_path)


def test_legacy_capture_keeps_every_new_authority_pending(tmp_path: Path) -> None:
    from avengine.contracts.json_io import load_json
    from test_describe_mp3d_viewpoint_tracks import _args
    from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor

    args, output, *_unused = _args(tmp_path, with_dynamic=True)

    descriptor.run(args)
    raw = load_json(output / "viewpoint_raw_track_descriptors.json")["viewpoints"][0][
        "dynamic_raw_clues"
    ]

    assert raw["status"] == "legacy_capture_pending_current_pose_authority"
    assert raw["has_current_post_render_pose_authority"] is False
    assert raw["frames"][0]["camera"]["actual_post_render_agent_camera"] is None
    assert raw["frames"][0]["listener"]["actual_derived_world_from_listener"] is None
    assert (
        raw["frames"][0]["sources"]["source0"]["actual_skin_root_readback_matrix"]
        is None
    )
    assert raw["audio_timing"]["audio_program"] is None
    assert raw["p1_amodal_visibility"]["off_screen_by_frame"] is None
    assert raw["frames"][0]["pts_ticks"] is None
    assert raw["frames"][0]["pts_status"] == (
        "pending_no_legacy_frame_records_artifact"
    )
    emitter_motion = raw["motion_timing"]["actual_emitter"]
    assert emitter_motion["pts_authority"] is None
    assert (
        emitter_motion["by_identity"]["source0"]["frame_to_frame_speed_mps"]
        == [None] * 75
    )


def test_legacy_capture_reads_frame_pts_and_compares_m2_action_time(
    tmp_path: Path,
) -> None:
    from avengine.contracts.json_io import load_json
    from test_describe_mp3d_viewpoint_tracks import _args
    from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor

    args, output, *_unused = _args(tmp_path, with_dynamic=True)
    dynamic_root = args.dynamic_capture_dir[0]
    receipt_path = dynamic_root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    frames = [
        {
            "frame_index": index,
            "pts_ticks": index * 4_000,
            "action_id": "walk",
            "action_time_ticks": index * 2_000,
            "action_sample_index": index,
        }
        for index in range(75)
    ]
    _write_json(dynamic_root / "frame_records.json", {"frames": frames})
    m2_path = tmp_path / "legacy_m2.json"
    _write_json(
        m2_path,
        {
            "request_id": "legacy_m2",
            "states": [
                {
                    "frame_index": frame["frame_index"],
                    "pts_ticks": frame["pts_ticks"],
                    "action_id": frame["action_id"],
                    "action_time_ticks": frame["action_time_ticks"],
                }
                for frame in frames
            ],
        },
    )
    receipt["artifacts"]["frame_records"] = "frame_records.json"
    receipt["inputs"]["m2_request"] = str(m2_path.resolve())
    _write_json(receipt_path, receipt)

    descriptor.run(args)
    raw = load_json(output / "viewpoint_raw_track_descriptors.json")["viewpoints"][0][
        "dynamic_raw_clues"
    ]

    assert raw["frames"][1]["pts_ticks"] == 4_000
    assert raw["frames"][1]["pts_status"] == (
        "validated_from_legacy_frame_records_artifact"
    )
    planned = raw["motion_timing"]["planned"]
    assert planned["labels_by_frame"][1]["action_time_ticks"] == 2_000
    assert planned["labels_by_frame"][1]["action_time_ticks_comparison_status"] == (
        "exact_match_between_m2_and_frame_record"
    )
    assert planned["labels_by_frame"][1]["action_sample_index_status"] == (
        "retained_not_independently_rederived"
    )
    assert raw["motion_timing"]["actual_emitter"]["pts_authority"] == (
        "legacy_dynamic_capture.frame_records.pts_ticks"
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("pts", "PTS must be strictly increasing"),
        ("action_time", "action_time_ticks differ"),
    ),
)
def test_legacy_capture_rejects_damaged_timing(
    tmp_path: Path, damage: str, message: str
) -> None:
    from test_describe_mp3d_viewpoint_tracks import _args
    from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor

    args, _output, *_unused = _args(tmp_path, with_dynamic=True)
    dynamic_root = args.dynamic_capture_dir[0]
    receipt_path = dynamic_root / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    frames = [
        {
            "frame_index": index,
            "pts_ticks": index * 3_200,
            "action_id": "walk",
            "action_time_ticks": index * 3_200,
            "action_sample_index": index,
        }
        for index in range(75)
    ]
    m2_states = [dict(frame) for frame in frames]
    for state in m2_states:
        state.pop("action_sample_index")
    if damage == "pts":
        frames[1]["pts_ticks"] = 0
        m2_states[1]["pts_ticks"] = 0
    else:
        m2_states[1]["action_time_ticks"] += 1
    _write_json(dynamic_root / "frame_records.json", {"frames": frames})
    m2_path = tmp_path / "damaged_legacy_m2.json"
    _write_json(m2_path, {"request_id": "damaged", "states": m2_states})
    receipt["artifacts"]["frame_records"] = "frame_records.json"
    receipt["inputs"]["m2_request"] = str(m2_path.resolve())
    _write_json(receipt_path, receipt)

    with pytest.raises(descriptor.MP3DTrackDescriptorError, match=message):
        descriptor.run(args)


def test_probe_subset_adapter_outputs_same_two_records_in_order(tmp_path: Path) -> None:
    from avengine.contracts.json_io import load_json
    from test_describe_mp3d_viewpoint_tracks import (
        ROOM_ID,
        VIEWPOINT_ID,
        _inputs,
        _write_json,
    )
    from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor

    bank, route, probes, *_unused = _inputs(tmp_path)
    bank_root = bank.parent
    first_request_path = bank_root / "requests" / f"{VIEWPOINT_ID}.json"
    first_request = json.loads(first_request_path.read_text())
    second_id = "mp3d_camera_placement_00002"
    second_request = json.loads(json.dumps(first_request))
    second_request["request_id"] = "request-view-02"
    _write_json(bank_root / "requests" / f"{second_id}.json", second_request)
    first_receipt = json.loads((probes / VIEWPOINT_ID / "receipt.json").read_text())
    second_receipt = json.loads(json.dumps(first_receipt))
    second_receipt["request_id"] = "request-view-02"
    _write_json(probes / second_id / "receipt.json", second_receipt)
    selected = [
        {
            "selection_order": 1,
            "placement_id": VIEWPOINT_ID,
            "ranking": {"selection_score": 0.9},
            "m1_capture_request_path": f"requests/{VIEWPOINT_ID}.json",
        },
        {
            "selection_order": 2,
            "placement_id": second_id,
            "ranking": {"selection_score": 0.1},
            "m1_capture_request_path": f"requests/{second_id}.json",
        },
    ]
    subset = bank_root / "probe_subset.json"
    _write_json(
        subset,
        {
            "research_only": True,
            "episode_counted": False,
            "room_id": ROOM_ID,
            "selected_candidates": selected,
            "m1_capture_request_paths": {
                item["placement_id"]: item["m1_capture_request_path"]
                for item in selected
            },
        },
    )
    output = tmp_path / "subset-output"
    args = descriptor.parse_args(
        [
            "--probe-subset",
            str(subset),
            "--route-explanation",
            str(route),
            "--probe-root",
            str(probes),
            "--output",
            str(output),
        ]
    )

    descriptor.run(args)
    result = load_json(output / "viewpoint_raw_track_descriptors.json")

    assert result["input_candidate_count"] == 2
    assert result["output_viewpoint_count"] == 2
    assert result["input_count_equals_output_count"] is True
    assert [item["viewpoint_id"] for item in result["viewpoints"]] == [
        VIEWPOINT_ID,
        second_id,
    ]
    assert [item["input_record"] for item in result["viewpoints"]] == selected
    assert all(item["candidate_retained_by_bridge"] for item in result["viewpoints"])


def test_region_bank_adapter_flattens_and_preserves_region_records() -> None:
    from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor

    candidates = [
        {"placement_id": "placement_a", "selection_order": 1, "ranking": {"x": 1}},
        {"placement_id": "placement_b", "selection_order": 1, "ranking": {"x": 2}},
    ]
    document = {
        "research_only": True,
        "episode_counted": False,
        "room_id": ROOM_ID,
        "region_banks": [
            {"region_instance_id": "region_a", "selected_candidates": [candidates[0]]},
            {"region_instance_id": "region_b", "selected_candidates": [candidates[1]]},
        ],
        "m1_capture_request_paths": {
            "placement_a": "requests/a.json",
            "placement_b": "requests/b.json",
        },
    }

    room_id, records, paths = descriptor._viewpoint_join_input(
        document, input_kind="region_viewpoint_banks"
    )

    assert room_id == ROOM_ID
    assert paths == document["m1_capture_request_paths"]
    assert [item["input_record"] for item in records] == candidates
    assert [item["input_parent_record"]["region_instance_id"] for item in records] == [
        "region_a",
        "region_b",
    ]
