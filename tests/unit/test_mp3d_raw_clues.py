from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.m5.mp3d_raw_clues import (
    MP3DRawClueError,
    load_dynamic_capture_raw_clues,
)
from test_m5_current_visual_review import _write_capture_output


ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _current_inputs(tmp_path: Path, *, name: str) -> tuple[Path, dict, Path]:
    root = _write_capture_output(tmp_path / name, pose_readback=True)
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


def test_current_capture_retains_close_exchange_and_zero_modal_pixels(
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
            modal = source["native_modal_semantic"]
            assert modal["visible_pixel_count"] == 0
            assert modal["modal_bbox_xyxy"] is None
            assert modal["occlusion"] is None
            assert modal["off_screen"] is None
    assert raw["audio_timing"] == {
        "status": "pending_no_audio_program_or_native_audio_receipt",
        "audio_program": None,
        "audio_receipt": None,
        "source_event_timing_by_frame": None,
        "p1_or_rir_identity": None,
    }
    assert raw["p1_amodal_visibility"]["status"] == ("pending_no_p1_or_amodal_receipt")


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
