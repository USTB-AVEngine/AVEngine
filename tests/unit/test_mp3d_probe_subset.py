from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import load_json
from avengine.m5.mp3d_probe_subset import (
    MP3DProbeSubsetError,
    analyze_mp3d_probe_candidates,
    build_placement_descriptor,
)
from tools.m5 import build_mp3d_probe_subset as builder


REPOSITORY = Path(__file__).resolve().parents[2]
ROOM = REPOSITORY / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
REQUEST = REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json"
ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"
REQUEST_ID = "m1_habitat_mp3d_single_view_v1"


def _source_positions() -> np.ndarray:
    progress = np.linspace(0.0, 0.5, 75)
    return np.stack(
        (
            np.stack((progress, np.full(75, 0.8), -3.0 + progress), axis=1),
            np.stack((0.5 + progress, np.full(75, 0.8), -4.0 + progress), axis=1),
        ),
        axis=1,
    )


def _placement(
    ordinal: int,
    *,
    position: list[float],
    yaw: float,
    semantic_room_label=None,
) -> dict:
    return {
        "placement_id": f"mp3d_camera_placement_{ordinal:05d}",
        "base_m1_request_id": REQUEST_ID,
        "floor_position_m": [position[0], 0.0, position[2]],
        "floor_sample_index": ordinal - 1,
        "height_id": "base",
        "position_m": position,
        "yaw_deg": yaw,
        "room_id": ROOM_ID,
        "semantic_room_label": semantic_room_label,
        "semantic_target": None,
        "placement_status": "declared_floor_position_crossed_with_height_and_yaw",
        "source_path_join_status": "pending",
        "native_dynamic_visibility_status": "pending",
        "difficulty_status": "not_evaluated",
    }


def _catalog() -> dict:
    placements = [
        _placement(1, position=[0.0, 1.5, 0.0], yaw=0.0),
        _placement(
            2,
            position=[0.0, 1.5, 0.0],
            yaw=180.0,
            semantic_room_label="unseen_room_type_is_posthoc_only",
        ),
        _placement(3, position=[2.0, 1.5, 0.0], yaw=30.0),
        _placement(4, position=[-2.0, 1.5, 0.0], yaw=330.0),
    ]
    return {
        "artifact_status": "complete_target_agnostic_camera_placement_catalog",
        "candidate_filtering": "none",
        "episode_counted": False,
        "floor_position_authority": ("live_habitat_pathfinder_declared_navmesh_export"),
        "placement_count": len(placements),
        "placements": placements,
        "research_only": True,
        "room_id": ROOM_ID,
        "studio_used": False,
    }


def _request() -> dict:
    return load_json(REQUEST)


def test_all_legal_placements_keep_descriptors_without_visibility_gate() -> None:
    catalog = _catalog()
    analysis = analyze_mp3d_probe_candidates(
        catalog=catalog,
        base_request=_request(),
        source_positions_m=_source_positions(),
        selection_count=4,
    )

    assert len(analysis.candidates) == 4
    assert len(analysis.selected) == 4
    by_id = {item.placement_id: item for item in analysis.candidates}
    assert by_id["mp3d_camera_placement_00002"].coverage_preference_score == 0.0
    assert {item["placement_id"] for item in analysis.selected} == set(by_id)
    assert analysis.to_record()["selection_hard_gates"] == []
    assert analysis.to_record()["candidate_legality_effect"] == "none"


def test_ranking_is_deterministic_and_ignores_posthoc_semantic_label() -> None:
    catalog = _catalog()
    first = analyze_mp3d_probe_candidates(
        catalog=catalog,
        base_request=_request(),
        source_positions_m=_source_positions(),
        selection_count=3,
    )
    catalog["placements"] = list(reversed(catalog["placements"]))
    second = analyze_mp3d_probe_candidates(
        catalog=catalog,
        base_request=_request(),
        source_positions_m=_source_positions(),
        selection_count=3,
    )

    assert first.to_record() == second.to_record()
    assert len({item["camera"]["floor_position_m"][0] for item in first.selected}) > 1
    assert len({item["camera"]["yaw_deg"] for item in first.selected}) > 1


def test_descriptor_keeps_75_frame_relative_geometry_and_pending_pixels() -> None:
    positions = _source_positions()
    analysis = analyze_mp3d_probe_candidates(
        catalog=_catalog(),
        base_request=_request(),
        source_positions_m=positions,
        selection_count=2,
    )
    descriptor = build_placement_descriptor(
        analysis.candidates[0],
        base_request=_request(),
        source_positions_m=positions,
    )

    assert descriptor["candidate_legality"]["preserved"] is True
    assert descriptor["candidate_legality"]["descriptor_effect"] == "none"
    assert descriptor["frame_count"] == 75
    assert set(descriptor["per_source"]) == {"source0", "source1"}
    for source in descriptor["per_source"].values():
        assert len(source["listener_local_azimuth_deg"]) == 75
        assert len(source["listener_local_elevation_deg"]) == 75
        assert len(source["listener_distance_m"]) == 75
        assert len(source["center_in_frustum"]) == 75
        assert source["native_semantic_visible_pixels"] is None
    json.dumps(descriptor, allow_nan=False)


def _write_builder_inputs(tmp_path: Path) -> dict[str, Path]:
    catalog_path = tmp_path / "camera_catalog.json"
    catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
    capture = tmp_path / "capture"
    arrays = capture / "arrays"
    arrays.mkdir(parents=True)
    positions_path = arrays / "source_positions_m.npy"
    positions = _source_positions()
    np.save(positions_path, positions)
    frame_records_path = capture / "frame_records.json"
    frame_records_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": frame_index,
                        "source_positions_m": positions[frame_index].tolist(),
                    }
                    for frame_index in range(75)
                ]
            }
        ),
        encoding="utf-8",
    )
    source_m1 = deepcopy(_request())
    source_m1["request_id"] = f"{REQUEST_ID}_source_capture_view"
    source_m1["primary_camera_rig"]["world_from_rig"] = {
        "translation_m": [1.0, 1.5, 2.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    source_m1_path = tmp_path / "source_capture_m1.json"
    source_m1_path.write_text(json.dumps(source_m1), encoding="utf-8")
    source_m2_path = tmp_path / "primary_m2_request.json"
    source_m2_path.write_text("{}", encoding="utf-8")
    receipt_path = capture / "research_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "actors": {
                    "actor_ids": ["actor0", "actor1"],
                    "source_ids": ["source0", "source1"],
                    "emitter_link_names": ["mouth0", "mouth1"],
                },
                "artifacts": {
                    "source_positions_m": "arrays/source_positions_m.npy",
                    "frame_records": "frame_records.json",
                },
                "capture": {
                    "frame_count": 75,
                    "native_habitat_started": True,
                },
                "episode_counted": False,
                "inputs": {
                    "m1_request": str(source_m1_path),
                    "m2_request": str(source_m2_path),
                },
                "research_only": True,
                "selected_room": {
                    "room_id": ROOM_ID,
                    "room_kind": "habitat_native",
                },
                "status": "research_only",
            }
        ),
        encoding="utf-8",
    )
    return {
        "catalog": catalog_path,
        "positions": positions_path,
        "frame_records": frame_records_path,
        "receipt": receipt_path,
        "source_m1": source_m1_path,
        "source_m2": source_m2_path,
    }


def _builder_args(tmp_path: Path, *, selection_count: int = 3):
    files = _write_builder_inputs(tmp_path)
    output = tmp_path / "probe-subset"
    args = builder.parse_args(
        [
            "--room-manifest",
            str(ROOM),
            "--base-m1-request",
            str(REQUEST),
            "--camera-catalog",
            str(files["catalog"]),
            "--source-positions",
            str(files["positions"]),
            "--source-receipt",
            str(files["receipt"]),
            "--source-m2-request",
            str(files["source_m2"]),
            "--selection-count",
            str(selection_count),
            "--output",
            str(output),
        ]
    )
    return args, output, files


def test_builder_writes_descriptor_jsonl_line_by_line_and_materializes_requests(
    tmp_path: Path,
) -> None:
    args, output, _files = _builder_args(tmp_path)

    assert builder.run(args) == output
    summary = load_json(output / "probe_subset.json")
    lines = (
        (output / "placement_descriptors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert summary["input_legal_placement_count"] == 4
    assert summary["full_descriptor_count"] == 4
    assert summary["selected_probe_count"] == 3
    assert len(lines) == 4
    assert len(summary["m1_capture_request_paths"]) == 3
    assert summary["input_catalog_authority_verification"] == {
        "all_placements_match_required_status": True,
        "artifact_status": "complete_target_agnostic_camera_placement_catalog",
        "floor_position_authority": ("live_habitat_pathfinder_declared_navmesh_export"),
        "required_placement_status": (
            "declared_floor_position_crossed_with_height_and_yaw"
        ),
        "status": "verified_target_agnostic_native_floor_catalog",
        "verified_placement_count": 4,
    }
    provenance = summary["source_track_provenance_verification"]
    assert provenance["status"] == (
        "verified_current_native_habitat_emitter_link_readback"
    )
    assert provenance["m1_camera_independent_content_exact_match"] is True
    assert provenance["explicit_source_m2_request_exact_match"] is True
    assert provenance["frame_records_source_positions_exact_match"] is True
    tracks = load_json(output / summary["source_tracks_path"])
    assert tracks["track_provenance"] == (
        "verified_same_habitat_renderer_emitter_link_readback"
    )
    assert tracks["provenance_verification"] == provenance
    records = [json.loads(line) for line in lines]
    zero_record = next(
        item
        for item in records
        if item["placement"]["placement_id"] == "mp3d_camera_placement_00002"
    )
    assert zero_record["ranking_inputs"]["coverage_preference_score"] == 0.0
    assert zero_record["candidate_legality"]["preserved"] is True
    assert zero_record["candidate_legality"]["catalog_authority_status"] == (
        "verified_target_agnostic_native_floor_catalog"
    )
    for selected in summary["selected_candidates"]:
        request = load_json(output / selected["m1_capture_request_path"])
        assert request["primary_camera_rig"]["view_id"] == "view0"
        assert request["listener"]["attached_to"] == "camera_rig_0"
        assert request["qa_views"] == [
            {
                "height_m": selected["camera"]["floor_position_m"][1],
                "kind": "topdown",
                "meters_per_pixel": 0.05,
                "qa_id": "navmesh_topdown",
            }
        ]

    with pytest.raises(
        builder.MP3DProbeSubsetBuildError, match="refusing to replace output"
    ):
        builder.run(args)


def test_builder_rejects_receipt_with_reversed_source_axis_identity(
    tmp_path: Path,
) -> None:
    args, output, files = _builder_args(tmp_path)
    receipt = files["receipt"]
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["actors"]["source_ids"] = ["source1", "source0"]
    receipt.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(builder.MP3DProbeSubsetBuildError, match="source order"):
        builder.run(args)
    assert not output.exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing_status",
        "wrong_room_kind",
        "native_not_started",
        "missing_actor_ids",
        "missing_emitter_links",
        "missing_frame_records",
        "missing_m1_request",
    ],
)
def test_builder_rejects_incomplete_native_source_receipt(
    tmp_path: Path, case: str
) -> None:
    args, output, files = _builder_args(tmp_path)
    receipt_path = files["receipt"]
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if case == "missing_status":
        value.pop("status")
    elif case == "wrong_room_kind":
        value["selected_room"]["room_kind"] = "comparison_visual"
    elif case == "native_not_started":
        value["capture"]["native_habitat_started"] = False
    elif case == "missing_actor_ids":
        value["actors"].pop("actor_ids")
    elif case == "missing_emitter_links":
        value["actors"].pop("emitter_link_names")
    elif case == "missing_frame_records":
        value["artifacts"].pop("frame_records")
    elif case == "missing_m1_request":
        value["inputs"].pop("m1_request")
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(builder.MP3DProbeSubsetBuildError, match="source receipt"):
        builder.run(args)
    assert not output.exists()


def test_builder_rejects_frame_records_artifact_escape(tmp_path: Path) -> None:
    args, output, files = _builder_args(tmp_path)
    escaped = tmp_path / "escaped_frame_records.json"
    escaped.write_text(
        files["frame_records"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    receipt_path = files["receipt"]
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["artifacts"]["frame_records"] = "../escaped_frame_records.json"
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(builder.MP3DProbeSubsetBuildError, match="escapes"):
        builder.run(args)
    assert not output.exists()


def test_builder_rejects_frame_record_and_npy_position_mismatch(
    tmp_path: Path,
) -> None:
    args, output, files = _builder_args(tmp_path)
    value = json.loads(files["frame_records"].read_text(encoding="utf-8"))
    value["frames"][37]["source_positions_m"][1][2] += 0.001
    files["frame_records"].write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(builder.MP3DProbeSubsetBuildError, match="differ"):
        builder.run(args)
    assert not output.exists()


def test_builder_rejects_camera_independent_source_m1_drift(tmp_path: Path) -> None:
    args, output, files = _builder_args(tmp_path)
    value = json.loads(files["source_m1"].read_text(encoding="utf-8"))
    value["primary_camera_rig"]["shared_calibration"]["hfov_degrees"] = 80.0
    files["source_m1"].write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(builder.MP3DProbeSubsetBuildError, match="camera-independent"):
        builder.run(args)
    assert not output.exists()


def test_builder_rejects_receipt_m2_path_different_from_explicit_input(
    tmp_path: Path,
) -> None:
    args, output, files = _builder_args(tmp_path)
    other_m2 = tmp_path / "other_m2_request.json"
    other_m2.write_text("{}", encoding="utf-8")
    receipt_path = files["receipt"]
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["inputs"]["m2_request"] = str(other_m2)
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(builder.MP3DProbeSubsetBuildError, match="M2 request differs"):
        builder.run(args)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("artifact_status", "pending", "artifact_status"),
        ("floor_position_authority", "pending", "floor authority"),
        ("placement_status", "pending", "placement_status"),
        ("placement_status", "invalid", "placement_status"),
    ],
)
def test_catalog_authority_and_each_placement_status_are_required(
    field: str, value: str, match: str
) -> None:
    catalog = _catalog()
    if field == "placement_status":
        catalog["placements"][2][field] = value
    else:
        catalog[field] = value

    with pytest.raises(MP3DProbeSubsetError, match=match):
        analyze_mp3d_probe_candidates(
            catalog=catalog,
            base_request=_request(),
            source_positions_m=_source_positions(),
            selection_count=2,
        )


def test_source_positions_must_keep_75_by_two_by_three_shape() -> None:
    with pytest.raises(MP3DProbeSubsetError, match=r"\[75,2,3\]"):
        analyze_mp3d_probe_candidates(
            catalog=_catalog(),
            base_request=_request(),
            source_positions_m=np.zeros((74, 2, 3), dtype=np.float64),
            selection_count=2,
        )
