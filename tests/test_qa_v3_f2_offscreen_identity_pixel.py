
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import join_f2_offscreen_identity_pixel as binder  # noqa: E402


def _frame(frame_index: int, *, visible: bool) -> dict:
    if visible:
        return {
            "frame_index": frame_index,
            "state": "visible_clear",
            "target_pixels": 100,
            "visible_pixels": 100,
            "visible_fraction": 1.0,
            "occlusion_fraction": 0.0,
            "target_bbox_xyxy_px": [1, 1, 5, 5],
            "target_centroid_xy_px": [3.0, 3.0],
        }
    return {
        "frame_index": frame_index,
        "state": "out_of_view",
        "target_pixels": 0,
        "visible_pixels": 0,
        "visible_fraction": None,
        "occlusion_fraction": None,
        "target_bbox_xyxy_px": None,
        "target_centroid_xy_px": None,
    }


def _truth() -> dict:
    return {
        "authority": "native_pixel_test",
        "camera_contract_id": "camera_test_v1",
        "resolution_hw": [8, 8],
        "camera_pose_ids": ["camera_frame_0", "camera_frame_1",
                            "camera_frame_2", "camera_frame_3"],
        "frame_indices": [0, 1, 2, 3],
        "per_instance": {
            "source1": {
                "source_slot_id": "source1",
                "instance_id": "human_1",
                "frames": [
                    _frame(0, visible=False), _frame(1, visible=False),
                    _frame(2, visible=True), _frame(3, visible=True),
                ],
            },
            "source2": {
                "source_slot_id": "source2",
                "instance_id": "human_2",
                "frames": [
                    _frame(0, visible=False), _frame(1, visible=False),
                    _frame(2, visible=True), _frame(3, visible=True),
                ],
            },
        },
    }


def _fixture(tmp_path: Path):
    fact_path = tmp_path / "fact_record.json"
    main_selection = tmp_path / "actor_selection.json"
    gateb_selection = tmp_path / "actor_selection_gateB.json"
    main_timeline = tmp_path / "timeline.json"
    gateb_timeline = tmp_path / "timeline_gateB.json"
    gateb_endpoints = tmp_path / "source_endpoints_gateB.json"
    intervention = tmp_path / "gateB_intervention.json"
    main_truth = tmp_path / "main_pixel_truth.json"
    gateb_truth = tmp_path / "gateB_pixel_truth.json"
    main_evidence = tmp_path / "main_evidence.json"
    gateb_evidence = tmp_path / "gateB_evidence.json"

    main_selection.write_text(json.dumps({
        "actors": [
            {"source_slot_id": "source1", "asset_id": "asset_blue"},
            {"source_slot_id": "source2", "asset_id": "asset_burgundy"},
        ],
    }))
    gateb_selection.write_text(json.dumps({
        "actors": [
            {"source_slot_id": "source1", "asset_id": "asset_burgundy"},
            {"source_slot_id": "source2", "asset_id": "asset_blue"},
        ],
    }))
    main_timeline.write_text(json.dumps({
        "actor_selection": main_selection.name,
        "frames": [],
    }))
    gateb_timeline.write_text(json.dumps({
        "actor_selection": gateb_selection.name,
        "frames": [],
    }))
    gateb_endpoints.write_text(json.dumps({"source_endpoints": []}))
    intervention.write_text(json.dumps({
        "variant": "gateB",
        "actor_selection": gateb_selection.name,
        "timeline": gateb_timeline.name,
        "source_endpoints": gateb_endpoints.name,
        "audio_program": "audio_program.json",
        "audio_unchanged": True,
    }))
    fact = {
        "schema": "avengine_qa_v3_offscreen_identity_candidate_v1",
        "scene_id": "room_test",
        "profile_id": "profile_test",
        "point_id": "point_test",
        "appearance_by_slot": {
            "source1": "blue",
            "source2": "burgundy",
        },
        "artifacts": {
            "selection": main_selection.name,
            "timeline": main_timeline.name,
        },
        "geometry": {
            "route_reports": [
                {
                    "source_slot_id": "source1",
                    "early": {"frames": [0, 1]},
                    "late": {"frames": [2, 3]},
                },
                {
                    "source_slot_id": "source2",
                    "early": {"frames": [0, 1]},
                    "late": {"frames": [2, 3]},
                },
            ],
        },
    }
    fact_path.write_text(json.dumps(fact))
    main_value = _truth()
    gateb_value = _truth()
    main_truth.write_text(json.dumps(main_value))
    gateb_truth.write_text(json.dumps(gateb_value))
    main_evidence.write_text(json.dumps({
        "schema": "qa_v3_current_timeline_native_pixel_probe_v1",
        "status": "pass",
        "artifacts": {"truth": main_truth.name},
        "inputs": {
            "actor_selection": main_selection.name,
            "timeline": main_timeline.name,
        },
        "pixel_visibility": main_value,
    }))
    gateb_evidence.write_text(json.dumps({
        "schema": "qa_v3_current_timeline_native_pixel_probe_v1",
        "status": "pass",
        "artifacts": {"truth": gateb_truth.name},
        "inputs": {
            "actor_selection": gateb_selection.name,
            "timeline": gateb_timeline.name,
        },
        "pixel_visibility": gateb_value,
    }))
    return {
        "fact": dict(fact, _source_dir=str(tmp_path)),
        "main_truth": main_truth,
        "gateb_truth": gateb_truth,
        "main_evidence": main_evidence,
        "gateb_evidence": gateb_evidence,
        "main_timeline": main_timeline,
        "gateb_timeline": gateb_timeline,
        "main_selection": main_selection,
        "gateb_selection": gateb_selection,
        "intervention": intervention,
        "fact_path": fact_path,
    }


def test_f2_pixel_join_accepts_declared_windows_and_swapped_appearance(tmp_path):
    fixture = _fixture(tmp_path)
    result = binder.join(
        fixture["fact"],
        fixture["main_evidence"],
        fixture["intervention"],
        fixture["gateb_evidence"],
    )
    assert result["pixel_join_status"] == "pass"
    assert result["evidence_class"] == "pixel_qualified_candidate"
    assert result["report"]["checks"]["frame_indices_same"] is True
    assert result["report"]["checks"]["camera_identity_same"] is True
    assert result["report"]["checks"]["gateb_geometry_same"] is True
    source_match = result["report"]["checks"]["source_match"]
    assert source_match["main"]["matched"] is True
    assert source_match["gateB"]["matched"] is True
    appearance = result["report"]["appearance_binding"]
    assert appearance["changed_slots"] == ["source1", "source2"]
    assert appearance["expected_gateb_appearance_by_slot"] == {
        "source1": "burgundy",
        "source2": "blue",
    }
    assert result["joined_fact"]["truth_status"] == (
        "native_pixel_joined_research_candidate"
    )
    assert result["joined_fact"]["point_id"] == "point_test"


def test_f2_pixel_join_rejects_early_occlusion_as_not_out_of_view(tmp_path):
    fixture = _fixture(tmp_path)
    truth = json.loads(fixture["main_truth"].read_text())
    truth["per_instance"]["source1"]["frames"][0].update({
        "state": "fully_occluded",
        "target_pixels": 12,
    })
    fixture["main_truth"].write_text(json.dumps(truth))
    evidence = json.loads(fixture["main_evidence"].read_text())
    evidence["pixel_visibility"] = truth
    fixture["main_evidence"].write_text(json.dumps(evidence))
    result = binder.join(
        fixture["fact"],
        fixture["main_evidence"],
        fixture["intervention"],
        fixture["gateb_evidence"],
    )
    assert result["pixel_join_status"] == "pixel_rejected"
    assert any(
        reason == "main.source1.early.frame_0.not_out_of_view"
        for reason in result["report"]["rejection_reasons"]
    )
    assert result["joined_fact"]["truth_status"] == (
        "native_pixel_rejected_research_candidate"
    )


def test_f2_pixel_join_rejects_gateb_geometry_mismatch(tmp_path):
    fixture = _fixture(tmp_path)
    truth = json.loads(fixture["gateb_truth"].read_text())
    truth["per_instance"]["source2"]["frames"][2]["target_pixels"] = 99
    fixture["gateb_truth"].write_text(json.dumps(truth))
    evidence = json.loads(fixture["gateb_evidence"].read_text())
    evidence["pixel_visibility"] = truth
    fixture["gateb_evidence"].write_text(json.dumps(evidence))
    result = binder.join(
        fixture["fact"],
        fixture["main_evidence"],
        fixture["intervention"],
        fixture["gateb_evidence"],
    )
    assert result["pixel_join_status"] == "pixel_rejected"
    assert any(
        reason == "gateB_geometry_differs.source2.late.frame_2"
        for reason in result["report"]["rejection_reasons"]
    )


def test_f2_pixel_join_requires_route_report_source_slot_id(tmp_path):
    fixture = _fixture(tmp_path)
    fact = copy.deepcopy(fixture["fact"])
    del fact["geometry"]["route_reports"][0]["source_slot_id"]
    with pytest.raises(
        binder.F2PixelJoinError,
        match="must declare source_slot_id",
    ):
        binder.join(
            fact,
            fixture["main_evidence"],
            fixture["intervention"],
            fixture["gateb_evidence"],
        )


def test_f2_pixel_join_rejects_non_exchange_intervention(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["intervention"].write_text(json.dumps({
        "actor_selection": "actor_selection.json",
        "timeline": "timeline_gateB.json",
        "source_endpoints": "source_endpoints_gateB.json",
    }))
    with pytest.raises(
        binder.F2PixelJoinError,
        match="does not exchange",
    ):
        binder.join(
            fixture["fact"],
            fixture["main_evidence"],
            fixture["intervention"],
            fixture["gateb_evidence"],
        )


def test_f2_pixel_join_cli_writes_fresh_joined_report(tmp_path):
    fixture = _fixture(tmp_path)
    output = tmp_path / "joined.json"
    import subprocess
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "join_f2_offscreen_identity_pixel.py"),
            "--fact", str(fixture["fact_path"]),
            "--main-pixel-evidence", str(fixture["main_evidence"]),
            "--visual-intervention", str(fixture["intervention"]),
            "--gateb-pixel-evidence", str(fixture["gateb_evidence"]),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["status"] == "pass"
    value = json.loads(output.read_text())
    assert value["status"] == "research_candidate"
    assert value["joined_fact"]["qualification_claim"] is False
    assert value["report"]["inputs"]["main_pixel_truth"] == str(
        fixture["main_truth"].resolve()
    )
    second = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "join_f2_offscreen_identity_pixel.py"),
            "--fact", str(fixture["fact_path"]),
            "--main-pixel-evidence", str(fixture["main_evidence"]),
            "--visual-intervention", str(fixture["intervention"]),
            "--gateb-pixel-evidence", str(fixture["gateb_evidence"]),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert second.returncode == 2
    assert "refusing to overwrite" in second.stderr


def test_f2_pixel_join_rejects_wrong_evidence_selection_source(tmp_path):
    fixture = _fixture(tmp_path)
    wrong_selection = tmp_path / "wrong_selection.json"
    wrong_selection.write_text(json.dumps({
        "actors": [
            {"source_slot_id": "source1", "asset_id": "asset_other"},
            {"source_slot_id": "source2", "asset_id": "asset_burgundy"},
        ],
    }))
    evidence = json.loads(fixture["main_evidence"].read_text())
    evidence["inputs"]["actor_selection"] = wrong_selection.name
    wrong_evidence = tmp_path / "wrong_main_evidence.json"
    wrong_evidence.write_text(json.dumps(evidence))
    result = binder.join(
        fixture["fact"],
        wrong_evidence,
        fixture["intervention"],
        fixture["gateb_evidence"],
    )
    assert result["pixel_join_status"] == "pixel_rejected"
    assert result["report"]["checks"]["source_match"]["main"][
        "actor_selection"]["content_equal"] is False
    assert result["evidence_class"] == "pixel_rejected"


def test_f2_pixel_join_rejects_raw_truth_without_evidence_wrapper(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(binder.F2PixelJoinError, match="must have schema"):
        binder.join(
            fixture["fact"],
            fixture["main_truth"],
            fixture["intervention"],
            fixture["gateb_evidence"],
        )


def test_f2_pixel_join_allows_only_timeline_actor_selection_path_drift(tmp_path):
    fixture = _fixture(tmp_path)
    alternate_timeline = tmp_path / "timeline_v9.json"
    timeline = json.loads(fixture["main_timeline"].read_text())
    timeline["actor_selection"] = "/v8/actor_selection.json"
    alternate_timeline.write_text(json.dumps(timeline))
    evidence = json.loads(fixture["main_evidence"].read_text())
    evidence["inputs"]["timeline"] = alternate_timeline.name
    alternate_evidence = tmp_path / "main_evidence_v9.json"
    alternate_evidence.write_text(json.dumps(evidence))
    result = binder.join(
        fixture["fact"],
        alternate_evidence,
        fixture["intervention"],
        fixture["gateb_evidence"],
    )
    assert result["pixel_join_status"] == "pass"
    timeline_match = result["report"]["checks"]["source_match"]["main"]["timeline"]
    assert timeline_match["path_equal"] is False
    assert timeline_match["content_equal"] is True
