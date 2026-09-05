from __future__ import annotations

import json
from pathlib import Path

from avengine.qa.mp3d_candidate_adapter import adapt_mp3d_candidate


def _inputs():
    tracks = []
    for index in range(2):
        tracks.append(
            {
                "actor_id": f"actor{index + 1}",
                "source_slot_id": f"source{index + 1}",
                "source_endpoint_id": f"endpoint{index + 1}",
                "semantic_id": 10 + index,
                "asset": {"asset_id": f"asset{index + 1}", "revision": "r1"},
                "emitter": {"anchor_id": "muzzle", "joint_id": "j0"},
                "frames": [
                    {
                        "frame_index": 0,
                        "pts_ticks": 0,
                        "action_id": "idle",
                        "planned_route_center_m": [float(index), 0.0, 1.0],
                        "planned_world_from_skin_root": {
                            "translation_m": [float(index), 0.0, 1.0],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            }
        )
    case = {
        "schema": "avengine_mp3d_region_actor_track_case_v1",
        "native_observed": False,
        "clock": {
            "frame_count": 1,
            "frame_rate_hz": 15,
            "ticks_per_frame": 3200,
            "time_base_hz": 48000,
            "sample_rate_hz": 16000,
            "sample_count": 1067,
        },
        "region": {"house_id": "house", "region_index": 0},
        "route_family_id": "family",
        "motion_case": "both_moving",
        "tracks": tracks,
    }
    room = {
        "room_id": "habitat_native_house",
        "room_kind": "habitat_native",
        "scene": {"scene_id": "house.glb"},
    }
    m1 = {
        "room_id": "habitat_native_house",
        "primary_camera_rig": {
            "world_from_rig": {
                "translation_m": [0.0, 1.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "sources": [
            {"source_id": "endpoint1"},
            {"source_id": "endpoint2"},
        ],
    }
    return case, room, m1


def test_adapter_preserves_planned_boundary_and_reports_missing_runtime_artifacts():
    candidate = adapt_mp3d_candidate(*_inputs())
    assert candidate["backend_id"] == "habitat_native"
    assert candidate["fact"]["backend_inputs"] == {
        "case_manifest": None,
        "room_manifest": None,
        "m1_request": None,
        "audio_program": None,
    }
    assert candidate["fact"]["scene_id"] == "house.glb"
    assert candidate["fact"]["runtime_consumer_status"] == "pending_question_facts"
    assert candidate["timeline"]["artifact_role"] == "planned_timeline_not_native_capture"
    assert candidate["fact"]["tracks"]["source1"]["planned_route_center_m_by_frame"] == [[0.0, 0.0, 1.0]]
    assert candidate["fact"]["tracks"]["source1"]["observed_emitter_position_m_by_frame"] is None
    assert "audio_program" in candidate["missing"]
    assert any("observed native emitter" in item for item in candidate["missing"])
    assert candidate["position_authority"]["observed"].endswith("source_positions_m")


def test_adapter_does_not_require_two_or_numbered_source_slots():
    case, room, m1 = _inputs()
    track = case["tracks"][0]
    track["source_slot_id"] = "moving_primary"
    track["source_endpoint_id"] = "mouth_anchor"
    case["tracks"] = [track]
    m1["sources"] = [{"source_id": "mouth_anchor"}]
    candidate = adapt_mp3d_candidate(case, room, m1)
    assert [
        actor["source_slot_id"] for actor in candidate["selection"]["actors"]
    ] == ["moving_primary"]
    assert len(candidate["timeline"]["frames"]) == 1



def test_adapter_fact_records_point_owned_runtime_input_paths(tmp_path: Path):
    case, room, m1 = _inputs()
    case_path = tmp_path / "case.json"
    room_path = tmp_path / "room.json"
    m1_path = tmp_path / "m1.json"
    program_path = tmp_path / "program.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    room_path.write_text(json.dumps(room), encoding="utf-8")
    m1_path.write_text(json.dumps(m1), encoding="utf-8")
    program_path.write_text(json.dumps({
        "program_id": "program",
        "candidate_source_endpoint_ids": ["endpoint1", "endpoint2"],
        "events": [],
    }), encoding="utf-8")

    candidate = adapt_mp3d_candidate(
        case_path,
        room_path,
        m1_path,
        audio_program_path=program_path,
    )

    assert candidate["fact"]["backend_inputs"] == {
        "case_manifest": str(case_path.resolve()),
        "room_manifest": str(room_path.resolve()),
        "m1_request": str(m1_path.resolve()),
        "audio_program": str(program_path.resolve()),
    }



def test_adapter_rejects_symlinked_manifest(tmp_path: Path):
    case, room, m1 = _inputs()
    target = tmp_path / "case-target.json"
    target.write_text(json.dumps(case), encoding="utf-8")
    alias = tmp_path / "case-link.json"
    alias.symlink_to(target)

    try:
        adapt_mp3d_candidate(alias, room, m1)
    except ValueError as error:
        assert "must not be a symlink" in str(error)
    else:
        raise AssertionError("symlinked case manifest was accepted")


def test_adapter_rejects_relative_track_escape(tmp_path: Path):
    case, room, m1 = _inputs()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    outside_track = tmp_path / "outside-track.json"
    outside_track.write_text(
        json.dumps(case["tracks"][0]), encoding="utf-8"
    )
    case["actor_tracks"] = [{"track_path": "../outside-track.json"}]
    case.pop("tracks")
    case_path = case_dir / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")

    try:
        adapt_mp3d_candidate(case_path, room, m1)
    except ValueError as error:
        assert "escapes the case directory" in str(error)
    else:
        raise AssertionError("case-relative track escape was accepted")
