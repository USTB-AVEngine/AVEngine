from __future__ import annotations

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
    assert candidate["fact"]["scene_id"] == "house.glb"
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
