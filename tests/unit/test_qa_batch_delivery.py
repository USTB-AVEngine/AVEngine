from copy import deepcopy
import pytest
import inspect
from avengine.qa.batch_delivery import (
    achieved_from_facts,
    attach_visibility_semantics,
    finalize_batch_episode,
    _review_frames,
)
from avengine.rooms.qa_delivery import finalize_qa_episode


def facts_fixture():
    basis = {"forward": [0, 0, 1], "right": [1, 0, 0], "up": [0, 1, 0]}
    return {"time": {"frame_count": 10, "frame_rate_hz": 1, "sample_rate_hz": 10,
                     "sample_count": 100, "duration_seconds": 10},
            "listener": {"positions_m": [[0, 0, 0]] * 10, "basis_m3": [basis] * 10},
            "actors": {"source1": {"emitter_positions_m": [[0, 1, 2]] * 10, "moving": [False] * 10},
                       "source2": {"emitter_positions_m": [[1, 1, 2]] * 10, "moving": [False] * 10}},
            "events": [{"event_id": "e1", "actor_id": "source1", "sound_asset_id": "s1",
                        "source_activity_intervals_samples": [{"start_sample": 0, "end_sample_exclusive": 20}]},
                       {"event_id": "e2", "actor_id": "source2", "sound_asset_id": "s2",
                        "source_activity_intervals_samples": [{"start_sample": 30, "end_sample_exclusive": 50}]}],
            "visibility": {actor: {str(i): {"state": "visible_clear", "target_pixels": 8} for i in range(10)}
                           for actor in ["source1", "source2"]},
            "audio": {"wet_tail_intervals": [{"end_s": 6}]}}
PROFILE = {"total_count": 2, "anchor_indices": [0], "separation_bin_deg": [15, 30], "separation_floor_deg": 15}


def test_achieved_fields_use_actual_emitters_and_preserve_unmeasured_los():
    facts = facts_fixture()
    result = achieved_from_facts(facts, PROFILE, {})
    assert result["camera_static"] is True
    assert result["maximum_concurrent_source_activity"] == 1
    assert result["minimum_gap_between_event_activity_spans_s"] == 1
    anchor = result["anchor_event_measurements"][0]
    assert anchor["separation"]["inside_requested_bin_all_frames"] is True
    assert anchor["static_emitter_los_counts"] == {"unmeasured": 2}
    assert anchor["body_proxy_los"]["status"] == "unmeasured"
    assert result["profile_certified"] is False
    changed = deepcopy(facts)
    changed["actors"]["source2"]["emitter_positions_m"] = [[0.01, 1, 2]] * 10
    changed["planned_conditions"] = {"separation": 25}
    actual = achieved_from_facts(changed, PROFILE, {})
    assert actual["anchor_event_measurements"][0]["separation"]["inside_requested_bin_all_frames"] is False


def test_missing_activity_does_not_become_measured_silence():
    facts = facts_fixture()
    del facts["events"][0]["source_activity_intervals_samples"]
    result = achieved_from_facts(facts, PROFILE, {})
    assert result["source_activity_missing_event_ids"] == ["e1"]
    assert result["speaking_count"] is None
    assert result["anchor_event_measurements"] == []


def test_review_frames_keep_frame_zero_and_the_actual_query_endpoint():
    questions = {"items": [{"question_id": "q", "evidence": {"query_frame": 0}},
                           {"question_id": "q2", "evidence": {"post_sound": {"query_frame": 8}}}]}
    result = _review_frames(questions, facts_fixture())
    assert set(result) == {0, 8}
    assert "q:query_frame" in result[0]
    assert "q2:query_frame" in result[8]
    with pytest.raises(ValueError, match="query frame"):
        _review_frames({"items": [{"question_id": "bad", "evidence": {"query_frame": 10}}]}, facts_fixture())



def test_attach_visibility_semantics_writes_pixel_and_achieved_fields() -> None:
    facts = facts_fixture()
    achieved = achieved_from_facts(facts, PROFILE, {})
    assert "visible_pixel_frames" not in achieved["anchor_event_measurements"][0]
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "resolution_hw": [20, 20],
        "per_instance": {
            "source1": {
                "semantic_id": 11,
                "frames": [
                    {
                        "frame_index": 0,
                        "target_pixels": 8,
                        "visible_pixels": 8,
                        "target_bbox_xyxy_px": [0, 0, 4, 4],
                        "state": "visible_clear",
                    },
                    {
                        "frame_index": 1,
                        "target_pixels": 8,
                        "visible_pixels": 0,
                        "target_bbox_xyxy_px": [0, 0, 4, 4],
                        "state": "fully_occluded",
                    },
                ],
            }
        },
    }
    annotated_achieved, annotated_truth = attach_visibility_semantics(achieved, truth)
    source = annotated_truth["per_instance"]["source1"]
    assert source["visible_pixel_frames"] == 1
    assert source["bbox_touches_frame_edge_frames"] == 2
    assert source["in_fov_frame_count"] == 2
    row = annotated_achieved["anchor_event_measurements"][0]
    assert row["visible_pixel_frames"] == 1
    assert row["bbox_touches_frame_edge_frames"] == 2
    assert row["in_fov_frame_count"] == 2


def test_delivery_call_sites_wire_visibility_annotators() -> None:
    batch_src = inspect.getsource(finalize_batch_episode)
    assert "attach_visibility_semantics" in batch_src
    assert "annotate_pixel_visibility_semantics" in inspect.getsource(attach_visibility_semantics)
    qa_src = inspect.getsource(finalize_qa_episode)
    assert "annotate_pixel_visibility_semantics" in qa_src

def test_exposure_gate_import_error_fails_review(monkeypatch, tmp_path):
    from avengine.qa import batch_delivery as module

    def _boom():
        raise ImportError("simulated missing avengine.qa.exposure_gate")

    monkeypatch.setattr(module, "_import_apply_exposure_gate", _boom)
    result = module.apply_review_exposure_gate(
        {"status": "delivered", "episode_id": "x"}, tmp_path
    )
    assert result["status"] == "review_failed"
    assert result["exposure_gate"]["status"] == "unavailable"
    assert "unavailable" in result["reason"]
    assert result["frame_source"]["kind"] == "unavailable"

