from copy import deepcopy
import json
from pathlib import Path
import wave

import pytest

from avengine.qa.binding_conditions import (
    TASK_FAMILIES,
    TASK_QA_IDS,
    BindingConditionError,
    motion_scope_note,
    task_family_requirements,
    validate_binding_episode,
    verify_task_family_evidence,
)


def _fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    clock = {"frame_count": 10, "frame_rate_hz": 2, "sample_rate_hz": 100,
             "sample_count": 500, "time_base_hz": 48000, "ticks_per_frame": 24000,
             "duration_seconds": 5}
    camera = {"position_m": [0, 1.5, 0],
              "basis": {"forward": [0, 0, -1], "right": [1, 0, 0], "up": [0, 1, 0]}}
    native = {"schema": "avengine_neutral_readback_v1", "clock": clock,
              "coordinate_frame": {"linear_unit": "meter", "up_axis": "+Y", "handedness": "right"},
              "camera": [{**deepcopy(camera), "frame_index": i, "pts_ticks": i*24000} for i in range(10)],
              "entities": {actor: [{"root": [x, 0, -2], "emitter": [x, 1, -2], "moving": False,
                                    "frame_index": i, "pts_ticks": i*24000} for i in range(10)]
                           for actor, x in (("a1", -1), ("a2", 1))},
              "producer": {"source_readbacks": ["unit_fixture"]}}
    clip = tmp_path / "prepared.wav"
    with wave.open(str(clip), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(100)
        stream.writeframes(b"\x10\x00"*100)
    pool = {"sounds": [{"sound_asset_id": "s1", "path": str(clip), "sample_count": 100,
                        "sample_rate_hz": 100, "sound_class": "speech_playback", "gender": "M",
                        "compatible_asset_ids": ["human1", "human2"]}]}
    registry = {"assets": [{"asset_id": name, "entity_class": "articulated_human", "identity": {"species_id": "human"},
                            "realized_attributes": {"sex_or_gender_label": "male"}} for name in ("human1", "human2")]}
    pool_path, registry_path = write("pool.json", pool), write("registry.json", registry)
    request = {"frame_count": 10, "frame_rate_hz": 2, "sample_rate_hz": 100,
               "camera": {"motion": "static", "fov_deg": 85, "resolution_hw": [100, 100]},
               "profile": {"reserve_tail_s": 3}, "entities": {"total_count": 2},
               "sound_pool": str(pool_path), "source_registry": str(registry_path),
               "post_assembly_convolution_gain": 0.5}
    plan = {"clock": clock, "request": deepcopy(request), "visual_plan": {
        "camera": {"horizontal_fov_deg": 85},
        "frames": [{"camera_state": {"horizontal_fov_deg": 85}} for _ in range(10)]}}
    report = {"gain_application": {"post_assembly_convolution_gain": 0.5},
              "inputs": {"dry_assets": {"s1": {"path": str(clip)}}},
              "events": [{"event_id": "e1", "source_slice": {"start_sample": 0, "end_sample_exclusive": 100}}]}
    paths = {"plan": write("plan.json", plan), "neutral_readback": write("neutral.json", native),
             "research_report": write("report.json", report)}
    facts = {"time": clock, "actors": {"a1": {"asset_id": "human1"}, "a2": {"asset_id": "human2"}},
             "events": [{"event_id": "e1", "actor_id": "a1", "sound_asset_id": "s1", "end_s": 1.1}],
             "visibility_meta": {"resolution_hw": [100, 100]},
             "source_paths": {key: str(value) for key, value in paths.items()}}
    return facts, request, paths


def test_request_controls_duration_without_hardcoded_ten_seconds(tmp_path):
    facts, request, _ = _fixture(tmp_path)
    result = validate_binding_episode(facts, request=request, base=tmp_path, cache={})
    assert result["duration_seconds"] == 5
    changed = {**request, "frame_count": 20}
    with pytest.raises(BindingConditionError, match="frame_count"):
        validate_binding_episode(facts, request=changed, base=tmp_path, cache={})


def test_fixed_position_does_not_permit_camera_rotation(tmp_path):
    facts, request, paths = _fixture(tmp_path)
    native = json.loads(paths["neutral_readback"].read_text())
    native["camera"][5]["basis"] = {"forward": [0.5, 0, -3**0.5/2], "right": [3**0.5/2, 0, 0.5], "up": [0, 1, 0]}
    paths["neutral_readback"].write_text(json.dumps(native))
    with pytest.raises(BindingConditionError, match="camera.*changes"):
        validate_binding_episode(facts, request=request, base=tmp_path, cache={})


@pytest.mark.parametrize("kind", ["tail_rewritten", "tail_overrun", "old_pool", "incompatible", "gain", "clip_crop", "fov"])
def test_inconsistent_native_inputs_are_rejected(tmp_path, kind):
    facts, request, paths = _fixture(tmp_path)
    plan = json.loads(paths["plan"].read_text())
    report = json.loads(paths["research_report"].read_text())
    if kind == "tail_rewritten":
        plan["request"]["profile"]["reserve_tail_s"] = 0.5
    elif kind == "tail_overrun":
        facts["events"][0]["end_s"] = 4.8
    elif kind == "old_pool":
        facts["events"][0]["sound_asset_id"] = "superseded_old_sound"
    elif kind == "incompatible":
        facts["actors"]["a1"]["asset_id"] = "unregistered_or_incompatible_asset"
    elif kind == "gain":
        report["gain_application"]["post_assembly_convolution_gain"] = 1.0
    elif kind == "clip_crop":
        report["events"][0]["source_slice"]["start_sample"] = 10
    elif kind == "fov":
        plan["visual_plan"]["frames"][5]["camera_state"]["horizontal_fov_deg"] = 90
    paths["plan"].write_text(json.dumps(plan))
    paths["research_report"].write_text(json.dumps(report))
    with pytest.raises(BindingConditionError):
        validate_binding_episode(facts, request=request, base=tmp_path, cache={})


@pytest.mark.parametrize("speed,allowed", [(0.7, True), (0.6, False), (0.9, False)])
def test_native_walk_speed_is_measured_after_frame_quantization(tmp_path, speed, allowed):
    facts, request, paths = _fixture(tmp_path)
    request["binding_identity"] = {"walk_speed_range_mps": [0.65, 0.8]}
    native = json.loads(paths["neutral_readback"].read_text())
    for index, row in enumerate(native["entities"]["a1"]):
        row["root"][0] = -1 + index * speed / facts["time"]["frame_rate_hz"]
        row["emitter"][0] = row["root"][0]
        row["moving"] = True
    paths["neutral_readback"].write_text(json.dumps(native))
    if allowed:
        result = validate_binding_episode(facts, request=request, base=tmp_path, cache={})
        assert result["native_motion_speed_readback"]["a1"]["minimum_mps"] == pytest.approx(speed)
    else:
        with pytest.raises(BindingConditionError, match="native walk speed"):
            validate_binding_episode(facts, request=request, base=tmp_path, cache={})


# --------------------------------------------------------------------------- task families


def test_every_task_family_separates_question_recipe_and_group_conditions():
    for family in TASK_FAMILIES:
        requirements = task_family_requirements(family)
        assert requirements["qa_id"] == TASK_QA_IDS[family]
        layers = {"question": requirements["question_conditions"],
                  "recipe": requirements["recipe_conditions"],
                  "group": requirements["group_conditions"]}
        for name, rows in layers.items():
            assert rows, f"{family} declares no {name} condition"
            assert all(row["layer"] == name for row in rows)
            assert all(row["judge"] and row["detail"] for row in rows)
        keys = [row["key"] for rows in layers.values() for row in rows]
        assert len(keys) == len(set(keys))
        assert set(requirements["claim_layers"]) == {
            "ordinary_question_valid", "core_group_valid",
            "dual_modality_necessity", "human_answerability"}


def test_only_cross_time_state_places_motion_after_the_measured_tail():
    placements = {
        family: [row["key"] for row in task_family_requirements(family)["recipe_conditions"]]
        for family in TASK_FAMILIES
    }
    assert "motion_window_after_measured_wet_tail" in placements["cross_time_state"]
    for family in TASK_FAMILIES:
        if family == "cross_time_state":
            continue
        assert "motion_window_after_measured_wet_tail" not in placements[family]


def test_motion_scope_keeps_the_three_constraints_apart():
    scope = motion_scope_note()
    assert scope["during_target_audible_window"]["applies_to_qa_ids"] == ["QA-06", "QA-15"]
    assert scope["post_sound_query"]["applies_to_qa_ids"] == ["QA-13", "QA-16", "QA-17"]
    assert scope["recipe_motion_window_placement"]["applies_to_task_families"] == [
        "cross_time_state"]
    # The post-sound query constraint must not be stated as a motion placement.
    assert "placement" not in scope["post_sound_query"]
    assert (scope["recipe_motion_window_placement"]["placement"]
            != scope["during_target_audible_window"]["placement"])


def test_unknown_family_and_unrecorded_reading_both_fail_closed():
    with pytest.raises(BindingConditionError, match="unknown binding task family"):
        task_family_requirements("no_such_family")
    keys = [row["key"] for row in
            task_family_requirements("visible_binding")["question_conditions"]]
    complete = {key: {"checked": True} for key in keys}
    assert verify_task_family_evidence(complete, task_family="visible_binding")["status"] == "pass"
    incomplete = dict(complete)
    incomplete.pop(keys[-1])
    with pytest.raises(BindingConditionError, match="no recorded reading"):
        verify_task_family_evidence(incomplete, task_family="visible_binding")
    with pytest.raises(BindingConditionError, match="unknown .* condition readings"):
        verify_task_family_evidence({**complete, "invented": {}},
                                    task_family="visible_binding")


def test_verified_evidence_does_not_promote_the_later_claims():
    keys = [row["key"] for row in
            task_family_requirements("cross_time_state")["question_conditions"]]
    result = verify_task_family_evidence(
        {key: {"checked": True} for key in keys}, task_family="cross_time_state")
    assert result["core_group_valid"] == "not_run"
    assert result["dual_modality_necessity"] == "not_run"
    assert result["human_answerability"] == "not_run"


def test_episode_validation_records_the_clip_and_crop_that_were_rendered(tmp_path):
    facts, request, paths = _fixture(tmp_path)
    pool_path = Path(request["sound_pool"])
    pool = json.loads(pool_path.read_text())
    pool["sounds"][0].update({
        "prepared_audio_id": "prepared_speech_band_deadbeef_v1",
        "source_relative": "speech/original_recording.wav",
        "source_sha256": "a" * 64,
        "source_crop_start_sample": 15040,
        "source_crop_end_sample_exclusive": 15140,
        "source_offset_s": 150.4,
        "source_crop_duration_s": 1.0,
        "source_normalization": "peak_dbfs_-3",
        "sound_identity_id": "identity_deadbeef",
    })
    pool_path.write_text(json.dumps(pool))
    result = validate_binding_episode(facts, request=request, base=tmp_path, cache={})
    identity = result["audio_input_identity"]["e1"]
    assert identity["crop_provenance"] == "sound_pool_row"
    assert identity["actor_id"] == "a1"
    assert identity["source_crop_start_sample"] == 15040
    assert identity["source_crop_end_sample_exclusive"] == 15140
    assert identity["rendered_slice"] == {"start_sample": 0, "end_sample_exclusive": 100}


def test_a_pool_row_without_crop_coordinates_is_reported_as_partial(tmp_path):
    facts, request, _ = _fixture(tmp_path)
    result = validate_binding_episode(facts, request=request, base=tmp_path, cache={})
    identity = result["audio_input_identity"]["e1"]
    assert identity["crop_provenance"] == "partial"
    assert "source_crop_start_sample" in identity["crop_fields_absent_in_pool_row"]
