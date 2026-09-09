from copy import deepcopy
import math

import pytest

from test_qa_unified_catalog import _fixture
from avengine.qa.angular_questions import candidates, public_camera_calibration
from avengine.qa.unified_catalog import generate_unified_questions, normalize_episode_bundle, model_input_questions, iter_unified_items
from avengine.qa.unified_scoring import score_unified_question_set, score_unified_item


def _angle_fixture():
    raw = _fixture()
    raw["camera_calibration"] = {"projection": "pinhole", "public": True,
        "width_px": 100, "height_px": 100, "fx_px": 50.0, "cx_px": 49.5}
    for actor_id, actor in raw["pixel_visibility_truth"]["per_instance"].items():
        for row in actor["frames"]:
            if actor_id == "a0" and row["frame_index"] >= 8:
                row["state"] = "fully_occluded"
            if row["state"] == "visible_clear":
                row.update(visible_fraction=1.0, target_centroid_xy_px=[74.5, 50.0])
    for field in ("actors", "emitters"):
        for row in raw["frame_readbacks"][field]["a0"]:
            frame = row["frame_index"]
            if frame >= 8:
                row["position_m"] = [-5 + 0.05 * (frame - 8) ** 2, 0.0, -2.0]
    raw["audio_program"]["events"][0]["start_sample"] = 0
    raw["audio_program"]["events"][0]["end_sample_exclusive"] = 28800
    raw["audio_readback"]["source_activity_intervals_samples"] = {
        event["event_id"]: [{"start_sample": event["start_sample"], "end_sample_exclusive": event["end_sample_exclusive"]}]
        for event in raw["audio_program"]["events"]}
    return raw


def test_all_three_subsets_have_explicit_numeric_questions_and_public_visual_calibration():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-25"])
    by_subset = {item["angle_subset"]: item for item in result["items"]}
    assert set(by_subset) == {"A", "V", "AV"}
    assert result["counts"]["valid"] == 3
    assert by_subset["V"]["forms"]["open"]["truth"] == round(math.degrees(math.atan2(25, 50)))
    for subset, item in by_subset.items():
        assert set(item["forms"]) == {"open"}
        assert item["forms"]["open"]["answer_type"] == "angle_deg"
        assert item["certification"]["status"] == "not_run"
        assert "[-180, 180)" in item["model_input"]["open"]["question_en"]
        assert ("camera_calibration" in item["model_input"]["open"]) == (subset == "V")
    assert len(by_subset["AV"]["evidence"]["competing_event_ids"]) >= 1
    public = model_input_questions(result)
    for row in public["items"]:
        assert row["question_id"].startswith("question_")
        assert "source" not in row["question_id"] and "event" not in row["question_id"]
        assert set(row) == {"question_id", "qa_id", "forms", "required_modalities"}
        assert set(row["forms"]["open"]) <= {"question_en", "question_zh", "camera_calibration"}


@pytest.mark.parametrize("change", ["no_competitor", "same_sound", "static_target", "linear_target", "no_anchor"])
def test_av_does_not_emit_unobservable_or_single_source_shortcuts(change):
    facts = normalize_episode_bundle(_angle_fixture())
    if change == "no_competitor":
        for event in facts["events"][1:]:
            event["source_activity_intervals_samples"] = []
    if change == "same_sound":
        for event in facts["events"]:
            event["sound_asset_id"] = "same_sound"
    if change in {"static_target", "linear_target"}:
        for field in ("root_positions_m", "emitter_positions_m"):
            facts["actors"]["a0"][field] = [[f * 0.1 if change == "linear_target" else 0, 0, -2] for f in range(40)]
    if change == "no_anchor":
        for row in facts["visibility"]["a0"].values():
            row["state"] = "fully_occluded"
    assert not any(row["subset"] == "AV" for row in candidates(facts))


@pytest.mark.parametrize("change", ["private_calibration", "wrong_resolution", "occluded_centroid", "no_activity"])
def test_missing_native_evidence_is_not_invented(change):
    facts = normalize_episode_bundle(_angle_fixture())
    if change == "private_calibration":
        facts["camera_calibration"]["public"] = False
    elif change == "wrong_resolution":
        facts["camera_calibration"]["width_px"] = 200
    elif change == "occluded_centroid":
        for rows in facts["visibility"].values():
            for row in rows.values():
                row["visible_fraction"] = 0.5
    else:
        for event in facts["events"]:
            event["source_activity_intervals_samples"] = []
    subsets = {row["subset"] for row in candidates(facts)}
    assert (not subsets & {"A", "AV"}) if change == "no_activity" else "V" not in subsets


def test_followups_are_explicit_linked_and_do_not_duplicate_qa13():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-03", "QA-20", "QA-13", "QA-24"])
    parents = {item["question_id"] for item in result["items"]}
    assert result["angle_followups"]
    for item in result["angle_followups"]:
        assert item["parent_question_id"] in parents
        assert item["qa_id"] in {"QA-03", "QA-20", "QA-24"}
        assert item["forms"]["open"]["answer_type"] == "angle_deg"
    assert len(list(iter_unified_items(result))) == len(result["items"]) + len(result["angle_followups"])
    assert generate_unified_questions(_angle_fixture(), qa_ids=["QA-03"], include_angle_followups=False)["angle_followups"] == []
    # A hidden silent final target has no deterministic angle followup.
    assert not any(item["qa_id"] == "QA-24" for item in result["angle_followups"])


def test_continuous_metrics_wraparound_missing_answers_and_legacy_tolerances():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-25"])
    rows = result["items"]
    for item in rows:
        item["forms"]["open"]["truth"] = 179.0
    answers = {rows[0]["question_id"]: "-179 degrees", rows[1]["question_id"]: "cannot tell"}
    report = score_unified_question_set(result, answers)
    metrics = report["angle_metrics"]
    assert metrics["total"] == 3 and metrics["parsed"] == 1
    assert metrics["mae_deg"] == 2.0 and metrics["median_deg"] == 2.0
    assert metrics["accuracy_at_deg"] == {"1": 0.0, "3": 1 / 3, "5": 1 / 3, "10": 1 / 3}
    assert score_unified_item(rows[0], "-179 degrees")["score"] == pytest.approx(1 - 2 / 180)
    legacy = deepcopy(rows[0])
    legacy["forms"]["open"].pop("scoring_mode")
    legacy["forms"]["open"]["truth"] = 0
    assert score_unified_item(legacy, "20 degrees")["score"] == 0.5


def test_wrong_instance_cannot_pass_joint_metric_with_correct_angle():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-03"])
    parent = result["items"][0]
    child = result["angle_followups"][0]
    wrong = next(key for key in parent["forms"]["open"]["classes"] if key != parent["truth"]["value"])
    answers = {parent["question_id"]: wrong, child["question_id"]: str(child["forms"]["open"]["truth"])}
    metrics = score_unified_question_set(result, answers)["angle_metrics"]
    assert metrics["mae_deg"] == 0
    assert metrics["instance_binding_joint"]["total"] == 1
    assert metrics["instance_binding_joint"]["joint_accuracy_at_deg"]["5"] == 0


def test_numeric_only_questions_do_not_inflate_mcq_denominator():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-03", "QA-25"])
    assert result["angle_followups"] and len(result["items"]) == 4
    report = score_unified_question_set(result, {}, form="mcq")
    assert report["counts"]["total"] == 1


def test_pure_visual_subset_does_not_require_audio():
    raw = _angle_fixture()
    raw.pop("audio_program")
    raw.pop("audio_readback")
    result = generate_unified_questions(raw, qa_ids=["QA-25"])
    assert [item["angle_subset"] for item in result["items"]] == ["V"]


def test_public_answer_ids_round_trip_without_exposing_private_target_ids():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-25"])
    public = model_input_questions(result)
    answers = {row["question_id"]: str(item["forms"]["open"]["truth"])
               for row, item in zip(public["items"], result["items"])}
    score = score_unified_question_set(result, answers)
    assert score["angle_metrics"]["parsed"] == 3
    assert score["angle_metrics"]["accuracy_at_deg"]["1"] == 1


def test_correct_parent_and_angle_pass_joint_metric():
    result = generate_unified_questions(_angle_fixture(), qa_ids=["QA-03"])
    parent, child = result["items"][0], result["angle_followups"][0]
    option = next(option for option in parent["forms"]["mcq"]["options"] if option["value"] == parent["truth"]["value"])
    answers = {"question_000001": option["label_en"], "question_000002": str(child["truth"]["value"])}
    metrics = score_unified_question_set(result, answers)["angle_metrics"]
    assert metrics["instance_binding_joint"]["parent_accuracy"] == 1
    assert metrics["instance_binding_joint"]["joint_accuracy_at_deg"]["1"] == 1


def test_fine_audio_bearing_uses_full_listener_pitch_and_emitter_readback():
    from avengine.qa.angular_questions import _audio_bearing
    from avengine.qa.unified_catalog import _Deferred
    facts = normalize_episode_bundle(_angle_fixture())
    basis = {"right": [1.0, 0.0, 0.0], "forward": [0.0, -0.5, -math.sqrt(3) / 2],
             "up": [0.0, math.sqrt(3) / 2, -0.5]}
    facts["listener"]["basis_m3"][0] = basis
    facts["actors"]["a0"]["emitter_positions_m"][0] = [1.0, -0.5, -math.sqrt(3) / 2]
    assert _audio_bearing(facts, "a0", 0) == pytest.approx(45.0)
    facts["actors"]["a0"]["emitter_positions_m"] = None
    with pytest.raises(_Deferred, match="actual sound emitter"):
        _audio_bearing(facts, "a0", 0)


def test_competing_reverberant_tail_is_not_treated_as_a_silent_anchor():
    raw = _angle_fixture()
    event = raw["audio_program"]["events"][1]
    event.update(start_sample=0, end_sample_exclusive=1000)
    raw["audio_readback"]["source_activity_intervals_samples"]["e1"] = [{"start_sample": 0, "end_sample_exclusive": 1000}]
    raw["audio_readback"]["wet_tail_intervals"][1].update(start_s=0, end_s=2)
    facts = normalize_episode_bundle(raw)
    assert not any(row["subset"] == "AV" for row in candidates(facts))


def test_native_visual_calibration_uses_measured_image_dimensions(tmp_path):
    import json
    from avengine.qa.angular_questions import camera_calibration_from_capture
    (tmp_path / "visual_plan.json").write_text(json.dumps({"camera": {"horizontal_fov_deg": 90}, "frames": []}))
    (tmp_path / "pixel_visibility_truth.json").write_text(json.dumps({"resolution_hw": [720, 1280]}))
    result = camera_calibration_from_capture(tmp_path)
    assert result["public"] is True
    assert result["fx_px"] == pytest.approx(640.0)
    assert result["cx_px"] == 639.5


def test_generator_cli_accepts_normalized_native_facts(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    source = tmp_path / "facts.json"
    source.write_text(json.dumps(normalize_episode_bundle(_angle_fixture())))
    output, public = tmp_path / "questions.json", tmp_path / "model_inputs.json"
    repository = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(repository / "tools/qa/generate_unified_questions.py"),
                    "--input", str(source), "--qa-ids", "QA-25", "--out", str(output),
                    "--model-inputs-out", str(public)], check=True, capture_output=True, text=True)
    assert json.loads(output.read_text())["counts"]["valid"] == 3
    assert json.loads(public.read_text())["items"][0]["question_id"] == "question_000001"


def test_integer_queries_recompute_angles_at_selected_native_frames():
    import re
    from avengine.qa.angular_questions import _audio_bearing, _visual_bearing
    raw = _angle_fixture()
    facts = normalize_episode_bundle(raw)
    result = generate_unified_questions(facts, qa_ids=["QA-25"])
    for item in result["items"]:
        frame = item["evidence"]["query_frame"]
        assert frame % 10 == 0
        assert isinstance(item["truth"]["value"], int)
        assert not re.search(r"\d+\.\d+", item["question"]["en"] + item["question"]["zh"])
        actor = item["evidence"]["actor_id"]
        exact = (_visual_bearing(facts, actor, frame)[0] if item["angle_subset"] == "V"
                 else _audio_bearing(facts, actor, frame))
        assert item["truth"]["value"] == (round(exact) + 180) % 360 - 180
        assert item["evidence"]["answer_full_precision"] == pytest.approx(exact)
        if item["angle_subset"] == "AV":
            assert item["evidence"]["anchor_frame"] % 10 == 0


def test_default_time_ranges_use_whole_seconds_without_expanding_evidence():
    from avengine.qa.unified_catalog import _display_time_bounds, _display_time_range
    facts = {"time": {"frame_rate_hz": 10}}
    assert _display_time_bounds(facts, [12, 38]) == (2, 3)
    assert _display_time_bounds(facts, [12, 18]) is None
    assert _display_time_range(facts, [12, 38])[1] == "第2至第3秒的时间段（左闭右开）"


def test_default_public_questions_and_answers_have_integer_time_and_angle_units():
    import re
    raw = _angle_fixture()
    raw.pop("sampling", None)
    result = generate_unified_questions(raw)
    for item in iter_unified_items(result):
        for form in item["model_input"].values():
            for field in ("question_en", "question_zh"):
                assert not re.search(r"\d+\.\d+", form[field])
            for option in form.get("options", []):
                assert not re.search(r"\d+\.\d+", option["label_en"] + option["label_zh"])
        form = item["forms"].get("open", {})
        if form.get("answer_type") in {"angle_deg", "time_s"}:
            assert isinstance(form["truth"], int)
        if form.get("answer_type") == "time_range_s":
            assert all(isinstance(value, int) for value in form["truth"])


def test_integer_time_bands_classify_against_the_displayed_boundaries():
    from avengine.qa.unified_catalog import _time_bands
    facts = {"time": {"duration_seconds": 10}}
    assert _time_bands(facts) == [(0, 2), (2, 5), (5, 8), (8, 10)]
