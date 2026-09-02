"""Card1 conditional tables and best-response baselines are descriptive only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from report_qa_v3_card1_conditional_baseline import (  # noqa: E402
    analyse_group,
    analyse_items,
    analyse_smoke_report,
    item_from_fact,
    main,
)

LABELS = ["[-52.5, -17.5)", "[-17.5, 17.5)", "[17.5, 52.5)"]
CENTRES = {"[-52.5, -17.5)": -35.0, "[-17.5, 17.5)": 0.0, "[17.5, 52.5)": 35.0}
THETA = {"THETA_FULL": 15.0, "THETA_HALF": 30.0}


def _fact(point_id, room, anchor_label, answer_label, *, coat="yellow",
          moves_more=True, realized=True, profile="card1F"):
    anchor = CENTRES[anchor_label]
    query = CENTRES[answer_label]
    other = CENTRES[next(label for label in LABELS
                         if label not in (answer_label,))]
    fact = {
        "variant": "main", "point_id": point_id, "scene_id": room,
        "profile_id": profile, "target_coat": coat,
        "motion": {"target_moves_more": moves_more},
        "mcq": {"options_space": LABELS, "truth_option": answer_label},
        "truth": {"query_azimuth_deg": query, "other_slot_azimuth_deg": other},
        "generation_checks": {"az_anchor_deg": anchor + 0.9},
    }
    if realized:
        fact["realized_generation_checks"] = {
            "main": {"anchor_azimuth_deg": anchor}}
    return fact


def _items(facts, split="unsplit"):
    return [item_from_fact(fact, room=fact["scene_id"], split=split,
                           source="test") for fact in facts]


def test_empty_same_band_cells_lift_audio_only_baseline_above_one_third():
    facts, index = [], 0
    for anchor in LABELS:
        for answer in LABELS:
            if anchor == answer:
                continue            # same-band diagonal structurally empty
            for _ in range(2):
                facts.append(_fact(f"p{index}", "room_a", anchor, answer,
                                   moves_more=bool(index % 2),
                                   coat=("yellow" if index % 4 < 2
                                         else "black-and-white")))
                index += 1
    group = analyse_group(_items(facts), theta_full=15.0, theta_half=30.0,
                          step=0.5)
    audio = group["by_missing_modality"]["video"]["form"]["mcq"]
    assert audio["best_response_conditional_baseline"] == pytest.approx(0.5)
    assert audio["structural_exclusion_uniform_baseline"] == pytest.approx(0.5)
    assert audio["nominal_uniform_baseline"] == pytest.approx(1 / 3)
    assert audio["best_response_conditional_baseline"] > 1 / 3
    empty = group["joint_table_anchor_by_answer"]["structural_empty_cells"]
    assert len(empty) == 3 and all(cell["same_band_diagonal"] for cell in empty)
    assert group["anchor_angle_sources"] == {"realized_generation_checks": 12}
    open_audio = group["by_missing_modality"]["video"]["form"]["open"]
    # Under the actual two-tier scorer the Open best response is higher than
    # the MCQ one: the centre stratum has clusters at -35/+35 (one angle covers
    # only one cluster -> 0.5), but the outer strata have clusters 35 deg
    # apart, where an angle 5 deg from one cluster scores 1.0 there and 0.5
    # (within THETA_HALF) at the other -> 0.75.  (0.5*4 + 0.75*8) / 12 = 2/3.
    assert open_audio["best_response_conditional_expected_score"] == \
        pytest.approx(2 / 3)
    assert open_audio["repeat_anchor_angle_expected_score"] == 0.0


def test_conditionally_uniform_table_returns_theoretical_values():
    facts, index = [], 0
    for anchor in LABELS:
        for answer in LABELS:
            facts.append(_fact(f"p{index}", "room_a", anchor, answer,
                               moves_more=bool(index % 2),
                               coat=("yellow" if (index // 2) % 2 == 0
                                     else "black-and-white")))
            index += 1
    detail, rows = analyse_items(_items(facts), theta_full=15.0,
                                 theta_half=30.0, step=0.5)
    group = detail["card1F|unsplit"]
    audio = group["by_missing_modality"]["video"]["form"]["mcq"]
    assert audio["best_response_conditional_baseline"] == pytest.approx(1 / 3)
    assert audio["structural_exclusion_uniform_baseline"] == pytest.approx(1 / 3)
    text = group["by_missing_modality"]["audio_and_video"]["form"]["mcq"]
    assert text["majority_answer_baseline"] == pytest.approx(1 / 3)
    assert group["joint_table_anchor_by_answer"]["structural_empty_cells"] == []
    assert len(rows) == 6          # 3 missing modalities x 2 forms
    assert {row["missing_modality"] for row in rows} == {
        "video", "audio", "audio_and_video"}
    assert all(row["split"] == "unsplit" for row in rows)


def test_video_only_rules_are_scored_against_both_visible_dogs():
    # target is always the yellow dog: a video-only "pick yellow" rule is perfect
    facts = [_fact(f"p{i}", "room_a", LABELS[i % 3], LABELS[(i + 1) % 3],
                   coat="yellow", moves_more=bool(i % 2)) for i in range(6)]
    group = analyse_group(_items(facts), theta_full=15.0, theta_half=30.0,
                          step=0.5)
    video = group["by_missing_modality"]["audio"]
    assert video["per_rule"]["pick_yellow"]["mcq_accuracy"] == 1.0
    assert video["per_rule"]["pick_black_and_white"]["mcq_accuracy"] == 0.0
    assert video["form"]["mcq"]["best_single_rule_accuracy"] == 1.0
    assert video["structural_two_candidate_baseline"] == 0.5
    assert video["covariate_balance"]["target_coat"] == {"yellow": 6}


def test_per_room_report_does_not_hide_a_dominant_room():
    facts = [_fact(f"a{i}", "room_a", LABELS[0], LABELS[2]) for i in range(10)]
    facts += [_fact(f"b{i}", "room_b", LABELS[1], LABELS[0]) for i in range(2)]
    group = analyse_group(_items(facts), theta_full=15.0, theta_half=30.0,
                          step=0.5)
    rooms = group["rooms"]
    assert rooms["room_count"] == 2
    assert rooms["max_single_room_share"] == pytest.approx(10 / 12)
    assert rooms["rooms"]["room_a"]["n"] == 10
    assert rooms["rooms"]["room_b"]["audio_only_mcq_best_response"] == 1.0


def test_inputs_with_model_outcomes_are_refused():
    fact = _fact("p0", "room_a", LABELS[0], LABELS[2])
    fact["model_answer"] = "[17.5, 52.5)"
    with pytest.raises(ValueError, match="model outcome"):
        item_from_fact(fact, room="room_a", split="unsplit", source="test")
    smoke = {"results": {"room": {"card1F": {
        "anchor_answer_band_distribution": {}, "accuracy": 0.9}}}}
    with pytest.raises(ValueError, match="model outcome"):
        analyse_smoke_report(smoke)


def test_planned_anchor_is_used_only_when_no_realized_record_exists():
    item = item_from_fact(_fact("p0", "room_a", LABELS[1], LABELS[2],
                                realized=False),
                          room="room_a", split="unsplit", source="test")
    assert item["anchor_source"] == "planned_solver_value_no_realized_record"
    assert item["anchor_azimuth_deg"] == pytest.approx(0.9)


def test_smoke_report_reproduction_recomputes_best_response_from_table():
    smoke = {"results": {"apartment_0000": {"card1F": {
        "anchor_answer_band_distribution": {
            "(-17.5, 17.5) -> (-17.5, 17.5)": 2,
            "(-17.5, 17.5) -> (-52.5, -17.5)": 2,
            "(-17.5, 17.5) -> (17.5, 52.5)": 2,
            "(-52.5, -17.5) -> (-17.5, 17.5)": 2,
            "(-52.5, -17.5) -> (17.5, 52.5)": 2,
            "(17.5, 52.5) -> (-17.5, 17.5)": 2,
            "(17.5, 52.5) -> (-52.5, -17.5)": 2,
        }}}}}
    out = analyse_smoke_report(smoke)["apartment_0000|card1F"]
    assert out["n"] == 14
    assert out["audio_only_mcq_best_response_fraction"] == "6/14"
    assert out["audio_only_mcq_best_response"] == pytest.approx(6 / 14)
    assert len(out["structural_empty_cells"]) == 2
    assert out["open_baselines"] is None


def test_cli_writes_report_with_params_snapshot_and_refuses_clobber(tmp_path):
    facts = tmp_path / "facts.jsonl"
    facts.write_text("\n".join(json.dumps(_fact(
        f"p{i}", "room_a", LABELS[i % 3], LABELS[(i + 1) % 3]))
        for i in range(6)) + "\n")
    params = tmp_path / "params.json"
    params.write_text(json.dumps(THETA))
    output = tmp_path / "report.json"
    assert main(["--facts", str(facts), "--params", str(params),
                 "--grid-step-deg", "1.0", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["params_snapshot"]["THETA_HALF"] == 30.0
    assert report["is_gate"] is False
    assert report["item_count"] == 6
    assert len(report["rows"]) == 6
    assert report["inputs"][0]["kind"] == "facts"
    assert main(["--facts", str(facts), "--params", str(params),
                 "--output", str(output)]) == 2
