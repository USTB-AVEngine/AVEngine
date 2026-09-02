"""Generation-time Gate A checks for both MCQ and Open forms."""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "qa"))

from design_qa_v3_scene_batch import (  # noqa: E402
    GenerationConstraintError,
    audit_gatea_pair,
    balanced_binary_joint,
    build_cell_plan,
    build_answer,
    materialize_derived_params,
    realized_cross_time_checks,
    resolve_scene_render_context,
    validate_anchor_binding,
    validate_profiles,
    main as design_main,
)


PARAMS = {"THETA_HALF": 30.0, "T_HALF": 1.0, "T_FULL": 0.5,
          "T_FULL_status": "placeholder_research"}


def program(slots):
    endpoints = ["ep1", "ep2"]
    rows = []
    for index, endpoint in enumerate(slots):
        rows.append({
            "event_id": f"event_{index}_{endpoint}",
            "source_endpoint_id": endpoint,
            "sound_asset_id": "same-dry-sound",
            "start_sample": 1000 + index * 10000,
            "end_sample_exclusive": 5800 + index * 10000,
            "linear_gain": 0.18,
        })
    return {"candidate_source_endpoint_ids": endpoints, "events": rows}


def answer(mcq, value, scoring):
    return {
        "mcq": {"stem": "same question", "options_space": ["a", "b"],
                "truth_option": mcq},
        "open": {"stem": "same open question", "truth_value": value,
                 "scoring": scoring},
    }


def test_card1_requires_strictly_disjoint_open_credit_regions():
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    with pytest.raises(ValueError, match="open_gold_separated"):
        audit_gatea_pair(
            {"id": "card1F"}, main, gate,
            answer("left", 0.0, "circular_deg"),
            answer("right", 60.0, "circular_deg"), PARAMS)
    checks = audit_gatea_pair(
        {"id": "card1F"}, main, gate,
        answer("left", 0.0, "circular_deg"),
        answer("right", 60.001, "circular_deg"), PARAMS)
    assert checks["mcq_gold_flipped"]
    assert checks["open_gold_separated"]


def test_card8_uses_the_actual_time_scorer_threshold():
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    with pytest.raises(ValueError, match="open_gold_separated"):
        audit_gatea_pair(
            {"id": "card8"}, main, gate,
            answer("band0", 0.5, "absolute_time"),
            answer("band1", 1.5, "absolute_time"), PARAMS)
    checks = audit_gatea_pair(
        {"id": "card8"}, main, gate,
        answer("band0", 0.5, "absolute_time"),
        answer("band1", 1.501, "absolute_time"), PARAMS)
    assert checks["open_separation"] == pytest.approx(1.001)
    assert checks["open_min_separation"] == pytest.approx(1.0)
    assert "max(T_HALF, 2*T_FULL)" in checks["open_rule"]
    # a wider strict tolerance moves the derived minimum: 2 * 0.6 = 1.2
    with pytest.raises(ValueError, match="open_gold_separated"):
        audit_gatea_pair(
            {"id": "card8"}, main, gate,
            answer("band0", 0.5, "absolute_time"),
            answer("band1", 1.7, "absolute_time"), dict(PARAMS, T_FULL=0.6))
    with pytest.raises(ValueError, match="T_FULL"):
        audit_gatea_pair(
            {"id": "card8"}, main, gate,
            answer("band0", 0.5, "absolute_time"),
            answer("band1", 1.501, "absolute_time"), {"THETA_HALF": 30.0,
                                                      "T_HALF": 1.0})


CARD8_PARAMS = dict(PARAMS, GAP_MIN_S=0.3, FIRST_MIN_S=0.35,
                    CLIP_SECONDS=5.0, EVENT_SECONDS=0.3)
CARD8_PROFILE = {"id": "card8", "temporal": "instant",
                 "answer_kind": "time_band", "binding_frames": [12, 40],
                 "idle_choices": [0, 8], "anchor_binding": "first_caller"}


def _card8_answer(first_sample, second_sample, params=CARD8_PARAMS):
    slot_events = [("source1", first_sample), ("source2", second_sample)]
    schedule = SimpleNamespace(events=[object(), object()])
    return build_answer(
        "time_band", CARD8_PROFILE, {"target_band": 0}, None, schedule,
        slot_events, "source1", "source2",
        {"source1": "black-and-white", "source2": "yellow"}, 0.0, 12, params)


def test_card8_fact_records_scoring_chain_and_keeps_mcq_unaffected():
    result = _card8_answer(8000, 8000 + 20000)      # 0.5 s and 1.75 s
    open_block = result["open"]
    assert open_block["certification_policy"] == "strict_full_credit_only"
    assert open_block["wide_tolerance_role"] == "diagnostic_only"
    assert open_block["T_FULL"] == 0.5
    assert open_block["T_HALF"] == 1.0
    assert open_block["T_FULL_status"] == "placeholder_research"
    assert open_block["min_first_call_separation_s"] == pytest.approx(1.0)
    assert result["truth"]["first_call_separation_s"] == pytest.approx(1.25)
    # MCQ keeps its declared band answer space; the strict Open policy is
    # not attached to it.
    assert result["mcq"]["truth_option"] == "[0.35, 1.2875)"
    assert "certification_policy" not in result["mcq"]
    assert "wide_tolerance_role" not in result["mcq"]


def test_card8_fact_rejects_first_calls_not_strictly_above_minimum():
    with pytest.raises(GenerationConstraintError, match="not strictly above"):
        _card8_answer(8000, 8000 + 16000)          # exactly 1.0 s apart
    with pytest.raises(GenerationConstraintError, match="not strictly above"):
        _card8_answer(8000, 8000 + 17600, dict(CARD8_PARAMS, T_FULL=0.6))
    assert _card8_answer(8000, 8000 + 19201, dict(CARD8_PARAMS, T_FULL=0.6))[
        "truth"]["first_call_separation_above_minimum"] is True
    with pytest.raises(Exception, match="T_FULL"):
        _card8_answer(8000, 8000 + 20000,
                      {k: v for k, v in CARD8_PARAMS.items() if k != "T_FULL"})


def test_materialized_params_fail_closed_only_when_first_call_profiles_exist():
    base = {"BANDS_CARD8": [0.35, 1.1, 1.85, 2.6], "FIRST_MIN_S": 0.35,
            "GAP_MIN_S": 0.3, "T_HALF": 1.0, "CLIP_SECONDS": 5.0,
            "EVENT_SECONDS": 0.3}
    card1_only = [{"id": "card1F", "answer_kind": "azimuth_band"}]
    effective = materialize_derived_params(base, card1_only)
    assert "Not derived" in effective["BANDS_CARD8_note"]
    with pytest.raises(Exception, match="T_FULL"):
        materialize_derived_params(base, card1_only + [CARD8_PROFILE])
    with pytest.raises(Exception, match="T_FULL"):
        materialize_derived_params(base)
    derived = materialize_derived_params(dict(base, T_FULL=0.5), [CARD8_PROFILE])
    assert derived["BANDS_CARD8"] == [0.35, 1.2875, 2.225, 3.1625, 4.1]
    assert derived["CARD8_FIRST_CALL_SCORING"]["min_first_call_separation_s"] == 1.0


def _timeline(target_slot, other_slot, angles, distance=300.0):
    """75-frame timeline; camera at origin facing +x; angles per frame index."""
    import math
    frames = []
    for index in range(75):
        states = []
        for slot in (target_slot, other_slot):
            angle = angles[slot].get(index, 0.0)
            states.append({
                "source_slot_id": slot,
                "translation_ue_cm": [distance * math.cos(math.radians(angle)),
                                      distance * math.sin(math.radians(angle)),
                                      0.0]})
        frames.append({"frame_index": index,
                       "camera": {"translation_ue_cm": [0.0, 0.0, 147.0],
                                  "yaw_ue_deg": 0.0},
                       "actor_states": states})
    return {"frames": frames}


CARD1_PROFILE = {"id": "card1F", "temporal": "forward",
                 "answer_kind": "azimuth_band",
                 "answer_bands_deg": [[-52.5, -17.5], [-17.5, 17.5],
                                      [17.5, 52.5]],
                 "anchor_frame": 40, "idle_choices": [0], "anchor_binding": "target"}
CARD1_CELL = {"anchor_band": (-17.5, 17.5), "answer_band": (17.5, 52.5)}
CARD1_PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0}


def test_realized_timeline_rejects_a_plan_that_only_passed_on_paper():
    """Plan gap > THETA_HALF, realized gap <= THETA_HALF: must be refused."""
    plan_checks = {"az_anchor_deg": 9.415, "az_end_deg": 40.496}
    timeline = _timeline("source2", "source1", {
        "source2": {40: 15.0, 74: 40.5},      # realized gap 25.5 <= 30
        "source1": {40: -36.6, 74: -31.7},
    })
    with pytest.raises(GenerationConstraintError,
                       match="realized_anchor_answer_scores_zero") as exc:
        realized_cross_time_checks(
            timeline, profile=CARD1_PROFILE, cell=CARD1_CELL,
            target_slot="source2", other_slot="source1", anchor_frame=40,
            query_frame=74, params=CARD1_PARAMS, plan_checks=plan_checks)
    assert "planning value" in str(exc.value)


def test_realized_timeline_rejects_anchor_that_drifted_out_of_its_band():
    timeline = _timeline("source2", "source1", {
        "source2": {40: 20.0, 74: 52.0},      # gap 32 but anchor left the band
        "source1": {40: -36.6, 74: -31.7},
    })
    with pytest.raises(GenerationConstraintError,
                       match="realized_anchor_in_allocated_band"):
        realized_cross_time_checks(
            timeline, profile=CARD1_PROFILE, cell=CARD1_CELL,
            target_slot="source2", other_slot="source1", anchor_frame=40,
            query_frame=74, params=CARD1_PARAMS,
            plan_checks={"az_anchor_deg": 9.415, "az_end_deg": 40.496})


def test_realized_timeline_passes_and_reports_planning_deviation():
    timeline = _timeline("source2", "source1", {
        "source2": {40: 8.451, 74: 40.496},   # Kujiale card1F_002 realized
        "source1": {40: -36.576, 74: -31.711},
    })
    checks = realized_cross_time_checks(
        timeline, profile=CARD1_PROFILE, cell=CARD1_CELL,
        target_slot="source2", other_slot="source1", anchor_frame=40,
        query_frame=74, params=CARD1_PARAMS,
        plan_checks={"az_anchor_deg": 9.415, "az_end_deg": 40.496})
    assert checks["passed"] and checks["failed"] == []
    assert checks["provenance"] == "final_timeline_recompute_after_camera_pose"
    assert checks["main"]["anchor_azimuth_deg"] == pytest.approx(8.451, abs=1e-6)
    assert checks["main"]["query_azimuth_deg"] == pytest.approx(40.496, abs=1e-6)
    assert checks["gatea"]["query_azimuth_deg"] == pytest.approx(-31.711, abs=1e-6)
    assert checks["mcq_gold_flipped"] is True
    assert checks["open_gold_regions_disjoint"] is True
    deviation = checks["planned_vs_realized"]
    assert deviation["planned_anchor_azimuth_deg_planning_value_only"] == 9.415
    assert deviation["anchor_deviation_deg"] == pytest.approx(0.964, abs=1e-6)
    assert checks["realized_anchor_answer_scores_zero"] is True


def test_gatea_rejects_non_slot_audio_mutation():
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    gate = copy.deepcopy(gate)
    gate["events"][0]["linear_gain"] = 0.4
    with pytest.raises(ValueError, match="non_slot_event_fields_same"):
        audit_gatea_pair(
            {"id": "card9"}, main, gate,
            answer("black-and-white", "black-and-white", "closed_set"),
            answer("yellow", "yellow", "closed_set"), PARAMS)


def test_closed_set_gatea_flips_both_forms():
    checks = audit_gatea_pair(
        {"id": "card9"},
        program(["ep1", "ep2"]), program(["ep2", "ep1"]),
        answer("black-and-white", "black-and-white", "closed_set"),
        answer("yellow", "yellow", "closed_set"), PARAMS)
    assert checks["mcq_gold_flipped"]
    assert checks["open_gold_separated"]


def test_card2_instant_azimuth_stem_and_profile_validation():
    slot_coat = {"source1": "black-and-white", "source2": "yellow"}
    bands = [[-52.5, -17.5], [17.5, 52.5]]
    profile = {
        "id": "card2", "temporal": "instant",
        "answer_kind": "instant_azimuth_band",
        "binding_frames": [30], "idle_choices": [0, 8],
        "answer_bands_deg": bands, "anchor_binding": "query_caller",
    }
    validate_profiles([profile])
    result = build_answer(
        "instant_azimuth_band", profile,
        {"answer_band": (-52.5, -17.5)}, None, None, [],
        "source1", "source2", slot_coat, -30.0, 30, PARAMS)
    assert "frame index 30" in result["mcq"]["stem"]
    assert "dog barking at that frame" in result["mcq"]["stem"]
    assert result["mcq"]["truth_option"] == "[-52.5, -17.5)"
    assert result["open"]["scoring"] == "circular_deg"


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda gate: gate["events"].pop(), "event_count_same"),
        (lambda gate: gate.update(candidate_source_endpoint_ids=["x", "y"]),
         "candidate_endpoints_same"),
        (lambda gate: [event.update(source_endpoint_id=main)
                       for event, main in zip(
                           gate["events"], ["ep1", "ep2"])],
         "slot_sequence_changed"),
    ],
)
def test_each_gatea_structure_check_has_a_failing_control(mutation, failure):
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    mutation(gate)
    with pytest.raises(GenerationConstraintError, match=failure):
        audit_gatea_pair(
            {"id": "card9"}, main, gate,
            answer("black-and-white", "black-and-white", "closed_set"),
            answer("yellow", "yellow", "closed_set"), PARAMS)


def test_card1_stems_keep_the_audio_referent_and_time_explicit():
    slot_coat = {"source1": "black-and-white", "source2": "yellow"}
    cell = {"answer_band": (-52.5, -17.5)}
    bands = [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]]
    for temporal, query_frame, phrase in (
            ("forward", 74, "At the end of the video"),
            ("backward", 22,
             "At zero-based video frame index 22 (22/15 seconds)")):
        profile = {"id": f"card1-{temporal}", "temporal": temporal,
                   "answer_bands_deg": bands}
        main = build_answer(
            "azimuth_band", profile, cell, None, None, [], "source1",
            "source2", slot_coat, -30.0, query_frame, PARAMS)
        gate = build_answer(
            "azimuth_band", profile, cell, None, None, [], "source2",
            "source1", slot_coat, -30.0, query_frame, PARAMS)
        assert main["mcq"]["stem"] == gate["mcq"]["stem"]
        assert main["open"]["stem"] == gate["open"]["stem"]
        assert "dog that barked last" in main["mcq"]["stem"].lower()
        assert phrase in main["mcq"]["stem"]


def test_audit_rejects_a_changed_question_stem():
    main_answer = answer("black-and-white", "black-and-white", "closed_set")
    gate_answer = answer("yellow", "yellow", "closed_set")
    main_answer["mcq"].update(stem="same", options_space=["a", "b"])
    gate_answer["mcq"].update(stem="different", options_space=["a", "b"])
    main_answer["open"]["stem"] = gate_answer["open"]["stem"] = "same"
    with pytest.raises(GenerationConstraintError, match="mcq_stem_same"):
        audit_gatea_pair(
            {"id": "card9"}, program(["ep1", "ep2"]),
            program(["ep2", "ep1"]), main_answer, gate_answer, PARAMS)


def test_joint_allocator_covers_all_slot_coat_cells_for_six():
    rows = balanced_binary_joint(
        ["source1", "source2"], ["black-and-white", "yellow"], 6,
        "seed")
    assert set(rows) == {
        ("source1", "black-and-white"), ("source1", "yellow"),
        ("source2", "black-and-white"), ("source2", "yellow"),
    }


def test_joint_allocator_rotates_which_diagonal_receives_remainders():
    doubled = set()
    for seed in range(20):
        rows = balanced_binary_joint(
            ["source1", "source2"], ["black-and-white", "yellow"], 6,
            seed)
        counts = Counter(rows)
        doubled.add(tuple(sorted(cell for cell, count in counts.items()
                                 if count == 2)))
    assert len(doubled) == 2


def test_card1_allocates_slot_anchor_and_query_bands_jointly():
    bands = [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]]
    profile = {
        "id": "card1F", "temporal": "forward",
        "answer_kind": "azimuth_band", "answer_bands_deg": bands,
        "anchor_frame": 40, "idle_choices": [0, 8, 16],
        "anchor_binding": "target",
    }
    assets = [
        "generated_border_collie_black_white_medium_standard_adult_research_v1",
        "generated_labrador_yellow_medium_standard_adult_research_v1",
    ]
    rows = build_cell_plan(18, [profile], assets, {}, "room-seed")
    triples = {
        (row["target_slot"], tuple(row["anchor_band"]),
         tuple(row["answer_band"]))
        for row in rows
    }
    assert len(triples) == 18
    assert {tuple(row["anchor_band"]) for row in rows} == {
        tuple(band) for band in bands}
    assert {tuple(row["answer_band"]) for row in rows} == {
        tuple(band) for band in bands}


def test_unknown_anchor_binding_fails_instead_of_falling_through():
    with pytest.raises(GenerationConstraintError, match="unknown anchor_binding"):
        validate_anchor_binding(
            {"id": "bad", "anchor_binding": "first_callerr"},
            SimpleNamespace(), [], target_slot="source1", query_frame=0,
            answer={})


def test_render_context_requires_explicit_scene_map_and_transform():
    scene = SimpleNamespace(scene_id="new-room", render_config={})
    with pytest.raises(ValueError, match="no apartment fallback"):
        resolve_scene_render_context(scene)
    scene.render_config = {
        "native_map": "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000",
        "room_profile_id": "spear_debug_0000",
        "world_transform": "ue_xyz_cm_to_xzy_m_v1",
        "ground_z_ue_cm": 0.0,
    }
    resolved = resolve_scene_render_context(scene)
    assert resolved["native_map"].endswith("/debug_0000")
    assert resolved["world_transform"]([100, 200, 300]) == [1.0, 3.0, 2.0]
    assert resolved["ground_z_ue_cm"] == 0.0


def test_profile_typo_is_a_preflight_error():
    profile = {
        "id": "bad", "temporal": "instant", "answer_kind": "time_band",
        "anchor_binding": "first_callerr",
    }
    with pytest.raises(ValueError, match="invalid anchor_binding"):
        validate_profiles([profile])


def test_missing_ground_fails_before_output_directory_is_created(tmp_path):
    route_bank = tmp_path / "routes.json"
    route_bank.write_text(json.dumps({
        "schema": "avengine_apartment_route_bank_v1",
        "routes": [{"route_id": "r1", "implied_speed_mps": 0.5,
                    "samples_ue_cm": [[float(i), 0.0] for i in range(75)]}],
    }))
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({
        "primary_camera_rig": {
            "world_from_rig": {"translation_m": [0.0, 1.471, 0.0]},
            "shared_calibration": {"hfov_degrees": 105.0},
        },
        "listener": {"rig_from_listener": {"translation_m": [0.0, 0.0, 0.0]}},
    }))
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({
        "scene_id": "missing-ground", "backend": "ue_spear",
        "route_bank": str(route_bank), "camera_base_request": str(camera),
        "render": {"native_map": "/Game/Test/Map",
                   "room_profile_id": "test-room",
                   "world_transform": "ue_xyz_cm_to_xzy_m_v1"},
    }))
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps([{
        "id": "card1F", "temporal": "forward",
        "answer_kind": "azimuth_band", "anchor_binding": "target",
        "anchor_frame": 40, "idle_choices": [0],
        "answer_bands_deg": [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]],
    }]))
    params = tmp_path / "params.json"
    params.write_text("{}")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="ground_z_ue_cm"):
        design_main([
            "--scene-config", str(scene), "--profiles", str(profiles),
            "--params", str(params), "--out-root", str(output),
            "--seed", "test",
        ])
    assert not output.exists()


def test_card15b_gatea_preserves_count_gold_under_slot_swap():
    main_answer = answer(3, 3, "count_single")
    gate_answer = answer(3, 3, "count_single")
    checks = audit_gatea_pair(
        {"id": "card15b", "gatea_gold_relation": "preserve"},
        program(["ep1", "ep2", "ep1"]),
        program(["ep2", "ep1", "ep2"]),
        main_answer, gate_answer, PARAMS)
    assert checks["mcq_gold_flipped"] is False
    assert checks["mcq_gold_preserved"] is True
    assert checks["open_gold_preserved"] is True
    assert checks["mcq_gold_relation_satisfied"] is True
    assert checks["open_gold_relation_satisfied"] is True


def test_card15b_profile_and_count_answer_are_machine_checkable():
    profile = {
        "id": "card15b", "temporal": "instant",
        "answer_kind": "event_count", "binding_frames": [12, 40],
        "idle_choices": [0, 8], "answer_values": [3, 4],
        "anchor_binding": "none", "gatea_gold_relation": "preserve",
    }
    validate_profiles([profile])
    result = build_answer(
        "event_count", profile, {"answer_value": 3}, None,
        SimpleNamespace(events=[1, 2, 3]), [], "source1", "source2",
        {"source1": "black-and-white", "source2": "yellow"},
        0.0, 12, PARAMS)
    assert result["truth"]["event_count"] == 3
    assert result["mcq"]["truth_option"] == 3
    assert result["open"]["scoring"] == "count_single"
    assert validate_anchor_binding(
        profile, SimpleNamespace(), [], target_slot="source1",
        query_frame=12, answer=result)["selected_slot"] is None


def test_card4r_distance_answer_uses_final_timeline_frame():
    profile = {
        "id": "card4R", "temporal": "instant",
        "answer_kind": "distance_at_query", "binding_frames": [30],
        "idle_choices": [0, 8],
        "answer_labels": ["black-and-white", "yellow"],
        "min_distance_gap_cm": 50.0, "anchor_binding": "none",
        "gatea_gold_relation": "preserve",
    }
    validate_profiles([profile])
    frames = [{} for _ in range(31)]
    frames[30] = {
        "camera": {"translation_ue_cm": [0.0, 0.0, 147.0]},
        "actor_states": [
            {"source_slot_id": "source1",
             "translation_ue_cm": [100.0, 0.0, 0.0]},
            {"source_slot_id": "source2",
             "translation_ue_cm": [200.0, 0.0, 0.0]},
        ],
    }
    result = build_answer(
        "distance_at_query", profile, {}, {"frames": frames},
        SimpleNamespace(events=[1, 2, 3]), [], "source1", "source2",
        {"source1": "black-and-white", "source2": "yellow"},
        0.0, 30, PARAMS)
    assert result["truth"]["closer_coat"] == "black-and-white"
    assert result["truth"]["distance_gap_cm"] == 100.0
    assert result["mcq"]["truth_option"] == "black-and-white"
    assert result["open"]["scoring"] == "closed_set"
