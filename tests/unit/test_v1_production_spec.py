"""One V1 configuration must yield ordinary Episodes and the four core groups.

These are hermetic contract checks on parsing and the stage protocol. They do
not execute a planner, renderer, RLR propagation or media readback, so they
prove nothing about native feasibility or delivered coverage.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from avengine.dataset.production_spec import (
    CORE_TASK_FAMILIES,
    GROUP_RECIPES,
    MOTION_TIMINGS,
    SCHEMA,
    STAGE_PUBLISHED_FACTS,
    STAGE_RESOURCE_KIND,
    STAGE_RESOURCE_KINDS,
    AudioLayoutSpec,
    EntityInstanceSpec,
    EventSelector,
    ProductionSpecError,
    QaTargetSpec,
    ResourceRequest,
    RetryPolicy,
    StageResult,
    group_blockers,
    group_round_state,
    group_stage_units,
    group_unit_scope_id,
    initial_group_work_items,
    initial_stage_work_items,
    measured_motion_window,
    next_group_work_items,
    next_stage_work_items,
    parse_production_config,
    production_request_from_legacy,
    recipe_for_task_family,
    request_round_state,
    retry_stage_work_item,
    stage_protocol_summary,
    work_item_id,
)
from avengine.qa.unified_catalog import QA_IDS

HUMAN = "articulated_human"
DEVICE = "rigid_static_object"


def instances(*asset_ids, source_class=HUMAN, speaking=None):
    return [
        {
            "instance_id": f"source{index + 1}",
            "asset_id": asset_id,
            "source_class": source_class,
            **({} if speaking is None else {"speaking": speaking[index]}),
        }
        for index, asset_id in enumerate(asset_ids)
    ]


def core_group(group_id, task_family, room_id, *, members=4):
    return {
        "group_id": group_id,
        "task_family": task_family,
        "room_id": room_id,
        "members": [
            {
                "request_id": f"{group_id}_m{index + 1}",
                "member_role": f"member_{index + 1}",
                "instances": instances("human_blue_v1", "human_green_v1"),
            }
            for index in range(members)
        ],
        "shared_audio_member_ids": [[f"{group_id}_m1", f"{group_id}_m2"]],
        "shared_visual_member_ids": [[f"{group_id}_m1", f"{group_id}_m3"]],
    }


@pytest.fixture
def config():
    return {
        "schema": SCHEMA,
        "batch_id": "v1_smoke",
        "seed": 20260910,
        "defaults": {
            "clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000},
            "rig": {"resolution_hw": [720, 1280], "fov_deg": 85, "height_above_floor_m": 1.55},
            "audio_layouts": [{"type": "binaural", "role": "primary", "indirect_sh_order": 1}],
            "reserve_tail_s": 3.0,
            "post_assembly_convolution_gain": 0.5,
            "resources": {
                "graphics_adapter": 0,
                "capture": {"min_free_vram_mb": 12000, "rpc_port": 39782},
                "audio": {"rlr_threads": 1},
                "audio_tail_probe": {"rlr_threads": 1},
            },
            "retry": {"attempts_per_stage": 2, "attempts_within_profile": 200},
            "sound": {"pool": "/pool/batch_sounds.json", "selection": {"max_clip_s": 5.0}},
            "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1},
            # The values a retained cross-time request actually declares.
            "request_extras": {"binding_motion": {
                "minimum_motion_s": 2.0, "end_hold_s": 0.5, "angle_tolerance_deg": 10,
                "minimum_entity_separation_m": 0.95, "source_start_s": 0.1,
                "walk_speed_range_mps": [0.5, 0.8]}},
        },
        "episodes": [
            {
                "request_id": "v1_episode_0001",
                "room_id": "legacy_ue_apartment_0000_v1",
                "instances": instances("human_blue_v1", "human_green_v1"),
            }
        ],
        "core_groups": [
            core_group("g_visible", "visible_binding", "legacy_ue_apartment_0000_v1"),
            core_group("g_relation", "visual_conditioned_relation", "kujiale_0020_full_home_v1"),
            core_group("g_identity", "cross_event_identity", "hm3d_val_00800_TEEsavR23oF"),
            core_group("g_state", "cross_time_state", "habitat_mp3d_example_17DRP5sb8fy"),
        ],
        "coverage_quota": {"min_main_questions_per_qa_id": 8, "min_worlds_per_qa_id": 2},
    }


def test_one_config_yields_an_episode_and_four_core_groups(config):
    before = deepcopy(config)
    parsed = parse_production_config(config)
    assert config == before
    assert len(parsed.episodes) == 1
    assert parsed.episodes[0].kind == "episode"
    assert parsed.episodes[0].task_family is None
    assert [group.task_family for group in parsed.core_groups] == list(CORE_TASK_FAMILIES)
    assert {group.room_id for group in parsed.core_groups} == {
        "legacy_ue_apartment_0000_v1", "kujiale_0020_full_home_v1",
        "hm3d_val_00800_TEEsavR23oF", "habitat_mp3d_example_17DRP5sb8fy",
    }
    assert all(len(group.members) == 4 for group in parsed.core_groups)
    assert len(parsed.all_requests()) == 17
    assert parsed.coverage_quota["min_main_questions_per_qa_id"] == 8
    assert len(parsed.request_by_id()) == 17
    for member in parsed.core_groups[0].members:
        assert member.kind == "core_group_member"
        assert member.group_id == "g_visible"
        assert member.member_role
    assert len({member.seed for member in parsed.core_groups[0].members}) == 1


def test_core_group_member_seed_conflict_is_rejected(config):
    config["core_groups"] = [config["core_groups"][0]]
    config["core_groups"][0]["members"][1]["seed"] = 999
    with pytest.raises(ProductionSpecError, match="differs in seed"):
        parse_production_config(config)


def test_full_qa_ids_and_real_targets_reach_every_request(config):
    parsed = parse_production_config(config)
    request = parsed.episodes[0]
    assert request.qa_ids == tuple(QA_IDS)
    assert len(request.qa_ids) == 25
    assert len(request.qa_targets) == 25
    for target in request.qa_targets:
        assert target.target_instance_ids == ("source1", "source2")
        assert target.event.kind == "target_audible_window"
        assert target.event.ordinal is None
        assert target.target_source == "derived_from_speaking_instances"
    legacy = request.to_legacy_request()
    assert request.qa_targets_source == "derived_for_catalog"
    assert request.qa_intent_source == "qa_ids_only"
    assert legacy["qa_ids"] == list(QA_IDS)
    assert "qa_targets" not in legacy
    item = initial_stage_work_items(request)[0]
    assert item.payload["qa_intent_source"] == "qa_ids_only"
    assert "qa_targets" not in item.payload


def test_declared_targets_name_the_entity_and_the_event(config):
    config["episodes"][0]["qa_ids"] = ["QA-05", "QA-13"]
    config["episodes"][0]["qa_targets"] = [
        {"qa_id": "QA-05", "target_instance_ids": ["source1", "source2"],
         "event": {"kind": "event_ordinal", "ordinal": 2}, "items": 3, "forms": ["mcq"]},
        {"qa_id": "QA-13", "target_instance_ids": ["source2"], "event": "clip_tail", "items": 2},
    ]
    request = parse_production_config(config).episodes[0]
    first, second = request.qa_targets
    assert (first.qa_id, first.event.kind, first.event.ordinal, first.items) == ("QA-05", "event_ordinal", 2, 3)
    assert first.forms == ("mcq",)
    assert (second.qa_id, second.event.kind, second.target_instance_ids) == ("QA-13", "clip_tail", ("source2",))
    assert request.quota_by_qa == {"QA-05": 3, "QA-13": 2}
    assert request.quota_source == "derived_from_qa_targets"
    assert request.qa_targets_declared is True
    assert request.qa_targets_source == "explicit"
    assert request.qa_intent_source == "explicit_conditions"
    assert request.to_legacy_request()["qa_targets"][0]["qa_id"] == "QA-05"


def test_qa_ids_only_round_trip_does_not_emit_catalog_targets(config):
    request = parse_production_config(config).episodes[0]
    legacy = request.to_legacy_request()
    reparsed = production_request_from_legacy(legacy)
    assert request.qa_targets_source == "derived_for_catalog"
    assert reparsed.qa_targets_source == "derived_for_catalog"
    assert "qa_targets" not in legacy
    assert "qa_targets" not in reparsed.to_legacy_request()
    assert reparsed.to_legacy_request() == legacy


def test_explicit_branch_and_drive_round_trip_without_catalog_targets(config):
    config["episodes"][0]["qa_ids"] = ["QA-06"]
    config["episodes"][0]["question_branches"] = {"QA-06": "moving"}
    config["episodes"][0]["drive_sampler_from_questions"] = True
    request = parse_production_config(config).episodes[0]
    legacy = request.to_legacy_request()
    assert request.qa_intent_source == "explicit_conditions"
    assert "qa_targets" not in legacy
    assert legacy["question_branches"] == {"QA-06": "moving"}
    assert legacy["drive_sampler_from_questions"] is True
    reparsed = production_request_from_legacy(legacy)
    assert reparsed.qa_intent_source == "explicit_conditions"
    assert reparsed.to_legacy_request() == legacy


def test_unknown_legacy_target_source_remains_explicit(config):
    config["episodes"][0]["qa_ids"] = ["QA-05"]
    config["episodes"][0]["qa_targets"] = [{
        "qa_id": "QA-05",
        "target_instance_ids": ["source1"],
        "event": {"kind": "target_audible_window"},
        "target_source": "legacy_unknown_source",
    }]
    request = parse_production_config(config).episodes[0]
    legacy = request.to_legacy_request()
    assert legacy["qa_targets"][0]["target_source"] == "legacy_unknown_source"
    reparsed = production_request_from_legacy(legacy)
    assert reparsed.qa_targets_declared is True
    assert reparsed.to_legacy_request()["qa_targets"][0]["target_source"] == (
        "legacy_unknown_source"
    )


def test_config_quota_is_data_not_algorithm(config):
    config["episodes"][0]["qa_ids"] = ["QA-05", "QA-25"]
    config["episodes"][0]["quota_by_qa"] = {"QA-05": 8, "QA-25": 3}
    request = parse_production_config(config).episodes[0]
    assert request.quota_by_qa == {"QA-05": 8, "QA-25": 3}
    assert request.quota_source == "config"
    assert request.to_legacy_request()["quota_by_qa"] == {"QA-05": 8, "QA-25": 3}


def test_explicit_duration_resolution_fov_and_resources_reach_the_work_item(config):
    config["episodes"][0].update({
        "clock": {"frame_count": 300, "frame_rate_hz": 30, "sample_rate_hz": 48000},
        "rig": {"resolution_hw": [1080, 1920], "fov_deg": 60},
        "reserve_tail_s": 4.0,
        "resources": {"capture": {"graphics_adapter": 1, "min_free_vram_mb": 24000, "rpc_port": 40001}},
    })
    request = parse_production_config(config).episodes[0]
    assert request.duration_seconds == pytest.approx(10.0)
    assert request.clock.frame_count == 300 and request.clock.sample_count == 480000
    assert request.available_program_seconds == pytest.approx(6.0)

    item = initial_stage_work_items(request)[0]
    assert item.payload["clock"]["frame_count"] == 300
    assert item.payload["clock"]["frame_rate_hz"] == 30
    assert item.payload["clock"]["sample_rate_hz"] == 48000
    assert item.payload["duration_seconds"] == pytest.approx(10.0)
    assert item.payload["reserve_tail_s"] == pytest.approx(4.0)
    assert item.payload["available_program_seconds"] == pytest.approx(6.0)
    assert item.payload["rig"]["resolution_hw"] == [1080, 1920]
    assert item.payload["rig"]["fov_deg"] == 60
    assert item.inputs["request"]["frame_count"] == 300
    assert item.inputs["request"]["camera"] == {
        "fov_deg": 60.0, "resolution_hw": [1080, 1920], "motion": "static",
        "height_above_floor_m": 1.55}

    capture = next_stage_work_items(request, [_plan_pass(request)])[0]
    assert capture.stage == "capture"
    assert capture.resource.to_dict() == {
        "kind": "gpu_native_visual", "execution": "gpu",
        "runtime_context": "renderer_native", "graphics_adapter": 1,
        "min_free_vram_mb": 24000, "rpc_port": 40001}
    assert capture.payload["rig"]["resolution_hw"] == [1080, 1920]


def test_shared_resource_block_reaches_each_stage(config):
    request = parse_production_config(config).episodes[0]
    assert request.resource_for("plan").to_dict() == {
        "kind": "cpu", "execution": "cpu", "runtime_context": "pure_python",
        "graphics_adapter": 0}
    assert request.resource_for("capture").to_dict() == {
        "kind": "gpu_native_visual", "execution": "gpu",
        "runtime_context": "renderer_native", "graphics_adapter": 0,
        "min_free_vram_mb": 12000, "rpc_port": 39782}
    for stage, resource in request.resources.items():
        assert resource.kind in STAGE_RESOURCE_KINDS[stage]


def test_acoustics_defaults_to_a_cpu_slot_inside_the_native_context(config):
    """RLR propagates on the CPU, so an audio column must not hold a GPU slot."""
    request = parse_production_config(config).episodes[0]
    audio = request.resource_for("audio")
    assert audio.kind == "cpu_native_acoustic"
    assert audio.execution == "cpu"
    assert audio.runtime_context == "rlr_native"
    assert audio.to_dict() == {
        "kind": "cpu_native_acoustic", "execution": "cpu",
        "runtime_context": "rlr_native", "graphics_adapter": 0, "rlr_threads": 1}
    assert request.resource_for("capture").execution == "gpu"
    assert request.resource_for("plan").runtime_context == "pure_python"


def test_a_gpu_acoustic_backend_stays_declarable(config):
    """The CPU default must not permanently rule out a GPU acoustic backend."""
    config["defaults"]["resources"]["audio"] = {
        "kind": "gpu_native_acoustic", "min_free_vram_mb": 4000, "rlr_threads": 1}
    request = parse_production_config(config).episodes[0]
    audio = request.resource_for("audio")
    assert audio.kind == "gpu_native_acoustic"
    assert (audio.execution, audio.runtime_context) == ("gpu", "rlr_native")
    assert audio.min_free_vram_mb == 4000


def test_an_explicit_cpu_acoustic_declaration_is_accepted(config):
    config["defaults"]["resources"]["audio"] = {"kind": "cpu_native_acoustic", "rlr_threads": 1}
    request = parse_production_config(config).episodes[0]
    assert request.resource_for("audio").execution == "cpu"


def test_a_late_plan_may_ask_for_the_native_geometry_context(config):
    config["defaults"]["resources"]["late_plan"] = {"kind": "cpu_native_geometry"}
    request = parse_production_config(config).episodes[0]
    late = request.resource_for("late_plan")
    assert (late.execution, late.runtime_context) == ("cpu", "habitat_native")


@pytest.mark.parametrize("stage,kind,message", [
    ("audio", "gpu_native_visual", "audio.kind must be one of"),
    ("capture", "cpu", "capture.kind must be one of"),
    ("plan", "gpu_native_acoustic", "plan.kind must be one of"),
    ("assembly", "cpu_native_acoustic", "assembly.kind must be one of"),
])
def test_a_resource_kind_outside_the_stage_set_fails(config, stage, kind, message):
    config["defaults"]["resources"][stage] = {"kind": kind}
    with pytest.raises(ProductionSpecError, match=message):
        parse_production_config(config)


def test_a_cpu_slot_cannot_reserve_vram(config):
    config["defaults"]["resources"]["audio"] = {"kind": "cpu_native_acoustic",
                                                 "min_free_vram_mb": 8000}
    with pytest.raises(ProductionSpecError, match="runs on the CPU"):
        parse_production_config(config)


def test_resources_kind_at_the_root_is_refused(config):
    config["defaults"]["resources"]["kind"] = "cpu"
    with pytest.raises(ProductionSpecError, match="kind is per stage"):
        parse_production_config(config)


def test_two_instances_may_share_one_asset(config):
    config["episodes"][0]["instances"] = instances("human_blue_v1", "human_blue_v1")
    request = parse_production_config(config).episodes[0]
    assert request.instance_count == 2
    assert request.distinct_asset_ids == ("human_blue_v1",)
    assert request.shares_assets_across_instances is True
    assert [item["instance_id"] for item in request.to_legacy_request()["entity_instances"]] == [
        "source1", "source2"]
    assert request.to_legacy_request()["source_asset_ids"] == ["human_blue_v1", "human_blue_v1"]
    assert request.to_legacy_request()["entities"]["total_count"] == 2


def test_distinct_assets_do_not_claim_sharing(config):
    request = parse_production_config(config).episodes[0]
    assert request.shares_assets_across_instances is False
    assert len(request.distinct_asset_ids) == request.instance_count


PLAN_FACTS = {"episode_plan_path": "plan/episode_plan.json", "renderer": "habitat"}
CAPTURE_FACTS = {"capture_receipt_path": "v0_capture.json", "captured_frame_count": 150}
DELIVERY_FACTS = {"questions_path": "q.json", "facts_path": "f.json"}


def audio_facts(*ends, member="v0_a0"):
    return {"facts_path": f"{member}/facts.json",
            "audio_report_path": f"{member}/audio_report.json",
            "wet_tail_intervals": [{"start_s": max(0.0, end - 1.0), "end_s": end}
                                   for end in ends]}


def _plan_pass(request, *, attempt=1):
    return StageResult(
        work_item_id=work_item_id(request.request_id, "plan", attempt),
        stage="plan", request_id=request.request_id, status="pass",
        facts={**PLAN_FACTS, "clock": request.clock.to_dict()},
    )


def unit_result(group, unit_id, stage, *, attempt=1, status="pass", facts=None,
                depends_on=(), reason=None):
    scope = group_unit_scope_id(group.group_id, unit_id)
    return StageResult(
        work_item_id=work_item_id(scope, stage, attempt),
        stage=stage, request_id=scope, status=status,
        facts={} if facts is None else facts,
        depends_on=tuple(depends_on),
        reason=reason if status != "pass" else None,
    )


def state_group(config):
    parsed = parse_production_config(config)
    return next(g for g in parsed.core_groups if g.task_family == "cross_time_state")


def test_initial_items_are_planning_only_for_an_ordinary_episode(config):
    request = parse_production_config(config).episodes[0]
    items = initial_stage_work_items(request)
    assert [item.stage for item in items] == ["plan"]
    assert items[0].depends_on == ()
    assert items[0].resource.execution == "cpu"
    assert items[0].fresh_output_relative == f"{request.request_id}/plan/attempt_01"
    assert request.stage_plan() == ("plan", "capture", "audio", "delivery")


def test_a_core_member_has_no_schedule_of_its_own(config):
    """Four members are one group's cross combinations, not four worlds."""
    member = parse_production_config(config).core_groups[0].members[0]
    with pytest.raises(ProductionSpecError, match="core group member"):
        member.stage_plan()
    with pytest.raises(ProductionSpecError, match="initial_group_work_items"):
        initial_stage_work_items(member)


def test_a_standalone_after_tail_episode_is_refused(config):
    """The after-tail shape needs early audio columns, so it is group only."""
    config["core_groups"] = []
    config["episodes"][0]["motion_timing"] = "after_wet_tail"
    request = parse_production_config(config).episodes[0]
    with pytest.raises(ProductionSpecError, match="only exists as a core group"):
        request.stage_plan()


def test_the_cross_time_recipe_matches_the_real_group_builder(config):
    recipe = recipe_for_task_family("cross_time_state")
    assert recipe.motion_timing == "after_wet_tail"
    assert "binding_group_motion.py" in recipe.source
    assert [unit.unit_id for unit in recipe.units_in_dependency_order()] == [
        "v0", "v0_capture", "v0_a0", "v0_a1", "v1", "v1_capture", "v1_a0", "v1_a1", "group"]
    late = recipe.unit("v1")
    assert late.stage == "late_plan"
    assert late.consumes_wet_tails_of == ("v0_a0", "v0_a1")
    assert recipe.member_unit_ids == ("v0_a0", "v0_a1", "v1_a0", "v1_a1")
    assert recipe.unit("v0_a1").visual_unit_id == "v0_capture"
    assert recipe.unit("v1_a0").visual_unit_id == "v1_capture"


def test_the_parallel_two_visual_recipe_shares_one_visual_per_two_members(config):
    recipe = recipe_for_task_family("visible_binding")
    assert recipe.motion_timing == "none"
    assert [unit.unit_id for unit in recipe.units_in_dependency_order()] == [
        "v0", "v1", "v0_capture", "v1_capture", "v0_a0", "v0_a1", "v1_a0", "v1_a1", "group"]
    assert recipe.unit("v0").depends_on_units == ()
    assert recipe.unit("v1").depends_on_units == ()
    assert not any(unit.consumes_wet_tails_of for unit in recipe.units)


def test_one_visual_unit_serves_the_two_members_that_share_it(config):
    group = parse_production_config(config).core_groups[0]
    units = {row["unit_id"]: row for row in group_stage_units(group)}
    member_ids = [member.request_id for member in group.members]
    assert units["v0_capture"]["member_request_ids"] == member_ids[:2]
    assert units["v1_capture"]["member_request_ids"] == member_ids[2:]
    assert units["v0_a0"]["member_request_ids"] == [member_ids[0]]
    assert units["group"]["member_request_ids"] == member_ids
    assert units["v0_capture"]["scope_id"] == f"{group.group_id}/v0_capture"


def test_the_state_group_reaches_a_late_plan_only_from_real_early_tails(config):
    group = state_group(config)
    first = initial_group_work_items(group)
    assert [item.unit_id for item in first] == ["v0"]
    assert first[0].stage == "plan"

    v0 = unit_result(group, "v0", "plan",
                     facts={**PLAN_FACTS, "clock": group.members[0].clock.to_dict()})
    after_plan = next_group_work_items(group, [v0])
    assert [item.unit_id for item in after_plan] == ["v0_capture"]
    assert after_plan[0].resource.execution == "gpu"
    assert after_plan[0].member_request_ids == tuple(
        member.request_id for member in group.members[:2])

    cap = unit_result(group, "v0_capture", "capture", facts=CAPTURE_FACTS,
                      depends_on=(v0.work_item_id,))
    after_capture = next_group_work_items(group, [v0, cap])
    assert sorted(item.unit_id for item in after_capture) == ["v0_a0", "v0_a1"]
    assert all(item.resource.execution == "cpu" for item in after_capture)
    assert all(item.resource.runtime_context == "rlr_native" for item in after_capture)

    # One early column alone is not enough: the late plan reads both.
    a0 = unit_result(group, "v0_a0", "audio", facts=audio_facts(3.1, member="v0_a0"),
                     depends_on=(cap.work_item_id,))
    assert [item.unit_id for item in next_group_work_items(group, [v0, cap, a0])] == ["v0_a1"]

    a1 = unit_result(group, "v0_a1", "audio", facts=audio_facts(2.6, 3.4, member="v0_a1"),
                     depends_on=(cap.work_item_id,))
    after_early = next_group_work_items(group, [v0, cap, a0, a1])
    assert [item.unit_id for item in after_early] == ["v1"]
    late = after_early[0]
    assert late.stage == "late_plan"
    window = late.payload["measured_motion_window"]
    assert window["measured_wet_end_s"] == pytest.approx(3.4)
    assert window["first_motion_frame"] == math.ceil(3.4 * 15) + 1
    assert window["boundary_formula"] == "ceil(max(wet_tail end_s) * fps) + 1"
    assert window["last_motion_frame"] == 150 - 1 - 8
    assert window["minimum_motion_frames"] == 30
    assert window["sufficient"] is True
    assert window["authority"] == "actual early binaural readbacks"
    assert window["requested_terminal_tail_s"] == 3.0
    assert sorted(window["wet_tail_source_work_item_ids"]) == sorted(
        [a0.work_item_id, a1.work_item_id])
    assert "measured_tail_seconds" not in window


def test_tail_length_absolute_time_and_frames_are_named_apart(config):
    clock = parse_production_config(config).episodes[0].clock
    window = measured_motion_window(
        wet_tail_end_seconds_by_unit={"v0_a0": [4.0], "v0_a1": [4.2]},
        clock=clock, minimum_motion_s=2.0, end_hold_s=0.5, reserve_tail_s=3.0,
        source_work_item_ids=["g/v0_a0:audio:01"])
    assert window["measured_wet_end_s"] == pytest.approx(4.2)
    assert window["measured_wet_end_sample"] == 67200
    assert window["first_motion_frame"] == 64
    assert window["last_motion_frame"] == 141
    assert window["available_motion_frames"] == 78
    assert window["frame_rate_hz"] == 15.0
    assert "is an absolute clip time, not a tail duration" in window["measured_tail_seconds_note"]


def test_a_tail_that_leaves_no_movement_time_blocks_the_group(config):
    group = state_group(config)
    clock = group.members[0].clock
    v0 = unit_result(group, "v0", "plan", facts={**PLAN_FACTS, "clock": clock.to_dict()})
    cap = unit_result(group, "v0_capture", "capture", facts=CAPTURE_FACTS,
                      depends_on=(v0.work_item_id,))
    a0 = unit_result(group, "v0_a0", "audio", facts=audio_facts(9.2, member="v0_a0"),
                     depends_on=(cap.work_item_id,))
    a1 = unit_result(group, "v0_a1", "audio", facts=audio_facts(9.4, member="v0_a1"),
                     depends_on=(cap.work_item_id,))
    results = [v0, cap, a0, a1]
    assert next_group_work_items(group, results) == []
    blockers = group_blockers(group, results)
    assert [row["code"] for row in blockers] == [
        "measured_reverberation_leaves_insufficient_movement_time"]
    assert blockers[0]["measured_motion_window"]["sufficient"] is False
    assert "9.4" in blockers[0]["reason"]


def test_a_late_plan_without_declared_motion_timing_values_fails(config):
    # A nested override would still inherit the default, so drop the declaration.
    config["defaults"]["request_extras"]["binding_motion"].pop("minimum_motion_s")
    group = state_group(config)
    clock = group.members[0].clock
    v0 = unit_result(group, "v0", "plan", facts={**PLAN_FACTS, "clock": clock.to_dict()})
    cap = unit_result(group, "v0_capture", "capture", facts=CAPTURE_FACTS,
                      depends_on=(v0.work_item_id,))
    a0 = unit_result(group, "v0_a0", "audio", facts=audio_facts(3.0, member="v0_a0"),
                     depends_on=(cap.work_item_id,))
    a1 = unit_result(group, "v0_a1", "audio", facts=audio_facts(3.0, member="v0_a1"),
                     depends_on=(cap.work_item_id,))
    with pytest.raises(ProductionSpecError, match="minimum_motion_s"):
        next_group_work_items(group, [v0, cap, a0, a1])


def test_every_shared_unit_is_emitted_exactly_once(config):
    """A whole run must not re-plan or re-capture a unit it already produced."""
    parsed = parse_production_config(config)
    for group in parsed.core_groups:
        recipe = recipe_for_task_family(group.task_family)
        clock = group.members[0].clock
        results, emitted = [], []
        for _ in range(20):
            items = next_group_work_items(group, results)
            if not items:
                break
            for item in items:
                emitted.append(item.unit_id)
                unit = recipe.unit(item.unit_id)
                facts = {
                    "plan": {**PLAN_FACTS, "clock": clock.to_dict()},
                    "late_plan": {"episode_plan_path": "v1/plan.json", "first_motion_frame": 60,
                                  "last_motion_frame": 141, "measured_wet_end_s": 3.4},
                    "capture": CAPTURE_FACTS,
                    "audio": audio_facts(3.2, member=item.unit_id),
                    "assembly": {"group_spec_path": "group_spec.json",
                                 "assembled_path": "assembled/binding_groups.json",
                                 "validation": {"status": "pass"}},
                }[unit.stage]
                results.append(unit_result(group, item.unit_id, unit.stage, facts=facts,
                                           depends_on=item.depends_on))
        assert len(emitted) == len(set(emitted)) == len(recipe.units), group.task_family
        assert next_group_work_items(group, results) == []
        assert group_blockers(group, results) == []


def test_units_with_no_cross_dependency_are_offered_together(config):
    group = parse_production_config(config).core_groups[0]
    assert group.task_family == "visible_binding"
    first = initial_group_work_items(group)
    assert sorted(item.unit_id for item in first) == ["v0", "v1"]
    assert all(item.stage == "plan" for item in first)


# --- the five reproduced fault inputs ---------------------------------------


def test_fault_one_a_pass_with_all_none_facts_is_refused(config):
    request = parse_production_config(config).episodes[0]
    with pytest.raises(ProductionSpecError, match="without real values"):
        StageResult(work_item_id=work_item_id(request.request_id, "plan", 1), stage="plan",
                    request_id=request.request_id, status="pass",
                    facts={key: None for key in STAGE_PUBLISHED_FACTS["plan"]})


def test_fault_one_an_empty_wet_tail_list_is_not_a_measurement(config):
    group = state_group(config)
    with pytest.raises(ProductionSpecError, match="without real values"):
        unit_result(group, "v0_a0", "audio",
                    facts={"facts_path": "f.json", "audio_report_path": "r.json",
                           "wet_tail_intervals": []})


def test_fault_two_a_mismatched_work_item_id_is_refused(config):
    request = parse_production_config(config).episodes[0]
    with pytest.raises(ProductionSpecError, match="not the id this scope and stage produce"):
        StageResult(work_item_id="unrelated:audio:88", stage="plan",
                    request_id=request.request_id, status="pass",
                    facts={**PLAN_FACTS, "clock": request.clock.to_dict()})
    with pytest.raises(ProductionSpecError, match="not the id this scope and stage produce"):
        StageResult(work_item_id=f"{request.request_id}:plan:1", stage="plan",
                    request_id=request.request_id, status="pass",
                    facts={**PLAN_FACTS, "clock": request.clock.to_dict()})


def test_fault_three_a_newer_failure_stops_an_older_pass(config):
    request = parse_production_config(config).episodes[0]
    good = _plan_pass(request)
    failed = StageResult(work_item_id=work_item_id(request.request_id, "plan", 2), stage="plan",
                         request_id=request.request_id, status="fail",
                         reason="planning_failed")
    assert next_stage_work_items(request, [good, failed]) == []
    state = request_round_state(request, [good, failed])
    assert state.is_blocked
    assert state.blocked == (("plan", "planning_failed"),)


def test_fault_four_the_newest_attempt_is_authoritative(config):
    request = parse_production_config(config).episodes[0]
    first = _plan_pass(request)
    second = StageResult(
        work_item_id=work_item_id(request.request_id, "plan", 2), stage="plan",
        request_id=request.request_id, status="pass",
        facts={**PLAN_FACTS, "episode_plan_path": "replanned.json",
               "clock": request.clock.to_dict()})
    capture = next_stage_work_items(request, [first, second])[0]
    assert capture.stage == "capture"
    assert capture.depends_on == (second.work_item_id,)
    assert capture.inputs["plan"]["facts"]["episode_plan_path"] == "replanned.json"


def test_fault_four_two_different_results_for_one_id_are_refused(config):
    request = parse_production_config(config).episodes[0]
    other = StageResult(
        work_item_id=work_item_id(request.request_id, "plan", 1), stage="plan",
        request_id=request.request_id, status="pass",
        facts={**PLAN_FACTS, "episode_plan_path": "other.json",
               "clock": request.clock.to_dict()})
    with pytest.raises(ProductionSpecError, match="two different results"):
        next_stage_work_items(request, [_plan_pass(request), other])


def test_fault_five_a_downstream_result_from_a_superseded_round_is_stale(config):
    """A new plan round invalidates the capture produced from the old one."""
    request = parse_production_config(config).episodes[0]
    first = _plan_pass(request)
    capture = StageResult(
        work_item_id=work_item_id(request.request_id, "capture", 1), stage="capture",
        request_id=request.request_id, status="pass", facts=CAPTURE_FACTS,
        depends_on=(first.work_item_id,))
    assert [item.stage for item in next_stage_work_items(request, [first, capture])] == ["audio"]

    replanned = StageResult(
        work_item_id=work_item_id(request.request_id, "plan", 2), stage="plan",
        request_id=request.request_id, status="pass",
        facts={**PLAN_FACTS, "episode_plan_path": "replanned.json",
               "clock": request.clock.to_dict()})
    state = request_round_state(request, [first, capture, replanned])
    assert state.stale == ("capture",)
    again = next_stage_work_items(request, [first, capture, replanned])
    assert [item.stage for item in again] == ["capture"]
    assert again[0].attempt == 2
    assert again[0].depends_on == (replanned.work_item_id,)
    assert again[0].fresh_output_relative.endswith("capture/attempt_02")


def test_a_stale_group_visual_invalidates_its_audio_columns(config):
    group = state_group(config)
    clock = group.members[0].clock
    v0 = unit_result(group, "v0", "plan", facts={**PLAN_FACTS, "clock": clock.to_dict()})
    cap = unit_result(group, "v0_capture", "capture", facts=CAPTURE_FACTS,
                      depends_on=(v0.work_item_id,))
    a0 = unit_result(group, "v0_a0", "audio", facts=audio_facts(3.0, member="v0_a0"),
                     depends_on=(cap.work_item_id,))
    a1 = unit_result(group, "v0_a1", "audio", facts=audio_facts(3.0, member="v0_a1"),
                     depends_on=(cap.work_item_id,))
    assert [item.unit_id for item in next_group_work_items(group, [v0, cap, a0, a1])] == ["v1"]

    recaptured = unit_result(group, "v0_capture", "capture", attempt=2,
                             facts={**CAPTURE_FACTS, "capture_receipt_path": "again.json"},
                             depends_on=(v0.work_item_id,))
    results = [v0, cap, a0, a1, recaptured]
    state = group_round_state(group, results)
    assert sorted(state.stale) == ["v0_a0", "v0_a1"]
    assert "v1" not in state.done
    ready = next_group_work_items(group, results)
    assert sorted(item.unit_id for item in ready) == ["v0_a0", "v0_a1"]
    assert all(item.attempt == 2 for item in ready)
    assert all(item.depends_on == (recaptured.work_item_id,) for item in ready)


def test_a_result_from_another_scope_is_refused(config):
    parsed = parse_production_config(config)
    request = parsed.episodes[0]
    group = parsed.core_groups[0]
    foreign = unit_result(group, "v0", "plan",
                          facts={**PLAN_FACTS, "clock": group.members[0].clock.to_dict()})
    with pytest.raises(ProductionSpecError, match="belongs to"):
        next_stage_work_items(request, [foreign])
    with pytest.raises(ProductionSpecError, match="not a unit of"):
        next_group_work_items(group, [_plan_pass(request)])


def test_a_result_filed_against_the_wrong_unit_stage_is_refused(config):
    group = state_group(config)
    scope = group_unit_scope_id(group.group_id, "v0_capture")
    wrong = StageResult(work_item_id=work_item_id(scope, "audio", 1), stage="audio",
                        request_id=scope, status="pass",
                        facts=audio_facts(2.0, member="v0_capture"))
    with pytest.raises(ProductionSpecError, match="filed against a capture unit"):
        next_group_work_items(group, [wrong])


def test_the_removed_probe_stage_is_named_in_the_error(config):
    group = state_group(config)
    scope = group_unit_scope_id(group.group_id, "v0")
    with pytest.raises(ProductionSpecError, match="early audio columns"):
        StageResult(work_item_id=work_item_id(scope, "audio_tail_probe", 1),
                    stage="audio_tail_probe", request_id=scope, status="pass",
                    facts={"measured_tail_seconds": 2.4})


def test_stage_chain_finishes_for_an_ordinary_episode(config):
    request = parse_production_config(config).episodes[0]
    results = [_plan_pass(request)]
    for stage, facts in (("capture", CAPTURE_FACTS),
                         ("audio", audio_facts(3.0, member=request.request_id)),
                         ("delivery", DELIVERY_FACTS)):
        item = next_stage_work_items(request, results)[0]
        assert item.stage == stage
        results.append(StageResult(work_item_id=item.work_item_id, stage=stage,
                                   request_id=request.request_id, status="pass",
                                   facts=facts, depends_on=item.depends_on))
    assert next_stage_work_items(request, results) == []


def test_stage_results_serialize_to_the_same_schedule(config):
    """A persisted round has to resume to exactly the same next units."""
    group = state_group(config)
    clock = group.members[0].clock
    v0 = unit_result(group, "v0", "plan", facts={**PLAN_FACTS, "clock": clock.to_dict()})
    cap = unit_result(group, "v0_capture", "capture", facts=CAPTURE_FACTS,
                      depends_on=(v0.work_item_id,))
    a0 = unit_result(group, "v0_a0", "audio", facts=audio_facts(3.1, member="v0_a0"),
                     depends_on=(cap.work_item_id,))
    a1 = unit_result(group, "v0_a1", "audio", facts=audio_facts(3.4, member="v0_a1"),
                     depends_on=(cap.work_item_id,))
    live = [v0, cap, a0, a1]
    encoded = json.loads(json.dumps([result.to_dict() for result in live]))
    restored = [StageResult.from_mapping(value) for value in encoded]
    assert [result.to_dict() for result in restored] == [result.to_dict() for result in live]
    assert ([item.to_dict() for item in next_group_work_items(group, restored)]
            == [item.to_dict() for item in next_group_work_items(group, live)])
    assert group_round_state(group, restored).to_dict() == group_round_state(group, live).to_dict()


def test_stage_result_status_vocabulary_and_round_trip():
    payload = {"work_item_id": "e:plan:01", "stage": "plan", "request_id": "e",
               "status": "blocked", "facts": {}, "reason": "insufficient_free_vram"}
    result = StageResult.from_mapping(payload)
    assert result.to_dict() == {**payload, "outputs": {}, "scope_id": "e",
                                "attempt": 1, "depends_on": []}
    assert result.passed is False
    assert result.attempt == 1
    with pytest.raises(ProductionSpecError, match="must carry a reason"):
        StageResult(work_item_id="e:plan:01", stage="plan", request_id="e", status="fail")
    with pytest.raises(ProductionSpecError, match="stage status must be"):
        StageResult(work_item_id="e:plan:01", stage="plan", request_id="e", status="delivered")


def test_retry_repeats_one_stage_into_a_fresh_output_until_the_budget_is_gone(config):
    request = parse_production_config(config).episodes[0]
    first = initial_stage_work_items(request)[0]
    second = retry_stage_work_item(first, retry=request.retry, reason="rpc_port_unavailable")
    assert second is not None
    assert second.attempt == 2
    assert second.fresh_output_relative == f"{request.request_id}/plan/attempt_02"
    assert second.payload["retry_reason"] == "rpc_port_unavailable"
    assert second.payload["retry_of"] == first.work_item_id
    assert retry_stage_work_item(second, retry=request.retry, reason="again") is None
    assert retry_stage_work_item(first, retry=RetryPolicy(attempts_per_stage=1), reason="x") is None


def test_protocol_summary_declares_the_recipes_and_its_boundary():
    summary = stage_protocol_summary()
    assert summary["schema"] == SCHEMA
    assert summary["stages"] == ["plan", "capture", "audio", "late_plan", "assembly", "delivery"]
    assert "audio_tail_probe" in summary["removed_stages"]
    assert summary["legal_resource_kinds_by_stage"]["audio"] == [
        "cpu_native_acoustic", "gpu_native_acoustic"]
    assert summary["resource_kind_profile"]["cpu_native_acoustic"] == {
        "execution": "cpu", "runtime_context": "rlr_native"}
    assert summary["published_facts_by_stage"]["audio"] == [
        "facts_path", "audio_report_path", "wet_tail_intervals"]
    assert summary["motion_timing_by_task_family"]["cross_time_state"] == "after_wet_tail"
    assert set(summary["group_recipes"]) == set(GROUP_RECIPES)
    assert summary["group_recipes"]["cross_time_state"]["member_unit_ids"] == [
        "v0_a0", "v0_a1", "v1_a0", "v1_a1"]
    assert "superseded" in summary["round_policy"]
    assert "not an executed stage" in summary["claim_boundary"]


# --- during-sound motion stays a separate condition -------------------------


def test_cross_time_state_cannot_be_turned_into_during_sound_motion(config):
    for group_block in config["core_groups"]:
        if group_block["task_family"] == "cross_time_state":
            group_block["member_defaults"] = {"motion_timing": "during_audible_window"}
    with pytest.raises(ProductionSpecError, match="moves the entity after_wet_tail"):
        parse_production_config(config)


def test_during_sound_motion_is_expressible_on_an_ordinary_episode(config):
    config["core_groups"] = []
    config["episodes"][0]["motion_timing"] = "during_audible_window"
    config["episodes"][0]["profile"] = {"separation_bin_deg": [30, 60], "anchor_count": 1,
                                        "speech_motion": "speaker_moving"}
    request = parse_production_config(config).episodes[0]
    assert request.motion_timing == "during_audible_window"
    assert request.moving_speech_motion == "speaker_moving"
    assert request.stage_plan() == ("plan", "capture", "audio", "delivery")
    assert initial_stage_work_items(request)[0].payload["motion_timing"] == (
        "during_audible_window")


def test_after_tail_motion_refuses_a_moving_speech_profile(config):
    """The early columns of a cross-time group are stationary by construction."""
    for group_block in config["core_groups"]:
        if group_block["task_family"] == "cross_time_state":
            group_block["member_defaults"] = {
                "profile": {"speech_motion": "speaker_moving", "anchor_count": 1}}
    with pytest.raises(ProductionSpecError, match="the early columns are stationary"):
        parse_production_config(config)


def test_motion_timing_vocabulary_is_closed(config):
    config["episodes"][0]["motion_timing"] = "whenever"
    config["core_groups"] = []
    with pytest.raises(ProductionSpecError, match="motion_timing must be one of"):
        parse_production_config(config)
    assert MOTION_TIMINGS == ("none", "during_audible_window", "after_wet_tail")


# --- group members must actually be able to share media --------------------


@pytest.mark.parametrize("override,message", [
    ({"clock": {"frame_count": 300, "frame_rate_hz": 30, "sample_rate_hz": 16000}},
     "differs in clock"),
    ({"rig": {"resolution_hw": [480, 640], "fov_deg": 85}}, "differs in rig"),
    ({"reserve_tail_s": 2.0}, "differs in reserve_tail_s"),
    ({"post_assembly_convolution_gain": 0.25},
     "differs in post_assembly_convolution_gain"),
    ({"audio_layouts": [{"type": "mono", "role": "primary"}]}, "differs in audio_layouts"),
])
def test_members_of_one_group_cannot_disagree_on_shared_media(config, override, message):
    config["core_groups"] = [config["core_groups"][0]]
    config["core_groups"][0]["members"][2].update(override)
    with pytest.raises(ProductionSpecError, match=message):
        parse_production_config(config)


def test_two_members_of_one_visual_cannot_use_different_asset_orders(config):
    config["core_groups"] = [config["core_groups"][0]]
    config["core_groups"][0]["members"][1]["instances"] = instances(
        "human_green_v1", "human_blue_v1")
    with pytest.raises(ProductionSpecError, match="one capture is one video"):
        parse_production_config(config)


def test_group_dict_reports_its_recipe_and_units(config):
    group = parse_production_config(config).core_groups[0]
    payload = group.to_dict()
    assert payload["recipe"]["task_family"] == "visible_binding"
    assert [row["unit_id"] for row in payload["stage_units"]][:2] == ["v0", "v1"]
    assert payload["stage_units"][0]["default_resource_kind"] == "cpu"


def test_foa_is_an_attached_view_of_the_same_rig(config):
    config["defaults"]["audio_layouts"] = [
        {"type": "binaural", "role": "primary", "indirect_sh_order": 1},
        {"type": "ambisonics", "ambisonic_order": 1, "role": "attached_view", "indirect_sh_order": 1},
    ]
    request = parse_production_config(config).episodes[0]
    assert [layout.layout_type for layout in request.audio_layouts] == ["binaural", "ambisonics"]
    assert request.primary_audio_layout.layout_type == "binaural"
    foa = request.audio_layouts[1]
    assert foa.channel_count == 4
    assert foa.to_channel_layout() == {"type": "ambisonics", "channel_count": 4}
    assert request.to_legacy_request()["audio_layouts"][1]["ambisonic_order"] == 1
    assert request.to_legacy_request()["simulation"]["indirect_sh_order"] == 1


def test_binaural_only_default_stays_binaural(config):
    del config["defaults"]["audio_layouts"]
    request = parse_production_config(config).episodes[0]
    assert [layout.to_dict() for layout in request.audio_layouts] == [
        {"type": "binaural", "channel_count": 2, "role": "primary"}]
    assert "simulation" not in request.to_legacy_request()


def _episode_config(config, **overrides):
    config = deepcopy(config)
    config["core_groups"] = []
    config["episodes"][0].update(overrides)
    return config


@pytest.mark.parametrize("overrides,message", [
    ({"rig": {"resolution_hw": [720, 1280], "fov_deg": 85, "motion": "orbit"}}, "fixes the camera"),
    ({"rig": {"resolution_hw": [720, 1280], "fov_deg": 200}}, "fov_deg must be within"),
    ({"rig": {"resolution_hw": [720], "fov_deg": 85}}, "resolution_hw must be"),
    ({"rig": {"resolution_hw": [0, 1280], "fov_deg": 85}}, "resolution_hw\\[0\\] must be a positive"),
    ({"clock": {"frame_count": 1, "frame_rate_hz": 3, "sample_rate_hz": 16000}}, "integer sample count"),
    ({"reserve_tail_s": 10.0}, "leaves no program time"),
    ({"reserve_tail_s": 12.0}, "leaves no program time"),
    ({"reserve_tail_s": -1.0}, "finite and nonnegative"),
    ({"post_assembly_convolution_gain": 0}, "positive finite"),
    ({"qa_ids": ["QA-99"]}, "not in the unified catalog"),
    ({"qa_ids": ["QA-05", "QA-05"]}, "must be distinct"),
    ({"qa_ids": ["QA-05"], "quota_by_qa": {"QA-06": 2}}, "outside qa_ids"),
    ({"qa_ids": ["QA-05"], "quota_by_qa": {"QA-05": 0}}, "positive integer"),
    ({"qa_ids": ["QA-05"], "qa_targets": [
        {"qa_id": "QA-05", "target_instance_ids": ["source9"], "event": "clip_tail"}]},
     "absent from this request"),
    ({"qa_ids": ["QA-05"], "qa_targets": [
        {"qa_id": "QA-05", "target_instance_ids": [], "event": "clip_tail"}]},
     "must name at least one instance"),
    ({"qa_ids": ["QA-05"], "qa_targets": [
        {"qa_id": "QA-05", "target_instance_ids": ["source1"], "event": {"kind": "event_ordinal"}}]},
     "ordinal is required"),
    ({"qa_ids": ["QA-05"], "qa_targets": [
        {"qa_id": "QA-05", "target_instance_ids": ["source1"],
         "event": {"kind": "clip_tail", "ordinal": 1}}]},
     "ordinal does not apply"),
    ({"qa_ids": ["QA-05"], "qa_targets": [
        {"qa_id": "QA-05", "target_instance_ids": ["source1"], "event": {"kind": "loudest"}}]},
     "kind must be one of"),
    ({"qa_ids": ["QA-05"], "qa_targets": [
        {"qa_id": "QA-05", "target_instance_ids": ["source1"], "event": "clip_tail",
         "forms": ["essay"]}]},
     "not offered by QA-05"),
    ({"instances": instances("a_v1", "b_v1") + [
        {"instance_id": "source1", "asset_id": "c_v1", "source_class": HUMAN}]},
     "instance_id values must be distinct"),
    ({"instances": instances("a_v1")}, "2..4 entity instances"),
    ({"instances": instances("a_v1", "b_v1", speaking=[False, False])}, "keep a speaking entity"),
    ({"instances": [{"instance_id": "s1", "asset_id": "a", "source_class": "ghost"},
                    {"instance_id": "s2", "asset_id": "b", "source_class": HUMAN}]},
     "source_class must be one of"),
    ({"instances": instances("a_v1", "b_v1"), "source_asset_ids": ["a_v1", "z_v1"]},
     "declared twice with different values"),
    ({"audio_layouts": [{"type": "binaural", "channel_count": 4}]}, "requires channel_count=2"),
    ({"audio_layouts": [{"type": "surround"}]}, "must be mono, binaural or ambisonics"),
    ({"audio_layouts": [{"type": "ambisonics", "ambisonic_order": 1, "indirect_sh_order": 0}]},
     "cannot carry order 1 ambisonics"),
    ({"audio_layouts": [{"type": "binaural", "role": "attached_view"}]}, "role=primary"),
    ({"audio_layouts": [{"type": "binaural"}, {"type": "ambisonics"}]}, "role=primary"),
    ({"audio_layouts": [{"type": "binaural"}, {"type": "binaural"}]}, "role=primary"),
    ({"audio_layouts": [{"type": "binaural", "role": "sidecar"}]}, "role must be one of"),
    ({"retry": {"attempts_within_profile": 400}}, "within 1..200"),
    ({"retry": {"resume_from_stage": "review"}}, "resume_from_stage must be one of"),
    ({"resources": {"capture": {"rpc_port": 70000}}}, "rpc_port must be within"),
    ({"resources": {"capture": {"min_free_vram_mb": -1}}}, "positive integer"),
    ({"room_id": None}, "room_id must be nonempty"),
    ({"request_id": ""}, "request_id must be nonempty"),
])
def test_illegal_combinations_fail_explicitly(config, overrides, message):
    with pytest.raises(ProductionSpecError, match=message):
        parse_production_config(_episode_config(config, **overrides))


def test_missing_reserve_tail_is_an_error(config):
    config = deepcopy(config)
    config["core_groups"] = []
    del config["defaults"]["reserve_tail_s"]
    with pytest.raises(ProductionSpecError, match="reserve_tail_s is required"):
        parse_production_config(config)


def test_conflicting_reserve_tail_declarations_are_reported(config):
    config = _episode_config(config, reserve_tail_s=3.0, profile={"reserve_tail_s": 2.0})
    with pytest.raises(ProductionSpecError, match="declared twice with different values"):
        parse_production_config(config)


def test_group_shape_and_family_conflicts_fail(config):
    broken = deepcopy(config)
    broken["core_groups"] = [core_group("g", "visible_binding", "room_a", members=3)]
    with pytest.raises(ProductionSpecError, match="needs exactly 4 members"):
        parse_production_config(broken)

    broken = deepcopy(config)
    broken["core_groups"] = [core_group("g", "unknown_family", "room_a")]
    with pytest.raises(ProductionSpecError, match="task_family must be one of"):
        parse_production_config(broken)

    broken = deepcopy(config)
    broken["core_groups"] = [core_group("g", "visible_binding", "room_a")]
    broken["core_groups"][0]["members"][1]["task_family"] = "cross_time_state"
    with pytest.raises(ProductionSpecError, match="declared twice with different values"):
        parse_production_config(broken)

    broken = deepcopy(config)
    broken["core_groups"] = [core_group("g", "visible_binding", "room_a")]
    broken["core_groups"][0]["members"][2]["room_id"] = "room_b"
    with pytest.raises(ProductionSpecError, match="is in another room"):
        parse_production_config(broken)

    broken = deepcopy(config)
    broken["core_groups"] = [core_group("g", "visible_binding", "room_a")]
    broken["core_groups"][0]["shared_audio_member_ids"] = [["g_m1", "g_m9"]]
    with pytest.raises(ProductionSpecError, match="names members outside this group"):
        parse_production_config(broken)


def test_duplicate_request_ids_across_the_config_fail(config):
    broken = deepcopy(config)
    broken["episodes"].append(deepcopy(broken["episodes"][0]))
    with pytest.raises(ProductionSpecError, match="request_id values must be unique"):
        parse_production_config(broken)


def test_schema_and_empty_config_are_rejected(config):
    with pytest.raises(ProductionSpecError, match="config.schema must be"):
        parse_production_config({**config, "schema": "something_else"})
    with pytest.raises(ProductionSpecError, match="must declare episodes"):
        parse_production_config({"schema": SCHEMA, "batch_id": "b"})
    with pytest.raises(ProductionSpecError, match="batch_id must be nonempty"):
        parse_production_config({"schema": SCHEMA})


def test_legacy_request_stays_readable_and_round_trips(config):
    request = parse_production_config(config).episodes[0]
    legacy = request.to_legacy_request()
    assert legacy["schema"] == "avengine_native_qa_room_request_v1"
    assert legacy["sampling_policy"] == "conditioned_static_v2"
    assert legacy["episode_id"] == "v1_episode_0001"
    assert legacy["frame_count"] == 150 and legacy["frame_rate_hz"] == 15.0
    assert legacy["sample_rate_hz"] == 16000
    assert legacy["profile"]["reserve_tail_s"] == 3.0
    assert legacy["profile"]["separation_bin_deg"] == [30, 60]
    assert legacy["post_assembly_convolution_gain"] == 0.5
    assert legacy["sound_pool"] == "/pool/batch_sounds.json"
    assert legacy["sound_selection"] == {"max_clip_s": 5.0}
    assert legacy["entities"] == {"total_count": 2, "silent_count": 0, "min_articulated_count": 0}

    reparsed = production_request_from_legacy(legacy)
    assert reparsed.request_id == request.request_id
    assert reparsed.clock.to_dict() == request.clock.to_dict()
    assert reparsed.rig == request.rig
    assert reparsed.qa_ids == request.qa_ids
    assert reparsed.reserve_tail_s == request.reserve_tail_s
    assert [i.to_dict() for i in reparsed.instances] == [i.to_dict() for i in request.instances]
    assert reparsed.resources == request.resources
    assert reparsed.retry == request.retry
    assert reparsed.to_legacy_request() == legacy
    # The saved request states its quota outright, so a re-read reports it as
    # declared even where the config had left the spec to fill a unit default.
    assert request.quota_source == "unit_default"
    assert reparsed.quota_source == "config"
    assert legacy["production"]["stage_resources"]["capture"] == {
        "kind": "gpu_native_visual", "execution": "gpu", "runtime_context": "renderer_native",
        "graphics_adapter": 0, "min_free_vram_mb": 12000, "rpc_port": 39782}
    assert legacy["production"]["stage_resources"]["audio"]["execution"] == "cpu"
    assert legacy["production"]["retry"] == {"attempts_per_stage": 2,
                                             "attempts_within_profile": 200}


def test_a_pre_two_axis_saved_resource_kind_falls_back_to_the_stage_default():
    """An earlier build forced acoustics onto a GPU slot; that is not a choice."""
    legacy = {
        "episode_id": "e", "room_id": "r",
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000,
        "entities": {"total_count": 2, "silent_count": 0},
        "profile": {"reserve_tail_s": 3.0},
        "qa_ids": ["QA-05"],
        "production": {"stage_resources": {
            "audio": {"kind": "gpu_native_acoustic", "graphics_adapter": 0, "rlr_threads": 1},
            "capture": {"kind": "gpu_native_visual", "graphics_adapter": 0,
                        "min_free_vram_mb": 12000},
        }},
    }
    request = production_request_from_legacy(legacy)
    audio = request.resource_for("audio")
    assert audio.kind == "cpu_native_acoustic"
    assert audio.execution == "cpu"
    assert audio.graphics_adapter == 0
    assert audio.rlr_threads == 1
    assert request.resource_for("capture").min_free_vram_mb == 12000


def test_a_two_axis_saved_resource_kind_is_honoured():
    legacy = {
        "episode_id": "e", "room_id": "r",
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000,
        "entities": {"total_count": 2, "silent_count": 0},
        "profile": {"reserve_tail_s": 3.0},
        "qa_ids": ["QA-05"],
        "production": {"stage_resources": {
            "audio": {"kind": "gpu_native_acoustic", "execution": "gpu",
                      "runtime_context": "rlr_native", "min_free_vram_mb": 4000},
        }},
    }
    request = production_request_from_legacy(legacy)
    assert request.resource_for("audio").kind == "gpu_native_acoustic"
    assert request.resource_for("audio").min_free_vram_mb == 4000


def test_old_style_request_without_the_new_blocks_is_still_parsed():
    legacy = {
        "schema": "avengine_native_qa_room_request_v1",
        "episode_id": "binding_relation_mp3d_20260909_g01_v0",
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "seed": 202609090901,
        "sampling_policy": "conditioned_static_v2",
        "camera": {"fov_deg": 85, "height_above_floor_m": 1.55, "motion": "static",
                   "resolution_hw": [720, 1280]},
        "frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000,
        "entities": {"min_articulated_count": 0, "silent_count": 0, "total_count": 3},
        "profile": {"anchor_count": 3, "event_relation": "overlap", "reserve_tail_s": 3.0},
        "qa_ids": ["QA-05"],
        "qa_sampling": {"items_per_type": 1, "query_time_policy": "uniform_in_legal_window"},
        "source_asset_ids": ["human_a", "human_b", "human_c"],
        "sound_pool": "/pool/batch_sounds.json",
        "sound_selection": {"max_clip_s": 5.0},
        "post_assembly_convolution_gain": 0.5,
        "diffraction": False, "rir_stride": 5, "max_diffraction_order": 0,
        "runtime": {"graphics_adapter": 0, "rpc_port": 39782},
        "allow_research_candidate_assets": True,
    }
    request = production_request_from_legacy(legacy)
    assert request.instance_count == 3
    assert [i.instance_id for i in request.instances] == ["source1", "source2", "source3"]
    assert request.qa_ids == ("QA-05",)
    assert request.items_per_type == 1
    assert request.reserve_tail_s == 3.0
    assert request.duration_seconds == pytest.approx(10.0)
    assert request.primary_audio_layout.layout_type == "binaural"
    assert request.qa_targets[0].event.kind == "target_audible_window"
    rendered = request.to_legacy_request()
    for key in ("diffraction", "rir_stride", "max_diffraction_order", "runtime",
                "allow_research_candidate_assets"):
        assert rendered[key] == legacy[key]
    assert rendered["qa_sampling"]["query_time_policy"] == "uniform_in_legal_window"
    assert initial_stage_work_items(request)[0].inputs["request"]["runtime"]["rpc_port"] == 39782


def test_silent_entities_are_read_from_a_legacy_request():
    request = production_request_from_legacy({
        "episode_id": "e", "room_id": "r",
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000,
        "entities": {"total_count": 2, "silent_count": 1},
        "profile": {"reserve_tail_s": 3.0},
        "qa_ids": ["QA-05"],
    })
    assert request.silent_count == 1
    assert [i.speaking for i in request.instances] == [True, False]
    assert request.to_legacy_request()["entities"]["silent_count"] == 1


def test_component_dataclasses_validate_on_direct_construction():
    with pytest.raises(ProductionSpecError, match="must name the entity instances"):
        QaTargetSpec(qa_id="QA-05", target_instance_ids=(), event=EventSelector(kind="clip_tail"))
    with pytest.raises(ProductionSpecError, match="not in the unified catalog"):
        QaTargetSpec(qa_id="QA-99", target_instance_ids=("source1",),
                     event=EventSelector(kind="clip_tail"))
    with pytest.raises(ProductionSpecError, match="requires channel_count=2"):
        AudioLayoutSpec.from_mapping({"type": "binaural", "channel_count": 3})
    assert EntityInstanceSpec.from_mapping(
        {"asset_id": "a", "source_class": DEVICE}, index=1).instance_id == "source2"
