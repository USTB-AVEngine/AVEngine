from copy import deepcopy
import json

import pytest

from avengine.dataset.binding_group_native import (
    BindingNativeError,
    GROUP_STAGE_SCHEMA,
    STAGE_RESULT_FILENAME,
    STAGE_RUNNERS,
    audio_column_of_unit,
    audio_render_inputs,
    build_audio_assignment_plan,
    build_variant_request,
    compare_controlled_visual_plans,
    compare_visual_plans,
    compare_visual_world,
    controlled_world_contract,
    declared_audio_delivery,
    finalize_audio_assignments,
    group_stage_context,
    group_world_equivalence,
    load_group_stage_results,
    plan_slot_identities,
    resolve_instance_runtime,
    run_group_stage_work_item,
    schedule_relation_audio_plan,
    shared_visual_evidence_root,
    verify_delivered_audio_layouts,
    verify_materialized_audio_root,
    verify_retained_visual_request,
    _group_spec,
    _neutral_endpoint_bindings,
)


ASSETS = (
    "asset_blue",
    "asset_green",
    "asset_yellow",
)
SOUNDS = (
    "sound_one",
    "sound_two",
    "sound_three",
)


def _plan():
    events = []
    bindings = []
    for index, (actor_id, sound_id, duration, local_start, local_end) in enumerate(
        (
            ("source1", SOUNDS[0], 47360, 15200, 61600),
            ("source2", SOUNDS[1], 50720, 15680, 65440),
            ("source3", SOUNDS[2], 56000, 14880, 69920),
        ),
        start=1,
    ):
        events.append(
            {
                "event_id": f"event_{index:03d}",
                "actor_id": actor_id,
                "sound_asset_id": sound_id,
                "path": f"/tmp/{sound_id}.wav",
                "sample_count": duration,
                "audible_start_sample": local_start,
                "audible_end_sample_exclusive": local_end,
                "source_activity_intervals_samples": [
                    [local_start, local_end]
                ],
                "source_metadata_manifest": None,
            }
        )
        bindings.append(
            {
                "actor_id": actor_id,
                "sound_asset_id": sound_id,
                "path": f"/tmp/{sound_id}.wav",
                "compatible_asset_ids": list(ASSETS),
                "compatible_object_categories": ["audio_playback"],
                "gender": "M",
                "sound_class": "speech_playback",
            }
        )
    return {
        "clock": {
            "frame_count": 150,
            "frame_rate_hz": 15.0,
            "sample_rate_hz": 16000,
            "sample_count": 160000,
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
        },
        "seed": 202609090901,
        "request": {
            "profile": {"reserve_tail_s": 3.0},
            "sound_pool": "final-pool.json",
        },
        "audio_events": events,
        "voice_bindings": bindings,
        "visual_plan": {
            "actors": [
                {
                    "actor_id": actor_id,
                    "asset_id": asset_id,
                    "entity_class": "articulated_human",
                    "identity": {"species_id": "human"},
                    "realized_attributes": {
                        "sex_or_gender_label": "male",
                    },
                    "source_endpoint_id": f"{actor_id}_emitter",
                }
                for actor_id, asset_id in zip(
                    ("source1", "source2", "source3"), ASSETS
                )
            ]
        },
        "planned_conditions": {
            "legal_event_start_ranges_samples": {
                "event_001": [[0, 64640]],
                "event_002": [[0, 61280]],
                "event_003": [[0, 56000]],
            }
        },
    }


def _request():
    return {
        "episode_id": "relation_v0",
        "room_id": "room",
        "seed": 202609090901,
        "entities": {"total_count": 3, "silent_count": 0},
        "profile": {"reserve_tail_s": 3.0},
        "camera": {"motion": "static"},
        "qa_ids": ["QA-05"],
        "sound_pool": "final-pool.json",
    }


def test_build_variant_request_accepts_three_sources_without_mutating_input():
    request = _request()
    result = build_variant_request(
        request,
        episode_id="relation_v1",
        source_asset_ids=(ASSETS[0], ASSETS[2], ASSETS[1]),
        rpc_port=39782,
        graphics_adapter=2,
        seed=request["seed"],
    )
    assert request["entities"]["total_count"] == 3
    assert result["entities"]["total_count"] == 3
    assert result["source_asset_ids"] == [ASSETS[0], ASSETS[2], ASSETS[1]]
    assert result["runtime"]["graphics_adapter"] == 2


def test_relation_schedule_preserves_declared_tail_and_randomizes_legal_times():
    plan = _plan()
    request = _request()
    original = deepcopy(plan)
    scheduled, scheduled_request, record = schedule_relation_audio_plan(
        plan, request, query_window_s=(4, 6), rng_seed=17
    )
    assert record["selection_mode"] == "seeded_random_legal"
    assert record["effective_reserve_tail_s"] == pytest.approx(3.0)
    assert max(row["end_sample_exclusive"] for row in record["events"]) <= 112000
    assert scheduled_request["profile"]["reserve_tail_s"] == pytest.approx(3.0)
    assert plan == original
    assert scheduled["request"]["sound_pool"] == request["sound_pool"]


def test_relation_schedule_rejects_fixed_schedule_that_breaks_three_second_tail():
    with pytest.raises(BindingNativeError, match="below declared reserve tail"):
        schedule_relation_audio_plan(
            _plan(), _request(), start_times_s=(3.0, 4.0, 6.0)
        )


def test_three_source_assignment_preserves_identity_and_checks_endpoint_and_compatibility():
    plan = _plan()
    request = _request()
    scheduled, scheduled_request, _ = schedule_relation_audio_plan(
        plan, request, start_times_s=(3.5, 3.0, 0.0)
    )
    original_events = {
        row["event_id"]: (
            row["start_sample"],
            row["end_sample_exclusive"],
            row["sound_asset_id"],
        )
        for row in scheduled["audio_events"]
    }
    assigned, rebound = build_audio_assignment_plan(
        scheduled,
        scheduled_request,
        "a1",
        assignment_targets={
            "a0": ("source1", "source2", "source3"),
            "a1": ("source1", "source3", "source2"),
        },
        expected_event_count=3,
        endpoint_by_actor={
            "source1": "source1_emitter",
            "source2": "source2_emitter",
            "source3": "source3_emitter",
        },
        require_authoritative_endpoints=True,
    )
    assert [
        (row["event_id"], row["actor_id"], row["source_endpoint_id"])
        for row in assigned["audio_events"]
    ] == [
        ("event_003", "source1", "source1_emitter"),
        ("event_002", "source3", "source3_emitter"),
        ("event_001", "source2", "source2_emitter"),
    ]
    for row in assigned["audio_events"]:
        assert (
            row["start_sample"],
            row["end_sample_exclusive"],
            row["sound_asset_id"],
        ) == original_events[row["event_id"]]
    assert rebound["audio_assignment_targets"] == {
        "event_003": "source1",
        "event_002": "source3",
        "event_001": "source2",
    }


def test_assignment_rejects_missing_authoritative_endpoint_and_incompatible_asset():
    plan = _plan()
    request = _request()
    scheduled, scheduled_request, _ = schedule_relation_audio_plan(
        plan, request, start_times_s=(3.5, 3.0, 0.0)
    )
    missing_endpoint = deepcopy(scheduled)
    for actor in missing_endpoint["visual_plan"]["actors"]:
        if actor["actor_id"] != "source1":
            actor.pop("source_endpoint_id", None)
    with pytest.raises(BindingNativeError, match="authoritative native source endpoint"):
        build_audio_assignment_plan(
            missing_endpoint,
            scheduled_request,
            "a0",
            assignment_targets={
                "a0": ("source1", "source2", "source3"),
                "a1": ("source1", "source3", "source2"),
            },
            expected_event_count=3,
            endpoint_by_actor={"source1": "source1_emitter"},
            require_authoritative_endpoints=True,
        )
    wrong_gender = deepcopy(scheduled)
    wrong_gender["voice_bindings"][0]["gender"] = "F"
    with pytest.raises(BindingNativeError, match="incompatible"):
        build_audio_assignment_plan(
            wrong_gender,
            scheduled_request,
            "a0",
            assignment_targets={
                "a0": ("source1", "source2", "source3"),
                "a1": ("source1", "source3", "source2"),
            },
            expected_event_count=3,
            endpoint_by_actor={
                "source1": "source1_emitter",
                "source2": "source2_emitter",
                "source3": "source3_emitter",
            },
            require_authoritative_endpoints=True,
        )
    incompatible = deepcopy(scheduled)
    incompatible["voice_bindings"][1]["compatible_asset_ids"] = [ASSETS[0]]
    with pytest.raises(BindingNativeError, match="incompatible"):
        build_audio_assignment_plan(
            incompatible,
            scheduled_request,
            "a0",
            assignment_targets={
                "a0": ("source1", "source2", "source3"),
                "a1": ("source1", "source3", "source2"),
            },
            expected_event_count=3,
            endpoint_by_actor={
                "source1": "source1_emitter",
                "source2": "source2_emitter",
                "source3": "source3_emitter",
            },
            require_authoritative_endpoints=True,
        )


def test_assignment_allows_multiple_events_for_one_physical_target():
    plan = _plan()
    request = _request()
    scheduled, scheduled_request, _ = schedule_relation_audio_plan(
        plan, request, start_times_s=(3.5, 3.0, 0.0)
    )
    two_event = deepcopy(scheduled)
    two_event["audio_events"] = two_event["audio_events"][:2]
    sound_ids = {row["sound_asset_id"] for row in two_event["audio_events"]}
    two_event["voice_bindings"] = [
        row for row in scheduled["voice_bindings"]
        if row["sound_asset_id"] in sound_ids
    ]
    assigned, _ = build_audio_assignment_plan(
        two_event,
        scheduled_request,
        "a0",
        assignment_targets={
            "a0": {
                "event_003": "source1",
                "event_002": "source1",
            },
            "a1": {
                "event_003": "source1",
                "event_002": "source1",
            },
        },
        expected_event_count=2,
        endpoint_by_actor={"source1": "source1_emitter"},
        require_authoritative_endpoints=True,
    )
    assert [row["actor_id"] for row in assigned["audio_events"]] == [
        "source1",
        "source1",
    ]


def test_assignment_reuses_same_sound_pcm_with_event_specific_bindings():
    plan = _plan()
    request = _request()
    first_event, second_event = plan["audio_events"][:2]
    second_event.update(
        sound_asset_id=first_event["sound_asset_id"],
        path=first_event["path"],
        sample_count=first_event["sample_count"],
    )
    first_binding, second_binding = plan["voice_bindings"][:2]
    first_binding["event_id"] = "event_001"
    first_binding["event_specific_marker"] = "first"
    second_binding.update(
        event_id="event_002",
        sound_asset_id=first_binding["sound_asset_id"],
        path=first_binding["path"],
        event_specific_marker="second",
    )
    assigned, _ = build_audio_assignment_plan(
        plan,
        request,
        "a0",
        assignment_targets={
            "a0": {
                "event_001": "source1",
                "event_002": "source2",
                "event_003": "source3",
            },
            "a1": {
                "event_001": "source1",
                "event_002": "source2",
                "event_003": "source3",
            },
        },
        expected_event_count=3,
        endpoint_by_actor={
            "source1": "source1_emitter",
            "source2": "source2_emitter",
            "source3": "source3_emitter",
        },
        require_authoritative_endpoints=True,
    )
    bindings = {row["event_id"]: row for row in assigned["voice_bindings"]}
    assert bindings["event_001"]["event_specific_marker"] == "first"
    assert bindings["event_002"]["event_specific_marker"] == "second"
    assert {
        (row["event_id"], row["actor_id"], row["source_endpoint_id"])
        for row in bindings.values()
    } == {
        ("event_001", "source1", "source1_emitter"),
        ("event_002", "source2", "source2_emitter"),
        ("event_003", "source3", "source3_emitter"),
    }


def test_assignment_rejects_duplicate_event_ids_before_target_mapping():
    plan = _plan()
    plan["audio_events"][1]["event_id"] = plan["audio_events"][0]["event_id"]
    with pytest.raises(BindingNativeError, match="unique event_id"):
        build_audio_assignment_plan(
            plan,
            _request(),
            "a0",
            assignment_targets={
                "a0": (
                    "source1",
                    "source2",
                    "source3",
                ),
                "a1": (
                    "source1",
                    "source2",
                    "source3",
                ),
            },
            expected_event_count=3,
            endpoint_by_actor={
                "source1": "source1_emitter",
                "source2": "source2_emitter",
                "source3": "source3_emitter",
            },
            require_authoritative_endpoints=True,
        )


def test_ue_endpoint_resolution_requires_actual_emitter_readback(tmp_path):
    frame_path = tmp_path / "frame_readbacks.json"
    frame_path.write_text(
        '{"emitters": {"source1": [{"frame_index": 0, "location_cm": [0, 0, 0]}]}}'
    )
    neutral_path = tmp_path / "neutral_readback.json"
    neutral_path.write_text(
        '{"entities": {"source1": [{"frame_index": 0, "emitter": [0, 0, 0], "root": [0, 0, 0], "moving": false}]},'
        f' "producer": {{"source_readbacks": ["{frame_path}"]}}}}'
    )
    plan = {
        "visual_plan": {
            "actors": [{
                "actor_id": "source1",
                "emitter_binding": {
                    "source_slot_id": "source_slot_1",
                    "semantic_anchor_id": "speaker",
                },
            }]
        }
    }
    assert _neutral_endpoint_bindings(neutral_path, plan=plan) == {
        "source1": "source_slot_1_speaker"
    }
    frame_path.write_text('{"emitters": {}}')
    with pytest.raises(BindingNativeError, match="emitter readback"):
        _neutral_endpoint_bindings(neutral_path, plan=plan)


# ---------------------------------------------------------------------------
# Shared native stages of a controlled four-member group
# ---------------------------------------------------------------------------

BLUE = "asset_blue"
GREEN = "asset_green"


def _legacy_request(request_id, order):
    return {
        "schema": "avengine_native_qa_room_request_v1",
        "episode_id": request_id,
        "room_id": "room_one",
        "seed": 7,
        "sampling_policy": "conditioned_static_v2",
        "camera": {"motion": "static", "fov_deg": 85, "resolution_hw": [720, 1280]},
        "frame_count": 150,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "entities": {"total_count": len(order), "silent_count": 0},
        "entity_instances": [
            {"instance_id": f"inst_{index:02d}", "source_class": "articulated_human",
             "asset_id": asset}
            for index, asset in enumerate(order)
        ],
        "profile": {"reserve_tail_s": 3.0},
        "qa_ids": ["QA-20"],
        "rir_stride": 5,
        "post_assembly_convolution_gain": 0.5,
        "runtime": {"hrtf": "/tmp/hrtf.sofa", "graphics_adapter": 1, "rpc_port": 39500},
        "sound_pool": "/tmp/pool.json",
    }


def _core_group(task_family="visible_binding", orders=None):
    from avengine.dataset.production_spec import (
        CoreGroupRequest, production_request_from_legacy)
    orders = orders or [[BLUE, GREEN], [BLUE, GREEN], [GREEN, BLUE], [GREEN, BLUE]]
    members = []
    for index, order in enumerate(orders):
        value = _legacy_request(f"member_{index}", order)
        value["source_asset_ids"] = list(order)
        value["task_family"] = task_family
        value["group_id"] = "group_one"
        members.append(production_request_from_legacy(
            value, request_id=f"member_{index}", kind="core_group_member"))
    return CoreGroupRequest(group_id="group_one", task_family=task_family,
                            room_id="room_one", members=tuple(members))


def test_controlled_world_contract_reads_the_declared_slot_swap():
    group = _core_group()
    context = group_stage_context(group=group)
    contract = context["contract"]
    assert contract["status"] == "pass"
    assert sorted(contract["visual_units"]) == ["v0_capture", "v1_capture"]
    assert contract["visual_units"]["v0_capture"]["source_asset_ids"] == [BLUE, GREEN]
    assert contract["visual_units"]["v1_capture"]["source_asset_ids"] == [GREEN, BLUE]
    assert contract["visual_units"]["v0_capture"]["member_request_ids"] == [
        "member_0", "member_1"]
    assert contract["audio_columns"] == {"a0": ["v0_a0", "v1_a0"],
                                         "a1": ["v0_a1", "v1_a1"]}
    assert contract["declared_interventions"]["visual_slot_permutation"] is True
    assert contract["world_population"] == sorted([BLUE, GREEN])
    assert contract["plan_equivalence_rule"] == (
        "identical_planned_world_under_declared_slot_identities")
    assert audio_column_of_unit(contract, "v1_a1") == "a1"


def test_controlled_world_contract_refuses_a_member_that_declares_another_world():
    group = _core_group()
    requests = {member.request_id: member.to_legacy_request() for member in group.members}
    requests["member_3"]["frame_count"] = 120
    with pytest.raises(BindingNativeError, match="one controlled world"):
        controlled_world_contract(group.to_dict(), requests)


def test_controlled_world_contract_refuses_a_different_entity_population():
    group = _core_group(orders=[[BLUE, GREEN], [BLUE, GREEN],
                                [BLUE, "asset_yellow"], [BLUE, "asset_yellow"]])
    requests = {member.request_id: member.to_legacy_request() for member in group.members}
    with pytest.raises(BindingNativeError, match="more than one world"):
        controlled_world_contract(group.to_dict(), requests)


def test_two_visual_units_need_a_declared_difference():
    group = _core_group(orders=[[BLUE, GREEN]] * 4)
    requests = {member.request_id: member.to_legacy_request() for member in group.members}
    with pytest.raises(BindingNativeError, match="not a controlled intervention"):
        controlled_world_contract(group.to_dict(), requests)


def test_a_declared_motion_timing_is_itself_the_intervention():
    group = _core_group(task_family="cross_time_state", orders=[[BLUE, GREEN]] * 4)
    contract = controlled_world_contract(
        group.to_dict(),
        {member.request_id: member.to_legacy_request() for member in group.members})
    assert contract["motion_timing"] == "after_wet_tail"
    assert contract["declared_interventions"]["visual_slot_permutation"] is False
    assert contract["plan_equivalence_rule"] == (
        "identical_planned_scene_clock_and_camera_route_only")


def test_audio_column_position_must_agree_with_an_explicit_unit_name():
    group = _core_group()
    spec = group.to_dict()
    for row in spec["stage_units"]:
        if row["unit_id"] == "v1_a0":
            row["unit_id"] = "v1_a1"
        elif row["unit_id"] == "v1_a1":
            row["unit_id"] = "v1_a0"
    with pytest.raises(BindingNativeError, match="assignment column"):
        controlled_world_contract(
            spec,
            {member.request_id: member.to_legacy_request() for member in group.members})


def test_variant_request_refuses_to_overwrite_a_declared_camera_motion():
    request = _legacy_request("episode", [BLUE, GREEN])
    request["camera"]["motion"] = "orbit"
    with pytest.raises(BindingNativeError, match="not silently replaced"):
        build_variant_request(request, episode_id="v0",
                              source_asset_ids=(BLUE, GREEN))


def test_variant_request_inherits_the_declared_question_selection():
    request = _legacy_request("episode", [BLUE, GREEN])
    request["qa_ids"] = ["QA-06", "QA-20"]
    result = build_variant_request(request, episode_id="v0",
                                   source_asset_ids=(BLUE, GREEN))
    assert result["qa_ids"] == ["QA-06", "QA-20"]
    assert result["binding_variant"]["qa_ids_source"] == "request"
    assert result["runtime"]["graphics_adapter"] == 1
    assert result["runtime"]["rpc_port"] == 39500
    assert result["binding_variant"]["instance_runtime_source"] == {
        "rpc_port": "request", "graphics_adapter": "request"}


def test_variant_request_needs_a_question_selection_from_somewhere():
    request = _legacy_request("episode", [BLUE, GREEN])
    request.pop("qa_ids")
    with pytest.raises(BindingNativeError, match="must declare qa_ids"):
        build_variant_request(request, episode_id="v0",
                              source_asset_ids=(BLUE, GREEN))


def test_variant_request_takes_the_lease_placement_when_one_is_supplied():
    request = _legacy_request("episode", [BLUE, GREEN])
    result = build_variant_request(request, episode_id="v0",
                                   source_asset_ids=(BLUE, GREEN),
                                   rpc_port=39999, graphics_adapter=2)
    assert result["runtime"]["rpc_port"] == 39999
    assert result["runtime"]["graphics_adapter"] == 2
    assert result["binding_variant"]["instance_runtime_source"] == {
        "rpc_port": "lease", "graphics_adapter": "lease"}


def test_instance_runtime_prefers_the_lease_then_the_work_item():
    item = {"resource": {"kind": "gpu_native_visual", "execution": "gpu",
                         "runtime_context": "renderer_native",
                         "graphics_adapter": 3, "rpc_port": 39100}}
    from_item = resolve_instance_runtime(item)
    assert (from_item["graphics_adapter"], from_item["rpc_port"]) == (3, 39100)
    assert from_item["sources"]["graphics_adapter"] == "work_item_resource"
    assert from_item["rlr_threads"] is None
    assert from_item["sources"]["rlr_threads"] == "request"
    leased = resolve_instance_runtime(
        item, lease={"graphics_adapter": 2, "rlr_threads": 1, "lease_id": "L1"})
    assert leased["graphics_adapter"] == 2
    assert leased["sources"]["graphics_adapter"] == "lease"
    assert leased["rpc_port"] == 39100
    assert leased["sources"]["rpc_port"] == "work_item_resource"
    assert leased["rlr_threads"] == 1
    assert leased["lease_id"] == "L1"


def _materialize(tmp_path, request, plan):
    root = tmp_path / "member"
    (root / "plan").mkdir(parents=True)
    (root / "capture").mkdir()
    (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (root / "plan/episode_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "plan/voice_bindings.json").write_text(
        json.dumps(plan["voice_bindings"]), encoding="utf-8")
    (root / "plan/audio_events.json").write_text(
        json.dumps(plan["audio_events"]), encoding="utf-8")
    (root / "native_linkage.json").write_text(
        json.dumps({"member_id": "v0_a0", "capture_source": str(root / "capture")}),
        encoding="utf-8")
    return root


def test_audio_root_verification_refuses_a_directory_that_was_only_created(tmp_path):
    bare = tmp_path / "v0_a0"
    bare.mkdir()
    with pytest.raises(BindingNativeError, match="materialize_audio_variant"):
        verify_materialized_audio_root(bare)


def test_audio_root_verification_accepts_a_materialized_root(tmp_path):
    request = _legacy_request("episode", [BLUE, GREEN])
    plan = _plan()
    root = _materialize(tmp_path, request, plan)
    checked = verify_materialized_audio_root(root, request=request)
    assert checked["status"] == "pass"
    assert checked["member_id"] == "v0_a0"
    assert checked["event_count"] == 3


def test_audio_root_verification_refuses_another_episode_request(tmp_path):
    plan = _plan()
    root = _materialize(tmp_path, _legacy_request("episode", [BLUE, GREEN]), plan)
    other = _legacy_request("another_episode", [BLUE, GREEN])
    with pytest.raises(BindingNativeError, match="differs from the request"):
        verify_materialized_audio_root(root, request=other)


def test_parallel_finalizer_refuses_an_unmaterialized_member_root(tmp_path):
    missing = tmp_path / "v1_a0"
    missing.mkdir()
    with pytest.raises(BindingNativeError, match="unusable root"):
        finalize_audio_assignments(
            {"v1_a0": {"root": missing,
                       "request": _legacy_request("episode", [BLUE, GREEN])}})


def test_declared_audio_delivery_reads_the_requested_layouts():
    request = _legacy_request("episode", [BLUE, GREEN])
    assert declared_audio_delivery(request)["primary_layout"] == "binaural"
    assert declared_audio_delivery(request)["foa_normalization"] == "native_n3d"
    request["audio_layouts"] = [
        {"type": "binaural", "channel_count": 2, "role": "primary"},
        {"type": "ambisonics", "channel_count": 4, "role": "attached_view",
         "ambisonic_order": 1},
    ]
    request["foa_normalization"] = "sn3d"
    declared = declared_audio_delivery(request)
    assert declared["attached_view_layouts"] == ["ambisonics"]
    assert declared["foa_normalization"] == "sn3d"


def test_a_declared_attached_view_that_was_never_delivered_is_a_blocker():
    declared = {"primary_layout": "binaural", "attached_view_layouts": ["ambisonics"],
                "foa_normalization": "native_n3d", "source_context_policy": "joint"}
    empty = verify_delivered_audio_layouts(declared, {"ancillary_audio_outputs": []})
    assert empty["status"] == "blocked"
    assert "--layouts" in empty["reason"]
    delivered = verify_delivered_audio_layouts(declared, {"ancillary_audio_outputs": [
        {"role": "ambisonic_wav", "path": "/tmp/audio/foa/ambisonics.wav"}]})
    assert delivered["status"] == "pass"
    assert delivered["undelivered_attached_view_layouts"] == []


def _write_plan(path, plan):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _visual_plan(actor_offset=0.0):
    return {
        "clock": {"frame_count": 2, "frame_rate_hz": 15.0, "sample_rate_hz": 16000},
        "scene": {"scene_id": "room_one"},
        "resources": {"room_package": {"family": "hm3d", "renderer": "habitat"}},
        "visual_plan": {
            "camera": {"motion": "static", "fov_deg": 85},
            "actors": [{"actor_id": "source1", "asset_id": BLUE},
                        {"actor_id": "source2", "asset_id": GREEN}],
            "frames": [
                {"frame_index": index, "pts_ticks": index * 3200,
                 "camera_state": {"position_m": [0, 0, 0]},
                 "actor_states": {"source1": {"position_m": [actor_offset, 0, 0]}}}
                for index in range(2)
            ],
        },
        "audio_events": [],
    }


def test_world_comparison_allows_declared_motion_but_strict_comparison_does_not(tmp_path):
    left = _write_plan(tmp_path / "left.json", _visual_plan())
    right = _write_plan(tmp_path / "right.json", _visual_plan(actor_offset=1.5))
    assert compare_visual_world(left, right)["status"] == "pass"
    with pytest.raises(BindingNativeError, match="transform-driving"):
        compare_visual_plans(left, right)
    moved = _visual_plan()
    moved["visual_plan"]["frames"][1]["camera_state"] = {"position_m": [2, 0, 0]}
    with pytest.raises(BindingNativeError, match="different worlds"):
        compare_visual_world(left, _write_plan(tmp_path / "moved.json", moved))


def test_group_world_equivalence_stops_a_group_whose_visuals_drifted(tmp_path):
    context = group_stage_context(group=_core_group())
    left = _write_plan(tmp_path / "v0.json", _visual_plan())
    drifted = _visual_plan()
    drifted["scene"] = {"scene_id": "another_room"}
    right = _write_plan(tmp_path / "v1.json", drifted)
    results = [
        {"work_item_id": "group_one/v0:plan:01", "stage": "plan",
         "scope_id": "group_one/v0", "status": "pass",
         "facts": {"episode_plan_path": str(left)}, "outputs": {}},
        {"work_item_id": "group_one/v1:plan:01", "stage": "plan",
         "scope_id": "group_one/v1", "status": "pass",
         "facts": {"episode_plan_path": str(right)}, "outputs": {}},
    ]
    verdict = group_world_equivalence(context, results)
    assert verdict["status"] == "fail"
    assert "did not plan one controlled world" in verdict["reason"]
    assert group_world_equivalence(context, results[:1])["status"] == "not_run"


def test_group_spec_pairs_the_members_that_share_one_modality():
    visual = {"vA": {"capture": "/tmp/capA", "visual_video": "/tmp/a.mp4"},
              "vB": {"capture": "/tmp/capB", "visual_video": "/tmp/b.mp4"}}
    variants = {name: {"facts": f"/tmp/{name}_facts.json", "audio": f"/tmp/{name}.wav"}
                for name in ("vA_a0", "vA_a1", "vB_a0", "vB_a1")}
    spec = _group_spec(
        "group_one", "world_one", "hm3d", "room_one", visual, variants,
        member_units=[("vA_a0", "vA", "vA_a0"), ("vA_a1", "vA", "vA_a1"),
                      ("vB_a0", "vB", "vB_a0"), ("vB_a1", "vB", "vB_a1")],
        task_family="visible_binding", split="pilot")
    group = spec["groups"][0]
    assert [member["member_id"] for member in group["members"]] == [
        "vA_a0", "vA_a1", "vB_a0", "vB_a1"]
    assert {tuple(row["members"]) + (row["shared_modality"],)
            for row in group["comparisons"]} == {
        ("vA_a0", "vA_a1", "video"), ("vB_a0", "vB_a1", "video"),
        ("vA_a0", "vB_a0", "audio"), ("vA_a1", "vB_a1", "audio")}


def _saved_result(root, group_id, unit_id, stage, attempt, status="pass"):
    path = (root / group_id / unit_id / stage / f"attempt_{attempt:02d}"
            / STAGE_RESULT_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": GROUP_STAGE_SCHEMA,
        "work_item_id": f"{group_id}/{unit_id}:{stage}:{attempt:02d}",
        "stage": stage, "request_id": f"{group_id}/{unit_id}",
        "scope_id": f"{group_id}/{unit_id}", "status": status,
        "facts": {}, "outputs": {}, "reason": None, "depends_on": [],
    }), encoding="utf-8")
    return path


def test_saved_stage_results_are_restored_in_attempt_order(tmp_path):
    _saved_result(tmp_path, "group_one", "v0", "plan", 1)
    _saved_result(tmp_path, "group_one", "v0_capture", "capture", 1, status="fail")
    _saved_result(tmp_path, "group_one", "v0_capture", "capture", 2)
    _saved_result(tmp_path, "other_group", "v0", "plan", 1)
    restored = load_group_stage_results(tmp_path, "group_one")
    assert [row["work_item_id"] for row in restored] == [
        "group_one/v0:plan:01",
        "group_one/v0_capture:capture:01",
        "group_one/v0_capture:capture:02",
    ]
    assert [row["status"] for row in restored] == ["pass", "fail", "pass"]
    assert "schema" not in restored[0]
    assert load_group_stage_results(tmp_path, "missing_group") == []


def _work_item(context, unit_id, stage, attempt=1):
    group_id = context["group_id"]
    return {
        "work_item_id": f"{group_id}/{unit_id}:{stage}:{attempt:02d}",
        "stage": stage, "request_id": f"{group_id}/{unit_id}",
        "scope_id": f"{group_id}/{unit_id}", "unit_id": unit_id,
        "group_id": group_id, "task_family": context["task_family"],
        "attempt": attempt,
        "resource": {"kind": "cpu", "execution": "cpu", "runtime_context": "pure_python"},
        "fresh_output_relative": f"{group_id}/{unit_id}/{stage}/attempt_{attempt:02d}",
        "depends_on": [], "inputs": {}, "payload": {}, "member_request_ids": [],
    }


def test_a_saved_attempt_is_restored_instead_of_rerun(tmp_path, monkeypatch):
    context = group_stage_context(group=_core_group())
    item = _work_item(context, "v0", "plan")
    _saved_result(tmp_path, context["group_id"], "v0", "plan", 1)
    calls = []
    monkeypatch.setitem(STAGE_RUNNERS, "visual_plan",
                        lambda *a, **k: calls.append(1))
    restored = run_group_stage_work_item(item, context, output_root=tmp_path)
    assert restored["status"] == "pass"
    assert calls == []


def test_an_interrupted_attempt_is_kept_and_a_new_attempt_is_asked_for(tmp_path):
    context = group_stage_context(group=_core_group())
    item = _work_item(context, "v0", "plan")
    partial = tmp_path / item["fresh_output_relative"]
    partial.mkdir(parents=True)
    (partial / "half_written.json").write_text("{}", encoding="utf-8")
    result = run_group_stage_work_item(item, context, output_root=tmp_path)
    assert result["status"] == "fail"
    assert "needs a new attempt" in result["reason"]
    assert (partial / "half_written.json").is_file()


def test_a_failing_unit_saves_its_failure_instead_of_raising(tmp_path, monkeypatch):
    context = group_stage_context(group=_core_group())
    item = _work_item(context, "v0", "plan")

    def explode(*args, **kwargs):
        raise BindingNativeError("planner refused this candidate")

    monkeypatch.setitem(STAGE_RUNNERS, "visual_plan", explode)
    result = run_group_stage_work_item(item, context, output_root=tmp_path)
    assert result["status"] == "fail"
    assert "planner refused this candidate" in result["reason"]
    saved = load_group_stage_results(tmp_path, context["group_id"])
    assert [row["status"] for row in saved] == ["fail"]
    assert "Traceback" in saved[0]["outputs"]["traceback"]


def test_a_work_item_from_another_group_is_refused(tmp_path):
    context = group_stage_context(group=_core_group())
    item = _work_item(context, "v0", "plan")
    item["group_id"] = "another_group"
    with pytest.raises(BindingNativeError, match="belongs to group"):
        run_group_stage_work_item(item, context, output_root=tmp_path)


def test_a_stage_filed_against_the_wrong_unit_is_refused(tmp_path):
    context = group_stage_context(group=_core_group())
    item = _work_item(context, "v0", "capture")
    with pytest.raises(BindingNativeError, match="filed against the"):
        run_group_stage_work_item(item, context, output_root=tmp_path)


def test_one_shared_visual_evidence_root_per_group(tmp_path):
    root = shared_visual_evidence_root(tmp_path, "group_one")
    assert root == (tmp_path / "group_one" / "shared_visual_evidence").resolve()


def test_retained_visual_roots_must_name_units_of_this_group(tmp_path):
    retained = tmp_path / "retained"
    (retained / "plan").mkdir(parents=True)
    (retained / "plan/episode_plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BindingNativeError, match="not visual units"):
        group_stage_context(group=_core_group(),
                            retained_visual_roots={"not_a_unit": retained})
    context = group_stage_context(group=_core_group(),
                                  retained_visual_roots={"v0": retained})
    assert context["retained_visual_roots"] == {"v0": str(retained.resolve())}


def test_an_undeclared_world_default_is_not_a_different_world():
    group = _core_group()
    requests = {member.request_id: member.to_legacy_request() for member in group.members}
    requests["member_2"]["motion_timing"] = None
    requests["member_2"]["audio_layouts"] = None
    contract = controlled_world_contract(group.to_dict(), requests)
    assert contract["status"] == "pass"
    assert contract["shared_world"]["motion_timing"] == "none"
    assert contract["shared_world"]["audio_layouts"] == [
        {"type": "binaural", "channel_count": 2, "role": "primary"}]


def test_a_retained_visual_root_from_another_seed_is_refused():
    group = _core_group()
    contract = controlled_world_contract(
        group.to_dict(),
        {member.request_id: member.to_legacy_request() for member in group.members})
    retained = _legacy_request("retained_episode", [BLUE, GREEN])
    retained["source_asset_ids"] = [BLUE, GREEN]
    assert verify_retained_visual_request(
        retained, contract, label="v0")["status"] == "pass"
    retained["seed"] = 9999
    with pytest.raises(BindingNativeError, match="different world"):
        verify_retained_visual_request(retained, contract, label="v0")


def test_a_retained_visual_root_keeps_its_own_episode_and_questions():
    group = _core_group()
    contract = controlled_world_contract(
        group.to_dict(),
        {member.request_id: member.to_legacy_request() for member in group.members})
    retained = _legacy_request("another_episode_id", [BLUE, GREEN])
    retained["qa_ids"] = ["QA-01", "QA-07"]
    retained["source_asset_ids"] = [GREEN, BLUE]
    checked = verify_retained_visual_request(retained, contract, label="v1")
    assert checked["retained_episode_id"] == "another_episode_id"
    assert "qa_ids" not in checked["checked_fields"]


def test_retained_candidate_defaults_to_zero_but_rotation_is_refused():
    group = _core_group()
    contract = controlled_world_contract(
        group.to_dict(),
        {member.request_id: member.to_legacy_request() for member in group.members})
    retained = _legacy_request("retained_candidate0", [BLUE, GREEN])
    assert verify_retained_visual_request(
        retained, contract, label="v0")["status"] == "pass"
    retained["sampling_candidate_index"] = 1
    with pytest.raises(BindingNativeError, match="sampling_candidate_index"):
        verify_retained_visual_request(retained, contract, label="v0")


def test_a_retained_visual_root_with_another_hrtf_is_refused():
    group = _core_group()
    contract = controlled_world_contract(
        group.to_dict(),
        {member.request_id: member.to_legacy_request() for member in group.members})
    retained = _legacy_request("retained_episode", [BLUE, GREEN])
    retained["runtime"]["hrtf"] = "/tmp/another.sofa"
    with pytest.raises(BindingNativeError, match="runtime.hrtf"):
        verify_retained_visual_request(retained, contract, label="v0")


def test_an_audio_column_is_reused_only_after_rechecking_this_run(tmp_path):
    from avengine.dataset.binding_group_native import _audio_reuse_candidate
    context = group_stage_context(group=_core_group())
    readback = tmp_path / "neutral.json"
    readback.write_text(json.dumps({
        "clock": {"frame_count": 2},
        "camera": [{"frame_index": 0, "position_m": [0, 0, 0]}],
        "entities": {"source1": [{"frame_index": 0, "emitter": [1, 0, 0]}]},
    }), encoding="utf-8")
    report = tmp_path / "research_report.json"
    report.write_text("{}", encoding="utf-8")
    plan_path = tmp_path / "assignment_plan.json"
    assignment = {"audio_events": [{"event_id": "event_001"}],
                  "voice_bindings": [{"sound_asset_id": "sound_one"}]}
    plan_path.write_text(json.dumps(assignment), encoding="utf-8")
    acoustics = {"rir_stride": 5}
    sibling = {
        "work_item_id": "group_one/v0_a0:audio:01", "stage": "audio",
        "scope_id": "group_one/v0_a0", "status": "pass",
        "facts": {"audio_report_path": str(report)},
        "outputs": {"acoustic_identity": acoustics,
                     "neutral_readback": str(readback),
                     "assignment_plan_path": str(plan_path)},
    }
    item = _work_item(context, "v1_a0", "audio")
    accepted = _audio_reuse_candidate(
        context, item, [sibling], column="a0", assignment_plan=assignment,
        acoustics=acoustics, neutral_readback=str(readback))
    assert accepted["reused"] is True
    assert accepted["source_unit_id"] == "v0_a0"
    assert "native_camera_clock_emitter_readback" in accepted["checked"]

    changed = _audio_reuse_candidate(
        context, item, [sibling], column="a0", assignment_plan=assignment,
        acoustics={"rir_stride": 3}, neutral_readback=str(readback))
    assert changed["reused"] is False
    assert changed["rejected"][0]["reason"] == (
        "acoustic input or configuration identity differs")

    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps({
        "clock": {"frame_count": 2},
        "camera": [{"frame_index": 0, "position_m": [3, 0, 0]}],
        "entities": {"source1": [{"frame_index": 0, "emitter": [1, 0, 0]}]},
    }), encoding="utf-8")
    drifted = _audio_reuse_candidate(
        context, item, [sibling], column="a0", assignment_plan=assignment,
        acoustics=acoustics, neutral_readback=str(moved))
    assert drifted["reused"] is False
    assert "readbacks differ" in drifted["rejected"][0]["reason"]

    other_events = {"audio_events": [{"event_id": "event_002"}],
                    "voice_bindings": assignment["voice_bindings"]}
    rebound = _audio_reuse_candidate(
        context, item, [sibling], column="a0", assignment_plan=other_events,
        acoustics=acoustics, neutral_readback=str(readback))
    assert rebound["reused"] is False
    assert rebound["rejected"][0]["reason"] == (
        "rebound audio events or voice bindings differ")

    report.unlink()
    gone = _audio_reuse_candidate(
        context, item, [sibling], column="a0", assignment_plan=assignment,
        acoustics=acoustics, neutral_readback=str(readback))
    assert gone["reused"] is False
    assert gone["rejected"][0]["reason"] == "no readable audio report"


def test_the_visual_compatibility_record_is_not_an_acoustic_input():
    left = {"audio_events": [{"event_id": "event_001", "start_sample": 100,
                              "target_sound_compatibility": [{"target_asset_id": BLUE}]}],
            "voice_bindings": [{"sound_asset_id": "sound_one", "path": "/tmp/one.wav",
                                "target_sound_compatibility": [{"target_asset_id": BLUE}]}]}
    right = deepcopy(left)
    right["audio_events"][0]["target_sound_compatibility"] = [{"target_asset_id": GREEN}]
    right["voice_bindings"][0]["target_sound_compatibility"] = [{"target_asset_id": GREEN}]
    assert audio_render_inputs(left) == audio_render_inputs(right)
    right["audio_events"][0]["start_sample"] = 200
    assert audio_render_inputs(left) != audio_render_inputs(right)
    moved_path = deepcopy(left)
    moved_path["voice_bindings"][0]["path"] = "/tmp/two.wav"
    assert audio_render_inputs(left) != audio_render_inputs(moved_path)


def _slotted_plan(order, *, camera_x=0.0, actor_x=0.0, moving=False):
    plan = _visual_plan()
    plan["visual_plan"]["actors"] = [
        {"actor_id": f"source{index + 1}", "asset_id": asset,
         "entity_instance_id": f"{asset}#instance01"}
        for index, asset in enumerate(order)
    ]
    for frame in plan["visual_plan"]["frames"]:
        frame["camera_state"] = {"position_m": [camera_x, 0, 0]}
        frame["actor_states"] = [
            {"actor_id": f"source{index + 1}", "asset_id": asset,
             "entity_instance_id": f"{asset}#instance01",
             "root_transform": {"translation_m": [actor_x + index, 0, 0]},
             "moving": moving}
            for index, asset in enumerate(order)
        ]
    plan["camera_condition_sampling"] = {
        "legal_candidate_ids": ["grid_00052_yaw_195"],
        "planned_query_window": {"windows": [
            {"actor_id": "source2", "entity_instance_id": f"{order[1]}#instance01"}]},
    }
    return plan


def test_a_swapped_source_slot_is_the_intervention_not_world_drift(tmp_path):
    left = _write_plan(tmp_path / "v0.json", _slotted_plan([BLUE, GREEN]))
    right = _write_plan(tmp_path / "v1.json", _slotted_plan([GREEN, BLUE]))
    with pytest.raises(BindingNativeError, match="transform-driving"):
        compare_visual_plans(left, right)
    result = compare_controlled_visual_plans(
        left, right, expected_slot_assets={"left": [BLUE, GREEN],
                                           "right": [GREEN, BLUE]})
    assert result["status"] == "pass"
    assert result["slot_identities"]["left"]["source1"]["asset_id"] == BLUE
    assert result["slot_identities"]["right"]["source1"]["asset_id"] == GREEN


def test_a_resampled_world_is_still_refused_under_the_slot_rewrite(tmp_path):
    left = _write_plan(tmp_path / "v0.json", _slotted_plan([BLUE, GREEN]))
    moved_camera = _write_plan(tmp_path / "camera.json",
                               _slotted_plan([GREEN, BLUE], camera_x=2.0))
    with pytest.raises(BindingNativeError, match="one controlled world"):
        compare_controlled_visual_plans(left, moved_camera)
    moved_actor = _write_plan(tmp_path / "actor.json",
                              _slotted_plan([GREEN, BLUE], actor_x=1.0))
    with pytest.raises(BindingNativeError, match="one controlled world"):
        compare_controlled_visual_plans(left, moved_actor)
    started_moving = _write_plan(tmp_path / "moving.json",
                                 _slotted_plan([GREEN, BLUE], moving=True))
    with pytest.raises(BindingNativeError, match="one controlled world"):
        compare_controlled_visual_plans(left, started_moving)


def test_a_plan_that_holds_another_asset_than_declared_is_refused(tmp_path):
    left = _write_plan(tmp_path / "v0.json", _slotted_plan([BLUE, GREEN]))
    right = _write_plan(tmp_path / "v1.json", _slotted_plan([GREEN, BLUE]))
    with pytest.raises(BindingNativeError, match="its group declared"):
        compare_controlled_visual_plans(
            left, right, expected_slot_assets={"left": [BLUE, "asset_yellow"],
                                               "right": [GREEN, BLUE]})


def test_plan_slot_identities_read_actors_then_the_first_frame():
    plan = _slotted_plan([BLUE, GREEN])
    for actor in plan["visual_plan"]["actors"]:
        actor.pop("entity_instance_id")
    identities = plan_slot_identities(plan)
    assert identities["source1"]["asset_id"] == BLUE
    assert identities["source1"]["entity_instance_id"] == f"{BLUE}#instance01"
