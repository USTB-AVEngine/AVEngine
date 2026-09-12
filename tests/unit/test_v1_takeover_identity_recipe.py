from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.dataset import binding_group_identity as identity
from avengine.dataset import binding_group_native as native
from avengine.dataset.production_spec import (
    CoreGroupRequest,
    group_stage_units,
    production_request_from_legacy,
    recipe_for_task_family,
)


REPOSITORY = Path(__file__).resolve().parents[2]
IDENTITY_ROOT = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "binding_dataset_20260909_v2/identity_first/group_v14"
)
MEMBER_IDS = ("m0", "m1", "m2", "m3")


def _identity_group() -> CoreGroupRequest:
    members = []
    shared_request = json.loads(
        (IDENTITY_ROOT / "variants/v0_a0/request.json").read_text(
            encoding="utf-8"
        )
    )
    for member_id in MEMBER_IDS:
        request = deepcopy(shared_request)
        request["task_family"] = "cross_event_identity"
        request["group_id"] = "identity_stage_probe"
        members.append(
            production_request_from_legacy(
                request, request_id=member_id, kind="core_group_member"
            )
        )
    return CoreGroupRequest(
        group_id="identity_stage_probe",
        task_family="cross_event_identity",
        room_id=members[0].room_id,
        members=tuple(members),
        shared_audio_member_ids=(("m0", "m2"), ("m1", "m3")),
    )


def _context() -> dict:
    group = _identity_group()
    summary = json.loads(
        (IDENTITY_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    probe_plan = json.loads(
        (IDENTITY_ROOT / "probe/visual/plan/episode_plan.json").read_text(
            encoding="utf-8"
        )
    )
    events = probe_plan["audio_events"]
    sound_pair = {
        "first_sound": deepcopy(events[0]),
        "second_sound": deepcopy(
            json.loads(
                (IDENTITY_ROOT / "visual/v0/plan/episode_plan.json").read_text(
                    encoding="utf-8"
                )
            )["audio_events"][1]
        ),
    }
    retained = {
        "identity_probe_plan": IDENTITY_ROOT / "probe/visual",
        "identity_probe_capture": IDENTITY_ROOT / "probe/visual",
        "identity_probe_audio": IDENTITY_ROOT / "probe/variant",
        "v0": IDENTITY_ROOT / "visual/v0",
        "v1": IDENTITY_ROOT / "visual/v1",
        "v0_a0": IDENTITY_ROOT / "variants/v0_a0",
        "v0_a1": IDENTITY_ROOT / "variants/v0_a1",
        "v1_a0": IDENTITY_ROOT / "variants/v1_a0",
        "v1_a1": IDENTITY_ROOT / "variants/v1_a1",
    }
    return identity.identity_group_stage_context(
        group=group,
        retained_visual_roots=retained,
        identity_sound_pair=sound_pair,
        identity_topology=summary["routes"],
        identity_base_episode_root=IDENTITY_ROOT / "cpu_materialize_20260910",
        world_id=summary["world_id"],
        split="pilot",
    )


def _item(context: dict, unit_id: str, *, attempt: int = 1, inputs=None) -> dict:
    row = next(
        row for row in context["group_spec"]["stage_units"]
        if row["unit_id"] == unit_id
    )
    stage = str(row["stage"])
    return {
        "work_item_id": (
            f"{context['group_id']}/{unit_id}:{stage}:{attempt:02d}"
        ),
        "request_id": f"{context['group_id']}/{unit_id}",
        "scope_id": f"{context['group_id']}/{unit_id}",
        "group_id": context["group_id"],
        "task_family": context["task_family"],
        "unit_id": unit_id,
        "stage": stage,
        "member_request_ids": list(row.get("member_request_ids") or ()),
        "fresh_output_relative": (
            f"{context['group_id']}/{unit_id}/{stage}/attempt_{attempt:02d}"
        ),
        "depends_on": list(row.get("depends_on_units") or ()),
        "inputs": inputs or {},
        "resource": {"kind": "cpu", "execution": "cpu"},
        "payload": {},
    }


def _run(
    context: dict,
    unit_id: str,
    results: list[dict],
    output_root: Path,
    *,
    inputs=None,
    attempt: int = 1,
) -> dict:
    item = _item(context, unit_id, attempt=attempt, inputs=inputs)
    result = identity.run_identity_group_stage_work_item(
        item, context, output_root=output_root, results=results
    )
    assert result["status"] == "pass", result
    results.append(result)
    return result


def test_identity_recipe_has_internal_probe_and_four_public_members():
    recipe = recipe_for_task_family("cross_event_identity")
    assert [unit.unit_id for unit in recipe.units_in_dependency_order()] == [
        "identity_probe_plan",
        "identity_probe_capture",
        "identity_probe_audio",
        "identity_topology",
        "v0",
        "v1",
        "v0_capture",
        "v1_capture",
        "v0_a0",
        "v0_a1",
        "v1_a0",
        "v1_a1",
        "group",
    ]
    internal = [unit for unit in recipe.units if unit.internal_only]
    assert [unit.unit_id for unit in internal] == [
        "identity_probe_plan",
        "identity_probe_capture",
        "identity_probe_audio",
        "identity_topology",
    ]
    assert all(unit.member_index is None for unit in internal)
    assert recipe.member_unit_ids == ("v0_a0", "v0_a1", "v1_a0", "v1_a1")


def test_real_retained_identity_pipeline_keeps_probe_internal_and_assembles(
    tmp_path: Path,
):
    context = _context()
    group = _identity_group()
    units = {row["unit_id"]: row for row in group_stage_units(group)}
    assert all(
        units[unit_id]["member_request_ids"] == []
        for unit_id in (
            "identity_probe_plan",
            "identity_probe_capture",
            "identity_probe_audio",
            "identity_topology",
        )
    )
    output_root = tmp_path / "identity_work"
    results: list[dict] = []
    stage_results: dict[str, dict] = {}

    probe_plan = _run(context, "identity_probe_plan", results, output_root)
    stage_results["identity_probe_plan"] = probe_plan
    probe_capture = _run(
        context,
        "identity_probe_capture",
        results,
        output_root,
        inputs={"identity_probe_plan": probe_plan},
    )
    stage_results["identity_probe_capture"] = probe_capture
    probe_audio = _run(
        context,
        "identity_probe_audio",
        results,
        output_root,
        inputs={
            "identity_probe_plan": probe_plan,
            "identity_probe_capture": probe_capture,
        },
    )
    stage_results["identity_probe_audio"] = probe_audio
    topology = _run(
        context,
        "identity_topology",
        results,
        output_root,
        inputs={
            "identity_probe_plan": probe_plan,
            "identity_probe_audio": probe_audio,
        },
    )
    stage_results["identity_topology"] = topology

    assert probe_audio["outputs"]["native_visual_worlds_created"] == 0
    assert probe_audio["outputs"]["native_acoustic_contexts_created"] == 0
    assert topology["outputs"]["internal_only"] is True
    assert topology["outputs"]["measured_wet_tail_end_s"] == pytest.approx(
        4.1574375
    )

    public_plans = {}
    for unit_id in ("v0", "v1"):
        public_plans[unit_id] = _run(
            context,
            unit_id,
            results,
            output_root,
            inputs={"identity_topology": topology},
        )
        stage_results[unit_id] = public_plans[unit_id]
    assert public_plans["v0"]["outputs"]["measured_wet_tail_end_s"] == pytest.approx(
        4.1574375
    )
    assert public_plans["v1"]["outputs"]["measured_wet_tail_end_s"] == pytest.approx(
        4.1574375
    )

    public_captures = {}
    for unit_id, plan_id in (
        ("v0_capture", "v0"),
        ("v1_capture", "v1"),
    ):
        public_captures[unit_id] = _run(
            context,
            unit_id,
            results,
            output_root,
            inputs={plan_id: public_plans[plan_id]},
        )
        stage_results[unit_id] = public_captures[unit_id]
        assert public_captures[unit_id]["outputs"]["native_visual_worlds_created"] == 0
        assert Path(
            public_captures[unit_id]["outputs"]["neutral_readback"]
        ).is_file()

    public_audio = {}
    for unit_id, visual_id in (
        ("v0_a0", "v0_capture"),
        ("v0_a1", "v0_capture"),
        ("v1_a0", "v1_capture"),
        ("v1_a1", "v1_capture"),
    ):
        public_audio[unit_id] = _run(
            context,
            unit_id,
            results,
            output_root,
            inputs={visual_id: public_captures[visual_id]},
        )
        stage_results[unit_id] = public_audio[unit_id]
        outputs = public_audio[unit_id]["outputs"]
        assert Path(outputs["visual_video"]).is_file()
        assert Path(outputs["audio"]).is_file()
        assert outputs["shared_audio_column"]["reused"] is False
        assert outputs["ancillary_audio_outputs"] is None

    assembly_inputs = {
        unit_id: row for unit_id, row in stage_results.items()
    }
    assembly_item = _item(
        context, "group", attempt=1, inputs=assembly_inputs
    )
    assembly = identity.run_identity_group_stage_work_item(
        assembly_item, context, output_root=output_root, results=results
    )
    assert assembly["status"] == "pass", assembly
    validation = assembly["facts"]["validation"]
    assert validation["status"] == "pass"
    assert validation["internal_probe_excluded"] is True
    assert validation["group_count"] == 1
    assert validation["world_count"] == 1
    assert validation["sample_count"] == 4
    assert validation["media_validation"] == "media_checked"
    assert validation["pcm_by_column"]["a0"]["same"] is True
    assert validation["pcm_by_column"]["a1"]["same"] is True
    group_validation = validation["group_validation"]
    assert group_validation["status"] == "pass"
    audio_comparisons = [
        row for row in group_validation["comparisons"]
        if row["shared_modality"] == "audio"
    ]
    assert len(audio_comparisons) == 2
    assert all(row["audio_reassignment_allowed"] is True for row in audio_comparisons)


# --- identity real geometry candidate pool ------------------------------------
# These read the same pools and plans production reads. The retained Apartment
# group and the fresh M21 plan are the two real cases: one has a pool that
# serves its planner placement, the other does not and must say so with an
# executable query request instead of a relocated actor.
APARTMENT_POOL = (
    "tmp/binding_dataset_20260909_v2/identity_room_preflight_v1/apartment/"
    "native_common_endpoint_query_v5.json"
)
RETAINED_APARTMENT_PLAN = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/identity_first/apartment_group_raw_v1/"
    "variants/v0_a0/plan/episode_plan.json"
)
FRESH_APARTMENT_PLAN = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/M21_CORE_SOURCE_BINDINGS/"
    "attempt_20260911T011155Z_pid932007/identity_probe_plan/base_plan/plan/"
    "episode_plan.json"
)
FRESH_APARTMENT_REQUEST = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/M21_CORE_SOURCE_BINDINGS/"
    "attempt_20260911T010638Z_pid927948/prepared/requests/"
    "takeover_cross_event_identity_apartment_g01_v0_a0.json"
)


def _retained_apartment_case() -> tuple[dict, dict, dict, dict]:
    plan = json.loads(RETAINED_APARTMENT_PLAN.read_text(encoding="utf-8"))
    request = deepcopy(plan["request"])
    request["binding_identity"].pop("native_polyline_query_path", None)
    request["binding_identity"]["native_geometry_pools"] = [
        {"path": APARTMENT_POOL, "room_id": "legacy_ue_apartment_0000_v1"},
    ]
    events = plan["audio_events"]
    sounds = tuple(
        {
            "sample_count": int(row["sample_count"]),
            "sample_rate_hz": int(row["sample_rate_hz"]),
        }
        for row in events[:2]
    )
    return plan, request, sounds[0], sounds[1]


def _fresh_apartment_case() -> tuple[dict, dict]:
    return (
        json.loads(FRESH_APARTMENT_PLAN.read_text(encoding="utf-8")),
        json.loads(FRESH_APARTMENT_REQUEST.read_text(encoding="utf-8")),
    )


@pytest.mark.skipif(
    not RETAINED_APARTMENT_PLAN.is_file(),
    reason="retained Apartment identity plan is not present",
)
def test_declared_geometry_pool_serves_the_retained_apartment_placement():
    plan, request, first, second = _retained_apartment_case()
    report = identity.identity_geometry_pool_report(plan, request)
    assert report["status"] == "compatible", report
    row = report["selected_pool"]
    assert row["candidate_count"] == 23, row
    assert row["pool_schema"] == (
        "avengine_binding_identity_apartment_common_endpoint_query_v1"
    )
    # The pool is only usable because it declares this plan's own starts.
    assert row["start_match_distances_m"] == {"source1": 0.0, "source2": 0.0}
    route, tracks = identity._select_topology(plan, request, None, first, second)
    assert int(route["native_polyline_query_candidate_index"]) == 778
    assert route["motion_counts"] == [27, 27]
    assert route["event2_start_s"] == pytest.approx(4.666666666666667)
    assert route["native_timing_preserved"] is False
    assert sorted(tracks) == ["v0", "v1"]


@pytest.mark.skipif(
    not RETAINED_APARTMENT_PLAN.is_file(),
    reason="retained Apartment identity plan is not present",
)
def test_legacy_single_pool_path_still_resolves_to_the_same_candidates():
    plan, request, first, second = _retained_apartment_case()
    legacy = deepcopy(request)
    legacy["binding_identity"].pop("native_geometry_pools")
    legacy["binding_identity"]["native_polyline_query_path"] = APARTMENT_POOL
    assert [row["path"] for row in identity._geometry_pool_specs(legacy)] == [
        APARTMENT_POOL
    ]
    route, _tracks = identity._select_topology(plan, legacy, None, first, second)
    assert int(route["native_polyline_query_candidate_index"]) == 778


@pytest.mark.skipif(
    not FRESH_APARTMENT_PLAN.is_file(),
    reason="fresh M21 Apartment identity plan is not present",
)
def test_pool_that_does_not_cover_the_planner_placement_requests_a_query():
    plan, request = _fresh_apartment_case()
    request["binding_identity"]["native_geometry_pools"] = [{"path": APARTMENT_POOL}]
    sound = {"sample_count": 37440, "sample_rate_hz": 16000}
    with pytest.raises(identity.IdentityGeometryQueryRequired) as raised:
        identity._select_topology(plan, request, None, sound, sound)
    report = raised.value.report
    assert report["status"] == "requires_native_geometry_query"
    assert report["selected_pool"] is None
    reasons = report["pools"][0]["reasons"]
    assert reasons and all("no_legal_common_endpoint" in text for text in reasons)
    # The request must carry the plan's own starts, never the pool's.
    query = report["query_request"]
    assert query["schema"] == identity.IDENTITY_GEOMETRY_QUERY_REQUEST_SCHEMA
    assert query["fixed_starts_m"] == [
        report["plan_starts_m"]["source1"], report["plan_starts_m"]["source2"],
    ]
    assert query["fixed_starts_m"] != report["pools"][0]["pool_fixed_starts_m"]
    assert query["native_resources"]["preferred_session"] == "identity_probe_capture"
    assert query["budget_claim"]["extra_visual_launches_if_shared"] == 0
    assert query["budget_claim"]["started_here"] == 0
    assert query["declared_filters"]["path_length_range_m"] == [0.8, 2.0]


@pytest.mark.skipif(
    not FRESH_APARTMENT_PLAN.is_file(),
    reason="fresh M21 Apartment identity plan is not present",
)
def test_route_bank_block_reports_the_measured_structure_not_just_a_string():
    plan, request = _fresh_apartment_case()
    assert identity._geometry_pool_specs(request) == []
    sound = {"sample_count": 37440, "sample_rate_hz": 16000}
    with pytest.raises(identity.IdentityGeometryQueryRequired) as raised:
        identity._select_topology(plan, request, None, sound, sound)
    bank = raised.value.report["route_bank"]
    assert bank["available"] is True
    assert bank["route_count"] == 113
    assert bank["distinct_route_pairs"] == 6328
    # A walker never leaves the retained route its start sits on, so a common
    # endpoint needs two routes sharing a vertex. The Apartment bank has one
    # such pair in 6328, which is why re-planning alone cannot fix this.
    assert bank["route_pairs_sharing_a_vertex"] == 1
    assert bank["plan_starts_on_a_shared_route"] is False


@pytest.mark.skipif(
    not FRESH_APARTMENT_PLAN.is_file(),
    reason="fresh M21 Apartment identity plan is not present",
)
def test_geometry_query_execution_refuses_without_authorization(tmp_path: Path):
    plan, request = _fresh_apartment_case()
    query = identity.identity_geometry_query_request(
        plan, request, reason="unit_test"
    )
    receipt = tmp_path / "receipt.json"
    refused = identity.execute_identity_geometry_query(query, receipt)
    assert refused["status"] == "refused"
    assert refused["reason"] == "native_geometry_query_not_authorized"
    assert refused["native_visual_worlds_created"] == 0
    assert refused["native_acoustic_contexts_created"] == 0
    assert not receipt.exists()
    # An authorized call still refuses to open a renderer of its own.
    with pytest.raises(identity.IdentityNativeError):
        identity.execute_identity_geometry_query(
            query, receipt,
            authorization=identity.IDENTITY_GEOMETRY_QUERY_AUTHORIZATION,
        )
    assert not receipt.exists()


@pytest.mark.skipif(
    not FRESH_APARTMENT_PLAN.is_file(),
    reason="fresh M21 Apartment identity plan is not present",
)
def test_authorized_geometry_query_writes_the_session_receipt(tmp_path: Path):
    plan, request = _fresh_apartment_case()
    query = identity.identity_geometry_query_request(
        plan, request, reason="unit_test", requested_common_endpoints=4
    )
    seen: list[dict] = []

    def _query(payload):
        seen.append(payload)
        return {
            "schema": sorted(identity.IDENTITY_GEOMETRY_POOL_SCHEMAS)[0],
            "status": "pass",
            "fixed_starts_m": payload["fixed_starts_m"],
            "candidate_pairs": [],
        }

    receipt = tmp_path / "receipt.json"
    done = identity.execute_identity_geometry_query(
        query, receipt,
        authorization=identity.IDENTITY_GEOMETRY_QUERY_AUTHORIZATION,
        session={"query": _query, "shared": True,
                 "native_visual_worlds_created": 0},
    )
    assert done["status"] == "pass"
    assert done["shared_session"] is True
    assert done["native_visual_worlds_created"] == 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "pass"
    assert seen[0]["fixed_starts_m"] == query["fixed_starts_m"]


@pytest.mark.skipif(
    not RETAINED_APARTMENT_PLAN.is_file(),
    reason="retained Apartment identity plan is not present",
)
def test_receipt_adoption_recovers_a_pool_without_new_native_work():
    plan, request, first, second = _retained_apartment_case()
    bare = deepcopy(request)
    bare["binding_identity"].pop("native_geometry_pools")
    adopted = identity.adopt_identity_geometry_query_receipt(
        APARTMENT_POOL, plan, bare
    )
    assert adopted["status"] == "pass"
    assert adopted["native_visual_worlds_created"] == 0
    assert adopted["native_acoustic_contexts_created"] == 0
    assert adopted["pool"]["candidate_count"] == 23
    route, _tracks = identity._select_topology(
        plan, adopted["request"], None, first, second
    )
    assert int(route["native_polyline_query_candidate_index"]) == 778


@pytest.mark.skipif(
    not FRESH_APARTMENT_PLAN.is_file(),
    reason="fresh M21 Apartment identity plan is not present",
)
def test_receipt_adoption_rejects_a_receipt_for_another_placement():
    plan, request = _fresh_apartment_case()
    with pytest.raises(identity.IdentityNativeError) as raised:
        identity.adopt_identity_geometry_query_receipt(
            APARTMENT_POOL, plan, request
        )
    assert "does not serve this plan" in str(raised.value)


def test_geometry_pool_declaration_errors_are_explicit():
    plan = json.loads(RETAINED_APARTMENT_PLAN.read_text(encoding="utf-8"))
    request = deepcopy(plan["request"])
    request["binding_identity"].pop("native_polyline_query_path", None)
    request["binding_identity"]["native_geometry_pools"] = "not-a-list"
    with pytest.raises(identity.IdentityNativeError):
        identity._geometry_pool_specs(request)
    request["binding_identity"]["native_geometry_pools"] = [{"kind": "x"}]
    with pytest.raises(identity.IdentityNativeError):
        identity._geometry_pool_specs(request)
    request["binding_identity"]["native_geometry_pools"] = [
        {"path": APARTMENT_POOL, "room_id": "some_other_room_v1"},
    ]
    report = identity.identity_geometry_pool_report(plan, request)
    assert report["status"] == "requires_native_geometry_query"
    assert any(
        "declared_pool_room_is_not_the_plan_room" in text
        for text in report["pools"][0]["reasons"]
    )


# --- generic audio recovery reachability -------------------------------------
# The ordinary runner recovers an interrupted audio attempt by looking
# recover_rendered_audio_attempt up on the recipe module
# (production_runner._invoke_recipe_audio_recovery). A module without it gets no
# recovery at all and the runner says nothing, so both the name and the attempt
# layout it resolves are part of this recipe's contract.
STATE_MATERIALIZED_ATTEMPT = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "binding_v1_parallel_20260910/TAKEOVER/T08_RELATION/"
    "attempt_20260910T205814Z_pid771755/state_audio_fresh/v0_a0/audio/attempt_01"
)


def test_recipe_module_exposes_the_runner_audio_recovery_hook():
    from avengine.dataset.production_runner import recipe_binding
    import importlib
    binding = recipe_binding("cross_event_identity")
    module = importlib.import_module(binding.module)
    recover = getattr(module, "recover_rendered_audio_attempt", None)
    assert callable(recover), binding.module
    assert "recover_rendered_audio_attempt" in module.__all__


def test_audio_recovery_resolves_this_recipe_variant_layout(tmp_path: Path):
    module = identity
    resolve = module._recipe_audio_attempt_root
    # episode layout, the shared native one
    episode = tmp_path / "attempt_01/episode"
    episode.mkdir(parents=True)
    (episode / "request.json").write_text("{}", encoding="utf-8")
    assert resolve(tmp_path / "attempt_01", {"unit_id": "v0_a0"}, {}) == episode
    # variants/<unit_id> layout, the one this recipe materializes
    variant = tmp_path / "attempt_02/variants/v0_a1"
    variant.mkdir(parents=True)
    (variant / "request.json").write_text("{}", encoding="utf-8")
    assert resolve(tmp_path / "attempt_02", {"unit_id": "v0_a1"}, {}) == variant
    # the unit id may only arrive through the lineage the runner builds
    assert resolve(tmp_path / "attempt_02", {}, {"unit_id": "v0_a1"}) == variant
    # a bare attempt root names what it looked at instead of failing deep inside
    with pytest.raises(Exception) as raised:
        resolve(tmp_path / "attempt_03", {"unit_id": "v0_a0"}, {})
    assert "looked at" in str(raised.value)


@pytest.mark.skipif(
    not STATE_MATERIALIZED_ATTEMPT.is_dir(),
    reason="retained materialized state audio attempt is not present",
)
def test_audio_recovery_delegation_reaches_the_shared_receipt_check(tmp_path: Path):
    module = identity
    item = {"unit_id": "v0_a0", "scope_id": "cross_time_state_hm3d_0001/v0_a0",
            "group_id": "cross_time_state_hm3d_0001"}
    # This retained attempt was materialized but never rendered, so the shared
    # implementation has to reject it by name rather than invent a delivery.
    with pytest.raises(native.BindingNativeError) as raised:
        module.recover_rendered_audio_attempt(
            item, {"group_id": "cross_time_state_hm3d_0001"},
            output_root=tmp_path / "recovery",
            previous_attempt_root=STATE_MATERIALIZED_ATTEMPT,
            lineage={"scope_id": "cross_time_state_hm3d_0001/v0_a0",
                     "unit_id": "v0_a0"},
        )
    message = str(raised.value)
    assert "no raw native audio receipt" in message
    assert message.endswith("variants/v0_a0")
    assert not (tmp_path / "recovery").exists()


# --- the shared-vertex blocker is Apartment-only ------------------------------
HM3D_IDENTITY_PROBE_PLAN = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "binding_dataset_20260909_v2/identity_first/group_v14/probe/visual/plan/"
    "episode_plan.json"
)


@pytest.mark.skipif(
    not HM3D_IDENTITY_PROBE_PLAN.is_file(),
    reason="retained HM3D identity probe plan is not present",
)
def test_navmesh_rooms_do_not_take_the_route_bank_identity_branch():
    plan = json.loads(HM3D_IDENTITY_PROBE_PLAN.read_text(encoding="utf-8"))
    package = (plan.get("resources") or {}).get("room_package") or {}
    walkable = package.get("walkable_space") or {}
    # The package carries a route_bank path, but the kind is what dispatches, so
    # this room paths through space.shortest_path and needs no shared vertex.
    assert "route_bank" in walkable
    assert walkable.get("kind") == "habitat_navmesh"
    request = json.loads(
        (HM3D_IDENTITY_PROBE_PLAN.parents[2] / "../variants/v0_a0/request.json")
        .resolve().read_text(encoding="utf-8")
    )
    from avengine.capture.qa_plan_adapters import load_planning_resources
    space, _mesh, _layout = load_planning_resources(plan["resources"], request)
    assert space.route_bank() is None
    capability = identity._route_bank_identity_capability(plan, request, space)
    assert capability["available"] is False


@pytest.mark.skipif(
    not FRESH_APARTMENT_PLAN.is_file(),
    reason="fresh M21 Apartment identity plan is not present",
)
def test_apartment_is_the_room_family_that_declares_a_route_bank():
    plan = json.loads(FRESH_APARTMENT_PLAN.read_text(encoding="utf-8"))
    package = (plan.get("resources") or {}).get("room_package") or {}
    walkable = package.get("walkable_space") or {}
    assert walkable.get("kind") == "route_bank"
    assert package.get("family") == "apartment"


# --- fresh pinned Apartment identity plan ------------------------------------
# The retained pool only serves the two starts it declares, so a fresh group has
# to be planned AT those starts. C01's profile.pinned_static_positions_m does
# that through the ordinary planner, which then solves its own camera; reusing
# the retained camera would prove nothing.
C04R1_ATTEMPT = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/CLAUDE_C04_R1/"
    "attempt_20260911T065736Z_pid1490314"
)
FRESH_PINNED_BASE_PLAN = C04R1_ATTEMPT / (
    "stage_run_prepared/takeover_cross_event_identity_apartment_g01/"
    "identity_probe_plan/plan/attempt_01/base_plan.json"
)
FRESH_PINNED_REQUEST = C04R1_ATTEMPT / (
    "stage_run_prepared/takeover_cross_event_identity_apartment_g01/"
    "identity_probe_plan/plan/attempt_01/request.json"
)


@pytest.mark.skipif(
    not FRESH_PINNED_BASE_PLAN.is_file(),
    reason="the fresh pinned Apartment identity base plan is not present",
)
def test_pinned_fresh_base_plan_sits_on_the_pool_starts_with_its_own_camera():
    plan = json.loads(FRESH_PINNED_BASE_PLAN.read_text(encoding="utf-8"))
    pool = json.loads(
        (REPOSITORY / APARTMENT_POOL).read_text(encoding="utf-8")
    )
    starts = {
        state["actor_id"]: state["root_transform"]["translation_m"]
        for state in plan["visual_plan"]["frames"][0]["actor_states"]
    }
    assert [starts["source1"], starts["source2"]] == pool["fixed_starts_m"]
    activity = plan.get("activity_plan") or {}
    assert activity["selected_route_ids"] == ["r00596", "r01752"]
    assert sorted(activity["pinned_static_actor_indices"]) == [0, 1]
    camera = plan["visual_plan"]["camera"]
    sampling = plan.get("camera_condition_sampling") or {}
    # the planner solved this camera for this placement, and it is not the one
    # the retained group used
    assert sampling["legal_candidate_count"] >= 1
    assert camera["candidate_id"] in sampling["legal_candidate_ids"]
    assert camera["candidate_id"] != "grid_00077_yaw_330"


@pytest.mark.skipif(
    not FRESH_PINNED_BASE_PLAN.is_file(),
    reason="the fresh pinned Apartment identity base plan is not present",
)
def test_fresh_pinned_plan_selects_a_topology_from_the_declared_pool():
    plan = json.loads(FRESH_PINNED_BASE_PLAN.read_text(encoding="utf-8"))
    request = json.loads(FRESH_PINNED_REQUEST.read_text(encoding="utf-8"))
    assert (request.get("profile") or {})["pinned_static_positions_m"] == {
        "source1": [-1.4833, 0.28, 0.1295],
        "source2": [0.7308, 0.28, 1.2741],
    }
    report = identity.identity_geometry_pool_report(plan, request)
    assert report["status"] == "compatible", report
    assert report["selected_pool"]["start_match_distances_m"] == {
        "source1": 0.0, "source2": 0.0,
    }
    attempt = FRESH_PINNED_BASE_PLAN.parent
    first = json.loads((attempt / "first_sound.json").read_text(encoding="utf-8"))
    second = json.loads((attempt / "second_sound.json").read_text(encoding="utf-8"))
    route, tracks = identity._select_topology(plan, request, None, first, second)
    assert route["motion_query_authority"] == (
        "native_spear_ue_recast_common_endpoint_query"
    )
    assert route["native_timing_preserved"] is False
    assert sorted(tracks) == ["v0", "v1"]
    assert len(tracks["v0"]["source1"]) == int(plan["clock"]["frame_count"])
    # this is a fresh solution, not the retained group's candidate 778
    assert int(route["native_polyline_query_candidate_index"]) != 778
    assert route["event2_frame"] % int(request.get("rir_stride", 3)) == 0


# --- identity capture delegation boundary ------------------------------------
# A helper that passes on its own proves nothing about the chain: the M23 stall
# was a plan unit whose reported directory the capture could not use. This walks
# the real units from the saved batch manifest and replaces only the external
# renderer launch, so plan materialization, capture_command and the post-launch
# artifact contract all run for real.
M27_MANIFEST = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/M27_CAPTURE_DELEGATION/"
    "attempt_20260911T074059Z_pid1589812/prepared/batch_manifest.json"
)
IDENTITY_MANIFEST_GROUP = "takeover_cross_event_identity_apartment_g01"


class _RecordingSubprocess:
    """Stands in for the subprocess module inside binding_group_native.

    A plan-only planning run still goes to the real subprocess module, because
    that is CPU planning. Any other argv is a native renderer launch: it is
    recorded and not started, and a normal zero return is handed back so the
    caller's own post-launch contract still runs.
    """

    def __init__(self):
        import subprocess
        self._real = subprocess
        self.STDOUT = subprocess.STDOUT
        self.PIPE = subprocess.PIPE
        self.CompletedProcess = subprocess.CompletedProcess
        self.launches: list[dict] = []

    def run(self, command, **kwargs):
        argv = [str(item) for item in command]
        if any(item == "--plan-only" for item in argv):
            return self._real.run(command, **kwargs)
        self.launches.append({"argv": argv, "cwd": str(kwargs.get("cwd"))})
        return self._real.CompletedProcess(argv, 0)

    def __getattr__(self, name):
        import subprocess
        return getattr(subprocess, name)


def _manifest_item(context, unit_id, done):
    row = next(r for r in context["group_spec"]["stage_units"]
               if r["unit_id"] == unit_id)
    stage = str(row["stage"])
    served = list(row.get("member_request_ids") or ())
    inputs = {}
    if stage in ("plan", "late_plan") and served:
        inputs["request"] = deepcopy(context["member_requests"][served[0]])
    for name in (row.get("depends_on_units") or ()):
        result = done.get(str(name))
        if result is not None:
            inputs[str(name)] = {
                "work_item_id": result.get("work_item_id"),
                "facts": deepcopy(result.get("facts") or {}),
                "outputs": deepcopy(result.get("outputs") or {}),
            }
    return {
        "work_item_id": f"{context['group_id']}/{unit_id}:{stage}:01",
        "request_id": f"{context['group_id']}/{unit_id}",
        "scope_id": f"{context['group_id']}/{unit_id}",
        "group_id": context["group_id"],
        "task_family": context["task_family"],
        "unit_id": unit_id,
        "stage": stage,
        "member_request_ids": served,
        "fresh_output_relative": (
            f"{context['group_id']}/{unit_id}/{stage}/attempt_01"
        ),
        "depends_on": list(row.get("depends_on_units") or ()),
        "inputs": inputs,
        "resource": dict(row.get("resource") or {"kind": "cpu", "execution": "cpu"}),
        "payload": dict(row.get("payload") or {}),
    }


@pytest.mark.skipif(
    not M27_MANIFEST.is_file(),
    reason="the prepared main-config batch manifest is not present",
)
def test_identity_chain_reaches_the_native_capture_with_a_legal_argv(
    tmp_path: Path, monkeypatch,
):
    manifest = json.loads(M27_MANIFEST.read_text(encoding="utf-8"))
    recorder = _RecordingSubprocess()
    monkeypatch.setattr(native, "subprocess", recorder)
    context = identity.identity_group_stage_context(
        manifest, IDENTITY_MANIFEST_GROUP,
        world_id="world_identity_capture_boundary_probe", split="pilot",
    )
    done: dict[str, dict] = {}
    plan_result = identity.run_identity_group_stage_work_item(
        _manifest_item(context, "identity_probe_plan", done),
        context, output_root=tmp_path / "work", results=[],
    )
    assert plan_result["status"] == "pass", plan_result
    done["identity_probe_plan"] = plan_result
    # the pinned placement and the pool both survived prepare into this plan
    base_plan = json.loads(
        Path((plan_result["outputs"])["base_plan"]).read_text(encoding="utf-8")
    )
    starts = {
        state["actor_id"]: state["root_transform"]["translation_m"]
        for state in base_plan["visual_plan"]["frames"][0]["actor_states"]
    }
    assert starts["source1"] == [-1.4833, 0.28, 0.1295]
    assert starts["source2"] == [0.7308, 0.28, 1.2741]

    capture_result = identity.run_identity_group_stage_work_item(
        _manifest_item(context, "identity_probe_capture", done),
        context, output_root=tmp_path / "work", results=[plan_result],
    )
    # The renderer never ran, so the unit must fail on the artifact contract -
    # and it must have got far enough to build one real launch first.
    assert capture_result["status"] == "fail"
    assert "native capture lacks" in str(capture_result["reason"])
    assert len(recorder.launches) == 1, recorder.launches
    argv = recorder.launches[0]["argv"]
    assert argv[1].endswith("tools/rooms/run_spear_residential_episode.py")
    flags = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}
    episode_root = Path(flags["--episode-root"])
    assert episode_root.is_dir()
    # the directory handed to the renderer carries the variant's own plan
    for name in ("episode_plan.json", "room_package.json", "audio_events.json",
                 "voice_bindings.json"):
        assert (episode_root / name).is_file(), name
    handed = json.loads((episode_root / "episode_plan.json").read_text(encoding="utf-8"))
    assert handed["visual_plan"]["camera"] == base_plan["visual_plan"]["camera"]
    assert int(flags["--width"]) == 1280 and int(flags["--height"]) == 720


@pytest.mark.skipif(
    not M27_MANIFEST.is_file(),
    reason="the prepared main-config batch manifest is not present",
)
def test_identity_audio_unit_declares_the_attached_foa_view_before_any_render():
    manifest = json.loads(M27_MANIFEST.read_text(encoding="utf-8"))
    context = identity.identity_group_stage_context(
        manifest, IDENTITY_MANIFEST_GROUP,
        world_id="world_identity_audio_boundary_probe", split="pilot",
    )
    for unit_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        unit_spec = native._unit_row(context, unit_id)
        item = {
            "work_item_id": f"{IDENTITY_MANIFEST_GROUP}/{unit_id}:audio:01",
            "scope_id": f"{IDENTITY_MANIFEST_GROUP}/{unit_id}",
            "group_id": IDENTITY_MANIFEST_GROUP,
            "task_family": context["task_family"],
            "unit_id": unit_id, "stage": "audio",
            "member_request_ids": list(unit_spec.get("member_request_ids") or ()),
            "inputs": {}, "depends_on": [],
        }
        _request_id, member_request = native._member_request_for_audio_unit(
            context, item, unit_spec
        )
        merged, _fields = native._apply_member_audio_view_fields(
            deepcopy(member_request), member_request, label=unit_id
        )
        delivery = native.declared_audio_delivery(merged)
        assert delivery["primary_layout"] == "binaural"
        assert delivery["attached_view_layouts"] == ["ambisonics"]
        assert delivery["foa_normalization"] == "native_n3d"
        attached = [row for row in delivery["layouts"]
                    if row["role"] == "attached_view"]
        assert len(attached) == 1
        assert attached[0]["ambisonic_order"] == 1
        assert attached[0]["channel_count"] == 4
        assert merged["post_assembly_convolution_gain"] == 0.5
