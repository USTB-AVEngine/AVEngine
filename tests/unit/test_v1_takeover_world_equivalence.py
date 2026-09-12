from __future__ import annotations

from copy import deepcopy
import json
from itertools import product
from pathlib import Path

import pytest

from avengine.dataset.binding_group_native import (
    BindingNativeError,
    audio_shared_content_signature,
    compare_group_native_visuals,
    compare_group_visual_plans,
    compare_visual_world,
    controlled_world_contract,
    run_group_stage_work_item,
)
from avengine.dataset.production_spec import parse_production_config

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY / "examples/dataset/qa_binding_first_version_20260910.json"
ROOM_CATALOG_PATH = REPOSITORY / "examples/rooms/packages/catalog.json"


def _parsed():
    return parse_production_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _plan(order=("asset_blue", "asset_green"), *, camera_x=0.0,
          clock_frames=2, geometry_id="mesh_one", actor_x=0.0):
    actors = [
        {"actor_id": f"source{index + 1}", "asset_id": asset,
         "entity_instance_id": f"{asset}#instance01"}
        for index, asset in enumerate(order)
    ]
    frames = [
        {"frame_index": index, "pts_ticks": index * 3200,
         "camera_state": {"position_m": [camera_x, 0.0, 0.0]},
         "actor_states": [
             {"actor_id": actor["actor_id"], "asset_id": actor["asset_id"],
              "entity_instance_id": actor["entity_instance_id"],
              "root_transform": {"translation_m": [actor_index + actor_x, 0.0, 0.0]},
              "moving": False}
             for actor_index, actor in enumerate(actors)
         ]}
        for index in range(clock_frames)
    ]
    return {
        "clock": {"frame_count": clock_frames, "frame_rate_hz": 15.0,
                  "sample_rate_hz": 16000},
        "scene": {"scene_id": "room_one"},
        "resources": {"room_package": {
            "family": "hm3d", "renderer": "habitat",
            "static_geometry": {"mesh_id": geometry_id}}},
        "coordinate_frame": {"handedness": "right", "linear_unit": "meter",
                             "up_axis": "+Y"},
        "condition_profile": {"sampling": "fixed"},
        "visual_plan": {
            "camera": {"motion": "static", "fov_deg": 85.0},
            "actors": actors, "frames": frames},
        "camera_condition_sampling": {
            "selection": "fixed",
            "planned_query_window": {"windows": [{
                "event_id": "event_001", "actor_id": "source2",
                "entity_instance_id": actors[1]["entity_instance_id"]}]}}
    }


def _assignment(sound_id="sound_one", start_sample=100,
                actor_id="source1", endpoint="source1_mouth"):
    event = {
        "event_id": "event_001", "actor_id": actor_id,
        "source_endpoint_id": endpoint, "sound_asset_id": sound_id,
        "path": f"/prepared/{sound_id}.wav", "sample_count": 32000,
        "start_sample": start_sample, "end_sample_exclusive": start_sample + 32000,
        "start_tick": start_sample * 3, "end_tick": (start_sample + 32000) * 3,
        "source_start_sample": 0, "source_end_sample_exclusive": 32000,
        "target_sound_compatibility": {"target_asset_id": "asset_blue"}}
    return {"audio_events": [event],
            "voice_bindings": [{**event, "voice_binding_actor_id": actor_id,
                                "assignment_variant": "a0"}]}


def test_config_has_sixteen_cells_and_one_seed_per_group():
    parsed = _parsed()
    catalog = json.loads(ROOM_CATALOG_PATH.read_text(encoding="utf-8"))
    family_of = {str(row["room_id"]): str(row["family"])
                 for row in catalog["rooms"]}
    assert len(parsed.core_groups) == 16
    assert {(g.task_family, family_of[g.room_id]) for g in parsed.core_groups} == set(
        product(("visible_binding", "visual_conditioned_relation",
                 "cross_event_identity", "cross_time_state"),
                ("apartment", "kujiale", "hm3d", "mp3d")))
    assert all(len({m.seed for m in g.members}) == 1 for g in parsed.core_groups)
    assert all(len({repr(m.sound_selection_policy) for m in g.members}) == 1
               for g in parsed.core_groups)


def test_identity_recipe_allows_same_asset_order_for_path_topology():
    group = next(g for g in _parsed().core_groups
                 if g.task_family == "cross_event_identity")
    requests = {m.request_id: m.to_legacy_request() for m in group.members}
    order = list(requests[group.members[0].request_id]["source_asset_ids"])
    for request in requests.values():
        request["source_asset_ids"] = list(order)
    contract = controlled_world_contract(group.to_dict(), requests)
    assert contract["plan_equivalence"] == "world"
    assert contract["visual_intervention"] == "identity_path_topology"
    assert contract["declared_interventions"]["visual_slot_permutation"] is False


def test_relation_recipe_keeps_declared_slot_rule():
    group = next(g for g in _parsed().core_groups
                 if g.task_family == "visual_conditioned_relation")
    contract = controlled_world_contract(
        group.to_dict(), {m.request_id: m.to_legacy_request() for m in group.members})
    assert contract["plan_equivalence"] == "controlled_slots"
    assert contract["visual_intervention"] == "source_slot_permutation"
    assert contract["query_identity_policy"] == "slot"


def test_visible_shared_audio_pair_rejects_selected_sound_change():
    group = next(g for g in _parsed().core_groups
                 if g.task_family == "visible_binding")
    requests = {m.request_id: m.to_legacy_request() for m in group.members}
    _, right_id = group.shared_audio_member_ids[0]
    requests[right_id]["sound_selection"] = {
        **requests[right_id]["sound_selection"],
        "selected_sound_asset_ids_by_actor": {"source1": ["changed_dry_sound"]}}
    with pytest.raises(BindingNativeError, match="declared dry sound content"):
        controlled_world_contract(group.to_dict(), requests)


def test_shared_candidate_allowlists_require_intersection():
    group = next(g for g in _parsed().core_groups
                 if g.task_family == "visible_binding")
    requests = {m.request_id: m.to_legacy_request() for m in group.members}
    left_id, right_id = group.shared_audio_member_ids[0]
    requests[left_id]["sound_selection"] = {
        **requests[left_id]["sound_selection"],
        "preallocated_sound_asset_ids_by_actor": {
            "source1": ["blue_common"], "source2": ["green_common"]}}
    requests[right_id]["sound_selection"] = {
        **requests[right_id]["sound_selection"],
        "preallocated_sound_asset_ids_by_actor": {
            "source1": ["green_common"], "source2": ["blue_common"]}}
    assert controlled_world_contract(group.to_dict(), requests)["status"] == "pass"
    requests[right_id]["sound_selection"][
        "preallocated_sound_asset_ids_by_actor"]["source1"] = ["other_green"]
    with pytest.raises(BindingNativeError, match="candidate_intersection"):
        controlled_world_contract(group.to_dict(), requests)


def test_candidate_rotation_is_shared_by_the_core_world():
    group = next(g for g in _parsed().core_groups
                 if g.task_family == "visible_binding")
    requests = {member.request_id: member.to_legacy_request() for member in group.members}
    for request in requests.values():
        request["sampling_candidate_index"] = 3
    contract = controlled_world_contract(group.to_dict(), requests)
    assert contract["sampling_candidate_index"] == 3
    assert contract["shared_world"]["sampling_candidate_index"] == 3
    requests[group.members[-1].request_id]["sampling_candidate_index"] = 4
    with pytest.raises(BindingNativeError, match="sampling_candidate_index"):
        controlled_world_contract(group.to_dict(), requests)


def test_same_seed_different_candidate_is_not_one_planned_world(tmp_path: Path):
    left = _plan()
    right = _plan()
    left["request"] = {"seed": 7, "sampling_candidate_index": 0}
    right["request"] = {"seed": 7, "sampling_candidate_index": 1}
    left_path = _write(tmp_path / "candidate0.json", left)
    right_path = _write(tmp_path / "candidate1.json", right)
    with pytest.raises(BindingNativeError, match="different worlds"):
        compare_visual_world(left_path, right_path)
    with pytest.raises(BindingNativeError, match="differing fields"):
        compare_group_visual_plans(
            left_path, right_path,
            contract={"plan_equivalence": "controlled_slots"},
        )


def test_query_identity_only_normalizes_under_slot_policy(tmp_path: Path):
    left = _write(tmp_path / "left.json", _plan())
    right = _write(tmp_path / "right.json", _plan(("asset_green", "asset_blue")))
    with pytest.raises(BindingNativeError, match="different worlds"):
        compare_visual_world(left, right)
    assert compare_visual_world(left, right, query_identity_policy="slot")["status"] == "pass"


def test_world_comparison_allows_identity_path_but_refuses_drift(tmp_path: Path):
    left = _write(tmp_path / "left.json", _plan())
    path_variant = _write(tmp_path / "path.json", _plan(actor_x=1.5))
    contract = {"plan_equivalence": "world", "query_identity_policy": "exact",
                "plan_equivalence_rule": "identical_planned_scene_clock_and_camera_route_only"}
    assert compare_group_visual_plans(left, path_variant, contract=contract)["status"] == "pass"
    for name, plan in (
        ("camera.json", _plan(camera_x=1.0)),
        ("clock.json", _plan(clock_frames=3)),
        ("geometry.json", _plan(geometry_id="mesh_two")),
    ):
        with pytest.raises(BindingNativeError, match="different worlds"):
            compare_group_visual_plans(left, _write(tmp_path / name, plan),
                                       contract=contract)


def test_shared_audio_signature_checks_consumed_content_and_timing():
    left = _assignment()
    target_variant = _assignment(actor_id="source2", endpoint="source2_mouth")
    target_variant["audio_events"][0]["target_sound_compatibility"] = {
        "target_asset_id": "asset_green"}
    target_variant["voice_bindings"][0]["target_sound_compatibility"] = {
        "target_asset_id": "asset_green"}
    assert audio_shared_content_signature(left) == audio_shared_content_signature(
        target_variant)
    assert audio_shared_content_signature(left) != audio_shared_content_signature(
        _assignment(sound_id="sound_two"))
    assert audio_shared_content_signature(left) != audio_shared_content_signature(
        _assignment(start_sample=101))


def test_scoped_stage_runner_hook_reuses_the_existing_save_resume_protocol(tmp_path: Path):
    called = []

    def runner(item, context, unit_root, *, output_root, results, lease):
        called.append((item["unit_id"], context["group_id"], output_root, results, lease))
        unit_root.mkdir(parents=True)
        return {
            "work_item_id": item["work_item_id"],
            "request_id": item["request_id"],
            "scope_id": item["request_id"],
            "status": "pass",
            "facts": {"hook": True},
            "outputs": {},
            "reason": None,
            "depends_on": [],
        }

    item = {
        "work_item_id": "g/v0:plan:01",
        "request_id": "g/v0",
        "group_id": "g",
        "unit_id": "v0",
        "stage": "plan",
        "fresh_output_relative": "g/v0/plan/attempt_01",
        "depends_on": [],
    }
    context = {
        "group_id": "g",
        "group_spec": {
            "stage_units": [{
                "unit_id": "v0",
                "unit_kind": "visual_plan",
                "stage": "plan",
            }]
        },
    }
    runners = {"visual_plan": runner}
    first = run_group_stage_work_item(
        item, context, output_root=tmp_path, stage_runners=runners
    )
    second = run_group_stage_work_item(
        item, context, output_root=tmp_path, stage_runners=runners
    )
    assert first["facts"] == {"hook": True}
    assert second["facts"] == {"hook": True}
    assert len(called) == 1
    assert (tmp_path / "g/v0/plan/attempt_01/stage_result.json").is_file()


def _native_contract(*, visual_intervention="identity_path_topology",
                     plan_equivalence="world", order=("rocketbox_human_male_adult_01_top_blue_research_v1",
                                                      "rocketbox_human_male_adult_01_top_green_research_v1")):
    return {
        "plan_equivalence": plan_equivalence,
        "visual_intervention": visual_intervention,
        "visual_units": {
            "v0_capture": {
                "source_asset_ids": list(order),
                "plan_unit_ids": ["v0"],
            },
            "v1_capture": {
                "source_asset_ids": list(order),
                "plan_unit_ids": ["v1"],
            },
        },
    }


@pytest.mark.parametrize(
    "root",
    [
        REPOSITORY / "tmp/binding_dataset_20260909_v2/state_first/hm3d/group_v1/visual",
        REPOSITORY / "tmp/binding_dataset_20260909_v2/identity_first/group_v14/visual",
    ],
)
def test_retained_state_and_identity_readbacks_allow_declared_paths(root: Path):
    left = root / "v0/capture/neutral_readback.json"
    right = root / "v1/capture/neutral_readback.json"
    assert compare_group_native_visuals(
        {"neutral_readback": left},
        {"neutral_readback": right},
        contract=_native_contract(),
        left_unit_id="v0_capture",
        right_unit_id="v1_capture",
    )["status"] == "pass"


@pytest.mark.parametrize(
    "root",
    [
        REPOSITORY / "tmp/binding_dataset_20260909_v2/state_first/hm3d/group_v1/visual",
        REPOSITORY / "tmp/binding_dataset_20260909_v2/identity_first/group_v14/visual",
    ],
)
def test_group_world_equivalence_uses_recipe_native_comparator(root: Path):
    contract = _native_contract()
    results = []
    for visual_id in ("v0", "v1"):
        plan = root / visual_id / "plan/episode_plan.json"
        readback = root / visual_id / "capture/neutral_readback.json"
        results.extend([
            {
                "work_item_id": f"g/{visual_id}:plan:01",
                "scope_id": f"g/{visual_id}",
                "status": "pass",
                "facts": {"episode_plan_path": str(plan)},
                "outputs": {},
            },
            {
                "work_item_id": f"g/{visual_id}_capture:capture:01",
                "scope_id": f"g/{visual_id}_capture",
                "status": "pass",
                "facts": {},
                "outputs": {"neutral_readback": str(readback)},
            },
        ])
    verdict = __import__(
        "avengine.dataset.binding_group_native",
        fromlist=["group_world_equivalence"],
    ).group_world_equivalence({"contract": contract}, results)
    assert verdict["status"] == "pass"


def test_group_native_comparison_rejects_camera_drift_even_for_path_recipes(tmp_path: Path):
    root = REPOSITORY / "tmp/binding_dataset_20260909_v2/identity_first/group_v14/visual"
    left = root / "v0/capture/neutral_readback.json"
    changed = json.loads((root / "v1/capture/neutral_readback.json").read_text())
    changed["camera"][0]["position_m"][0] += 1.0
    right = _write(tmp_path / "changed_readback.json", changed)
    with pytest.raises(BindingNativeError, match="group native camera"):
        compare_group_native_visuals(
            {"neutral_readback": left},
            {"neutral_readback": right},
            contract=_native_contract(),
            left_unit_id="v0_capture",
            right_unit_id="v1_capture",
        )


def test_visible_group_native_comparison_maps_declared_slot_swap_strictly():
    root = REPOSITORY / "tmp/binding_v1_parallel_20260910/P09/attempt_20260910T133054Z_pid345350/stage_run_fresh_world/p09_visible_binding_hm3d_fresh_g03"
    left = root / "v0_capture/capture/attempt_01/capture/neutral_readback.json"
    right = root / "v1_capture/capture/attempt_01/capture/neutral_readback.json"
    contract = _native_contract(
        visual_intervention="source_slot_permutation",
        plan_equivalence="controlled_slots",
        order=("rocketbox_human_male_adult_01_top_blue_research_v1",
                "rocketbox_human_male_adult_01_top_green_research_v1"),
    )
    contract["visual_units"]["v1_capture"]["source_asset_ids"] = [
        "rocketbox_human_male_adult_01_top_green_research_v1",
        "rocketbox_human_male_adult_01_top_blue_research_v1",
    ]
    result = compare_group_native_visuals(
        {"neutral_readback": left},
        {"neutral_readback": right},
        contract=contract,
        left_unit_id="v0_capture",
        right_unit_id="v1_capture",
    )
    assert result["status"] == "pass"
    assert result["entity_binding_source"] == {
        "left": "native_entity_identities",
        "right": "native_entity_identities",
    }
    assert result["native_entity_binding_verified"] is True
    assert result["authority"].endswith(
        "strict_entity_geometry_by_source_slot"
    )
