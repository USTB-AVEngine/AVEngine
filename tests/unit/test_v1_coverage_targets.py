"""What the shipped V1 first-version configuration has to keep true.

These tests read the configuration file itself, so a later edit that quietly
drops a QA type, lowers a target, changes the camera or lets a device accept
any sound fails here rather than in a production run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.dataset.production_spec import (
    parse_production_config, group_stage_units, recipe_for_task_family,
)
from avengine.dataset.source_capabilities import (
    combination_key, entity_combinations, normalize_sound_class_config,
    sound_class_asset_index, source_family,
)
from avengine.qa import batch_coverage as bc
from avengine.qa import generation_conditions as gc
from avengine.qa.unified_catalog import QA_IDS
from avengine.rooms import conditioned_sampler

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY / "examples/dataset/qa_binding_first_version_20260910.json"
REGISTRY_PATH = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
ROOM_CATALOG_PATH = REPOSITORY / "examples/rooms/packages/catalog.json"


@pytest.fixture(scope="module")
def document() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parsed(document):
    return parse_production_config(document)


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_the_configuration_resolves_into_episodes_and_four_member_groups(parsed) -> None:
    assert parsed.batch_id == "qa_binding_first_version_20260910"
    assert parsed.episodes, "the configuration must declare ordinary Episodes"
    assert len(parsed.core_groups) == 16
    for group in parsed.core_groups:
        assert len(group.members) == 4
        assert len(group_stage_units(group)) == len(
            recipe_for_task_family(group.task_family).units
        )
    assert len({request.request_id for request in parsed.all_requests()}) == len(
        parsed.all_requests()
    )


def test_relation_core_groups_use_three_source_permutations_and_common_candidates(
    parsed,
) -> None:
    expected_v0 = (
        "rocketbox_human_male_adult_01_top_blue_research_v1",
        "rocketbox_human_male_adult_01_top_green_research_v1",
        "rocketbox_human_male_adult_01_top_yellow_research_v1",
    )
    expected_v1 = (expected_v0[0], expected_v0[2], expected_v0[1])
    relation_groups = [
        group for group in parsed.core_groups
        if group.task_family == "visual_conditioned_relation"
    ]
    assert len(relation_groups) == 4
    for group in relation_groups:
        orders = [
            tuple(instance.asset_id for instance in member.instances)
            for member in group.members
        ]
        assert orders == [expected_v0, expected_v0, expected_v1, expected_v1]
        candidates_by_asset = {}
        for member in group.members:
            candidate_map = member.sound_selection[
                "preallocated_sound_asset_ids_by_actor"
            ]
            assert set(candidate_map) == {"source1", "source2", "source3"}
            for instance in member.instances:
                candidates = tuple(candidate_map[instance.instance_id])
                previous = candidates_by_asset.setdefault(
                    instance.asset_id, candidates
                )
                assert candidates == previous
        assert set(candidates_by_asset) == set(expected_v0)


def test_the_targets_load_and_ask_for_the_roadmap_first_version(document) -> None:
    targets = bc.load_v1_coverage_targets(document)
    assert sorted(targets["qa_ids"]) == sorted(QA_IDS)
    assert targets["target_core_group_count"] == 16
    assert targets["min_core_groups_per_task_family_and_room_family"] == 1
    assert targets["min_valid_main_questions_per_qa_id"] >= 8
    assert targets["min_distinct_worlds_per_qa_id"] >= 2
    assert targets["min_valid_main_questions_per_branch"] >= 2
    assert sorted(targets["entity_combinations"]) == sorted(
        combination_key(*pair) for pair in entity_combinations()
    )


def test_the_pilot_core_matrix_uses_one_group_per_cell(document) -> None:
    rounds = document["production_rounds"]
    assert rounds["core_groups_per_cell_in_this_configuration"] == rounds[
        "required_core_groups_per_cell"
    ] == 1
    assert "one core group in each of the sixteen" in rounds["note"]
    assert "not V1 coverage" in rounds["note"]
    assert "paper admission" in document["coverage_quota"]["claim_boundary"]


def test_every_qa_type_stays_generatable_in_every_episode(parsed) -> None:
    # No Episode narrows qa_ids, so a type the planner cannot yet be asked for
    # is still generated when the facts happen to support it.
    for request in parsed.all_requests():
        assert sorted(request.qa_ids) == sorted(QA_IDS)


def test_the_four_production_rooms_and_the_four_core_tasks_are_covered(
    document, parsed
) -> None:
    catalog = json.loads(ROOM_CATALOG_PATH.read_text(encoding="utf-8"))
    family_of = {str(room["room_id"]): str(room["family"]) for room in catalog["rooms"]}
    declared_families = set(document["coverage_quota"]["room_families"])
    episode_families = {family_of[request.room_id] for request in parsed.episodes}
    assert episode_families == declared_families
    cells = {
        (group.task_family, family_of[group.room_id]) for group in parsed.core_groups
    }
    assert cells == {
        (task, family)
        for task in document["coverage_quota"]["core_task_families"]
        for family in declared_families
    }


def test_every_two_entity_combination_appears_in_the_episodes(document, parsed, registry) -> None:
    family_by_asset = {
        str(record["asset_id"]): source_family(record) for record in registry["assets"]
    }
    seen = set()
    for request in parsed.episodes:
        families = sorted(family_by_asset[instance.asset_id] for instance in request.instances)
        assert len(families) == 2
        seen.add(combination_key(*families))
    assert seen == set(document["coverage_quota"]["entity_combinations"])


def test_every_requested_cell_reaches_at_least_two_rooms(document) -> None:
    rooms = {}
    for episode in document["episodes"]:
        for row in episode["qa_targets"]:
            key = f"{row['qa_id']}:{row.get('branch') or '-'}"
            rooms.setdefault(key, set()).add(episode["room_id"])
    thin = sorted(key for key, value in rooms.items() if len(value) < 2)
    assert thin == [], f"these requested cells cannot reach two worlds: {thin}"


def test_no_episode_asks_for_two_conditions_one_episode_cannot_hold(document, registry) -> None:
    class_of = {str(r["asset_id"]): str(r["entity_class"]) for r in registry["assets"]}
    token = {"articulated_human": "articulated_human",
             "articulated_animal": "articulated_animal",
             "rigid_object": "rigid_static_object"}
    for episode in document["episodes"]:
        instances = [
            {**spec, "entity_instance_id": spec["instance_id"],
             "entity_class": class_of[spec["asset_id"]],
             "source_class": token[class_of[spec["asset_id"]]]}
            for spec in episode["instances"]
        ]
        compiled = []
        for row in episode["qa_targets"]:
            compiled.extend(gc.compile_target_candidates(
                {"qa_id": row["qa_id"],
                 "target_instance_ids": row["target_instance_ids"],
                 "event": row["event"], "items": row["items"]},
                branch=row.get("branch"), instances=instances, registry=registry,
                capabilities=conditioned_sampler,
            ))
        assert all(entry.state == "available" for entry in compiled), (
            f"{episode['request_id']} requests a cell the planner refuses"
        )
        conflicts = gc.reject_conflicts(compiled)
        assert conflicts == [], f"{episode['request_id']} conflicts: {conflicts}"


def test_a_blocked_cell_is_recorded_with_its_own_state(document) -> None:
    blocked = document["known_blocked_qa_cells"]["cells"]
    assert blocked, "the configuration must record what the planner refuses"
    states = {row["state"] for row in blocked}
    assert states <= {
        "not_applicable_by_definition", "interface_not_implemented",
        "evidence_missing_or_unsampled",
    }
    # The two answers stay apart: a device that cannot walk is not the same
    # thing as a sampler that has no visibility-transition knob.
    assert "not_applicable_by_definition" in states
    assert "interface_not_implemented" in states
    for row in blocked:
        assert row["reason"], f"{row['qa_id']}:{row['branch']} has no reason"
    # A blocked cell is never also requested for the same source pair: retrying
    # a knob that does not exist would only spend a budget.
    requested = set()
    combination_of = {
        episode["request_id"]: episode["condition_group"].rsplit("_bundle_", 1)[0]
        for episode in document["episodes"]
    }
    for episode in document["episodes"]:
        for row in episode["qa_targets"]:
            requested.add(
                (combination_of[episode["request_id"]], row["qa_id"], row.get("branch"))
            )
    overlap = sorted(
        f"{row['entity_combination']}/{row['qa_id']}:{row['branch'] or '-'}"
        for row in blocked
        if (row["entity_combination"], row["qa_id"], row["branch"]) in requested
    )
    assert overlap == [], f"blocked cells are still being requested: {overlap}"


def test_the_fixed_episode_invariants_are_kept(parsed) -> None:
    for request in parsed.all_requests():
        assert request.rig.motion == "static"
        assert request.clock.frame_count == 150
        assert request.clock.frame_rate_hz == 15
        assert request.clock.sample_rate_hz == 16000
        assert request.reserve_tail_s == 3.0
        assert request.post_assembly_convolution_gain == 0.5
        assert request.resources["audio"].rlr_threads == 1
        layouts = {layout.role: layout for layout in request.audio_layouts}
        assert layouts["primary"].layout_type == "binaural"
        assert layouts["attached_view"].layout_type == "ambisonics"
        assert layouts["attached_view"].ambisonic_order == 1
        # A monaural reverb tail comes from an order-zero indirect field.
        assert all(layout.indirect_sh_order == 1 for layout in request.audio_layouts)


def test_whole_clip_questions_use_the_whole_clip_selector(document) -> None:
    for episode in document["episodes"]:
        for row in episode["qa_targets"]:
            expected = gc.DEFAULT_EVENT_SELECTORS.get(row["qa_id"], "target_audible_window")
            assert row["event"]["kind"] == expected


def test_a_self_motion_question_is_only_asked_of_a_source_that_can_move(
    document, registry
) -> None:
    class_of = {str(r["asset_id"]): str(r["entity_class"]) for r in registry["assets"]}
    for episode in document["episodes"]:
        movable = {
            spec["instance_id"] for spec in episode["instances"]
            if class_of[spec["asset_id"]] not in {"rigid_object", "rigid_static_object"}
        }
        for row in episode["qa_targets"]:
            if row["qa_id"] in gc.SELF_MOTION_QA_IDS:
                assert set(row["target_instance_ids"]) <= movable, (
                    f"{episode['request_id']} asks {row['qa_id']} of a source that cannot move"
                )


def test_the_telephone_events_get_a_device_the_evidence_supports(document, registry) -> None:
    sound = document["sound_sources"]
    index = sound_class_asset_index(registry, normalize_sound_class_config(sound))
    accepting = index.get("telephone") or []
    object_of = {
        str(record["asset_id"]): str((record.get("identity") or {}).get("object_type"))
        for record in registry["assets"]
    }
    types = {object_of[asset_id] for asset_id in accepting}
    assert types == {"desk_telephone", "landline_phone"}
    # A smartphone has no bell, so the generic telephone class does not reach it.
    assert "cellphone" not in types
    assert "FSD50K" in sound["telephone_binding_basis"]


def test_an_unsettled_sound_class_binds_to_no_device(document, registry) -> None:
    sound = document["sound_sources"]
    index = sound_class_asset_index(registry, normalize_sound_class_config(sound))
    undetermined = {row["sound_class"]: row
                    for row in sound["undetermined_sound_class_semantics"]}
    assert set(undetermined) == {"buzzer", "dial_tone"}
    for sound_class, row in undetermined.items():
        assert index.get(sound_class) in (None, [])
        assert row["state"] == "semantics_undetermined_pending_owner_decision"
        assert row["measured_basis"] and row["decision_needed"]


def test_no_device_is_given_a_blanket_accept(document, registry) -> None:
    sound = document["sound_sources"]
    resolved = normalize_sound_class_config(sound)
    index = sound_class_asset_index(registry, resolved)
    object_of = {
        str(record["asset_id"]): str((record.get("identity") or {}).get("object_type"))
        for record in registry["assets"]
    }
    # An air conditioner emits its own running noise and nothing else.
    for asset_id, object_type in object_of.items():
        if object_type != "air_conditioner":
            continue
        accepted = {name for name, assets in index.items() if asset_id in assets}
        assert accepted == {"air_conditioning"}
    # Speech playback stays with the declared playback categories.
    speech = {object_of[asset_id] for asset_id in index.get("speech_playback", [])}
    assert "air_conditioner" not in speech and "microwave_oven" not in speech


def test_five_seconds_is_not_a_cap_and_the_tail_reserve_is_intact(document) -> None:
    sound = document["sound_sources"]
    assert sound["clip_length_filter"]["max_clip_s"] is None
    budget = sound["segment_budget"]
    assert budget["episode_s"] == 10.0
    assert budget["reserve_tail_s"] == 3.0
    assert budget["derived_max_duration_s"] == pytest.approx(
        budget["episode_s"] - budget["reserve_tail_s"] - budget["earliest_start_s"]
    )
    assert "neither a per-clip cap" in budget["reason"]


def test_a_sparse_class_is_not_reclassified_to_lift_a_number(document) -> None:
    requests = document["sound_sources"]["activity_family_reclassification_requests"]
    row = next(item for item in requests if "clock_tick" in item["sound_classes"])
    assert row["declared_activity_family"] == "device_continuous"
    assert row["state"] == "declared_family_retained_pending_owner_decision"
    assert "not a reason to reclassify" in row["reason"]


def test_the_human_review_of_an_original_recording_does_not_travel(document) -> None:
    scope = document["sound_sources"]["human_review_scope"]
    assert "never overrides a machine QC fail" in scope
    assert "cropped segment" in scope


def test_the_retained_library_is_imported_and_counted_once(document) -> None:
    imported = document["imported_coverage"]
    snapshot = REPOSITORY / imported["retained_group_snapshot"]
    assert snapshot.exists(), snapshot
    assert "counts a member, a world and a question once" in imported["note"]


def test_the_capability_basis_is_measured_not_asserted(document, registry) -> None:
    """The configuration asks the planner what it honours, and checks the answer."""
    basis = document["planner_capability_basis"]
    measurement = gc.measure_sampler_capabilities(registry)
    solver_only = {"distance_trend_during_event", "target_moved_after_sound"}
    assert (
        measurement["agrees_with_declaration"]
        or set(measurement["unverified_knobs"]) == solver_only
    ), measurement["unverified_knobs"]
    assert sorted(basis["verified_knobs"]) == sorted(measurement["verified_knobs"])
    assert basis["unverified_knobs"] == []
    # The conservative fallback is behind the checked-in sampler; the
    # configuration must not silently inherit that as a coverage gap.
    behind = sorted(set(measurement["verified_knobs"]) - set(gc.BASELINE_SAMPLER_KNOBS))
    assert set(behind) <= set(basis["verified_knobs"])

