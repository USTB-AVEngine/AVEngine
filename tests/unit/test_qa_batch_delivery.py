from copy import deepcopy
import json
import pytest
import inspect
from avengine.qa import batch_delivery as bd
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



def test_review_frames_include_the_native_occluder_evidence_frame():
    questions = {"items": [{"question_id": "qa10", "evidence": {
        "target_actor_id": "source2", "frame": 7, "occluder_instance_ids": ["source1"]}}]}
    frames = _review_frames(questions, facts_fixture())
    assert set(frames) == {0, 7}
    assert "qa10:frame" in frames[7]


def test_review_frames_include_public_window_endpoints():
    questions = {"items": [{"question_id": "range", "evidence": {"query_window_frames": [3, 10]}}]}
    result = _review_frames(questions, facts_fixture())
    assert set(result) == {0, 3, 9}
    assert "range:query_window_start" in result[3]
    assert "range:query_window_end" in result[9]
    questions["items"][0]["evidence"]["query_window_frames"] = [3, 11]
    with pytest.raises(ValueError, match="query window"):
        _review_frames(questions, facts_fixture())


# --------------------------------------------------------------------------- V1 achieved coverage


def _item(qa_id, question_id, *, truth=None, forms=("mcq", "open"),
          status="pass", form_status="pass", modalities=None):
    block = {}
    for form in forms:
        block[form] = {"truth": truth} if truth is not None else {}
    return {
        "qa_id": qa_id,
        "question_id": question_id,
        "status": status,
        "forms": block,
        "form_status": {form: {"status": form_status} for form in forms},
        "required_modalities": modalities,
    }


def test_valid_forms_follow_the_delivery_publication_predicate() -> None:
    item = {
        "forms": {"mcq": {}, "open": {}},
        "form_status": {"mcq": {"status": "pass"}, "open": {"status": "deferred"}},
    }
    assert bd.valid_question_forms(item) == ["mcq"]
    assert bd.valid_question_forms({"forms": {"mcq": {}}, "form_status": {}}) == []


def test_observed_branch_reads_the_published_answer() -> None:
    yes_no = ("yes", "no")
    assert bd.observed_branch({"forms": {"open": {"truth": True}}}, yes_no, qa_id="QA-09") == "yes"
    assert bd.observed_branch({"forms": {"open": {"truth": "no"}}}, yes_no, qa_id="QA-17") == "no"
    four = ("out_of_view", "visible_clear", "visible_occluded", "fully_occluded")
    assert bd.observed_branch(
        {"forms": {"open": {"truth": "visible_occluded"}}}, four, qa_id="QA-08"
    ) == "visible_occluded"


def test_answer_vocabulary_that_differs_from_the_branch_name_is_declared() -> None:
    # QA-05 publishes yes/no for "did they overlap"; the branch names are the
    # event relation. QA-20 publishes an actor id or none_of_visible.
    assert bd.observed_branch(
        {"forms": {"open": {"truth": "yes"}}}, ("overlap", "disjoint"), qa_id="QA-05"
    ) == "overlap"
    assert bd.observed_branch(
        {"forms": {"open": {"truth": "no"}}}, ("overlap", "disjoint"), qa_id="QA-05"
    ) == "disjoint"
    both = ("visible_candidate", "none_of_them")
    assert bd.observed_branch(
        {"forms": {"open": {"truth": "source2"}}}, both, qa_id="QA-20"
    ) == "visible_candidate"
    assert bd.observed_branch(
        {"forms": {"open": {"truth": "none_of_visible"}}}, both, qa_id="QA-20"
    ) == "none_of_them"


def test_a_numeric_answer_never_guesses_its_branch() -> None:
    # QA-25 answers a bearing in whole degrees; its branch is the modality
    # subset and is unreadable from the number.
    modal = ("A", "V", "AV")
    assert bd.observed_branch({"forms": {"open": {"truth": 32}}}, modal, qa_id="QA-25") is None
    assert bd.observed_branch(
        {"forms": {"open": {"truth": 32}}, "required_modalities": ["audio"]},
        modal, qa_id="QA-25",
    ) == "A"
    assert bd.observed_branch(
        {"forms": {"open": {"truth": 32}}, "required_modalities": ["audio", "video"]},
        modal, qa_id="QA-25",
    ) == "AV"
    assert bd.observed_branch(
        {"forms": {"open": {"truth": "7.5"}}}, ("yes", "no"), qa_id="QA-17"
    ) is None


def test_question_rows_keep_angle_followups_out_of_the_main_count() -> None:
    question_set = {
        "items": [_item("QA-01", "q1", truth="yes"), _item("QA-20", "q2", truth="source1")],
        "angle_followups": [_item("QA-20", "q3", truth="12")],
        "deferred": [{"qa_id": "QA-06", "code": "missing_motion", "detail": "still"}],
    }
    rows = bd.question_set_rows(question_set)
    assert [(row["qa_id"], row["kind"]) for row in rows] == [
        ("QA-01", "main"), ("QA-20", "main"), ("QA-20", "angle_followup")
    ]
    assert bd.question_set_deferrals(question_set) == [
        {"qa_id": "QA-06", "code": "missing_motion", "detail": "still"}
    ]


def test_a_deferred_item_is_not_a_question() -> None:
    question_set = {"items": [_item("QA-01", "q1", truth="yes", status="deferred")]}
    assert bd.question_set_rows(question_set) == []


REGISTRY = {
    "assets": [
        {"asset_id": "human_a", "entity_class": "articulated_human"},
        {"asset_id": "human_b", "entity_class": "articulated_human"},
        {"asset_id": "dog_a", "entity_class": "articulated_animal"},
        {"asset_id": "speaker_a", "entity_class": "rigid_object"},
    ]
}


def test_source_family_is_resolved_through_the_registry() -> None:
    index = bd.source_family_index(REGISTRY)
    facts = {"actors": {"source1": {"asset_id": "human_a"}, "source2": {"asset_id": "dog_a"}}}
    resolved = bd.member_source_families(facts, family_by_asset=index)
    assert resolved["source_families"] == ["animal", "human"]
    assert resolved["entity_combination"] == "human+animal"
    assert resolved["entity_combinations"] == ["human+animal"]
    assert resolved["asset_ids_absent_from_registry"] == []


def test_an_unregistered_asset_is_reported_not_guessed() -> None:
    facts = {"actors": {"source1": {"asset_id": "human_a"}, "source2": {"asset_id": "mystery_dog"}}}
    resolved = bd.member_source_families(facts, family_by_asset=bd.source_family_index(REGISTRY))
    assert resolved["entity_combination"] == bd.UNRESOLVED_SOURCE_FAMILY
    assert resolved["entity_combinations"] == []
    assert resolved["asset_ids_absent_from_registry"] == ["mystery_dog"]


def test_a_three_source_episode_covers_every_pair_it_contains() -> None:
    facts = {
        "actors": {
            "source1": {"asset_id": "human_a"},
            "source2": {"asset_id": "human_b"},
            "source3": {"asset_id": "speaker_a"},
        }
    }
    resolved = bd.member_source_families(facts, family_by_asset=bd.source_family_index(REGISTRY))
    assert resolved["entity_combinations"] == ["human+device", "human+human"]
    # No single pair identifies a three-source room, so the scalar stays empty.
    assert resolved["entity_combination"] is None


def _member_row(**overrides):
    row = {
        "group_id": "g1", "member_id": "v0_a0", "world_id": "world_1",
        "task_family": "visible_binding", "room_family": "apartment",
        "room_id": "room_1", "core_sample_id": "sample_000001",
        "facts_path": "native/facts/facts_0001.json",
        "source_families": ["human", "human"], "entity_combination": "human+human",
        "entity_combinations": ["human+human"], "asset_ids_absent_from_registry": [],
        "generation_status": "pass",
        "questions": [
            {"qa_id": "QA-01", "question_id": "q1", "kind": "main",
             "forms": ["mcq", "open"], "branches_expected": [], "branch": None},
            {"qa_id": "QA-20", "question_id": "q2", "kind": "main", "forms": ["open"],
             "branches_expected": ["visible_candidate", "none_of_them"],
             "branch": "visible_candidate"},
            {"qa_id": "QA-20", "question_id": "q3", "kind": "angle_followup",
             "forms": ["open"], "branches_expected": [], "branch": None},
        ],
        "deferred": [{"qa_id": "QA-06", "code": "missing_motion", "detail": None}],
    }
    row.update(overrides)
    return row


def _survey(rows, kind="retained_group_library_census"):
    return {"schema": bd.V1_ACHIEVED_SCHEMA, "source_kind": kind, "members": rows}


def test_one_item_with_two_forms_is_one_valid_main_question() -> None:
    table = bd.achieved_coverage_table(_survey([_member_row()]))
    row = table["by_qa_id"]["QA-01"]
    assert row["valid_main_questions"] == 1
    assert row["form_counts"] == {"mcq": 1, "open": 1}
    assert table["by_qa_id"]["QA-20"]["valid_main_questions"] == 1
    assert table["by_qa_id"]["QA-20"]["valid_angle_followups"] == 1
    assert table["by_qa_branch"]["QA-20:visible_candidate"]["valid_main_questions"] == 1
    assert "QA-20:none_of_them" not in table["by_qa_branch"]
    assert table["deferred_codes_by_qa_id"]["QA-06"] == {"missing_motion": 1}
    assert table["by_qa_id"]["QA-06"]["valid_main_questions"] == 0


def test_a_world_serving_two_members_is_counted_once() -> None:
    table = bd.achieved_coverage_table(_survey([
        _member_row(member_id="v0_a0"), _member_row(member_id="v0_a1"),
    ]))
    assert table["member_count"] == 2
    assert table["world_count"] == 1
    assert table["by_qa_id"]["QA-01"]["valid_main_questions"] == 2
    assert table["by_qa_id"]["QA-01"]["distinct_worlds_with_main"] == 1
    assert table["core_task_by_room_family_group_counts"] == {"visible_binding|apartment": 1}


def test_a_generation_failure_is_a_failure_not_missing_coverage() -> None:
    table = bd.achieved_coverage_table(_survey([
        _member_row(generation_status="fail", generation_error="UnifiedQAError: broken",
                    questions=[], deferred=[]),
    ]))
    assert table["by_qa_id"]["QA-01"]["valid_main_questions"] == 0
    assert table["generation_failures"] == [
        {"group_id": "g1", "member_id": "v0_a0", "world_id": "world_1",
         "error": "UnifiedQAError: broken"}
    ]


def test_reimporting_the_same_member_does_not_count_it_twice() -> None:
    first = _survey([_member_row()])
    second = _survey([_member_row()], kind="delivered_export")
    merged = bd.merge_achieved_surveys(first, second)
    assert len(merged["members"]) == 1
    assert merged["already_imported_members_counted_once"] == [
        {"key": ["g1", "v0_a0"], "kept_source": "retained_group_library_census",
         "skipped_source": "delivered_export"}
    ]
    table = bd.achieved_coverage_table(merged)
    assert table["by_qa_id"]["QA-01"]["valid_main_questions"] == 1
    assert table["world_count"] == 1


def test_merging_refuses_a_document_that_is_not_an_achieved_survey() -> None:
    with pytest.raises(bd.V1AchievedCoverageError):
        bd.merge_achieved_surveys({"schema": "something_else", "members": []})


def test_surveying_one_member_reads_its_own_facts(tmp_path) -> None:
    facts = tmp_path / "facts_0001.json"
    facts.write_text(json.dumps({
        "actors": {"source1": {"asset_id": "human_a"}, "source2": {"asset_id": "speaker_a"}},
        "sampling": {"time_display_precision": 1},
    }), encoding="utf-8")
    seen = {}

    def generate(payload, *, seed, items_per_type):
        seen["precision"] = payload["sampling"]["time_display_precision"]
        seen["qa_precision"] = payload["sampling"]["qa_sampling"]["time_display_precision"]
        seen["seed"] = seed
        return {"items": [_item("QA-01", "q1", truth="yes")], "angle_followups": [],
                "deferred": []}

    row = bd.survey_group_member(
        {"group_id": "g1", "world_id": "w1", "task_family": "visible_binding",
         "room_family": "apartment", "room_id": "room_1"},
        {"member_id": "v0_a0", "sample_id": "sample_000001", "facts_path": "facts_0001.json"},
        base=tmp_path, family_by_asset=bd.source_family_index(REGISTRY),
        generate=generate, seed="unit-test",
    )
    # Public question times are whole seconds, so the survey asks for the same
    # display precision the export uses.
    assert seen["precision"] == 0 and seen["qa_precision"] == 0
    assert seen["seed"] == "unit-test"
    assert row["generation_status"] == "pass"
    assert row["entity_combinations"] == ["human+device"]
    assert row["questions"][0]["qa_id"] == "QA-01"


def test_a_member_whose_generator_raises_keeps_the_error(tmp_path) -> None:
    facts = tmp_path / "facts_0001.json"
    facts.write_text(json.dumps({"actors": {"source1": {"asset_id": "human_a"}}}),
                     encoding="utf-8")

    def generate(payload, *, seed, items_per_type):
        raise ValueError("no legal candidate")

    row = bd.survey_group_member(
        {"group_id": "g1", "world_id": "w1", "task_family": "visible_binding",
         "room_family": "apartment", "room_id": "room_1"},
        {"member_id": "v0_a0", "sample_id": "s1", "facts_path": "facts_0001.json"},
        base=tmp_path, family_by_asset=bd.source_family_index(REGISTRY),
        generate=generate, seed="unit-test",
    )
    assert row["generation_status"] == "fail"
    assert row["generation_error"] == "ValueError: no legal candidate"


def test_surveying_a_snapshot_refuses_a_repeated_member(tmp_path) -> None:
    facts = tmp_path / "facts_0001.json"
    facts.write_text(json.dumps({"actors": {"source1": {"asset_id": "human_a"}}}),
                     encoding="utf-8")
    bundle = tmp_path / "binding_groups.json"
    bundle.write_text(json.dumps({"groups": [{
        "group_id": "g1", "world_id": "w1", "task_family": "visible_binding",
        "room_family": "apartment", "room_id": "room_1",
        "members": [
            {"member_id": "v0_a0", "sample_id": "s1", "facts_path": "facts_0001.json"},
            {"member_id": "v0_a0", "sample_id": "s2", "facts_path": "facts_0001.json"},
        ],
    }]}), encoding="utf-8")
    snapshot = {"status": "partial_delivery",
                "groups": [{"group_id": "g1", "world_id": "w1",
                            "task_family": "visible_binding", "room_family": "apartment",
                            "bundle": str(bundle)}]}
    with pytest.raises(bd.V1AchievedCoverageError, match="twice"):
        bd.survey_retained_group_library(
            snapshot, registry=REGISTRY,
            generate=lambda facts, *, seed, items_per_type: {"items": []},
        )


# --------------------------------------------------------------------------
# C02-R1: the public index joins on the public sample id
#
# A real export permutes the two id namespaces: public sample_000001 carries
# core id sample_000003 and public sample_000003 carries core id
# sample_000001. Folding both into one lookup let one record's alias
# overwrite another record's primary key, so two public samples resolved to
# the same member and a delivered member vanished from the survey.
# --------------------------------------------------------------------------


def _write_export(tmp_path, public_samples, private_records):
    root = tmp_path / "delivery"
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir(parents=True)
    (root / "public" / "dataset_index.json").write_text(
        json.dumps({"samples": public_samples, "counts": {}}), encoding="utf-8")
    (root / "private" / "gold_index.json").write_text(
        json.dumps({"records": private_records}), encoding="utf-8")
    return root


def _public(sample_id):
    return {"sample_id": sample_id, "room_family": "hm3d", "questions": []}


def _private(sample_id, core_sample_id, member_id):
    return {
        "sample_id": sample_id,
        "core_sample_id": core_sample_id,
        "group_id": "g1",
        "member_id": member_id,
        "world_id": "w1",
        "core_task": "visible_binding",
        "facts_path": f"native/facts/{member_id}.json",
    }


def test_permuted_sample_id_namespaces_keep_every_member(tmp_path):
    """The shape of the real P09 export: the two id spaces cross over."""
    root = _write_export(
        tmp_path,
        [_public(f"sample_00000{n}") for n in (1, 2, 3, 4)],
        [
            _private("sample_000001", "sample_000003", "v1_a1"),
            _private("sample_000002", "sample_000002", "v1_a0"),
            _private("sample_000003", "sample_000001", "v0_a1"),
            _private("sample_000004", "sample_000004", "v0_a0"),
        ],
    )
    survey = bd.survey_delivery_export(root)
    joined = {row["delivered_sample_id"]: row["member_id"]
              for row in survey["members"]}
    assert joined == {
        "sample_000001": "v1_a1",
        "sample_000002": "v1_a0",
        "sample_000003": "v0_a1",
        "sample_000004": "v0_a0",
    }
    assert survey["private_join"]["join_key_counts"] == {"public_sample_id": 4}
    assert survey["private_join"]["unmatched_public_sample_ids"] == []
    merged = bd.merge_achieved_surveys(survey)
    assert len(merged["members"]) == 4
    assert merged["extra_samples_for_a_filled_member_slot"] == []
    assert bd.achieved_coverage_table(merged)["member_count"] == 4


def test_the_same_string_in_the_other_namespace_does_not_claim_a_sample(tmp_path):
    """A core id equal to another record's public id is not that record."""
    root = _write_export(
        tmp_path,
        [_public("sample_000001"), _public("sample_000002")],
        [
            _private("sample_000001", "sample_000002", "v0_a0"),
            _private("sample_000002", "sample_000001", "v0_a1"),
        ],
    )
    survey = bd.survey_delivery_export(root)
    joined = {row["delivered_sample_id"]: row["member_id"]
              for row in survey["members"]}
    assert joined == {"sample_000001": "v0_a0", "sample_000002": "v0_a1"}
    assert all(row["private_join_key"] == "public_sample_id"
               for row in survey["members"])


def test_an_older_export_with_only_core_ids_still_joins(tmp_path):
    """The alias stays usable where nothing else claims the sample."""
    legacy = _private("", "sample_000001", "v0_a0")
    legacy.pop("sample_id")
    root = _write_export(tmp_path, [_public("sample_000001")], [legacy])
    survey = bd.survey_delivery_export(root)
    assert survey["members"][0]["member_id"] == "v0_a0"
    assert survey["members"][0]["private_join_key"] == "core_sample_id_alias"


def test_an_alias_never_displaces_a_record_that_owns_that_public_id(tmp_path):
    """The alias is a fallback; it cannot outrank a primary key."""
    root = _write_export(
        tmp_path,
        [_public("sample_000001")],
        [
            _private("sample_000001", "sample_000009", "v0_a0"),
            _private("sample_000007", "sample_000001", "v1_a1"),
        ],
    )
    survey = bd.survey_delivery_export(root)
    assert survey["members"][0]["member_id"] == "v0_a0"
    assert survey["members"][0]["private_join_key"] == "public_sample_id"


def test_an_alias_that_two_records_share_is_refused_not_guessed(tmp_path):
    root = _write_export(
        tmp_path,
        [_public("sample_000005")],
        [
            _private("sample_000001", "sample_000005", "v0_a0"),
            _private("sample_000002", "sample_000005", "v0_a1"),
        ],
    )
    survey = bd.survey_delivery_export(root)
    assert survey["members"][0]["member_id"] is None
    assert survey["members"][0]["private_join_key"] == "unmatched"
    assert survey["private_join"]["unmatched_public_sample_ids"] == [
        "sample_000005"]
    assert survey["private_join"]["ambiguous_core_aliases"] == ["sample_000005"]


def test_an_alias_whose_record_is_already_joined_is_not_reused(tmp_path):
    """Using it would attribute one delivered member to two public samples."""
    root = _write_export(
        tmp_path,
        [_public("sample_000001"), _public("sample_000003")],
        [_private("sample_000001", "sample_000003", "v1_a1")],
    )
    survey = bd.survey_delivery_export(root)
    joined = {row["delivered_sample_id"]: row["member_id"]
              for row in survey["members"]}
    assert joined == {"sample_000001": "v1_a1", "sample_000003": None}
    assert survey["private_join"]["unmatched_public_sample_ids"] == [
        "sample_000003"]
