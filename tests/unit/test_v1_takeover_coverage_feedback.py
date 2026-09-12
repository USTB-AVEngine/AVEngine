import json
from pathlib import Path

from avengine.qa.batch_delivery import (
    V1_ACHIEVED_SCHEMA, achieved_coverage_table, merge_achieved_surveys,
)
from avengine.qa.batch_coverage import build_v1_coverage_feedback, outstanding_production_requests


def _targets():
    return json.loads((Path(__file__).resolve().parents[2] / "examples/dataset/qa_binding_first_version_20260910.json").read_text())


def _row(member, world, qa_id="QA-01", *, kind="main"):
    return {"member_id": member, "group_id": "g_" + world, "world_id": world,
            "generation_status": "pass", "entity_combinations": ["animal+device"],
            "source_families": ["animal", "device"],
            "questions": [{"qa_id": qa_id, "kind": kind, "forms": ["open", "mcq"]}],
            "deferred": []}


def test_many_members_and_question_forms_do_not_become_two_worlds():
    survey = {"members": [_row("m1", "w1"), _row("m2", "w1", "QA-02"),
                           _row("m3", "w2", kind="angle_followup")]}
    achieved = achieved_coverage_table(survey)
    feedback = build_v1_coverage_feedback(targets=_targets(), achieved=achieved)
    row = feedback["by_entity_combination"]["animal+device"]
    assert row["valid_main_questions"] == 2
    assert row["world_ids"] == ["w1"]
    assert row["distinct_worlds_with_main"] == 1
    assert row["state"] == "short_of_target"
    assert row["remaining_distinct_worlds"] == 1
    pending = [x for x in outstanding_production_requests(feedback)
               if x["kind"] == "entity_combination" and x["entity_combination"] == "animal+device"]
    assert pending[0]["remaining_distinct_worlds"] == 1


def test_two_proven_worlds_meet_the_pair_quota():
    achieved = achieved_coverage_table({"members": [_row("m1", "w1"), _row("m2", "w2")]})
    feedback = build_v1_coverage_feedback(targets=_targets(), achieved=achieved)
    row = feedback["by_entity_combination"]["animal+device"]
    assert row["world_ids"] == ["w1", "w2"]
    assert row["state"] == "met"
    assert row["remaining_distinct_worlds"] == 0


def test_old_aggregate_counts_cannot_prove_pair_worlds_or_hide_absent_qa():
    achieved = achieved_coverage_table({"members": [_row("m1", "w1"), _row("m2", "w2")]})
    achieved.pop("world_ids_by_entity_combination")
    feedback = build_v1_coverage_feedback(targets=_targets(), achieved=achieved)
    row = feedback["by_entity_combination"]["animal+device"]
    assert row["state"] == "evidence_missing_or_unsampled"
    assert row["distinct_worlds_with_main"] is None
    assert row["remaining_distinct_worlds"] is None
    assert len(feedback["by_qa_id"]) == 25
    assert feedback["by_qa_id"]["QA-17"]["valid_main_questions"] == 0
    assert feedback["by_qa_id"]["QA-17"]["remaining_valid_main_questions"] == 8


def test_a_string_is_not_a_sequence_of_proven_world_identities():
    import pytest
    from avengine.qa.batch_coverage import BatchCoverageError
    achieved = achieved_coverage_table({"members": [_row("m1", "w1")]})
    achieved["world_ids_by_entity_combination"]["animal+device"] = "w1"
    with pytest.raises(BatchCoverageError, match="list of nonempty strings"):
        build_v1_coverage_feedback(targets=_targets(), achieved=achieved)


# --------------------------------------------------------------------------
# C02: what collides on a member slot
#
# A real export in this programme carries an engineering replay: two
# different delivered samples for one core-group member. Both collide on the
# member key with a re-imported retained member, and the two mean opposite
# things -- one says nothing new arrived, the other says something new
# arrived that is not a new member. Neither may inflate the group.
# --------------------------------------------------------------------------


def _survey(source_kind, rows):
    return {"schema": V1_ACHIEVED_SCHEMA, "source_kind": source_kind,
            "members": rows}


def _member(member_id, sample_id, world="w1", group="g1"):
    row = _row(member_id, world)
    row["group_id"] = group
    row["delivered_sample_id"] = sample_id
    return row


def test_a_replay_of_one_member_does_not_make_the_group_bigger():
    export = _survey("delivered_export", [
        _member("v0_a0", "sample_000004"),
        _member("v0_a1", "sample_000001"),
        _member("v1_a0", "sample_000002"),
        _member("v0_a1", "sample_000003"),
    ])
    merged = merge_achieved_surveys(export)
    assert len(merged["members"]) == 3
    assert sorted(row["member_id"] for row in merged["members"]) == [
        "v0_a0", "v0_a1", "v1_a0"]
    assert achieved_coverage_table(merged)["world_count"] == 1


def test_an_extra_sample_is_named_as_one_not_filed_as_a_re_import():
    """The replay must stay visible: it was never imported before."""
    export = _survey("delivered_export", [
        _member("v0_a1", "sample_000001"),
        _member("v0_a1", "sample_000003"),
    ])
    merged = merge_achieved_surveys(export)
    assert merged["already_imported_members_counted_once"] == []
    extra = merged["extra_samples_for_a_filled_member_slot"]
    assert len(extra) == 1
    assert extra[0]["kept_sample_id"] == "sample_000001"
    assert extra[0]["skipped_sample_id"] == "sample_000003"
    assert extra[0]["world_id"] == "w1"


def test_the_same_member_imported_twice_is_a_re_import():
    retained = _survey("retained_bundle", [_member("v0_a1", "sample_000001")])
    export = _survey("delivered_export", [_member("v0_a1", "sample_000001")])
    merged = merge_achieved_surveys(retained, export)
    assert len(merged["members"]) == 1
    assert merged["extra_samples_for_a_filled_member_slot"] == []
    assert len(merged["already_imported_members_counted_once"]) == 1
    assert merged["already_imported_members_counted_once"][0][
        "kept_source"] == "retained_bundle"


def test_importing_the_same_export_again_changes_no_count():
    export = _survey("delivered_export", [
        _member("v0_a0", "s1"), _member("v0_a1", "s2"),
        _member("v1_a0", "s3"), _member("v1_a1", "s4"),
    ])
    once = merge_achieved_surveys(export)
    twice = merge_achieved_surveys(export, export)
    assert len(once["members"]) == len(twice["members"]) == 4
    assert (achieved_coverage_table(once)["world_count"]
            == achieved_coverage_table(twice)["world_count"] == 1)
    assert len(twice["already_imported_members_counted_once"]) == 4
