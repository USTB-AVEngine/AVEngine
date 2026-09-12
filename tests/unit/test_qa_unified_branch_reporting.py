"""The per-episode branch report has to name branches the way their owner does.

QA-20 publishes an actor id when a visible candidate made the sound and
``none_of_visible`` when none of them did. The branch owner
(``generation_conditions.BRANCHES``) calls that second case ``none_of_them``.
``_emitted_branch`` used to compare the published token to the branch name
directly, so a world that really produced the none_of_them question reported
``branches_seen=['visible_candidate']`` and ``branches_missing=['none_of_them']``
-- the branch looked unmet while the question existed, and a producer chasing
the gap would keep planning worlds it already had.

``batch_delivery.observed_branch`` already carried the mapping, so the delivery
table and the per-episode report disagreed about the same item. These tests pin
the mapping, pin that the two readers agree, and cover both branches: a fix that
only made the missing branch appear could just as easily have broken the branch
that was already satisfied.

The published truth, options and scoring are deliberately not changed by any of
this: ``none_of_visible`` stays the answer token.
"""
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from avengine.qa import unified_catalog as catalog
from avengine.qa.batch_delivery import BRANCH_OBSERVATION_RULES, observed_branch
from avengine.qa.generation_conditions import branches_for

REPOSITORY = Path(__file__).resolve().parents[2]
QA20_BRANCHES = ("visible_candidate", "none_of_them")


def unified_episode():
    """The repository's own unified-episode fixture, loaded by path.

    Reused rather than copied so this file cannot drift away from the shape the
    catalog tests keep honest. Only the audio program is edited below.
    """
    path = REPOSITORY / "tests/unit/test_qa_unified_catalog.py"
    spec = importlib.util.spec_from_file_location("unified_catalog_episode_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fixture()


def hidden_emitter_episode():
    """One sound event, made by the actor that is out of view at its own onset.

    ``a3`` is out_of_view for frames 0-2 in the shared fixture, so moving its
    event onto frame 0 and dropping the others leaves exactly one bound event
    whose emitter is not among the visible candidates. This is a reporting
    fixture, not evidence about any produced world.
    """
    raw = unified_episode()
    raw["audio_program"]["events"] = [event for event in raw["audio_program"]["events"]
                                      if event["event_id"] == "e3"]
    for event in raw["audio_program"]["events"]:
        event["start_sample"], event["end_sample_exclusive"] = 0, 3200
    return raw


def qa20_report(raw):
    out = catalog.generate_unified_questions(
        raw, qa_ids=["QA-20"], seed="branch-reporting", items_per_type=1,
        include_angle_followups=False)
    return out, (out.get("branch_state_by_qa") or {}).get("QA-20") or {}


def test_the_published_answer_token_maps_onto_the_declared_branch_name():
    assert catalog._emitted_branch("QA-20", {"truth": {"value": "none_of_visible"}}) == "none_of_them"
    assert catalog._emitted_branch("QA-20", {"truth": {"value": "a2"}}) == "visible_candidate"
    # A generator that one day publishes the branch token itself must still read.
    assert catalog._emitted_branch("QA-20", {"truth": {"value": "none_of_them"}}) == "none_of_them"
    # An item with no truth is not a branch observation.
    assert catalog._emitted_branch("QA-20", {}) is None


def test_every_mapped_value_is_a_branch_its_owner_declares():
    declared = set(branches_for("QA-20"))
    assert declared == set(QA20_BRANCHES)
    mapped = set(catalog.ANSWER_TOKEN_BRANCH_MAP["QA-20"].values())
    mapped.add(catalog.ANSWER_TOKEN_BRANCH_DEFAULT["QA-20"])
    assert mapped <= declared, "a reader must not invent a branch the owner never declared"


def test_the_two_branch_readers_agree_on_the_same_answer_token():
    """The delivery table and the per-episode report must not disagree.

    They are separate readers of one fact, so this is the guard that keeps them
    from drifting apart again rather than a restatement of the map.
    """
    for token in ("none_of_visible", "none_of_them", "a0", "a2"):
        from_catalog = catalog._emitted_branch("QA-20", {"truth": {"value": token}})
        from_delivery = observed_branch(
            {"forms": {"open": {"truth": token}}}, QA20_BRANCHES, qa_id="QA-20")
        assert from_catalog == from_delivery, token

    rule = BRANCH_OBSERVATION_RULES["QA-20"]
    assert rule["map"] == catalog.ANSWER_TOKEN_BRANCH_MAP["QA-20"]
    assert rule["default"] == catalog.ANSWER_TOKEN_BRANCH_DEFAULT["QA-20"]


def test_a_hidden_emitter_episode_reports_the_none_of_them_branch():
    out, report = qa20_report(hidden_emitter_episode())
    items = out["items"]
    assert len(items) == 1
    evidence = items[0]["evidence"]
    assert evidence["target_visible"] is False
    assert evidence["visible_candidate_actor_ids"], "a none_of_them item still needs candidates"

    assert report["branches_seen"] == ["none_of_them"]
    assert "none_of_them" not in report["branches_missing"]
    assert report["branch_values_unmapped"] == []


def test_a_visible_emitter_episode_still_reports_the_visible_candidate_branch():
    out, report = qa20_report(unified_episode())
    items = out["items"]
    assert len(items) == 1
    assert items[0]["evidence"]["target_visible"] is True

    assert report["branches_seen"] == ["visible_candidate"]
    assert report["branches_missing"] == ["none_of_them"]
    assert report["branch_values_unmapped"] == []


def test_the_published_answer_and_options_are_not_renamed():
    """Only the branch label is normalized; the answer vocabulary is untouched."""
    out, _ = qa20_report(hidden_emitter_episode())
    item = out["items"][0]
    assert item["forms"]["open"]["truth"] == "none_of_visible"
    option_values = {option["value"] for option in item["forms"]["mcq"]["options"]}
    assert "none_of_visible" in option_values
    assert "none_of_them" not in option_values

    visible, _ = qa20_report(unified_episode())
    visible_item = visible["items"][0]
    assert visible_item["forms"]["open"]["truth"] in set(visible_item["evidence"]
                                                         ["visible_candidate_actor_ids"])


def test_qa25_still_reads_its_branch_from_the_modality_subset():
    """The other type whose branch is not its answer keeps its own shape."""
    assert catalog._emitted_branch("QA-25", {"angle_subset": "AV", "truth": {"value": 31.0}}) == "AV"
    assert catalog._emitted_branch("QA-25", {"truth": {"value": 31.0}}) is None
    assert "QA-25" not in catalog.ANSWER_TOKEN_BRANCH_MAP


def test_types_whose_answer_is_their_branch_are_left_alone():
    for qa_id, value in (("QA-06", "moving"), ("QA-07", "left"), ("QA-09", "yes"),
                         ("QA-08", "out_of_view"), ("QA-24", "fully_occluded")):
        assert catalog._emitted_branch(qa_id, {"truth": {"value": value}}) == value
        assert qa_id not in catalog.ANSWER_TOKEN_BRANCH_MAP
