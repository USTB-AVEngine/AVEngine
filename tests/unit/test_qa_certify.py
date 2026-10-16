from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from avengine.qa.certify import (
    QACertifyError,
    certify_axis1,
    reanswer_on_fact_table,
    twin_fact_table,
)
from avengine.qa.miner import mine_fact_table

from test_qa_miner import _fact_table

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "avengine_qa_fact_table_v1.schema.json"


def _questions_by_type(fact_table: dict) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {}
    for question in mine_fact_table(fact_table):
        by_type.setdefault(question["type_id"], []).append(question)
    return by_type


def test_twin_swaps_voices_and_keeps_visual_facts() -> None:
    original = _fact_table()
    twin = twin_fact_table(original)

    assert twin["episode_id"] == original["episode_id"] + "__cf_route_swap"
    assert twin["counterfactual"]["twin_of"] == original["episode_id"]
    assert twin["instances"] == original["instances"]
    assert twin["tracks"] == original["tracks"]

    events = {event["source_slot_id"]: event for event in twin["sound_events"]}
    assert events["source1"]["sound_class"]["species_id"] == "cat"
    assert events["source1"]["voice_asset_id"] == "asset_cat"
    assert events["source1"]["asset_id"] == "asset_dog"
    assert events["source2"]["sound_class"]["species_id"] == "dog"
    assert twin["audio"]["mixture_sha256"] == "0" * 64

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(twin)
    # The original table is untouched by twin construction.
    assert original["sound_events"][0]["sound_class"]["species_id"] == "dog"


def test_axis1_certificate_granted_for_attr_and_avrel_flips() -> None:
    original = _fact_table()
    twin = twin_fact_table(original)
    by_type = _questions_by_type(original)

    dog_attr = next(
        question
        for question in by_type["Q-ATTR"]
        if question["evidence"]["instance_id"] == "source1"
    )
    record = certify_axis1(dog_attr, original, twin)
    assert record["status"] == "granted"
    assert record["original_answer"] == "black_white"
    assert record["twin_answer"] == "ruddy"

    dog_avrel = next(
        question
        for question in by_type["Q-AVREL"]
        if question["evidence"]["instance_id"] == "source1"
    )
    record = certify_axis1(dog_avrel, original, twin)
    assert record["status"] == "granted"
    assert record["original_answer"] == "left"
    assert record["twin_answer"] == "right"


def test_axis1_refuses_ambiguous_twin_and_marks_not_applicable() -> None:
    # Cat crosses the front axis: the dog's voice lands on an ambiguous route.
    original = _fact_table(cat_x_start=-0.4, cat_z=-1.0)
    twin = twin_fact_table(original)
    by_type = _questions_by_type(original)

    (dog_avrel,) = [
        question
        for question in by_type["Q-AVREL"]
        if question["evidence"]["instance_id"] == "source1"
    ]
    record = certify_axis1(dog_avrel, original, twin)
    assert record["status"] == "refused"
    assert record["twin_answer"] is None

    (cmp_question,) = by_type["Q-CMP"] if by_type.get("Q-CMP") else [None]
    if cmp_question is not None:
        record = certify_axis1(cmp_question, original, twin)
        assert record["status"] == "not_applicable"


def test_axis1_refuses_when_mined_answer_cannot_be_reproduced() -> None:
    original = _fact_table()
    twin = twin_fact_table(original)
    question = next(
        question
        for question in _questions_by_type(original)["Q-ATTR"]
        if question["evidence"]["instance_id"] == "source1"
    )
    tampered = dict(question)
    tampered["answer_value"] = "yellow"
    record = certify_axis1(tampered, original, twin)
    assert record["status"] == "refused"
    assert "disagrees" in record["reason"]


def test_reanswer_tracks_voice_not_slot() -> None:
    original = _fact_table()
    twin = twin_fact_table(original)
    question = next(
        question
        for question in _questions_by_type(original)["Q-ATTR"]
        if question["evidence"]["instance_id"] == "source2"
    )
    # Cat voice moved to the dog's slot in the twin.
    assert reanswer_on_fact_table(question, original) == "ruddy"
    assert reanswer_on_fact_table(question, twin) == "black_white"


def test_twin_requires_exactly_two_instances() -> None:
    original = _fact_table()
    original["instances"] = original["instances"][:1]
    with pytest.raises(QACertifyError):
        twin_fact_table(original)
