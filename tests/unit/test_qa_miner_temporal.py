from __future__ import annotations

import pytest

from avengine.qa.intermittent import event_records
from avengine.qa.miner import QAMinerError
from avengine.qa.miner_temporal import mine_temporal_fact_table

from test_qa_miner import _fact_table


def _intermittent_fact_table() -> dict:
    """Continuous fixture re-compiled with declared multi-window events."""

    declared = {
        "source1": event_records(
            slot_id="source1", windows=[(8000, 20000), (40000, 52000)]
        ),
        "source2": event_records(
            slot_id="source2", windows=[(16000, 28000), (56000, 68000)]
        ),
    }
    return _fact_table(declared_events_by_slot=declared)


def test_temporal_mining_yields_expected_answers() -> None:
    fact_table = _intermittent_fact_table()
    by_type: dict[str, list[dict]] = {}
    for question in mine_temporal_fact_table(fact_table):
        by_type.setdefault(question["type_id"], []).append(question)

    counts = {
        question["evidence"]["source_slot_id"]: question["answer_value"]
        for question in by_type["Q-NUM-CNT"]
    }
    assert counts == {"source1": "2", "source2": "2"}

    times = {
        question["evidence"]["source_slot_id"]: question["answer_numeric"]
        for question in by_type["Q-NUM-TIME"]
    }
    assert times["source1"] == pytest.approx(0.5)
    assert times["source2"] == pytest.approx(1.0)
    assert all(
        question["format"] == "numeric_banded" for question in by_type["Q-NUM-TIME"]
    )

    during = {
        question["qualifier_hint"]: question["answer_value"]
        for question in (
            {**q, "qualifier_hint": q["question_id"].split("__")[-1]}
            for q in by_type["Q-TEMP-DURING"]
        )
    }
    # Cat observed during dog barks: moving; dog observed during meows: static.
    assert during["dog0_cat"] == "moving"
    assert during["cat0_dog"] == "static"

    at_answers = {
        q["question_id"].split("__")[-1]: q["answer_value"]
        for q in by_type["Q-TEMP-AT"]
    }
    assert at_answers["dog0_cat"] == "right"
    assert at_answers["cat0_dog"] == "left"

    between = by_type["Q-TEMP-BETWEEN"]
    assert between and all(
        question["answer_value"] in {"moving", "static"} for question in between
    )

    (order,) = by_type["Q-TEMP-ORDER"]
    assert order["answer_value"] == "dog"
    assert order["evidence"]["margin_ticks"] == 24000


def test_temporal_mining_rejects_continuous_fact_tables() -> None:
    continuous = _fact_table()
    with pytest.raises(QAMinerError, match="declared intermittent"):
        mine_temporal_fact_table(continuous)
