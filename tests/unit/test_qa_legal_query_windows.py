from __future__ import annotations

from copy import deepcopy

import avengine.qa.unified_catalog as catalog
from avengine.qa.unified_catalog import (
    generate_unified_questions,
    normalize_episode_bundle,
)
from tests.unit.test_qa_unified_catalog import _fixture


def _uniform_sampling(facts: dict) -> dict:
    facts["sampling"] = {
        "qa_sampling": {
            "query_time_policy": "uniform_in_legal_window",
        }
    }
    return facts


def test_uniform_post_sound_windows_are_derived_per_event() -> None:
    facts = _uniform_sampling(normalize_episode_bundle(_fixture()))
    candidates = catalog._P8_CANDIDATES["QA-17"](facts)

    by_event: dict[str, list[int]] = {}
    by_window: dict[str, object] = {}
    for candidate in candidates:
        by_event.setdefault(candidate["event_id"], []).append(candidate["query_frame"])
        by_window[candidate["event_id"]] = candidate["legal_query_windows"]

    assert by_event == {
        "e0": [7, 8, 9],
        "e2": [17, 18, 19],
        "e3": list(range(24, 40)),
    }
    assert by_window == {
        "e0": [[7, 10]],
        "e2": [[17, 20]],
        "e3": [[24, 40]],
    }
    assert all(
        candidate["legal_window_authority"]
        == "native_audio_event_gap_and_wet_tail_readback_v1"
        for candidate in candidates
    )

    output = generate_unified_questions(
        facts, qa_ids=["QA-17"], seed="derived-post-sound"
    )
    item = output["items"][0]
    assert item["form_status"]["open"]["status"] == "pass"
    assert item["evidence"]["post_sound"]["legal_query_windows"] in (
        [[7, 10]],
        [[24, 40]],
    )
    assert (
        item["evidence"]["post_sound"]["legal_window_authority"]
        == "native_audio_event_gap_and_wet_tail_readback_v1"
    )


def test_explicit_post_sound_window_is_preserved() -> None:
    facts = _uniform_sampling(normalize_episode_bundle(_fixture()))
    facts["sampling"]["legal_window_by_qa"] = {"QA-17": [24, 28]}

    candidates = catalog._P8_CANDIDATES["QA-17"](facts)
    assert candidates
    assert {candidate["query_frame"] for candidate in candidates} == {24, 25, 26, 27}
    assert {tuple(map(tuple, candidate["legal_query_windows"])) for candidate in candidates} == {
        ((24, 28),)
    }
    assert all(
        candidate["legal_window_authority"] == "caller_declared_sampling_window"
        for candidate in candidates
    )


def test_uniform_qa18_uses_wet_tail_complement_union() -> None:
    raw = _fixture()
    raw["audio_readback"]["source_activity_intervals_samples"] = [
        {
            "event_id": event["event_id"],
            "start_sample": event["start_sample"],
            "end_sample_exclusive": event["end_sample_exclusive"],
        }
        for event in raw["audio_program"]["events"]
    ]
    facts = _uniform_sampling(normalize_episode_bundle(raw))
    candidates = catalog._P8_CANDIDATES["QA-18"](facts)

    assert len(candidates) == 24
    assert {candidate["query_frame"] for candidate in candidates} == {
        *range(0, 2),
        *range(7, 10),
        *range(17, 20),
        *range(24, 40),
    }
    assert all(candidate["legal_query_windows"] == [[0, 2], [7, 10], [17, 20], [24, 40]]
                  for candidate in candidates)
    assert all(
        candidate["legal_window_authority"]
        == "native_wet_tail_complement_frame_clock_v1"
        for candidate in candidates
    )

    output = generate_unified_questions(
        facts, qa_ids=["QA-18"], seed="derived-qa18"
    )
    assert output["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    item = output["items"][0]
    assert item["evidence"]["legal_query_windows"] == [
        [0, 2],
        [7, 10],
        [17, 20],
        [24, 40],
    ]
    assert (
        item["evidence"]["legal_window_authority"]
        == "native_wet_tail_complement_frame_clock_v1"
    )


def test_uniform_temporal_queries_stay_deferred_without_wet_tail_readback() -> None:
    raw = _fixture()
    raw["audio_readback"].pop("wet_tail_intervals")
    facts = _uniform_sampling(normalize_episode_bundle(raw))

    for qa_id in ("QA-13", "QA-16", "QA-17", "QA-18"):
        output = generate_unified_questions(
            facts, qa_ids=[qa_id], seed=f"missing-tail-{qa_id}"
        )
        assert output["counts"]["valid"] == 0
        assert output["counts"]["deferred"] == 1
        assert output["deferred"][0]["code"] in {
            "no_valid_post_sound_window",
            "sampling_window_missing",
        }
