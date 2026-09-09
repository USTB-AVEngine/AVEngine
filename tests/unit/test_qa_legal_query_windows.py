from __future__ import annotations

from copy import deepcopy
import re

import pytest

import avengine.qa.unified_catalog as catalog
from avengine.qa.unified_catalog import (
    generate_unified_questions,
    normalize_episode_bundle,
)
from avengine.qa.unified_scoring import score_time_range, score_unified_item
from tests.unit.test_qa_unified_catalog import _fixture


def _uniform_sampling(facts: dict) -> dict:
    facts["sampling"] = {
        "qa_sampling": {
            "query_time_policy": "uniform_in_legal_window",
            "time_display_precision": 2,  # This fixture exercises subsecond legal windows.
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



def test_reviewed_human_label_is_used_for_an_unseen_appearance_value() -> None:
    raw = _fixture()
    raw["actors"]["a0"]["realized_attributes"]["coat_profile"]["value"] = "custom_pattern_v2"
    raw["actors"]["a0"]["display_label"] = "gold striped person v2"
    raw["appearance_review"]["actors"]["a0"]["value"] = "custom_pattern_v2"
    facts = normalize_episode_bundle(raw)
    output = generate_unified_questions(
        facts, qa_ids=["QA-19"], seed="unseen-appearance", items_per_type=4
    )
    assert output["items"]
    for item in output["items"]:
        surface = " ".join(
            [
                item["question"]["en"],
                item["question"]["zh"],
                item["truth"]["label"],
                *[
                    str(option.get(key, ""))
                    for option in item.get("forms", {}).get("mcq", {}).get("options", [])
                    for key in ("label_en", "label_zh")
                ],
            ]
        )
        assert "custom_pattern_v2" not in surface
        assert " v2" not in surface.casefold()
    assert any("gold striped person" in item["question"]["en"] for item in output["items"])


def test_version_stripped_appearance_labels_are_unique_checked() -> None:
    raw = _fixture()
    raw["actors"]["a0"]["realized_attributes"]["coat_profile"]["value"] = "custom_a"
    raw["actors"]["a1"]["realized_attributes"]["coat_profile"]["value"] = "custom_b"
    raw["actors"]["a0"]["display_label"] = "Shiba Inu v2"
    raw["actors"]["a1"]["display_label"] = "Shiba Inu v3"
    raw["appearance_review"]["actors"]["a0"]["value"] = "custom_a"
    raw["appearance_review"]["actors"]["a1"]["value"] = "custom_b"
    facts = normalize_episode_bundle(raw)
    output = generate_unified_questions(
        facts, qa_ids=["QA-19"], seed="duplicate-version-labels", items_per_type=1
    )
    assert output["counts"]["valid"] == 0
    assert output["deferred"][0]["code"] == "appearance_display_labels_not_unique"


def test_time_range_domain_scales_with_duration_and_scores_both_forms() -> None:
    facts = normalize_episode_bundle(_fixture())
    facts["time"]["duration_seconds"] = 8.0
    output = generate_unified_questions(
        facts, qa_ids=["QA-19"], seed="duration-20", items_per_type=1
    )
    assert output["items"]
    item = output["items"][0]
    assert item["forms"]["open"]["answer_type"] == "time_range_s"
    assert item["truth"]["value"] == [0.0, 2.0]
    assert "[0, 2) seconds" in item["question"]["en"]
    assert item["evidence"]["time_bands_s"] == item["forms"]["open"]["time_ranges_s"]
    assert score_time_range(
        "[0, 2) seconds",
        item["truth"]["value"],
        form=item["forms"]["open"],
    )["score"] == 1.0
    assert score_unified_item(
        item, "[0, 2) seconds", form="open"
    )["score"] == 1.0
    correct = item["forms"]["mcq"]["gold"]["correct_index"]
    assert score_unified_item(item, chr(ord("A") + correct), form="mcq")["score"] == 1.0


def test_time_range_domain_uses_sampling_granularity_and_decimal_precision() -> None:
    facts = normalize_episode_bundle(_fixture())
    facts["time"]["duration_seconds"] = 10.0
    facts["sampling"] = {
        "qa_sampling": {
            "time_band_count": 5,
            "time_display_precision": 1,
        }
    }
    output = generate_unified_questions(
        facts, qa_ids=["QA-19"], seed="configured-domain", items_per_type=1
    )
    item = output["items"][0]
    assert item["forms"]["open"]["time_ranges_s"] == [
        [0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]
    ]
    assert "5 time intervals" in item["question"]["en"]
    assert "[0, 2) seconds" in item["question"]["en"]
    assert len(item["forms"]["mcq"]["options"]) == 5


def test_public_query_window_is_inward_quantized_and_retains_exact_frames() -> None:
    facts = normalize_episode_bundle(_fixture())
    facts["time"]["frame_rate_hz"] = 3.0
    facts["sampling"] = {"qa_sampling": {"time_display_precision": 1}}
    fields = catalog._query_window_fields(facts, [1, 4])
    assert fields["query_window_frames"] == [1, 4]
    assert fields["query_window_s"] == [0.4, 1.3]
    assert fields["query_window_exact_s"] == [1 / 3, 4 / 3]
    assert fields["query_window_precision"] == 1
    assert fields["query_window_s"][0] >= fields["query_window_exact_s"][0]
    assert fields["query_window_s"][1] <= fields["query_window_exact_s"][1]


def test_interval_questions_keep_exact_frames_in_evidence_only() -> None:
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
    output = generate_unified_questions(
        facts,
        qa_ids=["QA-04", "QA-08", "QA-13", "QA-14", "QA-16", "QA-17", "QA-18", "QA-19", "QA-21", "QA-24"],
        seed="surface-intervals",
    )
    assert output["items"]
    exact_time = re.compile(r"\b\d+\.\d{3}\s*seconds?\b", re.I)
    exact_frame = re.compile(r"\b(?:video )?frame\s+\d+\b", re.I)
    for item in output["items"]:
        question = item["question"]["en"]
        assert not exact_time.search(question)
        assert not exact_frame.search(question)
        assert "validated" not in question.casefold()
        assert "during between" not in question.casefold()
        assert "经核验" not in item["question"]["zh"]
        if item["qa_id"] in {"QA-04", "QA-08", "QA-13", "QA-14", "QA-16", "QA-17", "QA-18"}:
            assert len(item["evidence"].get("query_window_frames", [])) == 2
        if item["qa_id"] == "QA-21":
            assert "_" not in item["truth"]["label"]
            assert all("_" not in option["label_en"] for option in item["forms"]["mcq"]["options"])
        if item["qa_id"] in {"QA-08", "QA-24"}:
            assert "_" not in item["truth"]["label"]



def test_generic_playback_capability_yields_to_bound_asset_class() -> None:
    raw = _fixture()
    raw["audio_program"]["events"][0]["sound_class"] = "any_audioset_class_playback"
    raw["sound_registry"] = {
        "sound_assets": [
            {
                "sound_asset_id": "s0",
                "semantic_sound_class": "music_playback",
            }
        ]
    }
    facts = normalize_episode_bundle(raw)
    assert facts["events"][0]["sound_class"] == "music_playback"
    assert facts["events"][0]["sound_class_explicit"] is True



def test_capability_only_sound_class_is_deferred_for_qa21() -> None:
    raw = _fixture()
    raw["audio_program"]["events"][0]["sound_class"] = "any_audioset_class_playback"
    facts = normalize_episode_bundle(raw)
    candidate = {
        "candidate_id": "QA-21:capability",
        "kind": "actor",
        "actor_id": "a0",
        "event_id": "e0",
    }
    candidate_facts = catalog._p8_facts_for_candidate(facts, candidate, "capability")
    with pytest.raises(catalog._Deferred) as raised:
        catalog._P8_BASE_GENERATORS["QA-21"](candidate_facts, "capability")
    assert raised.value.code == "sound_class_capability_only"



def test_duplicate_reviewed_display_labels_are_deferred() -> None:
    facts = normalize_episode_bundle(_fixture())
    facts["actors"]["a0"]["appearance"]["label"] = "same visible actor"
    facts["actors"]["a1"]["appearance"]["label"] = "same visible actor"
    with pytest.raises(catalog._Deferred) as raised:
        catalog._appearance_candidates(facts)
    assert raised.value.code == "appearance_display_labels_not_unique"


def test_public_time_formatter_preserves_integer_trailing_zeroes():
    from avengine.qa.unified_catalog import _format_public_seconds
    assert _format_public_seconds(10.0, precision=0) == "10"
    assert _format_public_seconds(100.0, precision=0) == "100"
    assert _format_public_seconds(10.0, precision=1) == "10"
