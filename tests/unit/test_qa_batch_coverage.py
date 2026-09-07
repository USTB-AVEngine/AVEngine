from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.qa.answerability import structural_baselines
from avengine.qa.batch_coverage import (
    COVERAGE_STATES,
    BatchCoverageError,
    build_batch_coverage,
    validate_batch_coverage,
)


def _asset(asset_id: str, entity_class: str, *, category: str = "source") -> dict:
    return {
        "asset_id": asset_id,
        "entity_class": entity_class,
        "category": category,
        "allowed_event_classes": ["speech"] if entity_class == "articulated_human" else [],
    }


def _runtime(asset_id: str, *, wall: bool = False) -> dict:
    surface = "wall" if wall else "floor"
    return {
        "asset_id": asset_id,
        "runtime_backends": {
            "spear_unreal": {"binding": "test"},
            "habitat": {
                "resting_pose": {"attachment_surface": surface}
            },
        },
    }


def _facts(actors: dict[str, dict], *, episode_id: str) -> dict:
    return {
        "schema": "avengine_qa_unified_episode_facts_v1",
        "status": "pass",
        "episode_id": episode_id,
        "actors": actors,
        "events": [
            {
                "event_id": "event_001",
                "actor_id": "source1",
                "source_endpoint_id": "source1_mouth",
                "sound_asset_id": "test_speech",
                "sound_class": "speech",
            }
        ],
        "source_paths": {
            "frame_readbacks": "frame_readbacks.json",
            "mixture_audio": "mixture.wav",
            "pixel_visibility_truth": "pixel_visibility_truth.json",
        },
    }


def _questions(
    episode_id: str,
    *,
    items: list[dict] | None = None,
    coverage: list[dict] | None = None,
) -> dict:
    return {
        "schema": "avengine_qa_unified_question_set_v1",
        "status": "research_candidate",
        "episode_id": episode_id,
        "counts": {
            "requested": 24,
            "valid": len(items or []),
            "deferred": len(coverage or []),
        },
        "items": items or [],
        "coverage": coverage or [],
        "actual_evidence_summary": {
            "actor_count": 2,
            "event_count": 1,
            "audio_validation_status": "pass",
        },
    }


def _target_item(qa_id: str = "QA-02", actor_id: str = "source1") -> dict:
    return {
        "qa_id": qa_id,
        "status": "pass",
        "question_id": f"question_{qa_id}",
        "forms": {
            "mcq": {
                "options": [
                    {"value": "blue"},
                    {"value": "green"},
                ]
            },
            "open": {"truth": "blue"},
        },
        "evidence": {
            "target_actor_id": actor_id,
            "event_id": "event_001",
        },
    }


def _global_item(qa_id: str = "QA-22") -> dict:
    return {
        "qa_id": qa_id,
        "status": "pass",
        "question_id": f"question_{qa_id}",
        "forms": {"mcq": {"options": [{"value": "2|2"}]}},
        "evidence": {
            "appeared_actor_ids": ["source1", "source2"],
            "entity_count": 2,
            "speaking_count": 2,
        },
    }


def _manifest(
    tmp_path: Path,
    *,
    assets: list[dict],
    rooms: list[dict],
    runtime: list[dict],
    episodes: list[tuple[str, str, str, dict, dict]],
) -> dict:
    asset_path = tmp_path / "assets.json"
    room_path = tmp_path / "rooms.json"
    runtime_path = tmp_path / "runtime.json"
    asset_path.write_text(json.dumps({"assets": assets}), encoding="utf-8")
    room_path.write_text(json.dumps({"rooms": rooms}), encoding="utf-8")
    runtime_path.write_text(json.dumps({"assets": runtime}), encoding="utf-8")
    manifest_episodes = []
    for episode_id, room_id, family, facts, questions in episodes:
        episode_dir = tmp_path / episode_id
        episode_dir.mkdir()
        facts_path = episode_dir / "facts.json"
        questions_path = episode_dir / "questions.json"
        facts_path.write_text(json.dumps(facts), encoding="utf-8")
        questions_path.write_text(json.dumps(questions), encoding="utf-8")
        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "facts": str(facts_path),
                "questions": str(questions_path),
                "room_id": room_id,
                "source_room_id": room_id,
                "family": family,
                "source_refs": {"test": str(episode_dir)},
            }
        )
    return {
        "schema": "avengine_qa_batch_episode_input_manifest_v1",
        "asset_inventory": str(asset_path),
        "room_catalog": str(room_path),
        "runtime_registry": str(runtime_path),
        "episodes": manifest_episodes,
    }


def _base_actors() -> dict[str, dict]:
    return {
        "source1": {
            "actor_id": "source1",
            "asset_id": "human",
            "appearance": {"field": "top_color", "value": "blue"},
        },
        "source2": {
            "actor_id": "source2",
            "asset_id": "device",
            "appearance": {"field": "finish", "value": "black"},
        },
    }


def test_five_states_are_exclusive_and_invalid_state_rejected(tmp_path: Path) -> None:
    assets = [
        _asset("human", "articulated_human"),
        _asset("device", "rigid_static_object"),
        _asset("wall_device", "rigid_static_object"),
    ]
    rooms = [
        {"room_id": "ue_room", "family": "authored", "renderer": "ue_spear"},
        {"room_id": "hab_room", "family": "mp3d", "renderer": "habitat"},
    ]
    runtime = [
        _runtime("human"),
        _runtime("device"),
        _runtime("wall_device", wall=True),
    ]
    facts = _facts(_base_actors(), episode_id="episode_ue")
    questions = _questions(
        "episode_ue",
        items=[_target_item()],
        coverage=[
            {
                "qa_id": "QA-03",
                "status": "deferred",
                "code": "insufficient_candidates",
                "detail": "test defer",
            }
        ],
    )
    manifest = _manifest(
        tmp_path,
        assets=assets,
        rooms=rooms,
        runtime=runtime,
        episodes=[("episode_ue", "ue_room", "authored", facts, questions)],
    )
    result = build_batch_coverage(manifest)
    states = {row["state"] for row in result["rows"]}
    assert states == set(COVERAGE_STATES)
    validate_batch_coverage(result)
    wall_motion = next(
        row
        for row in result["rows"]
        if row["asset_id"] == "wall_device"
        and row["room_id"] == "hab_room"
        and row["qa_id"] == "QA-06"
    )
    assert wall_motion["state"] == "not_applicable_by_definition"
    assert wall_motion["interface_gap"] is not None
    broken = deepcopy(result)
    broken["rows"][0]["state"] = "valid"
    with pytest.raises(BatchCoverageError, match="invalid state"):
        validate_batch_coverage(broken)


def test_denominator_is_stable_when_every_qa_is_deferred(tmp_path: Path) -> None:
    assets = [_asset("human", "articulated_human")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("human")]
    deferred = [
        {
            "qa_id": qa_id,
            "status": "deferred",
            "code": "insufficient_candidates",
            "detail": "deferred test",
        }
        for qa_id in [f"QA-{index:02d}" for index in range(1, 25)]
    ]
    manifest = _manifest(
        tmp_path,
        assets=assets,
        rooms=rooms,
        runtime=runtime,
        episodes=[
            (
                "deferred_episode",
                "room_a",
                "authored",
                _facts(
                    {
                        "source1": {
                            "actor_id": "source1",
                            "asset_id": "human",
                            "appearance": {"field": "top_color", "value": "blue"},
                        }
                    },
                    episode_id="deferred_episode",
                ),
                _questions("deferred_episode", coverage=deferred),
            )
        ],
    )
    result = build_batch_coverage(manifest)
    assert result["denominator"]["row_count"] == 24
    assert len(result["rows"]) == 24
    assert {row["state"] for row in result["rows"]} == {"deferred_by_rule"}


def test_same_family_rooms_remain_separate(tmp_path: Path) -> None:
    assets = [_asset("human", "articulated_human")]
    rooms = [
        {"room_id": "authored_a", "family": "authored", "renderer": "ue_spear"},
        {"room_id": "authored_b", "family": "authored", "renderer": "ue_spear"},
    ]
    runtime = [_runtime("human")]
    manifest = _manifest(
        tmp_path,
        assets=assets,
        rooms=rooms,
        runtime=runtime,
        episodes=[
            (
                "episode_a",
                "authored_a",
                "authored",
                _facts(
                    {
                        "source1": {
                            "actor_id": "source1",
                            "asset_id": "human",
                            "appearance": {"field": "top_color", "value": "blue"},
                        }
                    },
                    episode_id="episode_a",
                ),
                _questions("episode_a", items=[_target_item()]),
            )
        ],
    )
    result = build_batch_coverage(manifest)
    qa02 = {
        row["room_id"]: row
        for row in result["rows"]
        if row["qa_id"] == "QA-02"
    }
    assert qa02["authored_a"]["state"] == "produced"
    assert qa02["authored_b"]["state"] == "evidence_missing_or_unsampled"
    assert result["denominator"]["row_count"] == 48


def test_global_item_is_not_multiplied_over_mixed_pair(tmp_path: Path) -> None:
    assets = [
        _asset("human", "articulated_human"),
        _asset("animal", "articulated_animal"),
    ]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("human"), _runtime("animal")]
    actors = _base_actors()
    actors["source2"]["asset_id"] = "animal"
    facts = _facts(actors, episode_id="mixed_episode")
    questions = _questions(
        "mixed_episode",
        items=[_target_item("QA-02", "source1"), _global_item("QA-22")],
    )
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=runtime,
            episodes=[("mixed_episode", "room_a", "authored", facts, questions)],
        )
    )
    qa02 = {
        row["asset_id"]: row
        for row in result["rows"]
        if row["qa_id"] == "QA-02"
    }
    qa22 = {
        row["asset_id"]: row
        for row in result["rows"]
        if row["qa_id"] == "QA-22"
    }
    assert qa02["human"]["state"] == "produced"
    assert qa02["animal"]["state"] == "evidence_missing_or_unsampled"
    assert all(row["state"] == "evidence_missing_or_unsampled" for row in qa22.values())
    assert any(
        outcome["qa_id"] == "QA-22"
        and outcome["scope"] == "global"
        for outcome in result["global_outcomes"]
    )


def test_generated_item_without_evidence_is_not_produced(tmp_path: Path) -> None:
    assets = [_asset("human", "articulated_human")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("human")]
    item = _target_item()
    item["evidence"] = {}
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=runtime,
            episodes=[
                (
                    "episode_missing",
                    "room_a",
                    "authored",
                    _facts(
                        {
                            "source1": {
                                "actor_id": "source1",
                                "asset_id": "human",
                                "appearance": {"field": "top_color", "value": "blue"},
                            }
                        },
                        episode_id="episode_missing",
                    ),
                    _questions("episode_missing", items=[item]),
                )
            ],
        )
    )
    row = next(row for row in result["rows"] if row["qa_id"] == "QA-02")
    assert row["state"] == "evidence_missing_or_unsampled"
    assert row["reason_code"] == "missing_item_evidence"


def test_structural_six_counterexamples_have_explicit_counts() -> None:
    cases = [
        ({"a": "x", "b": "x", "c": "y"}, "a", True, False),
        ({"a": "x", "b": "y", "c": "y"}, "a", False, True),
        ({"a": "x", "b": "y", "c": "z"}, "a", False, False),
        ({"a": "x", "b": "x", "c": "x"}, "a", True, False),
        ({"a": "x", "b": "y"}, "a", False, False),
        ({"a": "x", "b": "y", "c": "y", "d": "y"}, "a", False, True),
    ]
    for values, gold, majority, minority in cases:
        result = structural_baselines(values, gold, answer_domain_size=4)
        assert result["status"] == "measured"
        assert result["gold_is_majority"] is majority
        assert result["gold_is_unique_minority"] is minority
        assert "majority_hits" in result
        assert "unique_minority_hits" in result
        assert "random_hits" in result



def test_missing_deferred_evidence_is_not_a_legal_defer(tmp_path: Path) -> None:
    assets = [_asset("human", "articulated_human")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("human")]
    facts = _facts(
        {
            "source1": {
                "actor_id": "source1",
                "asset_id": "human",
                "appearance": {"field": "top_color", "value": "blue"},
            }
        },
        episode_id="evidence_gap",
    )
    questions = _questions(
        "evidence_gap",
        coverage=[
            {
                "qa_id": "QA-18",
                "status": "deferred",
                "code": "missing_source_activity_readback",
                "detail": "source activity is absent",
            },
            {
                "qa_id": "QA-07",
                "status": "deferred",
                "code": "no_entry_transition",
                "detail": "no legal transition",
            },
        ],
    )
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=runtime,
            episodes=[("evidence_gap", "room_a", "authored", facts, questions)],
        )
    )
    rows = {row["qa_id"]: row for row in result["rows"] if row["asset_id"] == "human"}
    assert rows["QA-18"]["state"] == "evidence_missing_or_unsampled"
    assert rows["QA-07"]["state"] == "deferred_by_rule"


def test_episode_ids_and_catalog_family_are_authoritative(tmp_path: Path) -> None:
    assets = [_asset("human", "articulated_human")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("human")]
    facts = _facts(
        {
            "source1": {
                "actor_id": "source1",
                "asset_id": "human",
                "appearance": {"field": "top_color", "value": "blue"},
            }
        },
        episode_id="facts_id",
    )
    questions = _questions("questions_id", items=[_target_item()])
    manifest = _manifest(
        tmp_path,
        assets=assets,
        rooms=rooms,
        runtime=runtime,
        episodes=[("manifest_id", "room_a", "authored", facts, questions)],
    )
    with pytest.raises(BatchCoverageError, match="facts episode_id"):
        build_batch_coverage(manifest)


def test_speech_playback_and_animal_transcript_applicability(tmp_path: Path) -> None:
    assets = [
        _asset("animal", "articulated_animal", category="animal"),
        _asset("speaker", "rigid_static_object", category="audio_playback"),
    ]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("animal"), _runtime("speaker")]
    actors = {
        "source1": {
            "actor_id": "source1",
            "asset_id": "animal",
            "appearance": {"field": "coat_profile.value", "value": "standard"},
        },
        "source2": {
            "actor_id": "source2",
            "asset_id": "speaker",
            "appearance": {"field": "finish", "value": "black_ash"},
        },
    }
    facts = _facts(actors, episode_id="speech_mix")
    facts["events"][0]["actor_id"] = "source2"
    facts["events"][0]["sound_class"] = "speech"
    questions = _questions("speech_mix")
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=runtime,
            episodes=[("speech_mix", "room_a", "authored", facts, questions)],
        )
    )
    rows = {
        row["asset_id"]: row
        for row in result["rows"]
        if row["qa_id"] == "QA-12"
    }
    assert rows["animal"]["state"] == "not_applicable_by_definition"
    assert rows["speaker"]["state"] != "not_applicable_by_definition"


def test_structural_report_uses_p8_form_structure_and_open_k_is_null() -> None:
    from avengine.qa.batch_coverage import _structural_report

    contexts = []
    for index in range(6):
        values = {"a": "x", "b": "x", "c": "y"}
        p8_mcq = structural_baselines(values, "c", answer_domain_size=2)
        p8_open = structural_baselines(values, "c")
        contexts.append(
            {
                "outputs": [
                    {
                        "episode_id": f"e{index}",
                        "qa_id": "QA-02",
                        "question_id": f"q{index}",
                        "scope": "target",
                        "target_asset_ids": ["asset_c"],
                        "reference_asset_ids": [],
                    }
                ],
                "items_by_qa": {
                    "QA-02": [
                        {
                            "question_id": f"q{index}",
                            "forms": {
                                "mcq": {"options": [{"value": "x"}, {"value": "y"}]},
                                "open": {"truth": "y"},
                            },
                            "structure": {"mcq": p8_mcq, "open": p8_open},
                        }
                    ]
                },
                "actors": {},
                "actor_assets": {},
            }
        )
    report = _structural_report(contexts)
    mcq = next(
        row
        for row in report["by_qa_form_k"]
        if row["qa_id"] == "QA-02" and row["form"] == "mcq"
    )
    open_row = next(
        row
        for row in report["by_qa_form_k"]
        if row["qa_id"] == "QA-02" and row["form"] == "open"
    )
    assert mcq["unique_minority_available_rows"] == 6
    assert mcq["unique_minority_hits_all"] == 6.0
    assert open_row["k"] is None
    assert open_row["random_rate_calibration"].startswith("uncalibrated_open")


def test_joint_summary_has_three_classes_by_seven_rooms_by_24_qas(
    tmp_path: Path,
) -> None:
    assets = [
        _asset("human", "articulated_human"),
        _asset("animal", "articulated_animal"),
        _asset("device", "rigid_static_object"),
    ]
    rooms = [
        {"room_id": f"room_{index}", "family": "authored", "renderer": "ue_spear"}
        for index in range(7)
    ]
    runtime = [_runtime(asset["asset_id"]) for asset in assets]
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=runtime,
            episodes=[],
        )
    )
    joint = result["summaries"]["joint_asset_class_family_room_qa"]
    assert joint["row_count"] == 3 * 7 * 24
    assert len(joint["cells"]) == 504



def test_original_sound_paths_are_scoped_to_observed_assets(tmp_path: Path) -> None:
    assets = [_asset("human", "articulated_human"),
              _asset("device", "rigid_static_object"),
              _asset("unused", "articulated_human")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    facts = _facts(_base_actors(), episode_id="source_episode")
    questions = _questions("source_episode", items=[_target_item()])
    manifest = _manifest(tmp_path, assets=assets, rooms=rooms,
                         runtime=[_runtime(row["asset_id"]) for row in assets],
                         episodes=[("source_episode", "room_a", "authored", facts, questions)])
    source = tmp_path / "original.wav"
    source.write_bytes(b"source-path fixture")
    manifest["episodes"][0]["sound_events"] = [{
        "event_id": "event_001", "actor_id": "source1", "sound_asset_id": "test_speech",
        "dry_audio_origin": {"path": str(source)}, "speaker_id": "speaker_fixture"}]
    result = build_batch_coverage(manifest)
    rows = {row["asset_id"]: row for row in result["rows"] if row["qa_id"] == "QA-02"}
    assert rows["human"]["sound_origins"][0]["dry_audio_origin"]["path"] == str(source)
    assert rows["human"]["sound_origins"][0]["source_status"] == "dry_source_file_verified"
    assert rows["unused"]["sound_origins"] == []
    assert rows["unused"]["context_sound_origins"][0]["asset_id"] == "human"
    assert rows["unused"]["context_sound_origins"][0]["speaker_id"] == "speaker_fixture"

def test_failed_episode_gap_state_is_used_instead_of_asset_not_in_episode(tmp_path: Path) -> None:
    assets = [
        _asset("human", "articulated_human"),
        _asset("cat", "articulated_animal"),
        _asset("lamp", "rigid_static_object"),
    ]
    rooms = [{"room_id": "mp3d_room", "family": "mp3d", "renderer": "habitat"}]
    runtime = [_runtime("human"), _runtime("cat"), _runtime("lamp")]
    manifest = _manifest(
        tmp_path,
        assets=assets,
        rooms=rooms,
        runtime=runtime,
        episodes=[],
    )
    manifest["failed_episodes"] = [
        {
            "episode_id": "qa_pilot46_20260907_mp3d_human_animal",
            "room_id": "mp3d_room",
            "asset_ids": ["human", "cat"],
            "gap_state": "interface_not_implemented",
            "failure_stage": "audio",
            "failure_reason": "AudioProgram validation failed: sequential_sources events must not overlap",
        }
    ]
    result = build_batch_coverage(manifest)
    validate_batch_coverage(result)
    human_row = next(
        row for row in result["rows"]
        if row["asset_id"] == "human" and row["qa_id"] == "QA-02"
    )
    lamp_row = next(
        row for row in result["rows"]
        if row["asset_id"] == "lamp" and row["qa_id"] == "QA-02"
    )
    assert human_row["state"] == "interface_not_implemented"
    assert human_row["reason"].startswith("audio:")
    assert "AudioProgram validation failed" in human_row["reason"]
    assert lamp_row["state"] == "evidence_missing_or_unsampled"
    assert lamp_row["reason_code"] == "no_episode_input_for_room"

