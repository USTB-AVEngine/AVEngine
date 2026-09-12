from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.qa.answerability import structural_baselines
from avengine.qa import batch_coverage as bc
from avengine.qa.unified_catalog import QA_IDS
from avengine.qa.batch_coverage import (
    APPEARANCE_CLASSIFIER_GAP_REASON,
    COVERAGE_STATES,
    BatchCoverageError,
    _deferred_state,
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
        for qa_id in [f"QA-{index:02d}" for index in range(1, 26)]
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
    assert result["denominator"]["row_count"] == 25
    assert len(result["rows"]) == 25
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
    assert result["denominator"]["row_count"] == 50


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


def test_joint_summary_has_three_classes_by_seven_rooms_by_25_qas(
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
    assert joint["row_count"] == 3 * 7 * 25
    assert len(joint["cells"]) == 525



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



def test_codex_worktree_paths_are_rewritten_in_provenance(tmp_path: Path) -> None:
    from avengine.qa.batch_coverage import CODEX_WORKTREE_PREFIX, build_batch_coverage, validate_batch_coverage

    assets = [_asset("human", "articulated_human"), _asset("lamp", "rigid_static_object")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    runtime = [_runtime("human"), _runtime("lamp")]
    manifest = _manifest(tmp_path, assets=assets, rooms=rooms, runtime=runtime, episodes=[])
    repo = tmp_path / "prod"
    mapping = {
        "asset_inventory": "assets.json",
        "room_catalog": "rooms.json",
        "runtime_registry": "runtime.json",
    }
    for key, name in mapping.items():
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(manifest[key]).read_bytes())
        manifest[key] = f"{CODEX_WORKTREE_PREFIX}/{name}"
    result = build_batch_coverage(manifest, repository=repo)
    validate_batch_coverage(result)
    for key, name in mapping.items():
        value = str(result["provenance"][key])
        assert CODEX_WORKTREE_PREFIX not in value
        assert str((repo / name).resolve()) == value


def test_deferred_state_maps_classifier_gap_to_interface_not_implemented() -> None:
    assert _deferred_state(APPEARANCE_CLASSIFIER_GAP_REASON) == "interface_not_implemented"
    assert _deferred_state("missing_source_activity_readback") == "evidence_missing_or_unsampled"
    assert _deferred_state("appearance_review_missing") == "deferred_by_rule"


def test_classifier_gap_appearance_defer_is_interface_not_implemented(tmp_path: Path) -> None:
    assets = [_asset("silver_device", "rigid_static_object")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    facts = _facts(
        {
            "source1": {
                "actor_id": "source1",
                "asset_id": "silver_device",
                "appearance": {"field": "body_color", "value": "silver"},
            }
        },
        episode_id="classifier_gap",
    )
    facts["appearance_review"] = {
        "source1": {
            "status": "not_observable",
            "value": "silver",
            "reason": APPEARANCE_CLASSIFIER_GAP_REASON,
            "gap_category": "interface_not_implemented",
            "checks": [
                {
                    "status": "not_observable",
                    "reason": APPEARANCE_CLASSIFIER_GAP_REASON,
                    "frame_index": 0,
                }
            ],
        }
    }
    questions = _questions(
        "classifier_gap",
        coverage=[
            {
                "qa_id": "QA-01",
                "status": "deferred",
                "code": "appearance_review_missing",
                "detail": "no actor has a matching reviewed appearance value",
            },
            {
                "qa_id": "QA-18",
                "status": "deferred",
                "code": "no_distance_trend_during_event",
                "detail": "no event has a measurable distance trend",
            },
        ],
    )
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=[_runtime("silver_device")],
            episodes=[("classifier_gap", "room_a", "authored", facts, questions)],
        )
    )
    validate_batch_coverage(result)
    rows = {row["qa_id"]: row for row in result["rows"] if row["asset_id"] == "silver_device"}
    assert rows["QA-01"]["state"] == "interface_not_implemented"
    assert rows["QA-01"]["reason_code"] == APPEARANCE_CLASSIFIER_GAP_REASON
    assert rows["QA-18"]["state"] == "deferred_by_rule"


def test_appearance_review_missing_without_classifier_gap_stays_deferred(tmp_path: Path) -> None:
    assets = [_asset("white_device", "rigid_static_object")]
    rooms = [{"room_id": "room_a", "family": "authored", "renderer": "ue_spear"}]
    facts = _facts(
        {
            "source1": {
                "actor_id": "source1",
                "asset_id": "white_device",
                "appearance": {"field": "body_color", "value": "white"},
            }
        },
        episode_id="pixel_gap",
    )
    facts["appearance_review"] = {
        "source1": {
            "status": "not_observable",
            "value": "white",
            "reason": "no visible native RGB pixels in the target mask",
            "checks": [
                {
                    "status": "not_observable",
                    "reason": "no visible native RGB pixels in the target mask",
                    "frame_index": 0,
                }
            ],
        }
    }
    questions = _questions(
        "pixel_gap",
        coverage=[
            {
                "qa_id": "QA-01",
                "status": "deferred",
                "code": "appearance_review_missing",
                "detail": "no actor has a matching reviewed appearance value",
            }
        ],
    )
    result = build_batch_coverage(
        _manifest(
            tmp_path,
            assets=assets,
            rooms=rooms,
            runtime=[_runtime("white_device")],
            episodes=[("pixel_gap", "room_a", "authored", facts, questions)],
        )
    )
    row = next(
        item for item in result["rows"]
        if item["asset_id"] == "white_device" and item["qa_id"] == "QA-01"
    )
    assert row["state"] == "deferred_by_rule"
    assert row["reason_code"] == "appearance_review_missing"


# --------------------------------------------------------------------------- V1 targets and feedback


def _targets(**overrides):
    from avengine.qa.generation_conditions import branches_for
    from avengine.dataset.source_capabilities import combination_key, entity_combinations

    block = {
        "schema": bc.V1_TARGETS_SCHEMA,
        "qa_ids": list(QA_IDS),
        "branches_by_qa_id": {qa_id: list(branches_for(qa_id)) for qa_id in QA_IDS},
        "room_families": ["apartment", "kujiale", "hm3d", "mp3d"],
        "core_task_families": ["visible_binding", "visual_conditioned_relation",
                               "cross_event_identity", "cross_time_state"],
        "entity_combinations": sorted(combination_key(*pair) for pair in entity_combinations()),
        "min_valid_main_questions_per_qa_id": 8,
        "min_distinct_worlds_per_qa_id": 2,
        "min_valid_main_questions_per_branch": 2,
        "min_distinct_worlds_per_branch": 2,
        "min_core_groups_per_task_family_and_room_family": 4,
        "target_core_group_count": 64,
        "min_distinct_worlds_per_entity_combination": 2,
    }
    block.update(overrides)
    return block


def _achieved(by_qa=None, by_branch=None, **overrides):
    table = {
        "schema": "avengine_qa_v1_achieved_coverage_v1",
        "source_kind": "unit_test",
        "member_count": 4, "group_count": 1, "world_count": 1,
        "generation_failures": [],
        "by_qa_id": {
            qa_id: {"valid_main_questions": 0, "valid_angle_followups": 0,
                    "distinct_worlds_with_main": 0, "form_counts": {},
                    "task_families": {}, "room_families": {},
                    "entity_combinations": {}, "source_families": {},
                    "branch_unobservable_main": 0}
            for qa_id in QA_IDS
        },
        "by_qa_branch": {},
        "core_task_by_room_family_member_counts": {},
        "core_task_by_room_family_group_counts": {},
        "deferred_codes_by_qa_id": {},
        "source_families_unresolved": [],
        "counting_note": "unit test",
    }
    for qa_id, row in (by_qa or {}).items():
        table["by_qa_id"][qa_id].update(row)
    table["by_qa_branch"].update(by_branch or {})
    table.update(overrides)
    return table


def test_targets_must_name_every_qa_type() -> None:
    block = _targets(qa_ids=[qa_id for qa_id in QA_IDS if qa_id != "QA-21"])
    with pytest.raises(bc.BatchCoverageError, match="QA-21"):
        bc.load_v1_coverage_targets(block)


def test_targets_must_agree_with_the_shared_branch_table() -> None:
    branches = {qa_id: list(row) for qa_id, row in _targets()["branches_by_qa_id"].items()}
    branches["QA-06"] = ["moving"]
    with pytest.raises(bc.BatchCoverageError, match="branch table"):
        bc.load_v1_coverage_targets(_targets(branches_by_qa_id=branches))


def test_targets_reject_a_group_count_that_cannot_fill_the_matrix() -> None:
    with pytest.raises(bc.BatchCoverageError, match="target_core_group_count"):
        bc.load_v1_coverage_targets(_targets(target_core_group_count=12))


def test_targets_reject_an_unknown_core_task_family() -> None:
    with pytest.raises(bc.BatchCoverageError, match="shared-unit recipe"):
        bc.load_v1_coverage_targets(_targets(core_task_families=["walking_speech"]))


def test_targets_reject_an_unknown_two_entity_combination() -> None:
    with pytest.raises(bc.BatchCoverageError, match="two-entity"):
        bc.load_v1_coverage_targets(_targets(entity_combinations=["human+robot"]))


def test_targets_are_read_from_a_configuration_block() -> None:
    resolved = bc.load_v1_coverage_targets({"coverage_quota": _targets()})
    assert resolved["target_core_group_count"] == 64
    assert len(resolved["qa_ids"]) == len(QA_IDS)


REGISTRY = {
    "assets": [
        {"asset_id": "human_a", "entity_class": "articulated_human"},
        {"asset_id": "dog_a", "entity_class": "articulated_animal"},
        {"asset_id": "speaker_a", "entity_class": "rigid_object",
         "identity": {"category": "audio_playback"}},
        {"asset_id": "fan_a", "entity_class": "rigid_object",
         "identity": {"category": "climate_control"},
         "allowed_event_classes": ["air_conditioning"]},
    ]
}


def test_a_static_device_is_inapplicable_to_the_entry_side_question() -> None:
    # A fixed camera plus a device that cannot walk means "which side did it
    # enter from" has no answer by definition, not missing evidence.
    assert "QA-07" in bc.MOTION_TARGET_QA_IDS
    row = bc.source_family_applicability(REGISTRY)["QA-07"]
    assert row["device"]["state"] == "not_applicable_by_definition"
    assert row["human"]["state"] == "available"
    assert row["animal"]["state"] == "available"


def test_an_inapplicable_device_never_collapses_the_whole_qa_type() -> None:
    applicability = bc.source_family_applicability(REGISTRY)
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(), achieved=_achieved(), applicability=applicability,
    )
    # Every family inapplicable is the only route to an inapplicable type.
    assert feedback["by_qa_id"]["QA-06"]["state"] != "not_applicable_by_definition"
    assert feedback["by_qa_id"]["QA-06"]["source_family_applicability"]["device"][
        "state"] == "not_applicable_by_definition"


def test_a_met_target_needs_both_the_question_count_and_the_world_count() -> None:
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(),
        achieved=_achieved(by_qa={
            "QA-01": {"valid_main_questions": 12, "distinct_worlds_with_main": 3},
            "QA-02": {"valid_main_questions": 12, "distinct_worlds_with_main": 1},
        }),
    )
    assert feedback["by_qa_id"]["QA-01"]["state"] == "met"
    assert feedback["by_qa_id"]["QA-02"]["state"] == "short_of_target"
    assert feedback["by_qa_id"]["QA-02"]["remaining_distinct_worlds"] == 1
    assert feedback["by_qa_id"]["QA-02"]["remaining_valid_main_questions"] == 0


def test_a_partial_count_stays_a_shortfall_and_keeps_its_remainder() -> None:
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(),
        achieved=_achieved(by_qa={
            "QA-10": {"valid_main_questions": 2, "distinct_worlds_with_main": 2},
        }),
    )
    row = feedback["by_qa_id"]["QA-10"]
    assert row["state"] == "short_of_target"
    assert row["remaining_valid_main_questions"] == 6


def test_an_unimplemented_interface_is_not_reported_as_unsampled() -> None:
    planning = {
        "qa_ids": {
            "QA-15": {"candidates": [
                {"branch": "nearer", "state": "interface_not_implemented",
                 "reason": "distance_net_change: routes are sampled for a moving window"},
            ]},
        }
    }
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(), achieved=_achieved(), planning=planning,
    )
    assert feedback["by_qa_branch"]["QA-15:nearer"]["state"] == "interface_not_implemented"
    assert "routes are sampled" in feedback["by_qa_branch"]["QA-15:nearer"]["reason"]
    # An untouched branch of the same type keeps the plain unsampled state.
    assert feedback["by_qa_branch"]["QA-15:farther"]["state"] == "evidence_missing_or_unsampled"


def test_one_available_candidate_keeps_the_cell_plannable() -> None:
    planning = {
        "qa_ids": {
            "QA-09": {"candidates": [
                {"branch": "yes", "state": "interface_not_implemented", "reason": "no knob"},
                {"branch": "yes", "state": "available", "reason": None},
            ]},
        }
    }
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(), achieved=_achieved(), planning=planning,
    )
    assert feedback["by_qa_branch"]["QA-09:yes"]["state"] == "evidence_missing_or_unsampled"


def test_a_generation_failure_is_not_reported_as_missing_evidence() -> None:
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(),
        achieved=_achieved(generation_failures=[
            {"group_id": "g1", "member_id": "v0_a0", "world_id": "w1",
             "error": "UnifiedQAError: broken"},
        ]),
    )
    assert feedback["by_qa_id"]["QA-01"]["state"] == "generation_failed"
    assert feedback["generation_failures"][0]["error"] == "UnifiedQAError: broken"


def test_the_core_matrix_counts_groups_not_members() -> None:
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(),
        achieved=_achieved(
            core_task_by_room_family_member_counts={"visible_binding|apartment": 16},
            core_task_by_room_family_group_counts={"visible_binding|apartment": 4},
        ),
    )
    cell = feedback["core_task_by_room_family"]["visible_binding|apartment"]
    assert cell["complete_group_count"] == 4 and cell["state"] == "met"
    empty = feedback["core_task_by_room_family"]["cross_time_state|mp3d"]
    assert empty["state"] == "evidence_missing_or_unsampled"
    assert empty["remaining_group_count"] == 4


def test_outstanding_work_excludes_cells_a_worker_cannot_act_on() -> None:
    planning = {"qa_ids": {"QA-15": {"candidates": [
        {"branch": "nearer", "state": "interface_not_implemented", "reason": "no knob"},
        {"branch": "farther", "state": "interface_not_implemented", "reason": "no knob"},
    ]}}}
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(),
        achieved=_achieved(by_qa={
            "QA-01": {"valid_main_questions": 12, "distinct_worlds_with_main": 3},
            "QA-10": {"valid_main_questions": 2, "distinct_worlds_with_main": 2},
        }),
        planning=planning,
    )
    rows = bc.outstanding_production_requests(feedback)
    kinds = {(row["kind"], row.get("qa_id"), row.get("branch")) for row in rows}
    assert ("qa_id", "QA-01", None) not in kinds
    assert ("qa_id", "QA-10", None) in kinds
    assert ("qa_branch", "QA-15", "nearer") not in kinds
    assert ("qa_branch", "QA-05", "overlap") in kinds
    assert any(row["kind"] == "core_group_cell" for row in rows)
    assert any(row["kind"] == "entity_combination" for row in rows)


def test_outstanding_work_refuses_a_document_that_is_not_feedback() -> None:
    with pytest.raises(bc.BatchCoverageError):
        bc.outstanding_production_requests({"schema": "something_else"})


SOUND_CONFIG = {
    "species_sound_classes": {"dog": ["dog_bark"]},
    "object_sound_classes": {
        "air_conditioner": ["air_conditioning"],
        "desk_telephone": ["telephone_bell_ringing", "telephone"],
    },
    "speech_playback_categories": ["audio_playback"],
    "undetermined_sound_class_semantics": [
        {"sound_class": "buzzer", "state": "semantics_undetermined_pending_owner_decision",
         "measured_basis": "9 of 10 retained clips are labelled Buzz, not Buzzer"},
    ],
}
SOUND_REGISTRY = {
    "assets": [
        {"asset_id": "fan_a", "entity_class": "rigid_object",
         "identity": {"object_type": "air_conditioner", "category": "climate_control"}},
        {"asset_id": "phone_a", "entity_class": "rigid_object",
         "identity": {"object_type": "desk_telephone", "category": "communication_device"}},
        {"asset_id": "speaker_a", "entity_class": "rigid_object",
         "identity": {"object_type": "smart_speaker", "category": "audio_playback"}},
    ]
}


def test_a_sound_class_with_no_accepting_device_is_reported_not_made_usable() -> None:
    accounting = bc.sound_input_accounting(
        registry=SOUND_REGISTRY, sound_class_config=SOUND_CONFIG,
        registered_event_class_counts={"telephone": 16, "buzzer": 9, "air_conditioning": 20},
    )
    assert accounting["sound_classes_bound_to_a_device"]["telephone"][
        "accepting_asset_ids"] == ["phone_a"]
    unbound = accounting["sound_classes_without_accepting_device"]["buzzer"]
    assert unbound["state"] == bc.V1_UNDETERMINED_SEMANTICS_STATE
    assert unbound["registered_event_count"] == 9
    assert "labelled Buzz" in unbound["undetermined_semantics"]["measured_basis"]
    assert accounting["unbound_registered_event_count"] == 9


def test_an_undeclared_unbound_class_is_not_silently_undetermined() -> None:
    accounting = bc.sound_input_accounting(
        registry=SOUND_REGISTRY, sound_class_config=SOUND_CONFIG,
        registered_event_class_counts={"dial_tone": 4},
    )
    assert accounting["sound_classes_without_accepting_device"]["dial_tone"][
        "state"] == "no_registered_device_declares_this_sound_class"


def test_no_registered_device_accepts_an_arbitrary_sound_class() -> None:
    accounting = bc.sound_input_accounting(
        registry=SOUND_REGISTRY, sound_class_config=SOUND_CONFIG,
        registered_event_class_counts={"dog_bark": 166},
    )
    assert "dog_bark" not in accounting["sound_classes_bound_to_a_device"]


def test_sound_denominators_stay_separate() -> None:
    accounting = bc.sound_input_accounting(
        registry=SOUND_REGISTRY, sound_class_config=SOUND_CONFIG,
        registered_event_class_counts={"air_conditioning": 20},
        library_denominators={"library_inventory_clips": 1190,
                              "byte_identical_carry_over": 1168,
                              "truly_new_relative_paths": 22,
                              "pool_admissions_without_length_filter": 844},
        segment_rows=[
            {"selection_authorized": True, "crop_authorization": "owner_authorized"},
            {"selection_authorized": False},
        ],
    )
    assert accounting["denominators"]["library_inventory_clips"] == 1190
    assert accounting["denominators"]["truly_new_relative_paths"] == 22
    assert accounting["segment_candidates"]["cropped_candidate_rows"] == 2
    assert accounting["segment_candidates"]["authorized_rows"] == 1
    assert accounting["segment_candidates"]["rows_without_named_authorization"] == 1


def test_feedback_refuses_a_sound_document_of_the_wrong_kind() -> None:
    with pytest.raises(bc.BatchCoverageError, match="sound_inputs"):
        bc.build_v1_coverage_feedback(
            targets=_targets(), achieved=_achieved(), sound_inputs={"schema": "other"},
        )


def test_feedback_refuses_an_achieved_document_of_the_wrong_kind() -> None:
    with pytest.raises(bc.BatchCoverageError, match="achieved"):
        bc.build_v1_coverage_feedback(
            targets=_targets(), achieved={"schema": "other", "by_qa_id": {}},
        )


def test_written_feedback_keeps_the_outstanding_list_and_refuses_an_existing_output(
    tmp_path,
) -> None:
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(),
        achieved=_achieved(by_qa={
            "QA-10": {"valid_main_questions": 2, "distinct_worlds_with_main": 2}}),
    )
    paths = bc.write_v1_coverage_feedback(feedback, tmp_path / "out")
    written = json.loads(Path(paths["feedback"]).read_text(encoding="utf-8"))
    assert written["by_qa_id"]["QA-10"]["state"] == "short_of_target"
    outstanding = json.loads(Path(paths["outstanding"]).read_text(encoding="utf-8"))
    assert outstanding["count"] == len(bc.outstanding_production_requests(feedback))
    rows = Path(paths["csv"]).read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("scope,qa_id,branch,state")
    assert any(line.startswith("qa_id,QA-10,,short_of_target") for line in rows)
    with pytest.raises(FileExistsError):
        bc.write_v1_coverage_feedback(feedback, tmp_path / "out")


def test_feedback_never_claims_evaluation_or_answerability() -> None:
    feedback = bc.build_v1_coverage_feedback(targets=_targets(), achieved=_achieved())
    assert feedback["model_evaluation"] == "not_run"
    assert feedback["human_answerability"] == "not_run"
    assert "paper admission" in feedback["claim_boundary"]


def test_a_device_only_inapplicability_never_speaks_for_a_movable_source() -> None:
    """A report covering several source pairs keeps the least excusing reason.

    QA-15 is inapplicable to a device pair by definition and blocked on a
    missing route solver for a human pair. Letting the device reason represent
    the cell would excuse the gap that actually has to be built.
    """
    planning = {
        "qa_ids": {
            "QA-15": {"candidates": [
                {"branch": "nearer", "state": "not_applicable_by_definition",
                 "reason": "this pair has no source that can be a self-motion target"},
                {"branch": "nearer", "state": "interface_not_implemented",
                 "reason": "sample_routes: routes are sampled for a contiguous window"},
            ]},
        }
    }
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(), achieved=_achieved(), planning=planning,
    )
    row = feedback["by_qa_branch"]["QA-15:nearer"]
    assert row["state"] == "interface_not_implemented"
    assert "sample_routes" in row["reason"]


def test_the_order_of_the_candidates_does_not_change_the_reason() -> None:
    reversed_candidates = {
        "qa_ids": {
            "QA-15": {"candidates": [
                {"branch": "nearer", "state": "interface_not_implemented",
                 "reason": "sample_routes"},
                {"branch": "nearer", "state": "not_applicable_by_definition",
                 "reason": "device"},
            ]},
        }
    }
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(), achieved=_achieved(), planning=reversed_candidates,
    )
    assert feedback["by_qa_branch"]["QA-15:nearer"]["state"] == "interface_not_implemented"


def test_a_cell_every_pair_finds_inapplicable_stays_inapplicable() -> None:
    planning = {
        "qa_ids": {
            "QA-06": {"candidates": [
                {"branch": "moving", "state": "not_applicable_by_definition",
                 "reason": "no source of this pair can be a self-motion target"},
            ]},
        }
    }
    feedback = bc.build_v1_coverage_feedback(
        targets=_targets(), achieved=_achieved(), planning=planning,
    )
    assert feedback["by_qa_branch"]["QA-06:moving"][
        "state"] == "not_applicable_by_definition"

