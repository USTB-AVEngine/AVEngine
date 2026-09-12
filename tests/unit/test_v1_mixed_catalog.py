import json
from pathlib import Path
import pytest
from avengine.qa.binding_catalog import merge_binding_catalogs
from avengine.qa.binding_groups import BindingGroupError
from avengine.qa.binding_delivery import _join_inputs, BindingDeliveryError


def _catalog(root: Path, kind: str, seed: str):
    root.mkdir()
    (root / "facts.json").write_text(json.dumps({"episode_id": kind + "_episode"}))
    (root / "video.mp4").write_bytes(("video-" + kind).encode())
    (root / "audio.wav").write_bytes(("audio-" + kind).encode())
    private = {"items": [{"qa_id": "QA-02", "question_id": "private_question", "status": "pass",
                           "forms": {"open": {"question_en": "Which?", "truth": "blue"}}}], "angle_followups": []}
    (root / "questions.json").write_text(json.dumps(private))
    record = {"sample_id": "sample_000001", "record_kind": kind, "world_id": kind + "_world",
              "episode_id": kind + "_episode", "group_id": "g" if kind == "core_group_member" else None,
              "member_id": "m" if kind == "core_group_member" else None,
              "facts_path": "facts.json", "questions_path": "questions.json",
              "public_question_ids": ["question_000001"], "core_question_count": int(kind == "core_group_member")}
    (root / "catalog_index.json").write_text(json.dumps({"status": "research_candidate", "records": [record]}))
    (root / "request_config.json").write_text(json.dumps({"seed": seed, "qa_sampling": {"time_display_precision": 0}, "items_per_type": 1}))
    (root / "model_inputs.json").write_text(json.dumps({"samples": [{"sample_id": "sample_000001",
        "media": {"video_path": "video.mp4", "audio_path": "audio.wav"},
        "items": [{"question_id": "question_000001", "qa_id": "QA-02", "question": "Which?"}]}]}))
    return root / "catalog_index.json", private


def test_mixed_catalog_reassigns_public_ids_but_keeps_original_questions_and_seeds(tmp_path):
    core, core_questions = _catalog(tmp_path / "core", "core_group_member", "core-seed")
    episode, episode_questions = _catalog(tmp_path / "episode", "episode", "episode-seed")
    result = merge_binding_catalogs([core, episode], output=tmp_path / "merged", seed="public-identifiers")
    assert result["av_sample_count"] == 2
    assert result["world_count"] == 2
    assert result["group_count"] == 1
    assert len({r["sample_id"] for r in result["records"]}) == 2
    assert len({q for r in result["records"] for q in r["public_question_ids"]}) == 2
    for row in result["records"]:
        expected = core_questions if row["record_kind"] == "core_group_member" else episode_questions
        assert json.loads((tmp_path / "merged" / row["questions_path"]).read_text()) == expected
        assert row["generation_config"]["seed"] == ("core-seed" if row["record_kind"] == "core_group_member" else "episode-seed")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"schema": "avengine_binding_groups_v1", "status": "research_candidate",
        "groups": [{"group_id": "g", "members": [{"member_id": "m", "sample_id": "core_m", "facts_path": str(tmp_path / "core/facts.json")}]}]}))
    _, _, joined = _join_inputs(bundle, tmp_path / "merged/catalog_index.json")
    assert len(joined) == 2
    assert sum(row["core_member"] is None for row in joined) == 1
    # The ordinary row does not relax missing core-member protection.
    value = json.loads(bundle.read_text()); value["groups"][0]["members"].append({"member_id": "missing"}); bundle.write_text(json.dumps(value))
    with pytest.raises(BindingDeliveryError, match="join keys differ"):
        _join_inputs(bundle, tmp_path / "merged/catalog_index.json")


def test_duplicate_source_catalog_is_refused_before_output_creation(tmp_path):
    source, _ = _catalog(tmp_path / "one", "episode", "same")
    with pytest.raises(BindingGroupError, match="duplicate source"):
        merge_binding_catalogs([source, source], output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_multiple_ordinary_records_do_not_collapse_to_a_none_core_join_key(tmp_path):
    core, _ = _catalog(tmp_path / "core", "core_group_member", "core-seed")
    first, _ = _catalog(tmp_path / "first", "episode", "first-seed")
    second, _ = _catalog(tmp_path / "second", "episode", "second-seed")
    data = json.loads(second.read_text())
    data["records"][0]["episode_id"] = "second_episode"
    data["records"][0]["world_id"] = "second_world"
    second.write_text(json.dumps(data))
    result = merge_binding_catalogs((value for value in [core, first, second]), output=tmp_path / "mixed")
    assert result["av_sample_count"] == result["world_count"] == 3
    config = json.loads((tmp_path / "mixed/request_config.json").read_text())
    assert len(config["source_catalogs"]) == 3
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"schema": "avengine_binding_groups_v1", "status": "research_candidate",
        "groups": [{"group_id": "g", "members": [{"member_id": "m", "sample_id": "core_m", "facts_path": str(tmp_path / "core/facts.json")}]}]}))
    _, _, joined = _join_inputs(bundle, tmp_path / "mixed/catalog_index.json")
    assert len(joined) == 3
    assert len([r for r in joined if r["core_member"] is None]) == 2
