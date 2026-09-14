"""The bank merge carries the whole source bank, and its README describes the result."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.qa.binding_bank_merge import merge_binding_groups_into_bank
from avengine.qa.binding_groups import BindingGroupError


def _rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _source_bank(root: Path) -> Path:
    bank = root / "bank_in"
    _rows(bank / "public/questions.jsonl", [
        {"question_id": "question_000001", "qa_id": "QA-04", "forms": {"open": {"question_en": "Where?"}},
         "required_modalities": ["audio"], "media": {"video": "media/video_aa.mp4", "audio": "media/audio_aa.wav"}},
        {"question_id": "question_000002", "qa_id": "QA-08", "forms": {"open": {"question_en": "Seen?"}},
         "required_modalities": ["video"], "media": {"video": "media/video_aa.mp4", "audio": "media/audio_aa.wav"}},
    ])
    _rows(bank / "private/answers.jsonl", [{"question_id": "question_000001", "truth": {"value": "left"}},
                                           {"question_id": "question_000002", "truth": {"value": "yes"}}])
    _rows(bank / "private/sources.jsonl", [{"question_id": "question_000001", "episode_id": "ordinary_ep_1"},
                                           {"question_id": "question_000002", "episode_id": "ordinary_ep_1"}])
    _rows(bank / "private/strata.jsonl", [{"question_id": "question_000001", "visibility_at_query": "out_of_view"},
                                          {"question_id": "question_000002", "visibility_at_query": "visible_clear"}])
    (bank / "media").mkdir()
    (bank / "media/video_aa.mp4").write_bytes(b"ordinary video")
    (bank / "media/audio_aa.wav").write_bytes(b"ordinary audio")
    (bank / "README.md").write_text(
        "# AVEngine question bank\n\nQuestions: 2. Target ceiling: 20000.\n\n"
        "private/strata.jsonl carries two research-only fields per question.\n", encoding="utf-8")
    (bank / "report.json").write_text(json.dumps({"exported_question_count": 2}), encoding="utf-8")
    return bank


def _export(root: Path, tag: str) -> Path:
    export = root / f"export_{tag}"
    (export / "media").mkdir(parents=True)
    members = []
    for number, member_id in enumerate(("v0_a0", "v1_a1"), start=1):
        (export / f"media/video_{tag}_{number}.mp4").write_bytes(f"{tag} video {number}".encode())
        (export / f"media/audio_{tag}_{number}.wav").write_bytes(f"{tag} audio {number}".encode())
        members.append({
            "member_id": member_id, "sample_id": f"sample_{tag}_{number:06d}",
            "native_episode_id": f"private_episode_{tag}_{number}", "facts_path": f"/nowhere/{tag}/{number}/facts.json",
            "media": {"video_path": f"media/video_{tag}_{number}.mp4", "audio_path": f"media/audio_{tag}_{number}.wav"},
            "question": {"qa_id": "QA-20",
                         "model_input": {"open": {"question_en": "What colour is the shirt of whoever spoke first?"}},
                         "required_modalities": ["audio", "video"],
                         "truth": {"value": "blue" if number == 1 else "green"}},
            "interventions": {"applied": {}},
            "planned_answer": {"value": "blue" if number == 1 else "green"},
        })
    group = {
        "group_id": f"bind_group_{tag}_0001", "task_family": "visible_binding", "room_family": "test_family",
        "room_id": "test_room_0001", "world_id": f"bind_world_{tag}_0001", "split": "research",
        "query": {"qa_id": "QA-20"}, "question_recipe": {"qa_id": "QA-20"}, "members": members,
        "validation": {"comparisons": [{"members": [members[0]["sample_id"], members[1]["sample_id"]],
                                        "kind": "invariance", "media_check": "pass"}]},
    }
    (export / "binding_groups.json").write_text(
        json.dumps({"schema": "avengine_binding_groups_v1", "groups": [group]}), encoding="utf-8")
    return export


def test_private_files_the_merge_does_not_rebuild_are_carried_unchanged(tmp_path):
    bank = _source_bank(tmp_path)
    before = (bank / "private/strata.jsonl").read_bytes()
    summary = merge_binding_groups_into_bank(_export(tmp_path, "one"), tmp_path / "bank_out", bank_in=bank)
    assert summary["carried_private_files"] == ["private/strata.jsonl"]
    assert (tmp_path / "bank_out/private/strata.jsonl").read_bytes() == before
    # the source bank is read only
    assert (bank / "private/strata.jsonl").read_bytes() == before
    assert (bank / "README.md").read_text(encoding="utf-8").startswith("# AVEngine question bank\n\nQuestions: 2.")
    public = [json.loads(line) for line in (tmp_path / "bank_out/public/questions.jsonl").read_text().splitlines()]
    assert [row["question_id"] for row in public] == [f"question_{n:06d}" for n in range(1, 5)]
    assert summary["total_question_count"] == 4 and summary["binding_question_count"] == 2


def test_readme_counts_the_merged_bank_and_describes_the_merge_once(tmp_path):
    bank = _source_bank(tmp_path)
    merge_binding_groups_into_bank(_export(tmp_path, "one"), tmp_path / "step1", bank_in=bank)
    first = (tmp_path / "step1/README.md").read_text(encoding="utf-8")
    assert "Questions: 4." in first and "Questions: 2." not in first
    assert first.count("## Binding groups") == 1
    assert "2 binding-group member questions were appended, in 1 groups" in first
    assert "private/strata.jsonl" in first and "ordinary questions only" in first
    merge_binding_groups_into_bank(_export(tmp_path, "two"), tmp_path / "step2", bank_in=tmp_path / "step1")
    second = (tmp_path / "step2/README.md").read_text(encoding="utf-8")
    assert "Questions: 6." in second
    assert second.count("## Binding groups") == 1
    assert "4 binding-group member questions were appended, in 2 groups" in second
    index = json.loads((tmp_path / "step2/private/binding_groups.json").read_text(encoding="utf-8"))
    assert index["group_count"] == 2 and index["member_question_count"] == 4
    assert (tmp_path / "step2/private/strata.jsonl").read_bytes() == (bank / "private/strata.jsonl").read_bytes()


def test_a_bank_without_extra_private_files_or_readme_still_merges(tmp_path):
    bank = _source_bank(tmp_path)
    (bank / "private/strata.jsonl").unlink()
    (bank / "README.md").unlink()
    summary = merge_binding_groups_into_bank(_export(tmp_path, "one"), tmp_path / "bank_out", bank_in=bank)
    assert summary["carried_private_files"] == []
    readme = (tmp_path / "bank_out/README.md").read_text(encoding="utf-8")
    assert readme.startswith("# AVEngine question bank") and "Questions: 4." in readme
    assert "ordinary questions only" not in readme


def test_an_existing_output_run_is_refused(tmp_path):
    bank = _source_bank(tmp_path)
    (tmp_path / "bank_out").mkdir()
    with pytest.raises(BindingGroupError):
        merge_binding_groups_into_bank(_export(tmp_path, "one"), tmp_path / "bank_out", bank_in=bank)
