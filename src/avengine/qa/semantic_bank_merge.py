"""Append QA-26 to QA-28 from semantic audio variants to an ordinary question bank.

A semantic item comes from an audio variant of a visual world that already has
ordinary questions. Two identities matter and they differ:

* its episode is the variant, so a reader of the bank gets the variant's own
  binaural and first-order ambisonic mixtures, never the original speech;
* its world is the source world, so a splitter keeps the variant in the split
  that holds the same pictures.

Paired variants (binding groups rendered in both appearances and both speaker
orders) also form groups: one question stem across its four variants, with
the pairs whose answers must differ or must stay the same. Each member is an
ordinary row; the group index is private.

The bank that is read is never written.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Sequence

from avengine.qa.binding_bank_merge import (
    BANK_GROUP_INDEX, BANK_MEDIA_DIR, BANK_PUBLIC_FILES, BANK_ROW_FILES,
    _carry_private_files, _load, _next_question_number, _publish_media, _read_rows,
    _write, _write_rows,
)
from avengine.qa.binding_groups import BindingGroupError, check_public_export_files
from avengine.qa.choice_support import apply_choice_support
from avengine.qa.unified_catalog import EXTENSION_QA_IDS


def _binaural(delivery: Path) -> Path:
    path = delivery / "audio/audio/binaural/mixture.wav"
    if not path.is_file():
        raise BindingGroupError(f"semantic variant has no binaural mixture: {path}")
    if not (delivery / "audio/audio/foa/mixture.wav").is_file():
        raise BindingGroupError(f"semantic variant has no ambisonic mixture beside {path}")
    return path


def _silent_video(delivery: Path) -> Path:
    video = Path(json.loads((delivery / "input_refs.json").read_text())["visual_video"])
    if not video.is_file():
        raise BindingGroupError(f"semantic variant's capture is missing: {video}")
    return video


def _source_world(source: Path) -> dict[str, Any]:
    request = json.loads((source / "request.json").read_text())
    return {"episode_id": request.get("episode_id"), "world_id": request.get("world_id") or request.get("episode_id"),
            "room_id": request.get("room_id")}


def _room_family(room_id: str | None) -> str | None:
    text = str(room_id or "")
    for key, family in (("mp3d", "mp3d"), ("hm3d", "hm3d"), ("kujiale", "kujiale"), ("apartment", "apartment")):
        if key in text:
            return family
    return None


def _items(member_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((member_dir / "semantic_questions.json").read_text())
    return [apply_choice_support(item) for item in payload["items"] if item["qa_id"] in EXTENSION_QA_IDS]


def merge_semantic_into_bank(bank_in: str | Path, bank_out: str | Path, *,
                             single_runs: Sequence[str | Path] = (),
                             paired_runs: Sequence[str | Path] = ()) -> dict[str, Any]:
    source_bank = Path(bank_in).expanduser().resolve()
    out = Path(bank_out).expanduser().resolve()
    if out.exists():
        raise BindingGroupError(f"refusing existing bank run: {out}")
    out.mkdir(parents=True)
    rows = {name: _read_rows(source_bank / relative) for relative, name in BANK_ROW_FILES}
    media_dir = out / BANK_MEDIA_DIR
    media_dir.mkdir()
    if (source_bank / BANK_MEDIA_DIR).is_dir():
        for item in sorted((source_bank / BANK_MEDIA_DIR).iterdir()):
            if item.is_file():
                _publish_media(item, media_dir)
    for extra in ("report.json", "README.md", "run_config.json", "progress.json", "controller.json"):
        if (source_bank / extra).is_file():
            (out / extra).write_bytes((source_bank / extra).read_bytes())
    carried_private = _carry_private_files(source_bank, out)
    index = list(_load(source_bank / BANK_GROUP_INDEX).get("groups") or []) \
        if (source_bank / BANK_GROUP_INDEX).is_file() else []
    carried_groups = len(index)
    carried = len(rows["public"])
    number = _next_question_number(rows["public"])
    counts: dict[str, int] = defaultdict(int)
    skipped: list[dict[str, Any]] = []

    def append(item, delivery, *, episode_id, world, room_id, group=None):
        nonlocal number
        question_id = f"question_{number:06d}"
        number += 1
        video, audio = _silent_video(delivery), _binaural(delivery)
        rows["public"].append({"question_id": question_id, "qa_id": item["qa_id"],
                               "forms": deepcopy(item["model_input"]),
                               "required_modalities": deepcopy(item.get("required_modalities")),
                               "media": {"video": _publish_media(video, media_dir),
                                         "audio": _publish_media(audio, media_dir)}})
        rows["answers"].append({"question_id": question_id, "source_question_id": item.get("question_id"),
                                "qa_id": item["qa_id"], "truth": deepcopy(item["truth"]),
                                "forms": deepcopy(item["forms"]), "evidence": deepcopy(item.get("evidence") or {})})
        source = {"question_id": question_id, "episode_id": episode_id, "world_id": world,
                  "room_family": _room_family(room_id), "room_id": room_id,
                  "facts_path": str(delivery / "facts.json"),
                  "questions_path": str(delivery.parent / "semantic_questions.json"),
                  "video_path": str(video), "audio_path": str(audio),
                  "task_family": "semantic_qa", "source_tag": "semantic_audio_variant"}
        if group is not None:
            source.update(group)
        rows["sources"].append(source)
        counts[item["qa_id"]] += 1
        return question_id

    for run in single_runs:
        for result in _load(Path(run) / "progress.json")["results"]:
            if "error" in result:
                skipped.append({"member": result.get("member_id"), "reason": result["error"][:300]})
                continue
            member = Path(result["facts"]).parent.parent
            world = _source_world(Path(result["source"]))
            for item in _items(member):
                append(item, member / "delivery", episode_id=f"{world['episode_id']}__{member.name}",
                       world=world["world_id"], room_id=world["room_id"])

    for run in paired_runs:
        for result in _load(Path(run) / "progress.json")["results"]:
            if "error" in result:
                skipped.append({"group": result.get("group"), "scenario": result.get("scenario_id"),
                                "reason": result["error"][:300]})
                continue
            if not result["all_checks_pass"]:
                skipped.append({"group": result["group"], "scenario": result["scenario_id"],
                                "reason": "a paired answer did not change as its modality required"})
                continue
            first = Path(result["members"][0]["facts"]).parent.parent
            world_root = Path(result["members"][0]["source"])
            world = _source_world(world_root)
            group_world = f"semantic_pair__{result['group']}"
            stems: dict[str, list] = defaultdict(list)
            for m in result["members"]:
                member = Path(m["facts"]).parent.parent
                for item in _items(member):
                    stems[item["qa_id"] + "|" + item["forms"]["open"]["question_zh"]].append((member, item))
            for stem_index, (stem, entries) in enumerate(sorted(stems.items())):
                if len(entries) != 4:
                    continue  # a stem one variant could not ask cannot be compared
                qa_id = stem.split("|")[0]
                group_id = f"semantic__{result['group']}__{result['scenario_id']}__{stem_index:02d}"
                members, by_variant = [], {}
                for member, item in entries:
                    sample_id = f"{group_id}__{member.name}"
                    qid = append(item, member / "delivery",
                                 episode_id=f"{world['episode_id']}__{first.parent.name}__{member.name}",
                                 world=group_world, room_id=world["room_id"],
                                 group={"binding_group_id": group_id, "binding_member_id": member.name,
                                        "binding_sample_id": sample_id})
                    by_variant[member.name] = qid
                    members.append({"member_id": member.name, "sample_id": sample_id, "question_id": qid,
                                    "interventions": {"appearance": member.name[:2], "speaker_order": member.name[3:]},
                                    "planned_answer": {"value": item["truth"]["value"]}})
                appearance_bound = qa_id in ("QA-26", "QA-27")
                comparisons = []
                for held, pairs in (("audio", (("v0_a0", "v1_a0"), ("v0_a1", "v1_a1"))),
                                    ("video", (("v0_a0", "v0_a1"), ("v1_a0", "v1_a1")))):
                    for left, right in pairs:
                        comparisons.append({"held_identical": held,
                                            "expected": "different" if (held == "video" or appearance_bound) else "same",
                                            "question_ids": [by_variant[left], by_variant[right]]})
                index.append({"group_id": group_id, "task_family": "semantic_binding",
                              "qa_id": qa_id, "room_family": _room_family(world["room_id"]),
                              "room_id": world["room_id"], "world_id": group_world,
                              "scenario_id": result["scenario_id"], "members": members,
                              "comparisons": comparisons,
                              "validation": {"paired_checks": result["checks"], "all_checks_pass": True}})

    for relative, name in BANK_ROW_FILES:
        _write_rows(out / relative, rows[name])
    _write(out / BANK_GROUP_INDEX, {
        "schema": "avengine_bank_binding_group_index_v1", "source_bank": str(source_bank),
        "group_count": len(index), "carried_group_count": carried_groups,
        "member_question_count": sum(len(g["members"]) for g in index),
        "scoring": ("each member is an ordinary sample in public/questions.jsonl; a group "
                    "is scored by requiring every one of its members to be right"),
        "groups": index})
    check = check_public_export_files(out, [{"group_id": g["group_id"], "members": g["members"]}
                                            for g in index[carried_groups:]],
                                      files=BANK_PUBLIC_FILES, media_dirs=(BANK_MEDIA_DIR,))
    from avengine.qa.prior_audit import write_prior_receipt
    prior = write_prior_receipt(out)
    summary = {"schema": "avengine_bank_semantic_merge_v1", "bank_run": str(out), "source_bank": str(source_bank),
               "carried_question_count": carried, "semantic_question_count": len(rows["public"]) - carried,
               "semantic_counts_by_qa": dict(sorted(counts.items())),
               "semantic_groups_added": len(index) - carried_groups, "total_question_count": len(rows["public"]),
               "single_runs": [str(p) for p in single_runs], "paired_runs": [str(p) for p in paired_runs],
               "skipped": skipped, "carried_private_files": carried_private,
               "public_export_file_check": check,
               "answer_prior_audit": {"path": "private/answer_priors.json", "status": prior["status"]},
               "claim_boundary": "semantic items are ordinary samples from audio variants of retained visual worlds; "
                                 "no model or human evaluation is claimed"}
    _write(out / "semantic_merge.json", summary)
    return summary
