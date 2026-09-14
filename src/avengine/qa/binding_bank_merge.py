"""Put an assembled core-group export into the ordinary question bank.

A core group is not a twenty-sixth question type: its members are ordinary
samples that happen to come in fours, and an evaluation should be able to read
them from the same bank as everything else. This module copies an assembled
binding export into a bank run's own layout - content-addressed media, one
public row per sample, private answers and sources - and adds one group index
so an evaluator can also score by whole group.

The bank it reads from is never written to. A merge produces a new bank run
whose ordinary rows are the source's, so re-running it cannot damage a batch
another producer is still filling.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from avengine.qa.binding_groups import BindingGroupError, check_public_export_files

#: The files of a bank run that a model or an annotator actually receives.
BANK_PUBLIC_FILES = ("public/questions.jsonl",)
BANK_MEDIA_DIR = "media"

#: Where the group index lives. Private: it names members and interventions.
BANK_GROUP_INDEX = "private/binding_groups.json"

BANK_ROW_FILES = (
    ("public/questions.jsonl", "public"),
    ("private/answers.jsonl", "answers"),
    ("private/sources.jsonl", "sources"),
)

#: Private files this module writes itself. Every other file under the source
#: bank's ``private/`` (the research-only strata, for one) is carried unchanged:
#: it describes the ordinary rows, which the merge keeps as they were.
BANK_REBUILT_PRIVATE_FILES = frozenset({"answers.jsonl", "sources.jsonl",
                                        Path(BANK_GROUP_INDEX).name})

README_MERGE_HEADING = "## Binding groups"


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def content_address(path: Path) -> str:
    """The bank's own media name for a file: its kind and its content hash."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    suffix = Path(path).suffix.lower()
    kind = {".mp4": "video", ".webm": "video", ".wav": "audio", ".flac": "audio"}.get(suffix)
    if kind is None:
        raise BindingGroupError(f"the bank has no media kind for {suffix!r}")
    return f"{kind}_{digest.hexdigest()}{suffix}"


def _publish_media(source: Path, media_dir: Path) -> str:
    name = content_address(source)
    destination = media_dir / name
    if not destination.exists():
        media_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            destination.write_bytes(Path(source).read_bytes())
    return f"{BANK_MEDIA_DIR}/{name}"


def _carry_private_files(source_bank: Path, out: Path) -> list[str]:
    """Copy the source bank's private files that the merge does not rebuild."""
    carried = []
    private_dir = source_bank / "private"
    if not private_dir.is_dir():
        return carried
    for item in sorted(private_dir.iterdir()):
        if not item.is_file() or item.name in BANK_REBUILT_PRIVATE_FILES:
            continue
        (out / "private").mkdir(parents=True, exist_ok=True)
        (out / "private" / item.name).write_bytes(item.read_bytes())
        carried.append(f"private/{item.name}")
    return carried


def _describe_merge_in_readme(out: Path, *, total: int, member_total: int,
                              group_total: int, carried_private: Sequence[str]) -> None:
    """Make the run's README describe the merged bank, not the bank it came from.

    The README is copied from the source bank, so its question count would
    otherwise keep naming the ordinary rows alone. A run can be merged into more
    than once; the merge paragraph is rewritten, not appended, so one paragraph
    describes the whole bank however many merges produced it.
    """
    path = out / "README.md"
    text = (path.read_text(encoding="utf-8") if path.is_file()
            else f"# AVEngine question bank\n\nQuestions: {total}.\n")
    text = re.sub(r"(?m)^Questions: \d+\.", f"Questions: {total}.", text)
    head = text.split(f"\n{README_MERGE_HEADING}\n", 1)[0].rstrip("\n")
    carried_note = (
        f"The carried private files ({', '.join(carried_private)}) describe the "
        f"{total - member_total} ordinary questions only; group members have no rows there. "
        if carried_private else "")
    paragraph = (
        f"{README_MERGE_HEADING}\n\n"
        f"{total - member_total} ordinary questions were carried from the source bank and "
        f"{member_total} binding-group member questions were appended, in {group_total} groups. "
        f"Each member is an ordinary row in public/questions.jsonl; the private index "
        f"{BANK_GROUP_INDEX} maps members to their groups so an evaluator can also score by whole "
        f"group. {carried_note}Merge counts are in binding_merge.json.\n")
    path.write_text(f"{head}\n\n{paragraph}", encoding="utf-8")


def _next_question_number(rows: Sequence[Mapping[str, Any]]) -> int:
    highest = 0
    for row in rows:
        value = str(row.get("question_id") or "")
        tail = value.rsplit("_", 1)[-1]
        if tail.isdigit():
            highest = max(highest, int(tail))
    return highest + 1


def merge_binding_groups_into_bank(
    export_root: str | Path, bank_out: str | Path, *,
    bank_in: str | Path | None = None,
) -> dict[str, Any]:
    """Copy one assembled binding export into a bank run's layout.

    ``bank_in`` is read and never written. Its ordinary rows and media are
    carried into ``bank_out`` first, the group members are appended after them,
    and the public side of the result is scanned for anything naming a member or
    a group before the run is declared complete.
    """
    export = Path(export_root).expanduser().resolve()
    out = Path(bank_out).expanduser().resolve()
    if out.exists():
        raise BindingGroupError(f"refusing existing bank run: {out}")
    core = _load(export / "binding_groups.json")
    if core.get("schema") != "avengine_binding_groups_v1":
        raise BindingGroupError("the export is not an assembled binding group bundle")
    out.mkdir(parents=True)
    carried = {name: [] for _path, name in BANK_ROW_FILES}
    media_dir = out / BANK_MEDIA_DIR
    media_dir.mkdir()
    source_bank = Path(bank_in).expanduser().resolve() if bank_in is not None else None
    if source_bank is not None:
        for relative, name in BANK_ROW_FILES:
            carried[name] = _read_rows(source_bank / relative)
        existing = source_bank / BANK_MEDIA_DIR
        if existing.is_dir():
            for item in sorted(existing.iterdir()):
                if item.is_file():
                    _publish_media(item, media_dir)
        for extra in ("report.json", "README.md", "run_config.json", "progress.json",
                      "controller.json"):
            if (source_bank / extra).is_file():
                (out / extra).write_bytes((source_bank / extra).read_bytes())
    carried_private = _carry_private_files(source_bank, out) if source_bank is not None else []
    # An index already in the source bank is carried too, so merging several
    # exports one after another ends with one index over all of them instead of
    # the last one's.
    carried_index = []
    if source_bank is not None and (source_bank / BANK_GROUP_INDEX).is_file():
        carried_index = list(_load(source_bank / BANK_GROUP_INDEX).get("groups") or [])
    number = _next_question_number(carried["public"])
    index, added = list(carried_index), 0
    for group in core.get("groups", []):
        members = []
        for member in group["members"]:
            question = member["question"]
            question_id = f"question_{number:06d}"
            number += 1
            media = {
                "video": _publish_media(export / member["media"]["video_path"], media_dir),
                "audio": _publish_media(export / member["media"]["audio_path"], media_dir),
            }
            carried["public"].append({
                "question_id": question_id,
                "qa_id": question["qa_id"],
                "forms": deepcopy(question["model_input"]),
                "required_modalities": deepcopy(question.get("required_modalities")),
                "media": media,
            })
            carried["answers"].append({
                "question_id": question_id,
                "source_question_id": question.get("question_id"),
                "qa_id": question["qa_id"],
                "truth": deepcopy(question["truth"]),
            })
            carried["sources"].append({
                "question_id": question_id,
                "episode_id": member.get("native_episode_id"),
                "world_id": group.get("world_id"),
                "room_family": group.get("room_family"),
                "room_id": group.get("room_id"),
                "facts_path": member.get("facts_path"),
                "binding_group_id": group["group_id"],
                "binding_member_id": member["member_id"],
                "binding_sample_id": member["sample_id"],
            })
            members.append({
                "member_id": member["member_id"],
                "sample_id": member["sample_id"],
                "question_id": question_id,
                "interventions": deepcopy(member.get("interventions") or {}),
                **({"planned_answer": deepcopy(member["planned_answer"])}
                   if isinstance(member.get("planned_answer"), Mapping) else {}),
            })
            added += 1
        by_sample = {row["sample_id"]: row["question_id"] for row in members}
        index.append({
            "group_id": group["group_id"],
            "task_family": group.get("task_family"),
            "room_family": group.get("room_family"),
            "room_id": group.get("room_id"),
            "world_id": group.get("world_id"),
            "split": group.get("split"),
            "query": deepcopy(group.get("query")),
            "question_recipe": deepcopy(group.get("question_recipe")),
            "members": members,
            "comparisons": [
                {**deepcopy(row),
                 "question_ids": [by_sample[value] for value in row["members"]]}
                for row in group.get("validation", {}).get("comparisons", [])
                if all(value in by_sample for value in row.get("members", []))
            ],
            "validation": deepcopy(group.get("validation")),
        })
    for relative, name in BANK_ROW_FILES:
        _write_rows(out / relative, carried[name])
    member_total = sum(len(row["members"]) for row in index)
    _describe_merge_in_readme(out, total=len(carried["public"]), member_total=member_total,
                              group_total=len(index), carried_private=carried_private)
    _write(out / BANK_GROUP_INDEX, {
        "schema": "avengine_bank_binding_group_index_v1",
        "source_export": str(export),
        "source_bank": str(source_bank) if source_bank is not None else None,
        "group_count": len(index),
        "carried_group_count": len(carried_index),
        "member_question_count": sum(len(row["members"]) for row in index),
        "member_questions_added_here": added,
        "scoring": ("each member is an ordinary sample in public/questions.jsonl; a group "
                    "is scored by requiring every one of its members to be right"),
        "groups": index,
    })
    # The scan covers every group in the run, not only the ones this call added:
    # a carried group's identity must stay out of the public side as well.
    check = check_public_export_files(
        out, [row for row in core.get("groups", [])] + carried_index,
        files=BANK_PUBLIC_FILES, media_dirs=(BANK_MEDIA_DIR,))
    summary = {
        "schema": "avengine_bank_binding_merge_v1",
        "bank_run": str(out),
        "source_export": str(export),
        "source_bank": str(source_bank) if source_bank is not None else None,
        "carried_question_count": len(carried["public"]) - added,
        "binding_question_count": added,
        "binding_group_count": len(index),
        "binding_groups_added_here": len(index) - len(carried_index),
        "total_question_count": len(carried["public"]),
        "media_file_count": len(list(media_dir.iterdir())),
        "group_index": BANK_GROUP_INDEX,
        "carried_private_files": carried_private,
        "public_export_file_check": check,
        "claim_boundary": ("the group members are ordinary bank samples plus one private "
                           "group index; no model or human evaluation is claimed"),
    }
    _write(out / "binding_merge.json", summary)
    return summary
