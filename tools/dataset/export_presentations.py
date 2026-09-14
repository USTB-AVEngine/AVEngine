#!/usr/bin/env python3
"""给已导出的题库补上五种呈现的输入清单，以及自己下混的单声道

每道题现在只有一份输入（视频 + 双耳 + 题面）。这个工具在不动任何现有文件的前提下，往题库里
新增两份公开清单和若干单声道 wav：

    public/model_inputs_by_presentation.jsonl   每道题 5 行：av / a_only / v_only / t_only / av_mono
    public/presentation_media.jsonl             每个新媒体文件一行，带来源和下混收据
    media/audio_mono_<sha256>.wav               下混出来的单声道

用法：

    python tools/dataset/export_presentations.py <bank_run> [--mono-method binaural_mean|foa_w]

跑之前和跑之后各把 public 下已有的文件哈希一遍，有一个字节不一样就整个失败退出。CPU 跑。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.presentations import (  # noqa: E402
    MONO_METHODS,
    MONO_METHOD_NOTES,
    PRESENTATIONS,
    PRESENTATION_NOTES,
    PresentationError,
    file_sha256,
    presentation_rows,
    text_is_identical,
    write_mono_wav,
)

NEW_FILES = ("model_inputs_by_presentation.jsonl", "presentation_media.jsonl")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def public_digest(bank: Path, *, skip=()):
    root = bank / "public"
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in skip
    }


def foa_mixture_for(source_row) -> Path:
    """这道题对应 episode 的 FOA 混音。facts 自己记着 episode 在哪。"""

    facts_path = Path(source_row["facts_path"])
    candidate = facts_path.parent / "audio" / "audio" / "foa" / "mixture.wav"
    if candidate.is_file():
        return candidate
    raise PresentationError(f"找不到这道题的 FOA 混音：{candidate}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank_run", type=Path)
    parser.add_argument("--mono-method", choices=MONO_METHODS, default="binaural_mean")
    parser.add_argument("--presentations", nargs="*", default=list(PRESENTATIONS))
    args = parser.parse_args()

    bank = args.bank_run.resolve()
    questions_path = bank / "public" / "questions.jsonl"
    if not questions_path.is_file():
        raise SystemExit(f"{questions_path} 不在，这不像一个导出好的题库。")
    for name in NEW_FILES:
        if (bank / "public" / name).exists():
            raise SystemExit(
                f"{bank / 'public' / name} 已经在了。这个工具只新增不覆盖；"
                "要重做就先把它挪走。"
            )

    before = public_digest(bank)
    questions = read_jsonl(questions_path)
    sources = {}
    sources_path = bank / "private" / "sources.jsonl"
    if sources_path.is_file():
        sources = {row["question_id"]: row for row in read_jsonl(sources_path)}

    # 一个源音频只下混一次，按源文件的哈希去重。
    mono_by_source: dict[str, dict] = {}
    receipts: list[dict] = []
    rows: list[dict] = []
    for question in questions:
        audio_relative = (question.get("media") or {}).get("audio")
        if not audio_relative:
            raise SystemExit(f"{question['question_id']} 没有音频，五种呈现拼不出来。")
        if args.mono_method == "binaural_mean":
            origin = bank / audio_relative
        else:
            source_row = sources.get(question["question_id"])
            if source_row is None:
                raise SystemExit(
                    f"{question['question_id']} 在 private/sources.jsonl 里查不到，"
                    "foa_w 需要它才能找到 FOA 文件。"
                )
            origin = foa_mixture_for(source_row)
        key = file_sha256(origin)
        if key not in mono_by_source:
            destination = bank / "media" / f"audio_mono_{args.mono_method}_{key}.wav"
            receipt = write_mono_wav(origin, destination, method=args.mono_method,
                                     source_sha256=key)
            receipt["path"] = str(destination.relative_to(bank))
            receipt["derived_from"] = (str(Path(origin).relative_to(bank))
                                       if str(origin).startswith(str(bank)) else str(origin))
            receipt["kind"] = "audio"
            mono_by_source[key] = receipt
            receipts.append(receipt)
        rows.extend(presentation_rows(
            question, mono_audio=mono_by_source[key]["path"],
            presentations=args.presentations,
        ))

    write_jsonl(bank / "public" / "model_inputs_by_presentation.jsonl", rows)
    write_jsonl(bank / "public" / "presentation_media.jsonl", receipts)

    after = public_digest(bank, skip=NEW_FILES)
    if after != before:
        changed = sorted({k for k in set(before) | set(after) if before.get(k) != after.get(k)})
        raise SystemExit(f"公开产物被改动了，这不允许：{changed}")

    identity = text_is_identical(rows)
    if not identity["identical"]:
        raise SystemExit(f"同一道题的几份输入题面不一致：{identity['questions_with_differing_text'][:5]}")

    counts = {}
    for row in rows:
        counts[row["presentation"]] = counts.get(row["presentation"], 0) + 1
    report = {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "bank_run": str(bank),
        "questions": len(questions),
        "rows": len(rows),
        "rows_by_presentation": counts,
        "presentations": list(args.presentations),
        "presentation_notes": {k: PRESENTATION_NOTES[k] for k in args.presentations},
        "mono_method": args.mono_method,
        "mono_method_note": MONO_METHOD_NOTES[args.mono_method],
        "mono_files": len(receipts),
        "mono_bytes": sum(r["bytes"] for r in receipts),
        "mono_clipped_samples": sum(r["clipped_samples"] for r in receipts),
        "text_identity": identity,
        "public_files_unchanged": True,
        "public_files_added": list(NEW_FILES),
        "claim_boundary": (
            "呈现变体只换输入的模态组合，不换题、不换答案、不换金标；"
            "现有公开文件逐字节未变，新增文件不参与任何既有验收。"
        ),
    }
    path = bank / "presentations_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "questions", "rows", "rows_by_presentation", "mono_files", "mono_bytes",
        "mono_method", "public_files_unchanged")}, ensure_ascii=False, indent=1))
    print("报告写到了", path)


if __name__ == "__main__":
    main()
