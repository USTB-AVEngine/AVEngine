#!/usr/bin/env python3
"""Pull English speech with transcripts from the VCTK copy on this machine.

We need people talking, with the exact words known, because one of the
new question types asks what the speaker said. VCTK gives all of that:
a hundred and eleven speakers with their gender recorded, an exact
transcript for every utterance, and - the part that matters most for us -
recordings made in a hemi-anechoic chamber, so the material is dry and
our own room acoustics are the only reverberation in the result.

Two choices are deliberate and visible in the output. Speakers are split
into training and evaluation groups that share nobody, so a voice heard
while training cannot be recognised at test time for the wrong reason;
and each speaker's first twenty-three sentences are the same text as
every other speaker's, which the sidecar marks, giving a controlled set
where the words are identical and only the voice differs.

Audio is decoded from flac and resampled by ffmpeg; the clips land in the
library like any other material and go through the same QC afterwards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.assets.sound_harvest import (  # noqa: E402
    flac_duration_seconds,
    parse_vctk_speakers,
    plan_vctk_roster,
    plan_vctk_utterances,
    sidecar_for_speech,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vctk-root",
        type=Path,
        default=Path("/data/datasets/TextrolMix/VCTK-Corpus"),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path("/data/avengine_external/assets/sound_library_v1"),
    )
    parser.add_argument("--train-per-gender", type=int, default=8)
    parser.add_argument("--eval-per-gender", type=int, default=4)
    parser.add_argument("--per-speaker", type=int, default=25)
    parser.add_argument("--mic", default="mic1", choices=("mic1", "mic2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    info = args.vctk_root / "speaker-info.txt"
    audio_root = args.vctk_root / "wav48_silence_trimmed"
    text_root = args.vctk_root / "txt"
    for path in (info, audio_root, text_root):
        if not path.exists():
            raise SystemExit(f"VCTK 里找不到 {path}")

    roster = plan_vctk_roster(
        parse_vctk_speakers(info.read_text(encoding="utf-8", errors="replace")),
        train_per_gender=args.train_per_gender,
        eval_per_gender=args.eval_per_gender,
    )
    print("说话人名单(训练和考试互不重叠):")
    for split in ("train", "eval"):
        names = [
            f"{s.speaker_id}({s.gender},{s.accent})" for s in roster if s.split == split
        ]
        print(f"  {split:<6} {len(names)} 人: {'、'.join(names)}")

    written = skipped = 0
    for speaker in roster:
        speaker_audio = audio_root / speaker.speaker_id
        speaker_text = text_root / speaker.speaker_id
        if not speaker_audio.is_dir() or not speaker_text.is_dir():
            print(f"  跳过 {speaker.speaker_id}:语音或文本目录缺失")
            continue

        durations: dict[str, float] = {}
        for flac in speaker_audio.glob(f"*_{args.mic}.flac"):
            sentence_id = flac.name.split("_")[1]
            if not (speaker_text / f"{speaker.speaker_id}_{sentence_id}.txt").is_file():
                continue
            seconds = flac_duration_seconds(flac)
            if seconds is not None:
                durations[sentence_id] = seconds

        for pick in plan_vctk_utterances(
            speaker, durations, per_speaker=args.per_speaker
        ):
            name = f"{speaker.speaker_id}_{pick['sentence_id']}"
            target_dir = args.library_root / "speech_playback" / f"vctk_{name}"
            if (target_dir / "clip.wav").is_file():
                skipped += 1
                continue
            if args.dry_run:
                written += 1
                continue
            transcript = (
                text_root / speaker.speaker_id / f"{name}.txt"
            ).read_text(encoding="utf-8", errors="replace")
            target_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y",
                    "-i", str(speaker_audio / f"{name}_{args.mic}.flac"),
                    "-ac", "1", "-ar", "16000",
                    str(target_dir / "clip.wav"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                print(f"  {name} 解码失败:{completed.stderr.strip()[:80]}")
                continue
            (target_dir / "clip.json").write_text(
                json.dumps(
                    sidecar_for_speech(pick, transcript), ensure_ascii=False, indent=1
                )
                + "\n",
                encoding="utf-8",
            )
            written += 1

    print(
        f"\n{'（演习，未写盘）' if args.dry_run else ''}"
        f"新增 {written} 条人声{f'（{skipped} 条已存在，跳过）' if skipped else ''}\n"
        f"接下来跑质检和格式处理:\n"
        f"  python3 tools/assets/qc_sound_library.py\n"
        f"  python3 tools/assets/prepare_sound_library.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
