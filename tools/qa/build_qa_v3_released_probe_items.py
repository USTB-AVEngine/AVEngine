#!/usr/bin/env python3
"""Build MCQ/Open shortcut-probe items from released run02-style media.

Only question text, published gold, and final released audio/video paths enter
the output.  No timeline, dry sound, RIR, engine state or hidden fact field is
exposed to the downstream modality probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(selection, facts_root, audio_root, media_root):
    records = []
    for chosen in sorted(selection["selected"], key=lambda row: row["point_id"]):
        point_id = str(chosen["point_id"])
        fact = _read(Path(facts_root) / point_id / "fact_record.json")
        wav = Path(audio_root) / point_id / "audio" / "binaural" / "mixture.wav"
        video = Path(media_root) / point_id / "video_only.mp4"
        if not wav.is_file() or not video.is_file():
            raise FileNotFoundError(f"{point_id}: released audio/video missing")
        common = {
            "question_id": point_id,
            "group_id": point_id,
            "profile_id": str(fact["profile_id"]),
            "audio": str(wav.resolve()),
            "video": str(video.resolve()),
        }
        records.append(dict(
            common, form="mcq", task_type="classification",
            question=str(fact["mcq"]["stem"]),
            options=list(fact["mcq"]["options_space"]),
            truth=fact["mcq"]["truth_option"]))
        scoring = fact["open"]["scoring"]
        if scoring == "circular_deg":
            task_type = "numeric_angle"
        elif scoring == "absolute_time":
            task_type = "numeric_time"
        else:
            task_type = "classification"
        records.append(dict(
            common, form="open", task_type=task_type,
            question=str(fact["open"]["stem"]), options=[],
            truth=fact["open"]["truth_value"],
            certification_policy=fact["open"].get("certification_policy")))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--facts-root", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    result = build(
        _read(args.selection_manifest), args.facts_root,
        args.audio_root, args.media_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "record_count": len(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
