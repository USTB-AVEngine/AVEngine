#!/usr/bin/env python3
"""Build MCQ/Open shortcut-probe items from released run02-style media.

Only question text, published gold, and final released audio/video paths enter
the output.  No timeline, dry sound, RIR, engine state or hidden fact field is
exposed to the downstream modality probes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import sys
from pathlib import Path

from qa_v3_request import normalize_answer_forms


_PUBLIC_OPEN_FIELDS = ("truth_interval_deg", "convention", "certification_policy")
_DEFAULT_ANSWER_FORMS = ("mcq", "open")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _answer_forms(selection, answer_forms, params):
    """Resolve the public request while retaining the legacy two-form default."""
    if answer_forms is not None:
        return normalize_answer_forms(answer_forms)

    if isinstance(selection, Mapping):
        candidates = [
            selection.get("answer_forms"),
            selection.get("ANSWER_FORMS_DEFAULT"),
        ]
        request = selection.get("question_request")
        if isinstance(request, Mapping):
            candidates.extend([
                request.get("answer_forms"),
                request.get("ANSWER_FORMS_DEFAULT"),
            ])
            per_room = request.get("per_room")
            if isinstance(per_room, Mapping):
                candidates.extend([
                    per_room.get("answer_forms"),
                    per_room.get("ANSWER_FORMS_DEFAULT"),
                ])
        for value in candidates:
            if value is not None:
                return normalize_answer_forms(value)

    if params is not None:
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        if "ANSWER_FORMS_DEFAULT" in params:
            return normalize_answer_forms(params["ANSWER_FORMS_DEFAULT"])

    return list(_DEFAULT_ANSWER_FORMS)


def _open_public_fields(open_fact, mcq_fact):
    fields = {}
    for key in _PUBLIC_OPEN_FIELDS:
        source = open_fact
        if key == "convention" and key not in source:
            source = mcq_fact
        if key in source:
            fields[key] = source[key]
    return fields


def build(selection, facts_root, audio_root, media_root, *,
          answer_forms=None, params=None):
    forms = _answer_forms(selection, answer_forms, params)
    records = []
    for chosen in sorted(selection["selected"], key=lambda row: row["point_id"]):
        point_id = str(chosen["point_id"])
        fact = _read(Path(facts_root) / point_id / "fact_record.json")
        wav = Path(audio_root) / point_id / "audio" / "binaural" / "mixture.wav"
        video = Path(media_root) / point_id / "video_only.mp4"
        if not wav.is_file() or not video.is_file():
            raise FileNotFoundError(f"{point_id}: released audio/video missing")

        scene_id = fact.get("scene_id")
        point_key = f"{scene_id}__{point_id}" if scene_id else point_id
        episode = str(fact.get("episode_id", point_id))
        group_id = f"{scene_id}__{episode}" if scene_id else episode
        common = {
            "group_id": group_id,
            "profile_id": str(fact["profile_id"]),
            "audio": str(wav.resolve()),
            "video": str(video.resolve()),
        }
        mcq = fact["mcq"]
        if "mcq" in forms:
            records.append(dict(
                common, question_id=f"{point_key}__mcq", form="mcq",
                task_type="classification", question=str(mcq["stem"]),
                options=list(mcq["options_space"]),
                truth=mcq["truth_option"]))

        open_fact = fact["open"]
        scoring = open_fact["scoring"]
        if scoring in ("circular_deg", "circular_deg_interval"):
            task_type = "numeric_angle"
        elif scoring == "absolute_time":
            task_type = "numeric_time"
        else:
            task_type = "classification"
        if "open" in forms:
            record = dict(
                common, question_id=f"{point_key}__open", form="open",
                task_type=task_type, question=str(open_fact["stem"]),
                options=[], truth=open_fact["truth_value"])
            record.update(_open_public_fields(open_fact, mcq))
            records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--facts-root", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--answer-form", action="append", dest="answer_forms",
                        help="mcq or open; repeat to request both")
    parser.add_argument("--params", type=Path,
                        help="JSON params containing ANSWER_FORMS_DEFAULT")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    params = _read(args.params) if args.params is not None else None
    result = build(
        _read(args.selection_manifest), args.facts_root,
        args.audio_root, args.media_root,
        answer_forms=args.answer_forms, params=params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "record_count": len(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
