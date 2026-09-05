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
import math
import shutil
import subprocess
import sys
from pathlib import Path

from qa_v3_request import normalize_answer_forms


_PUBLIC_OPEN_FIELDS = ("truth_interval_deg", "convention", "certification_policy")
_DEFAULT_ANSWER_FORMS = ("mcq", "open")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _declared_answer_forms(selection):
    if not isinstance(selection, Mapping):
        return None
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
    return next((value for value in candidates if value is not None), None)


def _answer_forms(selection, answer_forms, params, candidate_forms=None):
    """Resolve explicit, manifest, candidate, params, then legacy forms."""
    if answer_forms is not None:
        return normalize_answer_forms(answer_forms)
    declared = _declared_answer_forms(selection)
    if declared is not None:
        return normalize_answer_forms(declared)
    if candidate_forms is not None:
        return normalize_answer_forms(candidate_forms)
    if params is not None:
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        if "ANSWER_FORMS_DEFAULT" in params:
            return normalize_answer_forms(params["ANSWER_FORMS_DEFAULT"])
    return list(_DEFAULT_ANSWER_FORMS)


def _open_public_fields(open_fact, mcq_fact=None):
    fields = {}
    for key in _PUBLIC_OPEN_FIELDS:
        source = open_fact
        if (key == "convention" and key not in source
                and isinstance(mcq_fact, Mapping)):
            source = mcq_fact
        if key in source:
            fields[key] = source[key]
    return fields


def _encode_component(value):
    """Encode one ID component without introducing a hash or delimiter ambiguity."""
    text = str(value)
    return f"{len(text)}:{text}"


def _scoped_key(prefix, *parts):
    """Length-prefix components so the resulting key is reversible."""
    return prefix + "".join(_encode_component(part) for part in parts)


def _has_scene(scene_id):
    return scene_id is not None and str(scene_id) != ""


def _question_id(scene_id, point_id, form):
    # Keep the old single-room spelling when no scene was recorded.  Once a
    # scene is present, length-prefix all components so scene/point/form cannot
    # collide even when a component contains the historical ``__`` separator.
    if _has_scene(scene_id):
        return _scoped_key("qa3:", scene_id, point_id, form)
    return f"{point_id}__{form}"


def _group_id(scene_id, episode):
    if _has_scene(scene_id):
        return _scoped_key("qa3g:", scene_id, episode)
    return episode


def _ffprobe_video(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe is required to validate released video duration")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames,r_frame_rate,duration",
        "-of", "json", str(path),
    ]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"ffprobe timed out for released video: {path}") from exc
    if done.returncode != 0:
        raise ValueError(f"ffprobe failed for released video {path}: {done.stderr[-400:]}")
    try:
        payload = json.loads(done.stdout)
        stream = payload["streams"][0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe returned no video stream for {path}") from exc
    frames = stream.get("nb_read_frames", stream.get("nb_frames"))
    if frames in (None, "N/A"):
        raise ValueError(f"ffprobe returned no video frame count for {path}")
    try:
        frame_count = int(frames)
        duration = float(stream["duration"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"ffprobe returned an invalid video clock for {path}") from exc
    return {"frame_count": frame_count, "duration_seconds": duration,
            "frame_rate": stream.get("r_frame_rate")}


def _declared_frame_clock(fact, point_dir: Path | None = None):
    """Read an existing QA-v3 clock from the fact or point-local program."""
    sources = [fact]
    if point_dir is not None:
        point_dir = Path(point_dir)
        for name in ("audio_program.json", "timeline.json"):
            path = point_dir / name
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read declared clock input {path}") from exc
            sources.append(value)
    for value in sources:
        if not isinstance(value, Mapping):
            continue
        for key in ("frame_clock", "timeline"):
            clock = value.get(key)
            if isinstance(clock, Mapping):
                normalized = dict(clock)
                # Audio programs use the existing video_fps spelling;
                # facts/render clocks use frame_rate_hz.
                if "frame_rate_hz" not in normalized and "video_fps" in normalized:
                    normalized["frame_rate_hz"] = normalized["video_fps"]
                return normalized
    return None


def _validate_media_clock(
    fact, wav: Path, video: Path, *, owner: str, point_dir: Path | None = None
):
    """Validate real released media when the fact or point declares a clock."""
    clock = _declared_frame_clock(fact, point_dir)
    if clock is None:
        return None
    if not isinstance(clock, Mapping):
        raise ValueError(f"{owner}: frame_clock must be an object")
    required = ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count")
    missing = [key for key in required if key not in clock]
    if missing:
        raise ValueError(f"{owner}: frame_clock missing {missing}")
    try:
        expected_frames = int(clock["frame_count"])
        expected_fps = float(clock["frame_rate_hz"])
        expected_rate = int(clock["sample_rate_hz"])
        expected_samples = int(clock["sample_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: frame_clock contains invalid values") from exc
    if (expected_frames < 1 or not math.isfinite(expected_fps)
            or expected_fps <= 0.0 or expected_rate < 1 or expected_samples < 1):
        raise ValueError(f"{owner}: frame_clock values must be positive and finite")
    try:
        import soundfile as sf
        audio_info = sf.info(wav)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{owner}: cannot inspect released audio {wav}") from exc
    if (audio_info.samplerate != expected_rate
            or audio_info.frames != expected_samples
            or audio_info.channels != 2):
        raise ValueError(
            f"{owner}: released audio clock differs from fact: "
            f"media={audio_info.samplerate}Hz/{audio_info.frames} samples/"
            f"{audio_info.channels}ch, expected={expected_rate}Hz/"
            f"{expected_samples} samples/2ch")
    expected_duration = expected_frames / expected_fps
    tolerance = max(1.0 / expected_fps, 0.05)
    audio_duration = audio_info.frames / expected_rate
    if (not math.isfinite(audio_duration)
            or abs(audio_duration - expected_duration) > tolerance):
        raise ValueError(
            f"{owner}: released audio duration {audio_duration:.6f}s "
            f"differs from expected {expected_duration:.6f}s")
    video_info = _ffprobe_video(video)
    if video_info["frame_count"] != expected_frames:
        raise ValueError(
            f"{owner}: released video has {video_info['frame_count']} frames, "
            f"expected {expected_frames}")
    if (not math.isfinite(video_info["duration_seconds"])
            or abs(video_info["duration_seconds"] - expected_duration) > tolerance):
        raise ValueError(
            f"{owner}: released video duration {video_info['duration_seconds']:.6f}s "
            f"differs from expected {expected_duration:.6f}s")
    return {
        "frame_count": expected_frames,
        "frame_rate_hz": expected_fps,
        "clip_seconds": expected_duration,
        "sample_rate_hz": expected_rate,
        "sample_count": expected_samples,
        "audio_media_frames": audio_info.frames,
        "audio_media_duration_seconds": audio_duration,
        "video_media_frames": video_info["frame_count"],
        "video_media_duration_seconds": video_info["duration_seconds"],
    }


def _entry_forms(candidate, profile, selection):
    for owner in (candidate, profile, selection):
        if isinstance(owner, Mapping):
            value = owner.get("answer_forms")
            if value is not None:
                return value
    return None


def _iter_entries(selection, facts_root):
    """Yield old selected entries or candidates from an assembler manifest."""
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be an object")
    if "selected" in selection:
        if facts_root is None:
            raise ValueError("facts_root is required for selected input")
        selected = selection["selected"]
        if not isinstance(selected, list):
            raise ValueError("selection.selected must be a list")
        for chosen in sorted(selected, key=lambda row: str(row["point_id"])):
            point_id = str(chosen["point_id"])
            yield {
                "point_id": point_id,
                "pilot_id": None,
                "media_id": point_id,
                "fact_path": Path(facts_root) / point_id / "fact_record.json",
                "answer_forms": _entry_forms(chosen, None, selection),
                "scene_id_hint": None,
            }
        return

    rooms = selection.get("rooms")
    if not isinstance(rooms, Mapping):
        raise ValueError("selection must contain selected or rooms")
    for scene_id, room in sorted(rooms.items(), key=lambda pair: str(pair[0])):
        if not isinstance(room, Mapping):
            raise ValueError(f"room {scene_id!r} must be an object")
        profiles = room.get("profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError(f"room {scene_id!r} profiles must be an object")
        for profile_id, profile in sorted(
                profiles.items(), key=lambda pair: str(pair[0])):
            if not isinstance(profile, Mapping):
                raise ValueError(f"profile {profile_id!r} must be an object")
            if profile.get("status") != "selected":
                continue
            candidates = profile.get("candidates", [])
            if not isinstance(candidates, list):
                raise ValueError(
                    f"room {scene_id!r}/{profile_id!r} candidates must be a list")
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise ValueError("assembler candidate must be an object")
                pilot_id = candidate.get("pilot_id")
                if pilot_id is None or str(pilot_id) == "":
                    raise ValueError(
                        f"room {scene_id!r}/{profile_id!r} candidate has no pilot_id")
                artifacts = candidate.get("artifacts") or {}
                fact_path = artifacts.get("fact")
                if fact_path is None:
                    source_point = candidate.get("source_point")
                    if source_point is None:
                        raise ValueError(
                            f"candidate {pilot_id!r} has no artifacts.fact or source_point")
                    fact_path = Path(source_point) / "fact_record.json"
                yield {
                    "point_id": str(candidate.get("source_point_id")
                                    or Path(fact_path).parent.name),
                    "pilot_id": str(pilot_id),
                    "media_id": str(pilot_id),
                    "fact_path": Path(fact_path).expanduser().resolve(),
                    "answer_forms": _entry_forms(candidate, profile, selection),
                    "scene_id_hint": str(scene_id),
                }


def build(selection, facts_root=None, audio_root=None, media_root=None, *,
          answer_forms=None, params=None):
    records = []
    seen_question_ids = set()
    for entry in _iter_entries(selection, facts_root):
        point_id = entry["point_id"]
        fact = _read(entry["fact_path"])
        if audio_root is None or media_root is None:
            raise ValueError("audio_root and media_root are required")
        media_id = entry["media_id"]
        wav = Path(audio_root) / media_id / "audio" / "binaural" / "mixture.wav"
        video = Path(media_root) / media_id / "video_only.mp4"
        if not wav.is_file() or not video.is_file():
            raise FileNotFoundError(f"{media_id}: released audio/video missing")
        media_clock = _validate_media_clock(
            fact, wav, video, owner=f"{media_id} released media",
            point_dir=entry["fact_path"].parent)

        scene_id = fact.get("scene_id")
        if ((scene_id is None or str(scene_id) == "")
                and entry.get("scene_id_hint")):
            scene_id = entry["scene_id_hint"]
        episode = str(fact.get("episode_id", point_id))
        common = {
            "group_id": _group_id(scene_id, episode),
            "point_id": point_id,
            "episode_id": episode,
            "profile_id": str(fact["profile_id"]),
            "audio": str(wav.resolve()),
            "video": str(video.resolve()),
        }
        if scene_id is not None and str(scene_id):
            common["scene_id"] = str(scene_id)
        if media_clock is not None:
            common["media_clock"] = media_clock
        if entry.get("pilot_id") is not None:
            common["pilot_id"] = entry["pilot_id"]
        forms = _answer_forms(
            selection, answer_forms, params,
            candidate_forms=(entry.get("answer_forms")
                             if entry.get("answer_forms") is not None
                             else fact.get("answer_forms")))
        if "mcq" in forms:
            mcq = fact["mcq"]
            append = dict(
                common,
                question_id=_question_id(scene_id, point_id, "mcq"),
                form="mcq", task_type="classification",
                question=str(mcq["stem"]),
                options=list(mcq["options_space"]),
                truth=mcq["truth_option"])
            if append["question_id"] in seen_question_ids:
                raise ValueError(
                    f"duplicate question_id: {append['question_id']!r}")
            seen_question_ids.add(append["question_id"])
            records.append(append)

        if "open" in forms:
            open_fact = fact["open"]
            scoring = open_fact["scoring"]
            if scoring in ("circular_deg", "circular_deg_interval"):
                task_type = "numeric_angle"
            elif scoring == "absolute_time":
                task_type = "numeric_time"
            elif scoring == "transcript_wer":
                task_type = "transcript_wer"
            else:
                task_type = "classification"
            mcq_fact = fact.get("mcq")
            record = dict(
                common,
                question_id=_question_id(scene_id, point_id, "open"),
                form="open", task_type=task_type,
                question=str(open_fact["stem"]), options=[],
                truth=open_fact["truth_value"])
            record.update(_open_public_fields(open_fact, mcq_fact))
            if record["question_id"] in seen_question_ids:
                raise ValueError(
                    f"duplicate question_id: {record['question_id']!r}")
            seen_question_ids.add(record["question_id"])
            records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument(
        "--facts-root", type=Path,
        help="legacy facts root; assembler manifests read artifacts.fact directly")
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
