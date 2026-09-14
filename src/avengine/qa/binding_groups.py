"""Assemble and check groups of native instance-binding observations.

Private groups carry questions/gold and relations. The public export contains
only media and sanitized question forms. Actual media equality is checked
without introducing content-hash contracts. Matching groups stay in one split.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import filecmp
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re
import subprocess
from typing import Any

from avengine.qa.binding_questions import generate_binding_question, BindingQuestionError
from avengine.qa.binding_conditions import (
    TASK_FAMILIES,
    validate_binding_episode,
    BindingConditionError,
)


class BindingGroupError(ValueError):
    pass


#: Identifiers a group carries through assembly so a later export never has to
#: guess a task or a room from a directory name. ``required`` ones make the
#: group unusable when absent; the rest are reported as ``unknown``.
IDENTITY_FIELDS = ("task_family", "room_family", "room_id", "world_id")
REQUIRED_IDENTITY_FIELDS = ("task_family", "room_family", "world_id")


def _identifier(group: Mapping[str, Any], field: str) -> dict:
    """Read one declared identifier and say where the value came from."""
    value = group.get(field)
    if isinstance(value, str) and value.strip():
        return {"value": value.strip(), "status": "declared",
                "provenance": f"group_spec.{field}"}
    if field in REQUIRED_IDENTITY_FIELDS:
        raise BindingGroupError(
            f"group {group.get('group_id')!r} must declare a nonempty {field}; "
            "this identifier is carried into the export and is never inferred "
            "from a directory name")
    # Absent is reported as absent. Filling it in from a path or a neighbouring
    # group would put an invented value in front of whoever counts coverage.
    return {"value": None, "status": "unknown",
            "provenance": f"absent in group_spec {group.get('group_id')!r}"}


def _source_identity(group: Mapping[str, Any]) -> dict:
    identity = {field: _identifier(group, field) for field in IDENTITY_FIELDS}
    task_family = identity["task_family"]["value"]
    if task_family not in TASK_FAMILIES:
        raise BindingGroupError(
            f"group {group.get('group_id')!r} declares task_family {task_family!r}, "
            f"which is not one of {list(TASK_FAMILIES)}")
    return identity


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BindingGroupError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise BindingGroupError("native input path must be nonempty")
    path = Path(value)
    path = (base / path if not path.is_absolute() else path).resolve()
    if not path.is_file():
        raise BindingGroupError(f"input file is absent: {path}")
    return path


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    videos = [s for s in result["streams"] if s.get("codec_type") == "video"]
    if len(videos) != 1:
        raise BindingGroupError(f"expected one video stream: {path}")
    return result


def _video_info(path: Path) -> tuple:
    result = _probe(path)
    if any(s.get("codec_type") == "audio" for s in result["streams"]):
        raise BindingGroupError("public video must not carry a hidden audio track")
    stream = next(s for s in result["streams"] if s["codec_type"] == "video")
    return stream["width"], stream["height"], stream["avg_frame_rate"]


def _same_video(left: Path, right: Path, *, public: bool = True) -> bool:
    def info(path):
        if public:
            return _video_info(path)
        stream = next(s for s in _probe(path)["streams"] if s["codec_type"] == "video")
        return stream["width"], stream["height"], stream["avg_frame_rate"]
    if info(left) != info(right):
        return False
    if left == right or filecmp.cmp(left, right, shallow=False):
        return True
    commands = [
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        for path in (left, right)
    ]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) for command in commands]
    same = True
    try:
        while True:
            blocks = [process.stdout.read(262144) for process in processes]
            if blocks[0] != blocks[1]:
                same = False
                break
            if not blocks[0]:
                break
    finally:
        if not same:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        for process in processes:
            process.stdout.close()
        codes = [process.wait() for process in processes]
    if same and any(codes):
        raise BindingGroupError("video decoding failed during equality check")
    return same


def _audio(path: Path):
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.shape[0] == 0 or samples.shape[1] not in (2, 4):
        raise BindingGroupError("public audio must be nonempty binaural or four-channel spatial audio")
    if not np.isfinite(samples).all() or not np.any(samples):
        raise BindingGroupError("audio contains invalid samples or is entirely silent")
    return samples, int(sample_rate)


def _same_audio(left: Path, right: Path) -> bool:
    import numpy as np

    a, ar = _audio(left)
    b, br = _audio(right)
    return ar == br and a.shape == b.shape and bool(np.array_equal(a, b))


def _validate_clock(facts: Mapping[str, Any], video: Path, audio: Path, cache: dict,
                    *, cutoff: float | None = None) -> dict:
    """Check decoded media clocks against native facts and an optional query cutoff."""
    import soundfile as sf
    key = ("decoded_video_clock", str(video))
    if key not in cache:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(video)],
            check=True, capture_output=True, text=True)
        streams = [s for s in json.loads(result.stdout)["streams"] if s.get("codec_type") == "video"]
        if len(streams) != 1:
            raise BindingGroupError("media clock requires one video stream")
        stream = streams[0]
        try:
            count = int(stream["nb_read_frames"])
            fps = float(Fraction(stream["avg_frame_rate"]))
            origin = float(stream.get("start_time", 0.0))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise BindingGroupError("video has no valid decoded frame clock") from error
        cache[key] = {"frame_count": count, "frame_rate_hz": fps, "start_s": origin,
                      "resolution_hw": [stream["height"], stream["width"]]}
    observed_video = cache[key]
    key = ("decoded_audio_clock", str(audio))
    if key not in cache:
        info = sf.info(audio)
        cache[key] = {"sample_rate_hz": info.samplerate, "sample_count": info.frames,
                      "channel_count": info.channels}
    observed_audio = cache[key]
    expected = facts["time"]
    sr, fps = int(expected["sample_rate_hz"]), float(expected["frame_rate_hz"])
    frames = int(expected["frame_count"]) if cutoff is None else int(round(cutoff * fps)) + 1
    samples = int(expected["sample_count"]) if cutoff is None else int(round(cutoff * sr))
    if (observed_video["frame_count"] != frames
            or not math.isclose(observed_video["frame_rate_hz"], fps, abs_tol=1e-6)
            or abs(observed_video["start_s"]) > 1e-6):
        raise BindingGroupError("decoded video clock differs from native facts or observation cutoff")
    if observed_video["resolution_hw"] != facts.get("visibility_meta", {}).get("resolution_hw"):
        raise BindingGroupError("delivered video resolution differs from pixel facts")
    if (observed_audio["sample_rate_hz"] != sr or observed_audio["sample_count"] != samples
            or observed_audio["channel_count"] != int(facts["audio"]["channel_count"])):
        raise BindingGroupError("audio sample clock or channel count differs from native facts")
    if cutoff is None and not math.isclose(
            samples / sr, float(expected["duration_seconds"]), abs_tol=1 / sr):
        raise BindingGroupError("native fact duration disagrees with its sample clock")
    return {"status": "pass", "video": deepcopy(observed_video), "audio": deepcopy(observed_audio),
            "observation_cutoff_s": cutoff}


def align_question_forms(questions: list[dict], seed: str) -> None:
    """Use identical options/order and question text within a controlled group."""
    if not questions:
        raise BindingGroupError("empty question group")
    forms = set(questions[0]["forms"])
    if any(set(question["forms"]) != forms for question in questions):
        raise BindingGroupError("answer-form availability differs between group members")
    if "mcq" in forms:
        options = deepcopy(questions[0]["forms"]["mcq"]["options"])
        canonical = lambda values: sorted(
            (str(o["value"]), str(o["label_en"]), str(o["label_zh"])) for o in values)
        expected = canonical(options)
        if any(canonical(q["forms"]["mcq"]["options"]) != expected for q in questions):
            raise BindingGroupError("MCQ options differ between controlled variants")
        options.sort(key=lambda value: str(value["value"]))
        random.Random(seed).shuffle(options)
        for question in questions:
            form = question["forms"]["mcq"]
            truth = form["gold"]["value"]
            form["options"] = deepcopy(options)
            form["gold"]["correct_index"] = [o["value"] for o in options].index(truth)
            question["model_input"]["mcq"]["options"] = [
                {"option": chr(65 + index), "label_en": o["label_en"], "label_zh": o["label_zh"]}
                for index, o in enumerate(options)
            ]
    expected_input = questions[0]["model_input"]
    if any(question["model_input"] != expected_input for question in questions[1:]):
        raise BindingGroupError("public question text, options or calibration differs within group")


def _audio_input_identity(member: Mapping[str, Any]) -> dict | None:
    """The clip and crop each event actually used, as recorded at validation."""
    conditions = member.get("episode_conditions")
    if not isinstance(conditions, Mapping):
        return None
    identity = conditions.get("audio_input_identity")
    return dict(identity) if isinstance(identity, Mapping) else None


def _clip_inventory(identity: Mapping[str, Any]) -> list:
    """Which clips and crops were used, ignoring which instance played them."""
    rows = []
    for value in identity.values():
        row = {key: item for key, item in value.items() if key != "actor_id"}
        rows.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    return sorted(rows)


def _audio_input_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
    """Compare what the two members fed the renderer, not what they claim."""
    left_identity, right_identity = _audio_input_identity(left), _audio_input_identity(right)
    if left_identity is None or right_identity is None:
        return {"status": "not_run",
                "reason": "a member carries no recorded audio input identity"}
    same_assignment = left_identity == right_identity
    same_inventory = _clip_inventory(left_identity) == _clip_inventory(right_identity)
    return {
        "status": "pass",
        "identical_crop_assignment": same_assignment,
        "identical_clip_inventory": same_inventory,
        # Reported, not gated: an intervention that swaps the assignment of the
        # same crops is a tighter control than one that brings in other clips,
        # but both are legitimate ways to change the audio.
        "audio_intervention_kind": (
            "same_audio" if same_assignment
            else "reassignment_of_same_crops" if same_inventory
            else "different_clip_inventory"),
    }


def _different_truth(left: dict, right: dict, form: str, angle_tolerance_deg: float) -> bool:
    lf, rf = left["forms"][form], right["forms"][form]
    if form == "mcq":
        return lf["gold"]["value"] != rf["gold"]["value"]
    if lf["answer_type"] == "angle_deg":
        delta = abs((float(lf["truth"]) - float(rf["truth"]) + 180.0) % 360.0 - 180.0)
        return delta > 2 * angle_tolerance_deg
    return lf["truth"] != rf["truth"]


def validate_group(group: Mapping[str, Any], *, base: Path, verify_media: bool = True) -> dict:
    members = list(group.get("members", []))
    if len(members) < 4:
        raise BindingGroupError("a bidirectional group requires at least four members")
    by_id = {m["sample_id"]: m for m in members}
    if len(by_id) != len(members):
        raise BindingGroupError("duplicate sample ID inside group")
    if len({json.dumps(m["question"]["model_input"], sort_keys=True) for m in members}) != 1:
        raise BindingGroupError("public question inputs differ")
    forms = set(members[0]["question"]["forms"])
    if any(set(m["question"]["forms"]) != forms for m in members):
        raise BindingGroupError("inconsistent forms")
    witnesses = {member_id: set() for member_id in by_id}
    comparisons = []
    comparison_keys = set()
    tolerance = float(group.get("angle_tolerance_deg", 10))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise BindingGroupError("angle tolerance must be finite and nonnegative")
    for comparison in group.get("comparisons", []):
        ids = comparison.get("members", [])
        if len(ids) != 2 or ids[0] == ids[1] or any(value not in by_id for value in ids):
            raise BindingGroupError("invalid comparison endpoints")
        left, right = (by_id[value] for value in ids)
        relation = comparison.get("answer_relation")
        kind = comparison.get("kind")
        if kind not in {"necessity", "invariance"}:
            raise BindingGroupError("comparison kind must be necessity or invariance")
        if relation not in {"same", "different"}:
            raise BindingGroupError("answer_relation must be same or different")
        if (kind == "necessity" and relation != "different") or (kind == "invariance" and relation != "same"):
            raise BindingGroupError("comparison kind disagrees with its answer relation")
        key = (tuple(sorted(ids)), comparison.get("shared_modality"), kind)
        if key in comparison_keys:
            raise BindingGroupError("duplicate group comparison")
        comparison_keys.add(key)
        if kind == "necessity" and comparison.get("shared_modality") not in {"audio", "video"}:
            raise BindingGroupError("necessity comparison must share one declared modality")
        for form in forms:
            differs = _different_truth(left["question"], right["question"], form, tolerance)
            if relation == "different" and not differs:
                raise BindingGroupError(f"answers are not distinguishable for {form} under its tolerance")
            if relation == "same":
                l = left["question"]["forms"][form]
                r = right["question"]["forms"][form]
                if (l["gold"]["value"] if form == "mcq" else l["truth"]) != (r["gold"]["value"] if form == "mcq" else r["truth"]):
                    raise BindingGroupError("declared invariant answers differ")
        shared = comparison.get("shared_modality")
        if shared not in {"audio", "video", None}:
            raise BindingGroupError("invalid shared modality")
        audio_inputs = _audio_input_relation(left, right)
        allow_audio_reassignment = (
            shared == "audio"
            and comparison.get("allow_audio_reassignment") is True
            and audio_inputs["status"] == "pass"
            and audio_inputs["identical_clip_inventory"]
        )
        if (
            shared == "audio"
            and audio_inputs["status"] == "pass"
            and not audio_inputs["identical_crop_assignment"]
            and not allow_audio_reassignment
        ):
            # Cropping a long recording is allowed. An identity recipe may
            # rebind the same complete clips to another persistent actor, but
            # it must declare that intervention and keep the clip/crop
            # inventory identical. Ordinary recipes remain exact-assignment.
            raise BindingGroupError(
                "declared shared audio was rendered from a different clip or crop "
                "per member")
        media_result = "not_run"
        if verify_media:
            left_video = _path(left["media"]["video_path"], base)
            right_video = _path(right["media"]["video_path"], base)
            left_audio = _path(left["media"]["audio_path"], base)
            right_audio = _path(right["media"]["audio_path"], base)
            audio_same = _same_audio(left_audio, right_audio)
            video_same = _same_video(left_video, right_video)
            if shared == "audio" and not audio_same:
                raise BindingGroupError("declared shared audio differs in actual PCM samples")
            if shared == "video" and not video_same:
                raise BindingGroupError("declared shared video differs in decoded RGB")
            if relation == "different" and audio_same and video_same:
                raise BindingGroupError("identical AV observations have contradictory answers")
            if shared == "audio" and relation == "different" and video_same:
                raise BindingGroupError("the visual intervention did not change video")
            if shared == "video" and relation == "different" and audio_same:
                raise BindingGroupError("the audio intervention did not change audio")
            media_result = "pass"
        if comparison.get("kind") == "necessity" and relation == "different" and shared:
            for member_id in ids:
                witnesses[member_id].add(shared)
        comparisons.append({
            **deepcopy(dict(comparison)),
            "media_check": media_result,
            "audio_inputs": audio_inputs,
            "audio_reassignment_allowed": allow_audio_reassignment,
        })
    if any(values != {"audio", "video"} for values in witnesses.values()):
        raise BindingGroupError("every member needs both an audio-shared and a video-shared answer-changing witness")
    intervention_kinds: dict[str, int] = {}
    for row in comparisons:
        kind = row["audio_inputs"].get("audio_intervention_kind")
        if kind:
            intervention_kinds[kind] = intervention_kinds.get(kind, 0) + 1
    return {
        "status": "pass" if verify_media else "structure_only",
        "sample_count": len(members), "comparisons": comparisons,
        "audio_intervention_kinds": intervention_kinds,
        "shared_audio_crop_check": (
            "pass" if all(row["audio_inputs"]["status"] == "pass" for row in comparisons)
            else "not_run"),
        "forms": sorted(forms), "angle_tolerance_deg": tolerance,
        "claim_boundary": "observed media/answer relations only; human answerability and model ablations remain separate",
        "human_answerability": "not_run", "model_evaluation": "not_run",
    }


def _public_strings(value: Any, keys: set, strings: set) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            _public_strings(item, keys, strings)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _public_strings(item, keys, strings)
    elif isinstance(value, str):
        strings.add(value)


#: Keys that only ever belong to the private side of a group.
PRIVATE_ONLY_KEYS = frozenset({
    "group_id", "member_id", "interventions", "task_family", "world_id",
    "facts_path", "native_episode_id", "source_identity", "episode_conditions",
    "gold", "truth", "comparisons",
})


def _check_public_payload(public: Sequence[Mapping[str, Any]],
                          packed: Sequence[Mapping[str, Any]]) -> dict:
    """Confirm the model-facing rows carry no intervention or grouping identity.

    Option labels are public by construction, so a gold *label* is not a leak;
    what must not appear is the identity that says which member this is or what
    was changed to make it differ from its siblings.
    """
    keys: set[str] = set()
    strings: set[str] = set()
    _public_strings(list(public), keys, strings)
    private_values: set[str] = set()
    for group in packed:
        for field in ("group_id", "world_id", "task_family"):
            if isinstance(group.get(field), str):
                private_values.add(group[field])
        for member in group["members"]:
            for field in ("member_id", "native_episode_id", "facts_path"):
                if isinstance(member.get(field), str):
                    private_values.add(member[field])
            for value in (member.get("interventions") or {}).values():
                if isinstance(value, str):
                    private_values.add(value)
    leaked_keys = sorted(keys & PRIVATE_ONLY_KEYS)
    leaked_values = sorted(strings & private_values)
    if leaked_keys or leaked_values:
        raise BindingGroupError(
            f"public model input leaks private identity: keys={leaked_keys} "
            f"values={leaked_values}")
    return {"status": "pass", "checked_key_count": len(keys),
            "checked_string_count": len(strings),
            "private_identity_values_checked": len(private_values),
            "note": "option labels are public by construction and are not leaks"}


#: The files an exported binding dataset hands to a model or an annotator.
#: Everything else under the export root is the private side. Named here so the
#: scan cannot quietly stop covering a file the export starts writing.
PUBLIC_EXPORT_FILES = ("model_inputs.json",)

#: Directories of the export whose file names travel with the public payload.
PUBLIC_EXPORT_MEDIA_DIRS = ("media",)


def _private_identity_values(packed: Sequence[Mapping[str, Any]]) -> dict:
    """Every string that would say which member of which group a sample is."""
    values: set[str] = set()
    variant_tokens: set[str] = set()
    for group in packed:
        for field in ("group_id", "world_id", "task_family", "room_id"):
            if isinstance(group.get(field), str) and group[field].strip():
                values.add(group[field])
        request = group.get("request") if isinstance(group.get("request"), Mapping) else {}
        for asset_id in request.get("source_asset_ids") or ():
            if isinstance(asset_id, str) and asset_id.strip():
                values.add(asset_id)
        if isinstance(request.get("episode_id"), str) and request["episode_id"].strip():
            values.add(request["episode_id"])
        for member in group.get("members") or ():
            if not isinstance(member, Mapping):
                continue
            for field in ("member_id", "native_episode_id", "facts_path"):
                if isinstance(member.get(field), str) and member[field].strip():
                    values.add(member[field])
            for item in (member.get("interventions") or {}).values():
                if isinstance(item, str) and item.strip():
                    values.add(item)
                    variant_tokens.add(item)
            if isinstance(member.get("member_id"), str):
                variant_tokens.update(
                    part for part in str(member["member_id"]).split("_") if part)
    return {"values": values, "variant_tokens": variant_tokens}


def check_public_export_files(output: Path, packed: Sequence[Mapping[str, Any]]) -> dict:
    """Scan the exported public files for anything naming a member or a group.

    The in-memory payload check runs before anything is written; this reads the
    files back off disk, so a later writer that adds a field, a filename or a
    provenance line cannot put the intervention where a reader would see it. A
    media file name is checked as its own text because the name travels with the
    sample even when the payload does not.
    """
    output = Path(output)
    identity = _private_identity_values(packed)
    values = sorted(identity["values"])
    tokens = sorted(token for token in identity["variant_tokens"]
                    if re.fullmatch(r"[a-z]+[0-9]+", token))
    scanned, leaks = [], []
    for name in PUBLIC_EXPORT_FILES:
        path = output / name
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        scanned.append({"file": name, "bytes": len(raw.encode("utf-8"))})
        for value in values:
            if value in raw:
                leaks.append({"file": name, "kind": "private_value", "value": value})
        payload = json.loads(raw)
        keys: set[str] = set()
        strings: set[str] = set()
        _public_strings(payload, keys, strings)
        for key in sorted(keys & PRIVATE_ONLY_KEYS):
            leaks.append({"file": name, "kind": "private_key", "value": key})
    names = []
    for directory in PUBLIC_EXPORT_MEDIA_DIRS:
        folder = output / directory
        if not folder.is_dir():
            continue
        for item in sorted(folder.iterdir()):
            names.append(item.name)
            for value in values:
                if value in item.name:
                    leaks.append({"file": f"{directory}/{item.name}",
                                  "kind": "private_value_in_filename", "value": value})
            for token in tokens:
                if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", item.name):
                    leaks.append({"file": f"{directory}/{item.name}",
                                  "kind": "variant_token_in_filename", "value": token})
    if leaks:
        raise BindingGroupError(
            f"the exported public files name private identity: {leaks[:8]}")
    return {
        "status": "pass",
        "files_scanned": scanned,
        "media_file_count": len(names),
        "private_values_checked": len(values),
        "variant_tokens_checked": tokens,
        "authority": ("read back from the written export; covers the public payload "
                      "file and every media file name"),
    }


def _identity_matrix(packed: Sequence[Mapping[str, Any]]) -> dict:
    """Count real declared task and room identity, so a matrix needs no guess."""
    by_room: dict[str, dict[str, dict[str, int]]] = {}
    unknown = []
    for group in packed:
        identity = group["source_identity"]
        family = identity["task_family"]["value"]
        room = identity["room_family"]["value"]
        row = by_room.setdefault(str(family), {}).setdefault(
            str(room), {"group_count": 0, "sample_count": 0})
        row["group_count"] += 1
        row["sample_count"] += len(group["members"])
        for field, value in identity.items():
            if value["status"] == "unknown":
                unknown.append({"group_id": group["group_id"], "field": field,
                                "provenance": value["provenance"]})
    return {
        "task_family_by_room_family": by_room,
        "task_families_present": sorted(by_room),
        "task_families_absent": [family for family in TASK_FAMILIES if family not in by_room],
        "unknown_source_identity": unknown,
        "note": ("counted from the identifiers each group_spec declared; a missing "
                 "identifier is listed as unknown rather than inferred"),
    }


def _export_media(source_video: Path, source_audio: Path, output: Path, name: str,
                  cutoff: float | None, frame_rate: float, cache: dict) -> dict:
    video_key = (str(source_video), cutoff, frame_rate)
    audio_key = (str(source_audio), cutoff)
    if video_key not in cache:
        target = output / "media" / f"video_{name}.mp4"
        command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source_video),
                   "-map", "0:v:0", "-an", "-map_metadata", "-1", "-map_chapters", "-1"]
        if cutoff is not None:
            # Decode before trimming: stream-copy packet order can otherwise
            # retain a future P-frame across a B-frame query boundary.
            command.extend(["-frames:v", str(int(round(cutoff * frame_rate)) + 1),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "0"])
        else:
            command.extend(["-c:v", "copy"])
        command.append(str(target))
        subprocess.run(command, check=True, capture_output=True)
        _video_info(target)
        cache[video_key] = str(target.relative_to(output))
    if audio_key not in cache:
        import soundfile as sf

        target = output / "media" / f"audio_{name}.wav"
        samples, sr = _audio(source_audio)
        if cutoff is not None:
            count = int(round(cutoff * sr))
            if count <= 0 or count > len(samples):
                raise BindingGroupError("observation cutoff is outside audio")
            samples = samples[:count]
        sf.write(target, samples, sr, subtype="FLOAT")
        cache[audio_key] = str(target.relative_to(output))
    return {"video_path": cache[video_key], "audio_path": cache[audio_key]}


def assemble_binding_dataset(spec: Mapping[str, Any], *, input_base: Path, output: Path,
                             seed: str = "binding-dataset", verify_media: bool = True) -> dict:
    """Package completed native variants; the original artifacts are read-only."""
    groups = list(spec.get("groups", []))
    if not groups:
        raise BindingGroupError("input specification has no groups")
    identities = {}
    for group in groups:
        identities[id(group)] = _source_identity(group)
    if any(not isinstance(group.get("request", spec.get("request")), Mapping) for group in groups):
        raise BindingGroupError("each binding group needs its caller's QA request (group.request or spec.request)")
    group_ids = [g["group_id"] for g in groups]
    if len(group_ids) != len(set(group_ids)):
        raise BindingGroupError("duplicate group IDs")
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    (output / "media").mkdir()
    total = sum(len(group["members"]) for group in groups)
    sample_ids = [f"sample_{index + 1:06d}" for index in range(total)]
    random.Random(seed).shuffle(sample_ids)
    media_names = list(range(total))
    random.Random(seed + ":media").shuffle(media_names)
    cache: dict[Any, Any] = {}
    visual_splits: dict[str, str] = {}
    world_splits: dict[str, str] = {}
    packed, public = [], []
    next_index = 0
    for group in groups:
        result = {key: deepcopy(group[key]) for key in
                  ("group_id", "world_id", "task_family", "room_family", "room_id", "split", "query", "angle_tolerance_deg")
                  if key in group}
        result.setdefault("split", "pilot")
        identity = identities[id(group)]
        result["source_identity"] = deepcopy(identity)
        for field, row in identity.items():
            if row["value"] is not None:
                result[field] = row["value"]
        world_id = identity["world_id"]["value"]
        result["world_id"] = world_id
        if world_id in world_splits and world_splits[world_id] != result["split"]:
            raise BindingGroupError("one physical world cannot cross dataset splits")
        world_splits[world_id] = result["split"]
        request = deepcopy(group.get("request", spec.get("request")))
        result["request"] = request
        members, names = [], {}
        for raw in group["members"]:
            member_name = str(raw["member_id"])
            if member_name in names:
                raise BindingGroupError("duplicate member_id")
            facts_path = _path(raw["facts_path"], input_base)
            facts = _load(facts_path)
            try:
                conditions = validate_binding_episode(
                    facts, request=request, base=facts_path.parent, cache=cache)
            except BindingConditionError as error:
                raise BindingGroupError(str(error)) from error
            question = generate_binding_question(
                facts, group["task_family"], group["query"], seed=seed + ":" + group["group_id"])
            native_video = _path(facts["source_paths"]["video"], facts_path.parent)
            native_audio = _path(facts["audio"]["path"], facts_path.parent)
            source_video = _path(raw.get("video_path") or str(native_video), input_base)
            source_audio = _path(raw.get("audio_path") or str(native_audio), input_base)
            if source_video != native_video and not _same_video(source_video, native_video, public=False):
                raise BindingGroupError("video override does not match native facts")
            if source_audio != native_audio and not _same_audio(source_audio, native_audio):
                raise BindingGroupError("audio override does not match native facts")
            for key in ("frame_readbacks", "pixel_visibility_truth", "audio_program"):
                _path(facts["source_paths"][key], facts_path.parent)
            visual_key = str(native_video)
            if visual_key in visual_splits and visual_splits[visual_key] != result["split"]:
                raise BindingGroupError("one native visual episode cannot cross dataset splits")
            visual_splits[visual_key] = result["split"]
            native_clock = _validate_clock(facts, source_video, source_audio, cache)
            sample_id = sample_ids[next_index]
            names[member_name] = sample_id
            cutoff = question["evidence"].get("observation_cutoff_s")
            media = _export_media(source_video, source_audio, output, str(media_names[next_index]),
                                  cutoff, float(facts["time"]["frame_rate_hz"]), cache)
            exported_clock = _validate_clock(
                facts, output / media["video_path"], output / media["audio_path"], cache, cutoff=cutoff)
            next_index += 1
            members.append({
                "sample_id": sample_id, "member_id": member_name,
                "group_id": group["group_id"],
                "task_family": identity["task_family"]["value"],
                "room_family": identity["room_family"]["value"],
                "room_id": identity["room_id"]["value"],
                "world_id": world_id,
                "source_identity": deepcopy(identity),
                "question": question, "media": media,
                "media_clock": {"native": native_clock, "exported": exported_clock},
                "episode_conditions": conditions,
                "facts_path": str(facts_path), "native_episode_id": facts["episode_id"],
                "interventions": deepcopy(raw.get("interventions", {})),
            })
        align_question_forms([member["question"] for member in members], seed + ":" + group["group_id"])
        result["members"] = members
        result["comparisons"] = [
            {**deepcopy(comparison), "members": [names[name] for name in comparison["members"]]}
            for comparison in group["comparisons"]
        ]
        result["validation"] = validate_group(result, base=output, verify_media=verify_media)
        for member in members:
            public.append({
                "sample_id": member["sample_id"], "media": member["media"],
                "question_id": member["sample_id"],
                "forms": deepcopy(member["question"]["model_input"]),
            })
        packed.append(result)
        _write(output / "groups" / f"{len(packed):04d}.json", result)
    random.Random(seed + ":order").shuffle(public)
    result = {
        "schema": "avengine_binding_groups_v1", "status": "research_candidate",
        "group_count": len(packed), "world_count": len(world_splits), "sample_count": len(public),
        "requested_group_count": len(groups), "groups": packed,
        "source_identity_coverage": _identity_matrix(packed),
        "public_payload_check": _check_public_payload(public, packed),
        "validation": "media_checked" if verify_media else "structure_only",
        "model_evaluation": "not_run", "human_answerability": "not_run",
        "input_protocol": "one sample at a time; identifiers and media filenames are routing metadata, not prompt content",
    }
    _write(output / "binding_groups.json", result)
    _write(output / "model_inputs.json", {"schema": "avengine_binding_public_inputs_v1", "samples": public})
    result["public_export_file_check"] = check_public_export_files(output, packed)
    _write(output / "binding_groups.json", result)
    return result
