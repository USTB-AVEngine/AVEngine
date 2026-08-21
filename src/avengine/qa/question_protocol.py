"""Compile and validate the paper-facing QuestionSpec coverage protocol.

The compiler deliberately starts from retained native Episode bundles, then
re-runs the current :mod:`avengine.qa.question_spec` evaluator.  Historical
``question_evaluations.json`` files are never counted as ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from avengine.qa.question_spec import (
    QUESTION_SPEC_SCHEMA,
    evaluate_question_specs,
    question_type_catalog,
)

PROTOCOL_SCHEMA = "avengine_question_spec_paper_protocol_v1"
EPISODE_CATALOG_SCHEMA = "avengine_native_question_episode_catalog_v1"
COVERAGE_SCHEMA = "avengine_question_spec_native_coverage_v1"
DELIVERY_SCHEMA = "avengine_question_spec_protocol_delivery_v1"

_VISIBLE_RGB = (74, 222, 128)
_OCCLUDED_RGB = (236, 92, 92)
_OVERLAY_ALPHA = 0.45
_OVERLAY_HEADER_HEIGHT = 48

_APPEARANCE_FIELDS = (
    "breed_id",
    "size",
    "body_build",
    "life_stage",
    "coat_value",
    "sex_or_gender_label",
)
_NATIVE_ROLES = (
    "native_rgb_binaural",
    "normal_object_ids",
    "pixel_masks",
    "pixel_visibility_truth",
    "runtime_readbacks",
    "metric_depth",
)
_EXPECTED_0807_ORDER = {
    "appearance_to_speaking": 1,
    "appearance_to_spoken_content": 2,
    "sound_to_appearance": 3,
    "speaker_side": 4,
    "who_spoke_first": 5,
    "overlapping_speech": 6,
    "speaking_while_moving": 7,
    "offscreen_to_onscreen": 8,
    "occlusion_while_speaking": 9,
    "occluder_identity": 10,
}
_EXTENSION_TYPES = {
    "reappeared_after_occlusion",
    "became_clear_after_partial_occlusion",
}
_PIXEL_STATES = {
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
}


class QuestionProtocolError(ValueError):
    """The protocol, evidence catalog, or compiled delivery is invalid."""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuestionProtocolError(f"cannot read JSON {path}: {error}") from error


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_repository_path(repository: Path, logical_path: Any, *, field: str) -> Path:
    if not isinstance(logical_path, str) or not logical_path:
        raise QuestionProtocolError(f"{field} must be a non-empty repository-relative path")
    candidate = Path(logical_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise QuestionProtocolError(f"{field} must stay repository-relative: {logical_path!r}")
    return repository / candidate


def _json_pointer(document: Any, pointer: Any, *, field: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise QuestionProtocolError(f"{field} must be a JSON pointer")
    current = document
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise QuestionProtocolError(f"{field} does not resolve: {pointer!r}")
        current = current[part]
    return current


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuestionProtocolError(f"{field} must be a non-empty string")
    return value


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the tracked protocol against the live QuestionSpec catalog."""

    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise QuestionProtocolError(f"protocol schema must be {PROTOCOL_SCHEMA!r}")
    definitions = protocol.get("question_types")
    if not _is_sequence(definitions) or len(definitions) != 12:
        raise QuestionProtocolError("protocol must define exactly 12 question types")
    current = question_type_catalog()
    current_signature = [
        (item["index"], item["question_type"], item["name_zh"]) for item in current
    ]
    protocol_signature = []
    historical: dict[str, int] = {}
    extensions: set[str] = set()
    for position, item in enumerate(definitions, start=1):
        if not isinstance(item, Mapping):
            raise QuestionProtocolError("question type definitions must be objects")
        protocol_signature.append(
            (item.get("catalog_index"), item.get("question_type"), item.get("name_zh"))
        )
        if item.get("catalog_index") != position:
            raise QuestionProtocolError("protocol catalog indices must be consecutive 1..12")
        for field in (
            "definition_zh",
            "answerability_zh",
            "positive_example_zh",
            "negative_example_zh",
        ):
            _require_string(item.get(field), field=f"question_types[{position}].{field}")
        authorities = item.get("gt_authority")
        if not _is_sequence(authorities) or not authorities:
            raise QuestionProtocolError(
                f"question_types[{position}].gt_authority must be non-empty"
            )
        balance = item.get("paper_balance")
        if not isinstance(balance, Mapping) or set(balance) not in (
            {"required_answers"},
            {"minimum_distinct_answers"},
        ):
            raise QuestionProtocolError(
                f"question_types[{position}].paper_balance has an invalid policy"
            )
        origin = item.get("historical_origin")
        question_type = item.get("question_type")
        original_order = item.get("original_0807_order")
        if origin == "0807_core":
            if not isinstance(original_order, int) or isinstance(original_order, bool):
                raise QuestionProtocolError(f"{question_type} needs an 0807 order")
            historical[question_type] = original_order
        elif origin == "post_0807_extension":
            if original_order is not None:
                raise QuestionProtocolError(f"extension {question_type} cannot have an 0807 order")
            extensions.add(question_type)
        else:
            raise QuestionProtocolError(f"unknown historical origin for {question_type}")
    if protocol_signature != current_signature:
        raise QuestionProtocolError(
            "protocol catalog signature differs from the live QuestionSpec catalog"
        )
    if historical != _EXPECTED_0807_ORDER:
        raise QuestionProtocolError("0807 semantic order mapping is incomplete or changed")
    if extensions != _EXTENSION_TYPES:
        raise QuestionProtocolError("post-0807 extension set is incomplete or changed")
    gate = protocol.get("minimum_gate")
    if not isinstance(gate, Mapping) or gate.get("native_pass_cases_per_question_type") != 1:
        raise QuestionProtocolError("minimum gate must require one native pass per type")
    if gate.get("allow_retained_evaluation_as_gt") is not False:
        raise QuestionProtocolError("retained evaluations must not be accepted as GT")
    canary_contract = protocol.get("visual_canary_contract")
    if not isinstance(canary_contract, Mapping):
        raise QuestionProtocolError("visual_canary_contract is required")
    if set(canary_contract.get("required_native_roles", [])) != set(_NATIVE_ROLES):
        raise QuestionProtocolError("visual canary native roles differ from the fixed contract")
    gap_templates = protocol.get("paper_gap_scene_templates")
    if not isinstance(gap_templates, Mapping) or set(gap_templates) != {
        item["question_type"] for item in current
    }:
        raise QuestionProtocolError("paper gap scene templates must cover all 12 types")


def validate_episode_catalog(
    catalog: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    """Validate the declarative native Episode and canary catalog."""

    if not isinstance(catalog, Mapping) or catalog.get("schema") != EPISODE_CATALOG_SCHEMA:
        raise QuestionProtocolError(
            f"episode catalog schema must be {EPISODE_CATALOG_SCHEMA!r}"
        )
    episodes = catalog.get("episodes")
    if not _is_sequence(episodes) or not episodes:
        raise QuestionProtocolError("episode catalog needs at least one Episode")
    keys: set[str] = set()
    ids: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise QuestionProtocolError("episode records must be objects")
        key = _require_string(episode.get("episode_key"), field="episode_key")
        episode_id = _require_string(episode.get("episode_id"), field="episode_id")
        if key in keys or episode_id in ids:
            raise QuestionProtocolError("episode keys and ids must be unique")
        keys.add(key)
        ids.add(episode_id)
        for field in (
            "facts_path",
            "sound_registry_path",
            "event_sound_bindings_path",
            "binding_manifest_path",
        ):
            path = Path(_require_string(episode.get(field), field=field))
            if path.is_absolute() or ".." in path.parts:
                raise QuestionProtocolError(f"{field} must be repository-relative")
        _require_string(
            episode.get("asset_registry_manifest_pointer"),
            field="asset_registry_manifest_pointer",
        )
        _require_string(episode.get("facts_manifest_pointer"), field="facts_manifest_pointer")
        roles = episode.get("native_role_pointers")
        if not isinstance(roles, Mapping) or set(roles) != set(_NATIVE_ROLES):
            raise QuestionProtocolError(f"{key} must bind all required native roles")
        for role, pointer in roles.items():
            _require_string(pointer, field=f"{key}.{role}")
    canaries = catalog.get("visual_canaries")
    if not _is_sequence(canaries):
        raise QuestionProtocolError("visual_canaries must be a list")
    required = set(protocol["visual_canary_contract"]["required_canary_ids"])
    found: set[str] = set()
    controlled_types = {item["question_type"] for item in question_type_catalog()}
    for canary in canaries:
        if not isinstance(canary, Mapping):
            raise QuestionProtocolError("visual canaries must be objects")
        canary_id = _require_string(canary.get("canary_id"), field="canary_id")
        if canary_id in found:
            raise QuestionProtocolError(f"duplicate canary {canary_id!r}")
        found.add(canary_id)
        if canary.get("episode_key") not in keys:
            raise QuestionProtocolError(f"{canary_id} references an unknown Episode")
        _require_string(canary.get("target_instance_id"), field="target_instance_id")
        states = canary.get("expected_state_sequence")
        if not _is_sequence(states) or not states or not set(states) <= _PIXEL_STATES:
            raise QuestionProtocolError(f"{canary_id} has an invalid pixel-state sequence")
        if canary.get("qa_question_type") not in controlled_types:
            raise QuestionProtocolError(f"{canary_id} has an unknown QA type")
        if not isinstance(canary.get("require_dynamic_listener"), bool):
            raise QuestionProtocolError(f"{canary_id} dynamic-listener flag must be bool")
    if found != required:
        raise QuestionProtocolError("visual canary catalog differs from the protocol contract")


def _verify_manifest_file(
    record: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise QuestionProtocolError(f"{label} manifest record must be an object")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value.startswith("/"):
        raise QuestionProtocolError(f"{label} manifest path must be absolute")
    path = Path(path_value)
    if not path.is_file():
        raise QuestionProtocolError(f"{label} evidence file is missing: {path}")
    if expected_path is not None:
        try:
            same = os.path.samefile(path, expected_path)
        except OSError as error:
            raise QuestionProtocolError(f"cannot compare {label} evidence path: {error}") from error
        if not same:
            raise QuestionProtocolError(f"{label} manifest points at a different file")
    declared_size = record.get("size_bytes")
    if not isinstance(declared_size, int) or declared_size <= 0:
        raise QuestionProtocolError(f"{label} needs a positive manifest size")
    actual_size = path.stat().st_size
    if actual_size != declared_size:
        raise QuestionProtocolError(
            f"{label} size mismatch: declared {declared_size}, actual {actual_size}"
        )
    declared_sha = record.get("sha256")
    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        raise QuestionProtocolError(f"{label} needs a manifest SHA-256")
    actual_sha = _sha256(path)
    if actual_sha != declared_sha:
        raise QuestionProtocolError(f"{label} bytes differ from the native manifest")
    return {"path": str(path), "size_bytes": actual_size, "sha256": actual_sha}


def _load_native_episode(repository: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        field: _safe_repository_path(repository, record[field], field=field)
        for field in (
            "facts_path",
            "sound_registry_path",
            "event_sound_bindings_path",
            "binding_manifest_path",
        )
    }
    for field, path in paths.items():
        if not path.is_file():
            raise QuestionProtocolError(f"{record['episode_key']} {field} is missing: {path}")
    manifest = _load_json(paths["binding_manifest_path"])
    facts = _load_json(paths["facts_path"])
    sounds = _load_json(paths["sound_registry_path"])
    bindings = _load_json(paths["event_sound_bindings_path"])
    if not isinstance(manifest, Mapping) or manifest.get("status") != "pass":
        raise QuestionProtocolError(f"{record['episode_key']} binding manifest is not pass")
    if manifest.get("episode_id") != record["episode_id"]:
        raise QuestionProtocolError(f"{record['episode_key']} manifest Episode id mismatch")
    if not isinstance(facts, Mapping) or facts.get("status") != "pass":
        raise QuestionProtocolError(f"{record['episode_key']} Facts are not pass")
    if facts.get("episode_id") != record["episode_id"]:
        raise QuestionProtocolError(f"{record['episode_key']} Facts Episode id mismatch")
    facts_record = _json_pointer(
        manifest,
        record["facts_manifest_pointer"],
        field=f"{record['episode_key']}.facts_manifest_pointer",
    )
    _verify_manifest_file(
        facts_record,
        label=f"{record['episode_key']}.facts",
        expected_path=paths["facts_path"],
    )
    asset_registry_record = _json_pointer(
        manifest,
        record["asset_registry_manifest_pointer"],
        field=f"{record['episode_key']}.asset_registry_manifest_pointer",
    )
    asset_registry_file = _verify_manifest_file(
        asset_registry_record,
        label=f"{record['episode_key']}.asset_registry",
    )
    assets = _load_json(Path(asset_registry_file["path"]))
    native_files: dict[str, dict[str, Any]] = {}
    for role in _NATIVE_ROLES:
        native_record = _json_pointer(
            manifest,
            record["native_role_pointers"][role],
            field=f"{record['episode_key']}.{role}",
        )
        native_files[role] = _verify_manifest_file(
            native_record,
            label=f"{record['episode_key']}.{role}",
        )
    return {
        "episode_key": record["episode_key"],
        "episode_id": record["episode_id"],
        "paths": paths,
        "facts": facts,
        "asset_registry": assets,
        "sound_registry": sounds,
        "event_sound_bindings": bindings,
        "manifest": manifest,
        "native_files": native_files,
    }


def _binding_sound_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("sound_asset_id") or value.get("sound_id")
        return candidate if isinstance(candidate, str) else None
    return None


def _fact_appearance(instance: Mapping[str, Any], field: str) -> Any:
    if field == "breed_id":
        return instance.get(field)
    attributes = instance.get("attributes")
    return attributes.get(field) if isinstance(attributes, Mapping) else None


def enumerate_episode_specs(
    facts: Mapping[str, Any], event_sound_bindings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Enumerate only selectors already observed in one Episode.

    Rejected cases remain useful negative-contract evidence but cannot satisfy
    native coverage.  No unobserved asset, sound, frame, or statement is made up.
    """

    instances = facts.get("instances")
    events = facts.get("sound_events")
    if not _is_sequence(instances) or not _is_sequence(events):
        raise QuestionProtocolError("Facts instances and sound_events must be lists")
    if not isinstance(event_sound_bindings, Mapping):
        raise QuestionProtocolError("event_sound_bindings must be an object")
    events_by_id = {
        event.get("event_id"): event
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    }
    sounds: list[str] = []
    sound_events: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event_id, binding in event_sound_bindings.items():
        sound_id = _binding_sound_id(binding)
        event = events_by_id.get(event_id)
        if sound_id is None or event is None:
            continue
        if sound_id not in sounds:
            sounds.append(sound_id)
        sound_events[sound_id].append(event)
    sounds.sort()
    raw: list[tuple[str, dict[str, Any]]] = []
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        for field in _APPEARANCE_FIELDS:
            value = _fact_appearance(instance, field)
            if isinstance(value, str) and value:
                selectors = {"appearance_field": field, "appearance_value": value}
                raw.append(("appearance_to_speaking", selectors))
                raw.append(("appearance_to_spoken_content", selectors))
    for sound_id in sounds:
        for field in _APPEARANCE_FIELDS:
            raw.append(
                (
                    "sound_to_appearance",
                    {"sound_asset_id": sound_id, "appearance_field": field},
                )
            )
    raw.append(("who_spoke_first", {}))
    for sound_id in sounds:
        active_frames = sorted(
            {
                frame
                for event in sound_events[sound_id]
                if isinstance(event.get("start_frame"), int)
                and isinstance(event.get("end_frame"), int)
                for frame in range(event["start_frame"], event["end_frame"])
                if frame >= 0
            }
        )
        for frame in active_frames:
            raw.append(("speaker_side", {"sound_asset_id": sound_id, "frame_index": frame}))
            raw.append(
                (
                    "occlusion_while_speaking",
                    {"sound_asset_id": sound_id, "frame_index": frame},
                )
            )
        raw.append(("speaking_while_moving", {"sound_asset_id": sound_id}))
    for first, second in combinations(sounds, 2):
        raw.append(("overlapping_speech", {"sound_asset_ids": [first, second]}))
    frame_count = facts.get("time", {}).get("frame_count") if isinstance(facts.get("time"), Mapping) else None
    for instance in instances:
        if not isinstance(instance, Mapping) or not isinstance(instance.get("instance_id"), str):
            continue
        instance_id = instance["instance_id"]
        raw.extend(
            [
                ("offscreen_to_onscreen", {"target_instance_id": instance_id}),
                ("reappeared_after_occlusion", {"target_instance_id": instance_id}),
                (
                    "became_clear_after_partial_occlusion",
                    {"target_instance_id": instance_id},
                ),
            ]
        )
        if isinstance(frame_count, int) and frame_count > 0:
            for frame in range(frame_count):
                raw.append(
                    (
                        "occluder_identity",
                        {"target_instance_id": instance_id, "frame_index": frame},
                    )
                )
    deduplicated: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for question_type, selectors in raw:
        key = json.dumps([question_type, selectors], sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            deduplicated.append((question_type, selectors))
    if len(deduplicated) > 999:
        raise QuestionProtocolError("one Episode produced more than 999 controlled specs")
    return [
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": f"QS-{index:03d}",
            "question_type": question_type,
            "selectors": selectors,
        }
        for index, (question_type, selectors) in enumerate(deduplicated, start=1)
    ]


def _answer_key(evaluation: Mapping[str, Any]) -> str | None:
    answer = evaluation.get("answer")
    value = answer.get("value") if isinstance(answer, Mapping) else None
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _display_answer(answer_key: str) -> Any:
    return json.loads(answer_key)


def _find_state_sequence(frames: Sequence[Any], states: Sequence[str]) -> list[int]:
    selected: list[int] = []
    cursor = 0
    for expected in states:
        match = None
        for position in range(cursor, len(frames)):
            frame = frames[position]
            if isinstance(frame, Mapping) and frame.get("state") == expected:
                match = position
                selected.append(int(frame.get("frame_index", position)))
                break
        if match is None:
            return []
        cursor = match + 1
    return selected


def _dynamic_listener_is_observed(facts: Mapping[str, Any]) -> bool:
    listener = facts.get("listener")
    if not isinstance(listener, Mapping) or listener.get("static") is not False:
        return False
    trajectory = listener.get("sensor_rig_trajectory")
    if not isinstance(trajectory, Mapping) or trajectory.get("dynamic") is not True:
        return False
    for field in ("yaw_deg_by_frame", "positions_m_by_frame", "orientations_wxyz_by_frame"):
        values = listener.get(field)
        if _is_sequence(values) and len(values) > 1 and values[0] != values[-1]:
            return True
    return False


def _qa_case_matches_canary(
    case: Mapping[str, Any],
    *,
    question_type: str,
    target_instance_id: str,
    frame_indices: Sequence[int],
    expected_states: Sequence[str],
) -> bool:
    if case.get("question_type") != question_type or case.get("status") != "pass":
        return False
    selectors = case.get("selectors")
    evidence = case.get("evidence")
    if question_type == "occlusion_while_speaking":
        answer = case.get("answer")
        return (
            isinstance(selectors, Mapping)
            and selectors.get("frame_index") == frame_indices[0]
            and isinstance(evidence, Mapping)
            and evidence.get("instance_id") == target_instance_id
            and isinstance(answer, Mapping)
            and answer.get("value") == expected_states[0]
        )
    if isinstance(selectors, Mapping) and selectors.get("target_instance_id") == target_instance_id:
        return True
    return isinstance(evidence, Mapping) and evidence.get("instance_id") == target_instance_id


def _overlay_font(size: int) -> Any:
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _decode_rgb_frame(
    video_path: Path,
    frame_index: int,
    *,
    width: int,
    height: int,
    ffmpeg: str | Path = "ffmpeg",
) -> Any:
    """Decode exactly one RGB frame so overlays sit on the real native pixels."""

    import numpy as np

    executable = os.fspath(ffmpeg)
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise QuestionProtocolError(
            f"native RGB decode could not run {executable!r} for {video_path}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise QuestionProtocolError(
            f"native RGB decode failed for {video_path} frame {frame_index}: {detail}"
        )
    payload = completed.stdout
    expected = width * height * 3
    if not isinstance(payload, bytes) or len(payload) != expected:
        raise QuestionProtocolError(
            f"native RGB decode returned {len(payload)} bytes for {video_path} "
            f"frame {frame_index}; expected {expected}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)


def _compose_overlay_panel(
    rgb: Any,
    visible: Any,
    target: Any,
    *,
    episode_id: str,
    instance_id: str,
    frame_index: int,
    state: str,
) -> Any:
    """Blend the visible/occluded masks over the native frame.

    Pixels outside both masks keep their exact native bytes.  A reviewer must be
    able to see the occluder itself, not only the target footprint.
    """

    import numpy as np
    from PIL import Image, ImageDraw

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise QuestionProtocolError("overlay base frame must be uint8 [height,width,3]")
    if visible.shape != rgb.shape[:2] or target.shape != rgb.shape[:2]:
        raise QuestionProtocolError(
            f"canary masks {visible.shape} do not match native RGB {rgb.shape[:2]}"
        )
    height, width = rgb.shape[:2]
    occluded = target & ~visible
    pixels = rgb.astype(np.float32).copy()
    for mask, colour in ((visible, _VISIBLE_RGB), (occluded, _OCCLUDED_RGB)):
        if not mask.any():
            continue
        tint = np.asarray(colour, dtype=np.float32)
        pixels[mask] = (
            pixels[mask] * (1.0 - _OVERLAY_ALPHA) + tint * _OVERLAY_ALPHA
        )
    blended = np.clip(pixels, 0, 255).astype(np.uint8)
    # Stroke full-opacity outlines so thin structures survive the blend.
    for mask, colour in ((visible, _VISIBLE_RGB), (occluded, _OCCLUDED_RGB)):
        blended[_mask_edge(mask)] = colour
    panel = Image.fromarray(blended, mode="RGB")

    canvas = Image.new("RGB", (width, height + _OVERLAY_HEADER_HEIGHT), (10, 15, 23))
    canvas.paste(panel, (0, _OVERLAY_HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        f"{episode_id} · {instance_id} · frame {frame_index}",
        fill=(232, 238, 247),
        font=_overlay_font(17),
    )
    draw.text(
        (12, 30),
        f"state={state}  green=visible  red=occluded target footprint",
        fill=(154, 174, 199),
        font=_overlay_font(14),
    )
    return canvas


def _mask_edge(mask: Any) -> Any:
    """Return the one-pixel boundary of a boolean mask."""

    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _render_canary_overlay(
    *,
    episode: Mapping[str, Any],
    target_instance_id: str,
    frame_indices: Sequence[int],
    expected_states: Sequence[str],
    output_path: Path,
    ffmpeg: str | Path = "ffmpeg",
) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - declared runtime dependencies
        raise QuestionProtocolError("native overlay rendering needs numpy and Pillow") from error
    instances = episode["facts"].get("instances")
    matches = [
        item
        for item in instances
        if isinstance(item, Mapping) and item.get("instance_id") == target_instance_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("source_slot_id"), str):
        raise QuestionProtocolError(
            f"{episode['episode_key']} cannot map canary target {target_instance_id!r}"
        )
    slot = matches[0]["source_slot_id"]
    mask_path = Path(episode["native_files"]["pixel_masks"]["path"])
    rgb_path = Path(episode["native_files"]["native_rgb_binaural"]["path"])
    panels: list[Any] = []
    counts: list[dict[str, int]] = []
    with np.load(mask_path, allow_pickle=False) as archive:
        modal_key = f"modal_visible_{slot}"
        target_key = f"target_only_{slot}"
        if modal_key not in archive.files or target_key not in archive.files:
            raise QuestionProtocolError(
                f"{episode['episode_key']} mask archive lacks {slot} normal/target-only arrays"
            )
        modal = archive[modal_key]
        target_only = archive[target_key]
        for frame_index, expected_state in zip(frame_indices, expected_states):
            if frame_index < 0 or frame_index >= modal.shape[0]:
                raise QuestionProtocolError("canary frame lies outside native mask archive")
            visible = modal[frame_index].astype(bool)
            target = target_only[frame_index].astype(bool)
            height, width = visible.shape
            rgb = _decode_rgb_frame(
                rgb_path,
                frame_index,
                width=width,
                height=height,
                ffmpeg=ffmpeg,
            )
            panels.append(
                _compose_overlay_panel(
                    rgb,
                    visible,
                    target,
                    episode_id=episode["episode_id"],
                    instance_id=target_instance_id,
                    frame_index=frame_index,
                    state=expected_state,
                )
            )
            counts.append(
                {
                    "frame_index": frame_index,
                    "visible_pixels": int(visible.sum()),
                    "target_pixels": int(target.sum()),
                    "occluded_target_pixels": int((target & ~visible).sum()),
                }
            )
    sheet_width = max(panel.width for panel in panels)
    sheet = Image.new(
        "RGB", (sheet_width, sum(panel.height for panel in panels)), (10, 15, 23)
    )
    offset = 0
    for panel in panels:
        sheet.paste(panel, (0, offset))
        offset += panel.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return {"path": output_path.name, "frames": counts}


def _compile_cases(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = enumerate_episode_specs(episode["facts"], episode["event_sound_bindings"])
    evaluations = evaluate_question_specs(
        specs,
        facts=episode["facts"],
        asset_registry=episode["asset_registry"],
        sound_registry=episode["sound_registry"],
        event_sound_bindings=episode["event_sound_bindings"],
    )
    cases = []
    for spec, evaluation in zip(specs, evaluations):
        cases.append(
            {
                "case_id": f"{episode['episode_key']}::{spec['spec_id']}",
                "episode_key": episode["episode_key"],
                "episode_id": episode["episode_id"],
                "spec_id": spec["spec_id"],
                "question_type": spec["question_type"],
                "selectors": spec["selectors"],
                "status": evaluation["status"],
                "question": evaluation.get("question"),
                "answer": evaluation.get("answer"),
                "evidence": evaluation.get("evidence", {}),
                "reason": evaluation.get("reason"),
            }
        )
    return cases


def _coverage_by_type(
    protocol: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for definition in protocol["question_types"]:
        question_type = definition["question_type"]
        matching = [case for case in cases if case["question_type"] == question_type]
        passed = [case for case in matching if case["status"] == "pass"]
        answers = sorted(
            {key for case in passed if (key := _answer_key(case)) is not None}
        )
        balance = definition["paper_balance"]
        missing_answers: list[Any] = []
        if "required_answers" in balance:
            present = {_display_answer(key) for key in answers}
            missing_answers = [
                value for value in balance["required_answers"] if value not in present
            ]
            balance_status = "pass" if not missing_answers else "gap"
            balance_detail = {
                "required_answers": balance["required_answers"],
                "missing_answers": missing_answers,
            }
        else:
            required = balance["minimum_distinct_answers"]
            balance_status = "pass" if len(answers) >= required else "gap"
            balance_detail = {
                "minimum_distinct_answers": required,
                "missing_distinct_answer_count": max(0, required - len(answers)),
            }
        minimum_status = "pass" if passed else "gap"
        rejection_counts = Counter(case["status"] for case in matching if case["status"] != "pass")
        result.append(
            {
                "catalog_index": definition["catalog_index"],
                "question_type": question_type,
                "name_zh": definition["name_zh"],
                "historical_origin": definition["historical_origin"],
                "original_0807_order": definition["original_0807_order"],
                "candidate_case_count": len(matching),
                "native_pass_case_count": len(passed),
                "native_pass_episode_keys": sorted({case["episode_key"] for case in passed}),
                "accepted_case_ids": [case["case_id"] for case in passed[:12]],
                "observed_answers": [_display_answer(key) for key in answers],
                "minimum_status": minimum_status,
                "paper_balance_status": balance_status,
                "paper_balance": balance_detail,
                "nonpass_status_counts": dict(sorted(rejection_counts.items())),
                "minimal_native_supplement_zh": (
                    None
                    if balance_status == "pass"
                    else protocol["paper_gap_scene_templates"][question_type]
                ),
            }
        )
    return result


def _report_markdown(coverage: Mapping[str, Any]) -> str:
    lines = [
        "# QuestionSpec 论文评测协议与原生覆盖审计",
        "",
        f"- 最低协议：**{coverage['minimum_protocol_status'].upper()}**（12 类各至少一个当前代码重算的 native pass）",
        f"- 五类视觉 canary：**{coverage['visual_canary_status'].upper()}**",
        f"- 论文答案平衡：**{coverage['paper_balance_status'].upper()}**",
        "- 边界：历史 question_evaluations.json 不作为 GT；所有答案均由当前 QuestionSpec 对保留 Facts/registry 重新计算。",
        "",
        "## 编号与 0807 语义来源",
        "",
        "当前 catalog 编号保持 API 兼容。QS-012（外貌→说了什么）来自 0807 原始第 2 类；真正的 0807 后扩展是 QS-009 与 QS-011。",
        "",
        "| Catalog | 类型 | 0807 顺序/来源 | native pass | 观测答案 | 最低门 | 论文平衡 |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for item in coverage["question_type_coverage"]:
        origin = (
            f"#{item['original_0807_order']}"
            if item["original_0807_order"] is not None
            else "扩展"
        )
        answers = ", ".join(str(value) for value in item["observed_answers"]) or "—"
        lines.append(
            f"| {item['catalog_index']} | `{item['question_type']}` / {item['name_zh']} | {origin} | "
            f"{item['native_pass_case_count']} | {answers} | {item['minimum_status']} | {item['paper_balance_status']} |"
        )
    lines.extend(
        [
            "",
            "## 五类原生像素 canary",
            "",
            "| Canary | Episode | 目标 | 状态序列/帧 | 当前 QA | Overlay | 状态 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in coverage["visual_canaries"]:
        transitions = " → ".join(
            f"{state}@{frame}"
            for state, frame in zip(item["expected_state_sequence"], item["frame_indices"])
        )
        lines.append(
            f"| `{item['canary_id']}` | `{item['episode_key']}` | `{item['target_instance_id']}` | "
            f"{transitions} | `{item['qa_case_id']}` | `{item['overlay_path']}` | {item['status']} |"
        )
    lines.extend(["", "## 仍缺的论文平衡场景", ""])
    gaps = [
        item
        for item in coverage["question_type_coverage"]
        if item["paper_balance_status"] != "pass"
    ]
    if not gaps:
        lines.append("无。")
    else:
        for item in gaps:
            detail = item["paper_balance"]
            if "missing_answers" in detail:
                missing = ", ".join(map(str, detail["missing_answers"]))
            else:
                missing = f"还需 {detail['missing_distinct_answer_count']} 个不同答案"
            lines.extend(
                [
                    f"- `{item['question_type']}`：{missing}。",
                    f"  最小 native 补充：{item['minimal_native_supplement_zh']}",
                ]
            )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "最低 0807 首批门与论文平衡门分开报告。最低门通过不等于答案分布已经适合最终论文统计；所有 gap 都保留为显式场景需求，不以合成 Facts 或手工答案填补。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_delivery_manifest(output: Path) -> dict[str, Any]:
    inventory = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "manifest.json":
            continue
        inventory.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return {
        "schema": DELIVERY_SCHEMA,
        "status": "pass",
        "claim_boundary": "Reproducible current-code audit over retained native evidence; no synthetic Episode admission",
        "files": inventory,
    }


def compile_question_protocol_coverage(
    *,
    repository: Path,
    protocol_path: Path,
    episode_catalog_path: Path,
    output: Path,
    ffmpeg: str | Path = "ffmpeg",
) -> dict[str, Any]:
    """Atomically compile current-code native coverage and human overlays."""

    repository = repository.resolve()
    protocol = _load_json(protocol_path)
    episode_catalog = _load_json(episode_catalog_path)
    validate_protocol(protocol)
    validate_episode_catalog(episode_catalog, protocol)
    if output.exists():
        raise QuestionProtocolError(f"output already exists (no clobber): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        episodes = [
            _load_native_episode(repository, record)
            for record in episode_catalog["episodes"]
        ]
        by_key = {episode["episode_key"]: episode for episode in episodes}
        cases = [case for episode in episodes for case in _compile_cases(episode)]
        type_coverage = _coverage_by_type(protocol, cases)
        canary_results = []
        for canary in episode_catalog["visual_canaries"]:
            episode = by_key[canary["episode_key"]]
            truth = episode["facts"].get("visibility", {}).get("pixel_truth", {})
            per_instance = truth.get("per_instance") if isinstance(truth, Mapping) else None
            target = per_instance.get(canary["target_instance_id"]) if isinstance(per_instance, Mapping) else None
            frames = target.get("frames") if isinstance(target, Mapping) else None
            if not _is_sequence(frames):
                raise QuestionProtocolError(f"{canary['canary_id']} target pixel truth is missing")
            selected = _find_state_sequence(frames, canary["expected_state_sequence"])
            if not selected:
                raise QuestionProtocolError(f"{canary['canary_id']} state sequence is not observed")
            if canary["require_dynamic_listener"] and not _dynamic_listener_is_observed(
                episode["facts"]
            ):
                raise QuestionProtocolError(f"{canary['canary_id']} lacks observed Listener motion")
            qa_matches = [
                case
                for case in cases
                if case["episode_key"] == episode["episode_key"]
                and _qa_case_matches_canary(
                    case,
                    question_type=canary["qa_question_type"],
                    target_instance_id=canary["target_instance_id"],
                    frame_indices=selected,
                    expected_states=canary["expected_state_sequence"],
                )
            ]
            if not qa_matches:
                raise QuestionProtocolError(f"{canary['canary_id']} lacks a current-code passing QA")
            overlay_relative = Path("canary_overlays") / f"{canary['canary_id']}.png"
            overlay = _render_canary_overlay(
                episode=episode,
                target_instance_id=canary["target_instance_id"],
                frame_indices=selected,
                expected_states=canary["expected_state_sequence"],
                output_path=temporary / overlay_relative,
                ffmpeg=ffmpeg,
            )
            canary_results.append(
                {
                    "canary_id": canary["canary_id"],
                    "episode_key": episode["episode_key"],
                    "episode_id": episode["episode_id"],
                    "target_instance_id": canary["target_instance_id"],
                    "expected_state_sequence": canary["expected_state_sequence"],
                    "frame_indices": selected,
                    "qa_case_id": qa_matches[0]["case_id"],
                    "qa_question_type": canary["qa_question_type"],
                    "dynamic_listener_observed": _dynamic_listener_is_observed(episode["facts"]),
                    "native_roles_verified": list(_NATIVE_ROLES),
                    "overlay_path": overlay_relative.as_posix(),
                    "overlay_frame_counts": overlay["frames"],
                    "status": "pass",
                }
            )
        minimum_status = (
            "pass" if all(item["minimum_status"] == "pass" for item in type_coverage) else "gap"
        )
        balance_status = (
            "pass"
            if all(item["paper_balance_status"] == "pass" for item in type_coverage)
            else "gap"
        )
        coverage = {
            "schema": COVERAGE_SCHEMA,
            "protocol_id": protocol["protocol_id"],
            "episode_catalog_id": episode_catalog["catalog_id"],
            "claim_boundary": episode_catalog["claim_boundary"],
            "minimum_protocol_status": minimum_status,
            "visual_canary_status": "pass",
            "paper_balance_status": balance_status,
            "episode_count": len(episodes),
            "candidate_case_count": len(cases),
            "question_type_coverage": type_coverage,
            "visual_canaries": canary_results,
            "episodes": [
                {
                    "episode_key": episode["episode_key"],
                    "episode_id": episode["episode_id"],
                    "status": "pass",
                    "native_roles_verified": list(_NATIVE_ROLES),
                }
                for episode in episodes
            ],
            "cases": cases,
        }
        _write_json(temporary / "protocol_snapshot.json", protocol)
        _write_json(temporary / "coverage.json", coverage)
        (temporary / "report.md").write_text(_report_markdown(coverage), encoding="utf-8")
        _write_json(temporary / "manifest.json", _build_delivery_manifest(temporary))
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_compiled_delivery(output)
    return coverage


def validate_compiled_delivery(output: Path, *, require_paper_ready: bool = False) -> dict[str, Any]:
    """Validate a compiled delivery without consulting mutable source inputs."""

    manifest = _load_json(output / "manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != DELIVERY_SCHEMA:
        raise QuestionProtocolError("compiled delivery manifest schema is invalid")
    if manifest.get("status") != "pass" or not _is_sequence(manifest.get("files")):
        raise QuestionProtocolError("compiled delivery manifest is not pass")
    declared_paths: set[str] = set()
    for record in manifest["files"]:
        if not isinstance(record, Mapping):
            raise QuestionProtocolError("compiled file inventory records must be objects")
        relative = record.get("path")
        path = _safe_repository_path(output, relative, field="delivery file")
        if relative in declared_paths or not path.is_file():
            raise QuestionProtocolError(f"compiled delivery file is missing or duplicated: {relative}")
        declared_paths.add(relative)
        if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
            raise QuestionProtocolError(f"compiled delivery file bytes changed: {relative}")
    actual_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != declared_paths:
        raise QuestionProtocolError("compiled delivery contains undeclared or missing files")
    protocol = _load_json(output / "protocol_snapshot.json")
    validate_protocol(protocol)
    coverage = _load_json(output / "coverage.json")
    if not isinstance(coverage, Mapping) or coverage.get("schema") != COVERAGE_SCHEMA:
        raise QuestionProtocolError("compiled coverage schema is invalid")
    if coverage.get("minimum_protocol_status") != "pass":
        raise QuestionProtocolError("compiled delivery does not satisfy the 12-type minimum")
    if coverage.get("visual_canary_status") != "pass":
        raise QuestionProtocolError("compiled delivery does not satisfy all five canaries")
    types = coverage.get("question_type_coverage")
    canaries = coverage.get("visual_canaries")
    if not _is_sequence(types) or len(types) != 12:
        raise QuestionProtocolError("compiled delivery does not contain 12 type rows")
    if not _is_sequence(canaries) or len(canaries) != 5:
        raise QuestionProtocolError("compiled delivery does not contain five canaries")
    if require_paper_ready and coverage.get("paper_balance_status") != "pass":
        raise QuestionProtocolError("compiled delivery still has paper-balance gaps")
    return dict(coverage)
