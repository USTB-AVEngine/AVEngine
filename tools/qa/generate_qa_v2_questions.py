#!/usr/bin/env python3
"""Compile constraint-driven QA v2 questions and report answer-key balance.

Production is strict by default: each point must resolve through a viewpoint
bank and every binary type must be ``50/50±1`` by ``options.index(answer)``.
Historical/global-pose auditing requires the explicit ``--report-only`` mode;
``--require-balanced-answer-keys`` remains as a compatibility spelling for a
strict balance check inside that audit mode.

T5 numeric DoA and formal off-screen questions are opt-in until their required
per-point UE readback and native target-only pixel evidence roots are supplied.
Refusal questions are selected deterministically so they occupy 10--15 percent
of the final batch without deleting any ordinary question.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avengine.qa.answer_balance import (
    AnswerBalanceError,
    build_answer_balance_report,
    require_balanced_answer_keys,
)
from avengine.m6.audio_program import AudioProgramError, load_audio_program
from avengine.qa.extended_questions import (
    ExtendedQuestionEvidenceError,
    T5_FULL_CREDIT_DEG,
    T5_PARTIAL_CREDIT_DEG,
    choose_refusal_quota,
    event_frame_index,
    event_frame_window,
    load_native_pixel_visibility,
    load_native_point_evidence,
    require_native_out_of_view,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PROGRAMS_DIR = REPOSITORY / "examples/m7/current_apartment/audio_programs/qa_v2"
DEFAULT_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
DEFAULT_CAMERA_POSITION_UE_CM = (-70.0, 65.0, 147.1)
DEFAULT_CAMERA_YAW_DEG = -145.0
DEFAULT_SIDE_MARGIN = 0.1
DEFAULT_DISTANCE_RATIO_MIN = 1.2
_MODALITY_NAMES = frozenset({"audio", "video"})


class QuestionGenerationError(ValueError):
    """Question inputs cannot support the requested production mode."""


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite_triplet(value: Any, *, owner: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise QuestionGenerationError(f"{owner} must be a three-number list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise QuestionGenerationError(f"{owner} must contain only finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise QuestionGenerationError(f"{owner} must contain only finite numbers")
        result.append(number)
    return result[0], result[1], result[2]


def _finite_yaw(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuestionGenerationError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise QuestionGenerationError(f"{owner} must be a finite number")
    return result


def load_viewpoint_bank(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the per-point UE camera authority used by production generation."""

    document = _load_json(path)
    viewpoints = document.get("viewpoints") if isinstance(document, Mapping) else None
    if not isinstance(viewpoints, list):
        raise QuestionGenerationError("viewpoint bank must contain a viewpoints list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(viewpoints):
        if not isinstance(record, Mapping):
            raise QuestionGenerationError(
                f"viewpoint bank row {index} must be an object"
            )
        viewpoint_id = record.get("viewpoint_id")
        if not isinstance(viewpoint_id, str) or not viewpoint_id:
            raise QuestionGenerationError(
                f"viewpoint bank row {index} has no non-empty viewpoint_id"
            )
        if viewpoint_id in by_id:
            raise QuestionGenerationError(
                f"duplicate viewpoint_id in bank: {viewpoint_id}"
            )
        camera = record.get("camera")
        if not isinstance(camera, Mapping):
            raise QuestionGenerationError(
                f"viewpoint {viewpoint_id} has no camera object"
            )
        position = _finite_triplet(
            camera.get("ue_location_cm"),
            owner=f"viewpoint {viewpoint_id} camera.ue_location_cm",
        )
        yaw_value = camera.get("ue_yaw_deg")
        if yaw_value is None:
            yaw_value = camera.get("yaw")
        yaw = _finite_yaw(
            yaw_value,
            owner=f"viewpoint {viewpoint_id} camera yaw",
        )
        by_id[viewpoint_id] = {
            "position_ue_cm": position,
            "yaw_ue_deg": yaw,
        }
    if not by_id:
        raise QuestionGenerationError("viewpoint bank contains no viewpoints")
    return by_id


def resolve_point_viewpoint(
    spec: Mapping[str, Any],
    *,
    viewpoint_bank: Mapping[str, Mapping[str, Any]] | None,
    report_only: bool,
    legacy_camera_position_ue_cm: Sequence[float],
    legacy_camera_yaw_deg: float,
) -> tuple[str | None, Sequence[float], float, str]:
    """Resolve one point's camera, failing closed outside report-only mode."""

    raw_id = spec.get("viewpoint_id")
    viewpoint_id = raw_id if isinstance(raw_id, str) and raw_id else None
    if viewpoint_bank is not None and viewpoint_id is not None:
        try:
            camera = viewpoint_bank[viewpoint_id]
        except KeyError as exc:
            raise QuestionGenerationError(
                f"spec viewpoint_id is absent from viewpoint bank: {viewpoint_id}"
            ) from exc
        return (
            viewpoint_id,
            camera["position_ue_cm"],
            float(camera["yaw_ue_deg"]),
            "viewpoint_bank",
        )
    if not report_only:
        if viewpoint_bank is None:
            raise QuestionGenerationError(
                "production generation requires an explicit --viewpoint-bank"
            )
        raise QuestionGenerationError(
            "production generation requires every spec.json to carry a non-empty viewpoint_id"
        )
    return (
        viewpoint_id,
        legacy_camera_position_ue_cm,
        legacy_camera_yaw_deg,
        "legacy_global_pose_report_only",
    )


def appearance(
    asset_id: str, assets: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    rec = assets[asset_id]
    attrs = rec.get("realized_attributes", {})
    if "top_color" in attrs:
        color = attrs["top_color"]
        zh = {"blue": "蓝色", "green": "绿色", "burgundy": "酒红色"}[color]
        return {
            "kind": "top_color",
            "value": color,
            "en": f"a {color} top",
            "zh": f"{zh}上衣",
            "label_en": color,
            "label_zh": zh,
        }
    breed = rec["identity"].get("breed_id") or rec["asset_id"]
    coat = {
        "border_collie": ("black_white", "black and white", "黑白色"),
        "labrador_retriever": ("yellow", "yellow", "黄色"),
    }[breed]
    return {
        "kind": "coat",
        "value": coat[0],
        "en": f"a {coat[1]} coat",
        "zh": f"{coat[2]}毛",
        "label_en": coat[1],
        "label_zh": coat[2],
    }


def side_of(
    point_xy: Sequence[float],
    *,
    camera_position_ue_cm: Sequence[float],
    camera_yaw_deg: float,
) -> tuple[str, float]:
    """Return calibrated image side and normalized lateral magnitude."""

    theta = math.radians(camera_yaw_deg)
    fx, fy = math.cos(theta), math.sin(theta)
    dx = float(point_xy[0]) - float(camera_position_ue_cm[0])
    dy = float(point_xy[1]) - float(camera_position_ue_cm[1])
    norm = math.hypot(dx, dy)
    cross = (fx * dy - fy * dx) / max(norm, 1e-9)
    return ("right" if cross > 0 else "left"), abs(cross)


def combo_program(
    spec: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]]
) -> str:
    if spec.get("program_id"):
        return str(spec["program_id"])

    def short(asset_id: str) -> str:
        asset = assets[asset_id]
        if "top_color" in asset.get("realized_attributes", {}):
            return str(asset["realized_attributes"]["top_color"])
        return {
            "border_collie": "collie",
            "labrador_retriever": "labrador",
        }[asset["identity"]["breed_id"]]

    source1, source2 = short(spec["source1_asset"]), short(spec["source2_asset"])
    kind = "two_human" if spec["pair_kind"] == "human" else "two_dog"
    return f"qa_v2_{kind}_{source1}_{source2}_turn_taking_v1"


def load_point_audio_program(
    point_path: Path,
    spec: Mapping[str, Any],
    *,
    programs_dir: str | Path,
    assets: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Path, str]:
    """Resolve a materialized point program, with legacy fallback for old inputs."""

    declared = spec.get("audio_program_path")
    if declared is None:
        path = Path(programs_dir) / f"{combo_program(spec, assets)}.json"
        return _load_json(path), path, "shared_base_program_legacy_input"
    if not isinstance(declared, str) or not declared:
        raise QuestionGenerationError("audio_program_path must be a relative path")
    relative = Path(declared)
    if relative.is_absolute():
        raise QuestionGenerationError(
            "audio_program_path must be relative to its point"
        )
    point_root = point_path.resolve(strict=True)
    try:
        path = (point_root / relative).resolve(strict=True)
        path.relative_to(point_root)
    except (OSError, ValueError) as exc:
        raise QuestionGenerationError(
            f"{point_path.name}: audio_program_path escapes or is missing"
        ) from exc
    try:
        program = load_audio_program(path)
    except (AudioProgramError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise QuestionGenerationError(
            f"{point_path.name}: materialized AudioProgram is invalid: {exc}"
        ) from exc
    schedule = spec.get("audio_schedule")
    if not isinstance(schedule, Mapping):
        raise QuestionGenerationError(
            f"{point_path.name}: materialized AudioProgram requires audio_schedule"
        )
    if schedule.get("base_program_id") != combo_program(spec, assets):
        raise QuestionGenerationError(
            f"{point_path.name}: audio_schedule base_program_id mismatch"
        )
    if schedule.get("materialized_program_id") != program.get("program_id"):
        raise QuestionGenerationError(
            f"{point_path.name}: audio_schedule materialized_program_id mismatch"
        )
    starts = [event["start_sample"] for event in program["events"]]
    ends = [event["end_sample_exclusive"] for event in program["events"]]
    if (
        schedule.get("event_start_samples") != starts
        or schedule.get("event_end_samples_exclusive") != ends
    ):
        raise QuestionGenerationError(
            f"{point_path.name}: audio_schedule event windows mismatch"
        )
    return program, path, "point_materialized_audio_program"


def _normalize_modality_list(value: Any, *, owner: str) -> set[str]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise QuestionGenerationError(
            f"{owner} must be a list of unique modality names"
        )
    normalized = {str(item) for item in value}
    unknown = normalized - _MODALITY_NAMES
    if unknown:
        raise QuestionGenerationError(
            f"{owner} contains unsupported modalities: {sorted(unknown)}"
        )
    return normalized


def _declared_modality_sets(
    record: Mapping[str, Any], *, owner: str
) -> tuple[set[str], set[str]] | None:
    provided_present = "provided_modalities" in record
    withheld_present = "withheld_modalities" in record
    input_present = "input_modalities" in record
    if not (provided_present or withheld_present or input_present):
        return None
    provided = (
        _normalize_modality_list(
            record.get("provided_modalities"), owner=f"{owner}.provided_modalities"
        )
        if provided_present
        else None
    )
    withheld = (
        _normalize_modality_list(
            record.get("withheld_modalities"), owner=f"{owner}.withheld_modalities"
        )
        if withheld_present
        else None
    )
    if provided is not None and withheld is not None and provided & withheld:
        raise QuestionGenerationError(
            f"{owner} provided_modalities and withheld_modalities overlap"
        )
    input_modalities = (
        _normalize_modality_list(
            record.get("input_modalities"), owner=f"{owner}.input_modalities"
        )
        if input_present
        else None
    )
    if input_modalities is not None:
        if provided is not None and provided != input_modalities:
            raise QuestionGenerationError(
                f"{owner} input_modalities conflicts with provided_modalities"
            )
        if withheld is not None and withheld != _MODALITY_NAMES - input_modalities:
            raise QuestionGenerationError(
                f"{owner} input_modalities conflicts with withheld_modalities"
            )
        provided = input_modalities
        withheld = _MODALITY_NAMES - input_modalities
    if provided is None:
        provided = _MODALITY_NAMES - (withheld or set())
    if withheld is None:
        withheld = _MODALITY_NAMES - provided
    if provided | withheld != _MODALITY_NAMES or provided & withheld:
        raise QuestionGenerationError(
            f"{owner} must account for both audio and video exactly once"
        )
    return provided, withheld


def _declared_modality_mask(
    record: Mapping[str, Any], *, owner: str
) -> dict[str, bool] | None:
    if "modality_mask" not in record:
        return None
    raw = record["modality_mask"]
    if not isinstance(raw, Mapping) or set(raw) != _MODALITY_NAMES:
        raise QuestionGenerationError(
            f"{owner}.modality_mask must contain exactly audio and video"
        )
    if any(isinstance(raw[name], bool) is False for name in _MODALITY_NAMES):
        raise QuestionGenerationError(
            f"{owner}.modality_mask audio/video values must be booleans"
        )
    return {name: bool(raw[name]) for name in _MODALITY_NAMES}


def resolve_modality_mask(
    row: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    point_id: str,
    audio_available: bool,
) -> dict[str, bool]:
    """Derive one row mask from explicit modality facts and validate it."""

    row_owner = f"{point_id} question {row.get('type_id', '<unknown>')}"
    row_sets = _declared_modality_sets(row, owner=row_owner)
    spec_sets = _declared_modality_sets(spec, owner=f"{point_id} spec")
    spec_mask = _declared_modality_mask(spec, owner=f"{point_id} spec")
    if spec_sets is not None and spec_mask is not None:
        spec_provided, _spec_withheld = spec_sets
        expected_spec_mask = {name: name in spec_provided for name in _MODALITY_NAMES}
        if spec_mask != expected_spec_mask:
            raise QuestionGenerationError(
                f"{point_id} spec modality_mask conflicts with modality declaration"
            )
    if row_sets is not None and spec_sets is not None:
        row_provided, _row_withheld = row_sets
        spec_provided, _spec_withheld = spec_sets
        if not row_provided.issubset(spec_provided):
            raise QuestionGenerationError(
                f"{row_owner} modality declaration conflicts with point spec"
            )
    if row_sets is not None and spec_mask is not None:
        row_provided, _row_withheld = row_sets
        spec_provided = {name for name, enabled in spec_mask.items() if enabled}
        if not row_provided.issubset(spec_provided):
            raise QuestionGenerationError(
                f"{row_owner} modality declaration conflicts with point spec"
            )
    declared_sets = row_sets or spec_sets
    if declared_sets is None and spec_mask is not None:
        declared_sets = (
            {name for name, enabled in spec_mask.items() if enabled},
            {name for name, enabled in spec_mask.items() if not enabled},
        )
    explicit_mask = _declared_modality_mask(row, owner=row_owner)
    if declared_sets is None and explicit_mask is None:
        raise QuestionGenerationError(
            f"{point_id}: modality_mask cannot be derived without explicit "
            "provided/withheld_modalities or input_modalities"
        )
    if declared_sets is None:
        mask = explicit_mask
    else:
        provided, _withheld = declared_sets
        mask = {name: name in provided for name in _MODALITY_NAMES}
    assert mask is not None
    if explicit_mask is not None and explicit_mask != mask:
        raise QuestionGenerationError(
            f"{row_owner} modality_mask conflicts with modality declaration"
        )
    if audio_available is not True and mask["audio"]:
        raise QuestionGenerationError(
            f"{point_id}: modality_mask declares audio without an audio input"
        )
    expectation = row.get("modality_expectation")
    if expectation in {"dual_required", "vision_dominant_control"} and mask != {
        "audio": True,
        "video": True,
    }:
        raise QuestionGenerationError(
            f"{row_owner} requires both audio and video modalities"
        )
    if expectation == "must_refuse_audio_only_visual_attribute" and mask != {
        "audio": True,
        "video": False,
    }:
        raise QuestionGenerationError(
            f"{row_owner} refusal must mask video while retaining audio"
        )
    if row.get("refusal_gold") is True and mask != {
        "audio": True,
        "video": False,
    }:
        raise QuestionGenerationError(
            f"{row_owner} refusal modality mask is not audio-only"
        )
    return mask


def actor_states(
    timeline: Mapping[str, Any], frame: int
) -> dict[str, Mapping[str, Any]]:
    states = timeline["frames"][frame]["actor_states"]
    by_slot: dict[str, Mapping[str, Any]] = {}
    for state in states:
        key = state.get("source_slot_id") or state.get("actor_id")
        slot = "source1" if "1" in str(key) else "source2"
        if slot in by_slot:
            raise QuestionGenerationError(
                f"timeline frame {frame} repeats actor state for {slot}"
            )
        by_slot[slot] = state
    if set(by_slot) != {"source1", "source2"}:
        raise QuestionGenerationError(
            f"timeline frame {frame} must contain source1 and source2 actor states"
        )
    return by_slot


def actor_positions(
    timeline: Mapping[str, Any], frame: int
) -> dict[str, Sequence[float]]:
    return {
        slot: state["translation_ue_cm"][:2]
        for slot, state in actor_states(timeline, frame).items()
    }


def timeline_motion_answer(
    timeline: Mapping[str, Any],
    *,
    source_slot: str,
    frame_indices: Sequence[int],
    owner: str,
) -> tuple[str, tuple[str, ...]]:
    """Read moving/still truth from authored Timeline action IDs."""

    if not frame_indices:
        raise QuestionGenerationError(f"{owner}: event touches no Timeline frames")
    action_ids = tuple(
        str(actor_states(timeline, frame_index)[source_slot].get("action_id"))
        for frame_index in frame_indices
    )
    unique = set(action_ids)
    if unique == {"walk"}:
        return "moving", action_ids
    if unique == {"idle"}:
        return "staying still", action_ids
    raise QuestionGenerationError(
        f"{owner}: Timeline action must be uniformly walk or idle, got {sorted(unique)}"
    )


def mover_slots(motion: str) -> set[str]:
    return {
        "s1_moving_s2_static": {"source1"},
        "s1_static_s2_moving": {"source2"},
        "both_moving": {"source1", "source2"},
        "both_static": set(),
    }[motion]


def generate_question_document(
    *,
    inputs_root: str | Path,
    programs_dir: str | Path = DEFAULT_PROGRAMS_DIR,
    registry_path: str | Path = DEFAULT_REGISTRY,
    viewpoint_bank_path: str | Path | None = None,
    report_only: bool = False,
    camera_position_ue_cm: Sequence[float] = DEFAULT_CAMERA_POSITION_UE_CM,
    camera_yaw_deg: float = DEFAULT_CAMERA_YAW_DEG,
    side_margin: float = DEFAULT_SIDE_MARGIN,
    distance_ratio_min: float = DEFAULT_DISTANCE_RATIO_MIN,
    native_capture_root: str | Path | None = None,
    pixel_visibility_root: str | Path | None = None,
    enable_t5_doa: bool = False,
    enable_offscreen: bool = False,
    enable_refusal: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compile questions without writing files.

    Production mode resolves every point through ``viewpoint_bank_path`` and
    its ``spec.viewpoint_id``. The global camera arguments are a legacy audit
    fallback and are reachable only with ``report_only=True``.
    """

    if len(camera_position_ue_cm) != 3:
        raise ValueError("camera_position_ue_cm must contain exactly three values")
    if (enable_t5_doa or enable_offscreen) and native_capture_root is None:
        raise QuestionGenerationError(
            "T5 DoA and off-screen questions require --native-capture-root"
        )
    if enable_offscreen and pixel_visibility_root is None:
        raise QuestionGenerationError(
            "off-screen questions require --pixel-visibility-root"
        )
    registry = _load_json(registry_path)
    assets = {asset["asset_id"]: asset for asset in registry["assets"]}
    viewpoint_bank = (
        load_viewpoint_bank(viewpoint_bank_path)
        if viewpoint_bank_path is not None
        else None
    )
    questions: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    refusal_candidates: list[dict[str, Any]] = []
    resolved_viewpoints: dict[str, dict[str, Any]] = {}
    resolved_audio_programs: dict[str, dict[str, Any]] = {}
    extended_evidence: dict[str, dict[str, Any]] = {}

    def emit(row: dict[str, Any], ok: bool, reason: str | None = None) -> None:
        row = dict(row)
        row["modality_mask"] = resolve_modality_mask(
            row,
            spec=spec,
            point_id=point_id,
            audio_available=audio_input_available,
        )
        if ok:
            questions.append(row)
        else:
            refused.append(dict(row, refusal_reason=reason))

    for point_path in sorted(Path(inputs_root).iterdir()):
        spec_path = point_path / "spec.json"
        if not spec_path.is_file():
            continue
        point_id = point_path.name
        spec = _load_json(spec_path)
        (
            viewpoint_id,
            point_camera_position_ue_cm,
            point_camera_yaw_deg,
            camera_authority,
        ) = resolve_point_viewpoint(
            spec,
            viewpoint_bank=viewpoint_bank,
            report_only=report_only,
            legacy_camera_position_ue_cm=camera_position_ue_cm,
            legacy_camera_yaw_deg=camera_yaw_deg,
        )
        resolved_viewpoints[point_id] = {
            "viewpoint_id": viewpoint_id,
            "camera_authority": camera_authority,
            "position_ue_cm": [float(value) for value in point_camera_position_ue_cm],
            "yaw_ue_deg": float(point_camera_yaw_deg),
        }
        timeline = _load_json(point_path / "timeline.json")
        program, program_path, program_authority = load_point_audio_program(
            point_path,
            spec,
            programs_dir=programs_dir,
            assets=assets,
        )
        if not isinstance(program, Mapping):
            raise QuestionGenerationError(
                f"{point_id}: audio program must be an object"
            )
        resolved_audio_programs[point_id] = {
            "program_id": program.get("program_id"),
            "path": str(program_path),
            "authority": program_authority,
        }
        endpoints = program.get("candidate_source_endpoint_ids")
        if (
            not isinstance(endpoints, list)
            or len(endpoints) != 2
            or any(not isinstance(value, str) or not value for value in endpoints)
            or endpoints[0] == endpoints[1]
        ):
            raise QuestionGenerationError(
                f"{point_id}: audio program must declare two distinct source endpoints"
            )
        endpoint_to_slot = {
            endpoints[0]: "source1",
            endpoints[1]: "source2",
        }
        raw_events = program.get("events")
        if not isinstance(raw_events, list) or len(raw_events) < 2:
            raise QuestionGenerationError(
                f"{point_id}: audio program needs at least two events"
            )
        events = []
        for event_index, event in enumerate(raw_events):
            if not isinstance(event, Mapping):
                raise QuestionGenerationError(
                    f"{point_id}: audio event {event_index} must be an object"
                )
            endpoint = event.get("source_endpoint_id")
            if not isinstance(endpoint, str) or endpoint not in endpoint_to_slot:
                raise QuestionGenerationError(
                    f"{point_id}: audio event {event_index} uses an unknown endpoint"
                )
            try:
                event_frame_index(event.get("start_sample"))
            except ExtendedQuestionEvidenceError as exc:
                raise QuestionGenerationError(
                    f"{point_id}: audio event {event_index} has invalid timing"
                ) from exc
            events.append(event)
        audio_input_available = bool(events)
        events.sort(key=lambda event: event["start_sample"])
        first = events[0]
        second = events[1]
        slot_a = endpoint_to_slot[first["source_endpoint_id"]]
        slot_b = "source2" if slot_a == "source1" else "source1"
        second_slot = endpoint_to_slot[second["source_endpoint_id"]]
        if second_slot != slot_b:
            raise QuestionGenerationError(
                f"{point_id}: the second event must belong to the other source slot"
            )
        first_frame = event_frame_index(first["start_sample"])
        second_frame = event_frame_index(second["start_sample"])
        movers = mover_slots(spec["motion_case"])
        first_motion_answer, first_action_ids = timeline_motion_answer(
            timeline,
            source_slot=slot_a,
            frame_indices=(first_frame,),
            owner=f"{point_id} first event",
        )
        second_event_frames = event_frame_window(
            second["start_sample"], second["end_sample_exclusive"]
        )
        second_motion_answer, second_action_ids = timeline_motion_answer(
            timeline,
            source_slot=slot_b,
            frame_indices=second_event_frames,
            owner=f"{point_id} second event",
        )
        expected_first = "moving" if slot_a in movers else "staying still"
        expected_second = "moving" if slot_b in movers else "staying still"
        if (
            first_motion_answer != expected_first
            or second_motion_answer != expected_second
        ):
            raise QuestionGenerationError(
                f"{point_id}: motion_case disagrees with authored Timeline action IDs"
            )
        appearances = {
            slot: appearance(spec[f"{slot}_asset"], assets)
            for slot in ("source1", "source2")
        }
        noun_en = "person" if spec["pair_kind"] == "human" else "dog"
        noun_zh = "人" if spec["pair_kind"] == "human" else "狗"
        verb_en = "spoke" if spec["pair_kind"] == "human" else "barked"
        verb_zh = "说话" if spec["pair_kind"] == "human" else "叫"
        base = {
            "point_id": point_id,
            "pair_kind": spec["pair_kind"],
            "motion_case": spec["motion_case"],
            "twin_of": spec["twin_of"],
            "offscreen_candidate": spec["offscreen_candidate"],
            "first_slot": slot_a,
            "first_frame": first_frame,
            "second_frame": second_frame,
            "first_event_start_sample": first["start_sample"],
            "second_event_start_sample": second["start_sample"],
            "first_second_onset_interval_samples": (
                second["start_sample"] - first["start_sample"]
            ),
            "audio_program_id": program.get("program_id"),
            "audio_program_authority": program_authority,
            "first_event_timeline_action_ids": list(first_action_ids),
            "second_event_frame_window": list(second_event_frames),
            "second_event_timeline_action_ids": list(second_action_ids),
            "motion_gold_authority": "authored_timeline_action_id",
            "viewpoint_id": viewpoint_id,
        }

        native_evidence = None
        if enable_t5_doa or (enable_offscreen and spec["offscreen_candidate"]):
            native_evidence = load_native_point_evidence(
                native_capture_root,
                point_id=point_id,
                expected_camera_position_ue_cm=point_camera_position_ue_cm,
                expected_camera_yaw_deg=point_camera_yaw_deg,
            )
            extended_evidence[point_id] = {
                "frame_records": str(native_evidence.frame_records_path),
                "research_receipt": str(native_evidence.receipt_path),
                "camera_viewpoint_crosscheck": "pass_1e-6",
            }

        options = sorted(
            {
                appearances["source1"]["label_en"],
                appearances["source2"]["label_en"],
            }
        )
        emit(
            dict(
                base,
                type_id="QV2-T2-ATTR",
                answer_arity="binary",
                question_en=(
                    f"Which {noun_en} {verb_en} first — the one with "
                    f"{appearances['source1']['en']} or the one with {appearances['source2']['en']}?"
                ),
                question_zh=(
                    f"先{verb_zh}的是穿戴{appearances['source1']['zh']}的那只/位，还是"
                    f"{appearances['source2']['zh']}的？"
                    if spec["pair_kind"] == "dog"
                    else f"先{verb_zh}的那个{noun_zh}穿的是{appearances['source1']['zh']}还是"
                    f"{appearances['source2']['zh']}？"
                ),
                options=options,
                answer=appearances[slot_a]["label_en"],
                semantic_answer_source_slot=slot_a,
                modality_expectation="dual_required",
            ),
            True,
        )

        if enable_refusal:
            refusal_answer = "cannot determine from audio alone"
            refusal_row = dict(
                base,
                type_id="QV2-TR-REFUSAL",
                answer_arity="nonbinary",
                question_en=(
                    f"Using only the audio, what visual appearance did the {noun_en} "
                    f"that {verb_en} first have?"
                ),
                question_zh=f"只根据音频，最先{verb_zh}的那个{noun_zh}是什么外观？",
                options=[*options, refusal_answer],
                answer=refusal_answer,
                semantic_answer_source_slot=None,
                provided_modalities=["audio"],
                withheld_modalities=["video"],
                modality_expectation="must_refuse_audio_only_visual_attribute",
                refusal_gold=True,
            )
            refusal_row["modality_mask"] = resolve_modality_mask(
                refusal_row,
                spec=spec,
                point_id=point_id,
                audio_available=audio_input_available,
            )
            refusal_candidates.append(refusal_row)

        emit(
            dict(
                base,
                type_id="QV2-TA-MOTION",
                answer_arity="binary",
                question_en=(
                    f"At the moment of the first {('speech' if noun_en == 'person' else 'bark')}, "
                    f"was the {noun_en} making it moving or staying still?"
                ),
                question_zh=f"第一声{verb_zh}响起时，发出它的那个{noun_zh}在走动还是静止？",
                options=["moving", "staying still"],
                answer=first_motion_answer,
                semantic_answer_source_slot=slot_a,
                modality_expectation="dual_required",
            ),
            True,
        )

        positions = actor_positions(timeline, first_frame)
        side_a, magnitude_a = side_of(
            positions[slot_a],
            camera_position_ue_cm=point_camera_position_ue_cm,
            camera_yaw_deg=point_camera_yaw_deg,
        )
        side_b, magnitude_b = side_of(
            positions[slot_b],
            camera_position_ue_cm=point_camera_position_ue_cm,
            camera_yaw_deg=point_camera_yaw_deg,
        )
        row = dict(
            base,
            type_id="QV2-T4-SIDE",
            answer_arity="binary",
            question_en=(
                f"Did the first {('speech' if noun_en == 'person' else 'bark')} come from the left "
                "or the right side of the view?"
            ),
            question_zh=f"第一声{verb_zh}来自画面的左侧还是右侧？",
            options=["left", "right"],
            answer=side_a,
            semantic_answer_source_slot=slot_a,
            modality_expectation="audio_sufficient_control",
            side_margins=[round(magnitude_a, 3), round(magnitude_b, 3)],
        )
        if magnitude_a < side_margin:
            emit(row, False, "G-side-margin")
        elif side_a == side_b:
            emit(row, False, "G-differential-same-side")
        else:
            emit(row, True)

        if enable_t5_doa:
            if native_evidence is None:
                raise ExtendedQuestionEvidenceError(
                    f"{point_id}: native evidence was not loaded for T5"
                )
            t5_azimuth = native_evidence.azimuth_deg(second_slot, second_frame)
            emit(
                dict(
                    base,
                    type_id="QV2-T5-DOA",
                    answer_arity="numeric_banded",
                    question_en=(
                        f"At the second {('speech' if noun_en == 'person' else 'bark')}, "
                        "what was the source azimuth relative to the camera in degrees? "
                        "Use positive values to the right and negative values to the left."
                    ),
                    question_zh=(
                        f"第二声{verb_zh}响起时，声源相对相机的方位角是多少度？"
                        "右侧为正，左侧为负。"
                    ),
                    options=None,
                    answer=round(t5_azimuth, 1),
                    answer_numeric=round(t5_azimuth, 1),
                    answer_unit="degree",
                    format="numeric_banded",
                    unit="degree",
                    scoring_bands=[T5_FULL_CREDIT_DEG, T5_PARTIAL_CREDIT_DEG],
                    semantic_answer_source_slot=second_slot,
                    modality_expectation="audio_spatial_numeric",
                    event_frame=second_frame,
                    event_start_sample=second["start_sample"],
                    azimuth_convention="listener_local_positive_right_degrees",
                    scoring={
                        "metric": "circular_absolute_error_degrees",
                        "full_credit_within_deg": T5_FULL_CREDIT_DEG,
                        "partial_credit_within_deg": T5_PARTIAL_CREDIT_DEG,
                        "domain_deg": [-180.0, 180.0],
                    },
                    gold_authority="ue_actor_anchor_and_camera_pose_readback",
                ),
                True,
            )

        if enable_offscreen and spec["offscreen_candidate"]:
            if native_evidence is None:
                raise ExtendedQuestionEvidenceError(
                    f"{point_id}: native evidence was not loaded for off-screen"
                )
            pixel_truth = load_native_pixel_visibility(
                pixel_visibility_root,
                point_id=point_id,
            )
            extended_evidence[point_id]["pixel_visibility_truth"] = str(
                (
                    Path(pixel_visibility_root)
                    / point_id
                    / "pixel_visibility_truth.json"
                ).resolve()
            )
            selected_offscreen = None
            for event in events:
                event_slot = endpoint_to_slot[event["source_endpoint_id"]]
                event_frame = event_frame_index(event["start_sample"])
                event_frames = event_frame_window(
                    event["start_sample"],
                    event["end_sample_exclusive"],
                )
                try:
                    pixel_frames = tuple(
                        require_native_out_of_view(
                            pixel_truth,
                            source_slot=event_slot,
                            frame_index=frame_index,
                        )
                        for frame_index in event_frames
                    )
                except ExtendedQuestionEvidenceError:
                    continue
                azimuth = native_evidence.azimuth_deg(event_slot, event_frame)
                if math.isclose(azimuth, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
                    continue
                selected_offscreen = (
                    event,
                    event_slot,
                    event_frame,
                    event_frames,
                    pixel_frames,
                    azimuth,
                )
                break
            offscreen_row = dict(
                base,
                type_id="QV2-TO-OFFSCREEN",
                answer_arity="binary",
                question_en=(
                    "While the sounding source was completely outside the camera view, "
                    "did its sound come from the left or the right of the camera?"
                ),
                question_zh="发声源完全位于画面外时，声音来自相机左侧还是右侧？",
                options=["left", "right"],
                modality_expectation="audio_required_native_pixel_offscreen",
            )
            if selected_offscreen is None:
                emit(
                    dict(
                        offscreen_row,
                        answer=None,
                        semantic_answer_source_slot=None,
                    ),
                    False,
                    "G-native-pixel-out-of-view-event",
                )
            else:
                (
                    offscreen_event,
                    offscreen_slot,
                    offscreen_frame,
                    offscreen_frames,
                    offscreen_pixel_frames,
                    offscreen_azimuth,
                ) = selected_offscreen
                emit(
                    dict(
                        offscreen_row,
                        answer="right" if offscreen_azimuth > 0.0 else "left",
                        semantic_answer_source_slot=offscreen_slot,
                        event_frame=offscreen_frame,
                        event_frame_window=list(offscreen_frames),
                        event_start_sample=offscreen_event["start_sample"],
                        event_end_sample_exclusive=offscreen_event[
                            "end_sample_exclusive"
                        ],
                        listener_local_azimuth_deg=round(offscreen_azimuth, 3),
                        native_pixel_state="out_of_view_entire_event",
                        native_target_pixels_max=max(
                            frame["target_pixels"] for frame in offscreen_pixel_frames
                        ),
                        gold_authority=(
                            "ue_readback_plus_same_renderer_target_only_pixel_truth"
                        ),
                    ),
                    True,
                )

        emit(
            dict(
                base,
                type_id="QV2-T7-DURING",
                answer_arity="binary",
                question_en=(
                    f"During the second {('speech' if noun_en == 'person' else 'bark')}, "
                    f"was the {noun_en} making it moving or staying still?"
                ),
                question_zh=f"第二声{verb_zh}期间，发出它的那个{noun_zh}在走动还是静止？",
                options=["moving", "staying still"],
                answer=second_motion_answer,
                semantic_answer_source_slot=slot_b,
                modality_expectation="dual_required",
            ),
            True,
        )

        distances = {
            slot: math.hypot(
                float(position[0]) - float(point_camera_position_ue_cm[0]),
                float(position[1]) - float(point_camera_position_ue_cm[1]),
            )
            for slot, position in positions.items()
        }
        ratio = max(distances.values()) / max(min(distances.values()), 1e-9)
        closer = min(distances, key=distances.get)
        row = dict(
            base,
            type_id="QV2-T9-CLOSER",
            answer_arity="binary",
            question_en=(
                f"At the first {('speech' if noun_en == 'person' else 'bark')}, which {noun_en} is "
                f"closer to the camera — the one with {appearances['source1']['en']} or the one with "
                f"{appearances['source2']['en']}?"
            ),
            question_zh=(
                f"第一声{verb_zh}时，离相机更近的是{appearances['source1']['zh']}的那个还是"
                f"{appearances['source2']['zh']}的？"
            ),
            options=options,
            answer=appearances[closer]["label_en"],
            semantic_answer_source_slot=closer,
            modality_expectation="vision_dominant_control",
            distance_ratio=round(ratio, 2),
        )
        emit(
            row,
            ratio >= distance_ratio_min,
            None if ratio >= distance_ratio_min else "G-distance-ratio",
        )

    refusal_quota: dict[str, Any] = {
        "status": "disabled",
        "minimum_fraction": 0.10,
        "maximum_fraction": 0.15,
        "selected_count": 0,
        "final_fraction": 0.0,
    }
    if enable_refusal:
        selected_count, selected_indices, refusal_fraction = choose_refusal_quota(
            len(questions),
            len(refusal_candidates),
        )
        questions.extend(refusal_candidates[index] for index in selected_indices)
        refusal_quota = {
            "status": "pass",
            "minimum_fraction": 0.10,
            "maximum_fraction": 0.15,
            "selected_count": selected_count,
            "candidate_count": len(refusal_candidates),
            "final_fraction": refusal_fraction,
            "selection_policy": "deterministic_evenly_spaced_points",
        }

    if enable_offscreen and not any(
        question["type_id"] == "QV2-TO-OFFSCREEN" for question in questions
    ):
        raise QuestionGenerationError(
            "off-screen was enabled but no native-pixel out-of-view question passed"
        )

    by_type = Counter(question["type_id"] for question in questions)
    answers_by_type: dict[str, Counter[str]] = {}
    for question in questions:
        answers_by_type.setdefault(question["type_id"], Counter())[
            str(question["answer"])
        ] += 1
    answer_balance = build_answer_balance_report(questions)
    return {
        "schema": "avengine_qa_v2_questions_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Constraint-driven v2 questions compiled from the pilot48 design, "
            "authored timelines and executed audio programs; visibility/pixel "
            "gates pending; no dataset admission."
        ),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "counts": {
            "accepted": len(questions),
            "refused": len(refused),
            "by_type": dict(by_type),
            "answers_by_type": {
                type_id: dict(counts) for type_id, counts in answers_by_type.items()
            },
        },
        "answer_balance": answer_balance,
        "extended_types": {
            "enabled": {
                "t5_doa": enable_t5_doa,
                "offscreen": enable_offscreen,
                "refusal": enable_refusal,
            },
            "native_capture_root": (
                None if native_capture_root is None else str(native_capture_root)
            ),
            "pixel_visibility_root": (
                None if pixel_visibility_root is None else str(pixel_visibility_root)
            ),
            "native_evidence_by_point": extended_evidence,
            "refusal_quota": refusal_quota,
        },
        "gate_params": {
            "side_margin": side_margin,
            "distance_ratio_min": distance_ratio_min,
        },
        "viewpoint_resolution": {
            "mode": "report_only" if report_only else "strict_production",
            "viewpoint_bank": (
                None if viewpoint_bank_path is None else str(viewpoint_bank_path)
            ),
            "points": resolved_viewpoints,
        },
        "audio_program_resolution": {
            "points": resolved_audio_programs,
            "materialized_point_count": sum(
                item["authority"] == "point_materialized_audio_program"
                for item in resolved_audio_programs.values()
            ),
        },
        "questions": questions,
        "refused": refused,
    }


def write_question_document(document: Mapping[str, Any], output: str | Path) -> Path:
    """Serialize in a sibling staging directory, then publish no-replace."""

    unresolved = Path(output).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots((parent,))
    destination = policy.resolve_output(
        parent / unresolved.name,
        owner="QA question output directory",
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    try:
        with (staging / "questions.json").open("x", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=1)
        published = atomic_publish_directory(policy, staging, destination)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return published / "questions.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--programs-dir", default=str(DEFAULT_PROGRAMS_DIR))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--viewpoint-bank",
        help="required in production; resolves each spec.viewpoint_id camera",
    )
    parser.add_argument(
        "--native-capture-root",
        help="per-point UE frame_records.json + research_receipt.json authority",
    )
    parser.add_argument(
        "--pixel-visibility-root",
        help="per-point native pixel_visibility_truth.json authority",
    )
    parser.add_argument("--enable-t5-doa", action="store_true")
    parser.add_argument("--enable-offscreen", action="store_true")
    parser.add_argument("--enable-refusal", action="store_true")
    parser.add_argument(
        "--camera-position-ue-cm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_CAMERA_POSITION_UE_CM,
    )
    parser.add_argument("--camera-yaw-deg", type=float, default=DEFAULT_CAMERA_YAW_DEG)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "audit legacy/global-pose or imbalanced inputs; required to bypass "
            "production viewpoint and answer-balance enforcement"
        ),
    )
    parser.add_argument(
        "--require-balanced-answer-keys",
        action="store_true",
        help="compatibility flag; production is already strict by default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.path.lexists(args.output):
        print(json.dumps({"error": "output exists", "out": args.output}))
        return 2
    try:
        document = generate_question_document(
            inputs_root=args.inputs_root,
            programs_dir=args.programs_dir,
            registry_path=args.registry,
            viewpoint_bank_path=args.viewpoint_bank,
            report_only=args.report_only,
            camera_position_ue_cm=args.camera_position_ue_cm,
            camera_yaw_deg=args.camera_yaw_deg,
            native_capture_root=args.native_capture_root,
            pixel_visibility_root=args.pixel_visibility_root,
            enable_t5_doa=args.enable_t5_doa,
            enable_offscreen=args.enable_offscreen,
            enable_refusal=args.enable_refusal,
        )
    except (OSError, KeyError, TypeError, QuestionGenerationError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}))
        return 2
    if not args.report_only or args.require_balanced_answer_keys:
        try:
            require_balanced_answer_keys(document["answer_balance"])
        except AnswerBalanceError as exc:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "answer_balance": document["answer_balance"],
                        "output_written": False,
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            )
            return 2
    destination = write_question_document(document, args.output)
    print(
        json.dumps(
            {
                "out": str(destination.parent),
                "counts": document["counts"],
                "answer_balance": document["answer_balance"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
