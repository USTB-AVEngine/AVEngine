#!/usr/bin/env python3
"""Compile constraint-driven QA v2 questions and report answer-key balance.

Production is strict by default: each point must resolve through a viewpoint
bank and every binary type must be ``50/50±1`` by ``options.index(answer)``.
Historical/global-pose auditing requires the explicit ``--report-only`` mode;
``--require-balanced-answer-keys`` remains as a compatibility spelling for a
strict balance check inside that audit mode.
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
            raise QuestionGenerationError(f"viewpoint bank row {index} must be an object")
        viewpoint_id = record.get("viewpoint_id")
        if not isinstance(viewpoint_id, str) or not viewpoint_id:
            raise QuestionGenerationError(
                f"viewpoint bank row {index} has no non-empty viewpoint_id"
            )
        if viewpoint_id in by_id:
            raise QuestionGenerationError(f"duplicate viewpoint_id in bank: {viewpoint_id}")
        camera = record.get("camera")
        if not isinstance(camera, Mapping):
            raise QuestionGenerationError(f"viewpoint {viewpoint_id} has no camera object")
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


def appearance(asset_id: str, assets: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
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


def combo_program(spec: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]]) -> str:
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


def actor_positions(timeline: Mapping[str, Any], frame: int) -> dict[str, Sequence[float]]:
    states = timeline["frames"][frame]["actor_states"]
    positions: dict[str, Sequence[float]] = {}
    for state in states:
        key = state.get("source_slot_id") or state.get("actor_id")
        slot = "source1" if "1" in str(key) else "source2"
        positions[slot] = state["translation_ue_cm"][:2]
    return positions


def mover_slots(motion: str) -> set[str]:
    return {
        "s1_moving_s2_static": {"source1"},
        "s1_static_s2_moving": {"source2"},
        "both_moving": {"source1", "source2"},
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
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compile questions without writing files.

    Production mode resolves every point through ``viewpoint_bank_path`` and
    its ``spec.viewpoint_id``. The global camera arguments are a legacy audit
    fallback and are reachable only with ``report_only=True``.
    """

    if len(camera_position_ue_cm) != 3:
        raise ValueError("camera_position_ue_cm must contain exactly three values")
    registry = _load_json(registry_path)
    assets = {asset["asset_id"]: asset for asset in registry["assets"]}
    viewpoint_bank = (
        load_viewpoint_bank(viewpoint_bank_path)
        if viewpoint_bank_path is not None
        else None
    )
    questions: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    resolved_viewpoints: dict[str, dict[str, Any]] = {}

    def emit(row: dict[str, Any], ok: bool, reason: str | None = None) -> None:
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
            "position_ue_cm": [
                float(value) for value in point_camera_position_ue_cm
            ],
            "yaw_ue_deg": float(point_camera_yaw_deg),
        }
        timeline = _load_json(point_path / "timeline.json")
        program = _load_json(Path(programs_dir) / f"{combo_program(spec, assets)}.json")
        events = sorted(program["events"], key=lambda event: event["start_sample"])
        endpoint_to_slot = {
            program["candidate_source_endpoint_ids"][0]: "source1",
            program["candidate_source_endpoint_ids"][1]: "source2",
        }
        first = events[0]
        _second = next(
            event
            for event in events
            if endpoint_to_slot[event["source_endpoint_id"]]
            != endpoint_to_slot[first["source_endpoint_id"]]
        )
        slot_a = endpoint_to_slot[first["source_endpoint_id"]]
        slot_b = "source2" if slot_a == "source1" else "source1"
        first_frame = min(74, round(first["start_sample"] * 75 / 80000))
        movers = mover_slots(spec["motion_case"])
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
            "viewpoint_id": viewpoint_id,
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

        if (slot_a in movers) != (slot_b in movers):
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
                    answer="moving" if slot_a in movers else "staying still",
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

        if (slot_b in movers) != (slot_a in movers) or spec["motion_case"] == "both_moving":
            answer = "moving" if slot_b in movers else "staying still"
            ok = (slot_a in movers) != (slot_b in movers)
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
                    answer=answer,
                    semantic_answer_source_slot=slot_b,
                    modality_expectation="dual_required",
                ),
                ok,
                None if ok else "G-differential-motion",
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
        emit(row, ratio >= distance_ratio_min, None if ratio >= distance_ratio_min else "G-distance-ratio")

    by_type = Counter(question["type_id"] for question in questions)
    answers_by_type: dict[str, Counter[str]] = {}
    for question in questions:
        answers_by_type.setdefault(question["type_id"], Counter())[str(question["answer"])] += 1
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
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent)
    )
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
