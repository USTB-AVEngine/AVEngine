#!/usr/bin/env python3
"""Finalize current MP3D multi-view QA after target-only and RLR completion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import load_json, write_json
from avengine.qa.answer_balance import build_answer_balance_report
from avengine.qa.extended_questions import choose_refusal_quota
from avengine.qa.question_spec import evaluate_offscreen_entry_truth
from avengine.security import WorkspacePathPolicy, atomic_publish_directory
from tools.qa.generate_qa_v2_questions import resolve_modality_mask


REFUSAL_ANSWER = "cannot determine from audio alone"
VISUAL_STATES = (
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
)
BASE_TYPES = {"QV2-TA-MOTION", "QV2-T7-DURING", "QV2-T5-DOA"}
POINT_INDEX_PATTERN = re.compile(r"_p([0-9]{3})_")


class MP3DMultiviewFinalizeError(RuntimeError):
    """The current visual/audio population is not ready for final QA."""


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DMultiviewFinalizeError(f"{owner} must be an object")
    return value


def _point_index(point_id: str) -> int:
    match = POINT_INDEX_PATTERN.search(point_id)
    if match is None:
        raise MP3DMultiviewFinalizeError(
            f"point id has no explicit point index: {point_id}"
        )
    return int(match.group(1))


def _load_audio_receipt(
    path: Path,
    *,
    expected_visual_capture: Path,
    expected_program_id: str,
) -> Mapping[str, Any]:
    receipt = load_json(path)
    outputs = receipt.get("outputs")
    view_identity = receipt.get("listener", {}).get("view_identity", {})
    if not (
        receipt.get("status") == "pass"
        and receipt.get("research_only") is True
        and receipt.get("episode_counted") is False
        and receipt.get("formal_dataset_count") == 0
        and receipt.get("audio", {}).get("sample_rate_hz") == 16_000
        and receipt.get("audio", {}).get("sample_count") == 80_000
        and receipt.get("rir", {}).get("prior_listener_rir_reused") is False
        and receipt.get("audio_program", {}).get("program_id") == expected_program_id
        and view_identity.get("visual_capture_dir") == str(expected_visual_capture)
        and isinstance(outputs, Mapping)
        and outputs
        and all((path.parent / relative).is_file() for relative in outputs)
    ):
        raise MP3DMultiviewFinalizeError(
            f"audio receipt does not bind fresh RLR to visual/program: {path}"
        )
    return receipt


def _load_target_capture(
    path: Path,
    *,
    plan_item: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    receipt = load_json(path / "research_receipt.json")
    visibility = _mapping(receipt.get("visibility"), owner="target visibility")
    artifacts = _mapping(receipt.get("artifacts"), owner="target artifacts")
    truth_path = path / str(artifacts.get("pixel_visibility_truth"))
    truth = load_json(truth_path)
    case = _mapping(receipt.get("case"), owner="target case")
    if not (
        receipt.get("status") == "research_only"
        and receipt.get("research_only") is True
        and receipt.get("episode_counted") is False
        and receipt.get("capture", {}).get("frame_count") == 75
        and receipt.get("capture", {}).get("actor_pair_collision_free") is True
        and case.get("region_instance_id") == plan_item["region_instance_id"]
        and case.get("motion_case") == plan_item["motion_case"]
        and case.get("route_family_id") == plan_item["route_family_id"]
        and Path(str(case.get("camera_m1_request"))).resolve()
        == Path(str(plan_item["m1_request_path"])).resolve()
        and visibility.get("candidate_filtering") == "disabled"
        and visibility.get("formal_modal_visibility") == truth
        and truth.get("authority")
        == "same_renderer_same_camera_modal_target_only_semantic_masks_v1"
    ):
        raise MP3DMultiviewFinalizeError(
            f"target-only capture does not bind the planned point: {path}"
        )
    return receipt, truth


def select_balanced_offscreen_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the maximum deterministic exact-left/right subset."""

    by_answer = {
        side: sorted(
            (dict(row) for row in candidates if row.get("answer") == side),
            key=lambda row: row["question_id"],
        )
        for side in ("left", "right")
    }
    selected_per_side = min(len(by_answer["left"]), len(by_answer["right"]))
    if selected_per_side == 0:
        raise MP3DMultiviewFinalizeError(
            "native off-screen candidates do not cover both entry sides"
        )
    selected = [
        *by_answer["left"][:selected_per_side],
        *by_answer["right"][:selected_per_side],
    ]
    selected.sort(key=lambda row: (row["point_id"], row["question_id"]))
    return selected, {
        "status": "pass",
        "candidate_counts": {side: len(by_answer[side]) for side in ("left", "right")},
        "selected_counts": {
            "left": selected_per_side,
            "right": selected_per_side,
        },
        "unselected_count": len(candidates) - len(selected),
        "selection_policy": "maximum_exact_left_right_subset_then_question_id",
        "candidate_rows_preserved_in_report": True,
    }


def finalize_multiview_batch(
    *,
    design_root: str | Path,
    target_only_root: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
) -> Path:
    design = Path(design_root).resolve(strict=True)
    target_root = Path(target_only_root).resolve(strict=True)
    audio = Path(audio_root).resolve(strict=True)
    design_manifest = load_json(design / "manifest.json")
    try:
        plan = json.loads((target_root / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MP3DMultiviewFinalizeError(
            f"target-only plan cannot be loaded: {exc}"
        ) from exc
    if design_manifest.get("status") != "pending_randomized_audio_render":
        raise MP3DMultiviewFinalizeError("design root is not awaiting randomized audio")
    if not isinstance(plan, list) or not plan:
        raise MP3DMultiviewFinalizeError("target-only plan is empty")
    plan_by_key = {
        (
            int(item["region_index"]),
            str(item["camera_slot_id"]),
            int(item["point_index"]),
        ): item
        for item in plan
    }
    views = design_manifest.get("views")
    if not isinstance(views, list) or not views:
        raise MP3DMultiviewFinalizeError("design manifest has no views")

    base_questions: list[dict[str, Any]] = []
    offscreen_candidates: list[dict[str, Any]] = []
    refusal_candidates: list[dict[str, Any]] = []
    specs: dict[str, dict[str, Any]] = {}
    point_records: list[dict[str, Any]] = []
    for view in views:
        view = _mapping(view, owner="design view")
        view_root = Path(str(view["output"])).resolve(strict=True)
        question_document = load_json(view_root / "questions.json")
        view_manifest = load_json(view_root / "manifest.json")
        rows_by_point: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in question_document.get("questions", []):
            if not isinstance(row, Mapping):
                raise MP3DMultiviewFinalizeError("design question row is invalid")
            rows_by_point[str(row.get("point_id"))].append(row)
        for point_id in view_manifest.get("points", []):
            point_id = str(point_id)
            key = (
                int(view["region_index"]),
                str(view["camera_slot_id"]),
                _point_index(point_id),
            )
            try:
                item = plan_by_key[key]
            except KeyError as exc:
                raise MP3DMultiviewFinalizeError(
                    f"design point is absent from target plan: {point_id}"
                ) from exc
            rows = rows_by_point.get(point_id, [])
            if {row.get("type_id") for row in rows} != BASE_TYPES:
                raise MP3DMultiviewFinalizeError(
                    f"point does not carry exactly TA/T7/T5: {point_id}"
                )
            target_capture = (
                target_root
                / f"region_{int(item['region_index']):03d}"
                / f"point_{int(item['point_index']):03d}_{item['motion_case']}"
            )
            target_receipt, pixel_truth = _load_target_capture(
                target_capture,
                plan_item=item,
            )
            program_path = view_root / point_id / "audio_program.json"
            program = load_json(program_path)
            program_id = str(program.get("program_id"))
            audio_receipt_path = audio / point_id / "research_receipt.json"
            audio_receipt = _load_audio_receipt(
                audio_receipt_path,
                expected_visual_capture=target_capture,
                expected_program_id=program_id,
            )
            case_manifest = load_json(item["case_manifest_path"])
            case_actors = sorted(
                case_manifest.get("actors", []),
                key=lambda actor: actor.get("source_id", ""),
            )
            if [actor.get("source_id") for actor in case_actors] != [
                "source1",
                "source2",
            ]:
                raise MP3DMultiviewFinalizeError(f"case source axes differ: {point_id}")
            actors = _mapping(target_receipt.get("actors"), owner="target actors")
            if actors.get("source_ids") != ["source1", "source2"]:
                raise MP3DMultiviewFinalizeError(
                    f"target source axes differ: {point_id}"
                )
            pair_kind = str(rows[0].get("pair_kind"))
            spec = {
                "point_id": point_id,
                "room_id": item["region_instance_id"],
                "viewpoint_id": item["placement_id"],
                "camera_slot_id": item["camera_slot_id"],
                "pair_kind": pair_kind,
                "source1_asset": case_actors[0]["asset_id"],
                "source2_asset": case_actors[1]["asset_id"],
                "motion_case": item["motion_case"],
                "program_id": program_id,
                "sound_id": program["events"][0]["sound_asset_id"],
                "twin_of": None,
                "m1_request_path": item["m1_request_path"],
                "case_manifest_path": item["case_manifest_path"],
                "target_only_visual_capture": str(target_capture),
                "randomized_audio_receipt": str(audio_receipt_path),
                "target_only_status": "pass",
                "randomized_audio_status": "pass",
                "research_only": True,
                "episode_counted": False,
            }
            if point_id in specs:
                raise MP3DMultiviewFinalizeError(f"point is repeated: {point_id}")
            specs[point_id] = spec
            normalized_rows: list[dict[str, Any]] = []
            for source_row in rows:
                row = deepcopy(dict(source_row))
                source_visual = row.get("native_visual_capture")
                row.update(
                    {
                        "question_id": f"{point_id}::{row['type_id']}",
                        "room_id": spec["room_id"],
                        "viewpoint_id": spec["viewpoint_id"],
                        "camera_slot_id": spec["camera_slot_id"],
                        "pair_kind": spec["pair_kind"],
                        "motion_case": spec["motion_case"],
                        "twin_of": None,
                        "source_native_visual_capture": source_visual,
                        "native_visual_capture": str(target_capture),
                        "randomized_audio_receipt": str(audio_receipt_path),
                        "randomized_audio_render_status": "pass",
                    }
                )
                normalized_rows.append(row)
                base_questions.append(row)
            t5 = next(row for row in normalized_rows if row["type_id"] == "QV2-T5-DOA")
            source_ids = actors["source_ids"]
            actor_ids = actors["actor_ids"]
            semantic_ids = actors["semantic_ids"]
            for actor_id, source_id, semantic_id in zip(
                actor_ids, source_ids, semantic_ids, strict=True
            ):
                evaluation = evaluate_offscreen_entry_truth(
                    pixel_truth,
                    target_instance_id=actor_id,
                )
                if evaluation["status"] != "pass":
                    continue
                offscreen_candidates.append(
                    {
                        "question_id": f"{point_id}::QV2-TO-OFFSCREEN::{actor_id}",
                        "point_id": point_id,
                        "type_id": "QV2-TO-OFFSCREEN",
                        "question_type": "offscreen_to_onscreen",
                        "answer_arity": "binary",
                        "question_en": (
                            f"Did controlled actor {actor_id} enter the camera "
                            "view from the left or the right?"
                        ),
                        "question_zh": (
                            f"受控个体 {actor_id} 从画面左侧还是右侧进入？"
                        ),
                        "options": ["left", "right"],
                        "answer": evaluation["answer"]["value"],
                        "semantic_answer_source_slot": source_id,
                        "target_actor_id": actor_id,
                        "target_source_id": source_id,
                        "target_semantic_id": semantic_id,
                        "entry_evidence": evaluation["evidence"],
                        "gold_authority": pixel_truth["authority"],
                        "modality_expectation": "video_pixel_visibility_required",
                        "modality_mask": {"audio": False, "video": True},
                        "room_id": spec["room_id"],
                        "viewpoint_id": spec["viewpoint_id"],
                        "camera_slot_id": spec["camera_slot_id"],
                        "pair_kind": spec["pair_kind"],
                        "motion_case": spec["motion_case"],
                        "twin_of": None,
                        "native_visual_capture": str(target_capture),
                    }
                )
            source_slot = t5["semantic_answer_source_slot"]
            try:
                actor_index = source_ids.index(source_slot)
            except ValueError as exc:
                raise MP3DMultiviewFinalizeError(
                    f"T5 source slot differs from target axes: {point_id}"
                ) from exc
            actor_id = actor_ids[actor_index]
            frame_index = int(t5["event_frame"])
            frame = pixel_truth["per_instance"][actor_id]["frames"][frame_index]
            state = frame["state"]
            if state not in VISUAL_STATES:
                raise MP3DMultiviewFinalizeError(
                    f"refusal visual state is invalid: {point_id}"
                )
            refusal = {
                "question_id": f"{point_id}::QV2-TR-REFUSAL",
                "point_id": point_id,
                "type_id": "QV2-TR-REFUSAL",
                "question_type": "audio_only_visual_state_refusal",
                "answer_arity": "nonbinary",
                "question_en": (
                    f"Using only the provided audio, what was the camera-image "
                    f"visibility state of controlled actor {actor_id} at video "
                    f"frame {frame_index}?"
                ),
                "question_zh": (
                    f"只使用给定音频，受控个体 {actor_id} 在视频第 "
                    f"{frame_index} 帧的画面可见状态是什么？"
                ),
                "options": [*VISUAL_STATES, REFUSAL_ANSWER],
                "answer": REFUSAL_ANSWER,
                "semantic_answer_source_slot": None,
                "provided_modalities": ["audio"],
                "withheld_modalities": ["video"],
                "modality_expectation": "must_refuse_audio_only_visual_attribute",
                "refusal_gold": True,
                "target_actor_id": actor_id,
                "target_source_id": source_slot,
                "frame_index": frame_index,
                "withheld_visual_gold_state": state,
                "withheld_visual_raw_counts": {
                    "visible_pixels": frame["visible_pixels"],
                    "target_pixels": frame["target_pixels"],
                },
                "visual_gold_authority": pixel_truth["authority"],
                "audio_receipt": str(audio_receipt_path),
                "audio_program_id": audio_receipt["audio_program"]["program_id"],
                "answerability_certificate": {
                    "required_fact": "camera_image_modal_and_target_only_pixels",
                    "required_modality": "video",
                    "provided_modalities": ["audio"],
                    "required_modality_withheld": True,
                    "structural_not_model_performance_claim": True,
                },
                "room_id": spec["room_id"],
                "viewpoint_id": spec["viewpoint_id"],
                "camera_slot_id": spec["camera_slot_id"],
                "pair_kind": spec["pair_kind"],
                "motion_case": spec["motion_case"],
                "twin_of": None,
                "native_visual_capture": str(target_capture),
            }
            refusal["modality_mask"] = resolve_modality_mask(
                refusal,
                spec={"input_modalities": ["audio", "video"]},
                point_id=point_id,
                audio_available=True,
            )
            refusal_candidates.append(refusal)
            point_records.append(
                {
                    "point_id": point_id,
                    "target_visual": str(target_capture),
                    "audio_receipt": str(audio_receipt_path),
                    "program_id": program_id,
                }
            )

    selected_offscreen, offscreen_report = select_balanced_offscreen_candidates(
        offscreen_candidates
    )
    ordinary_questions = [*base_questions, *selected_offscreen]
    selected_count, selected_indices, refusal_fraction = choose_refusal_quota(
        len(ordinary_questions),
        len(refusal_candidates),
    )
    selected_refusals = [refusal_candidates[index] for index in selected_indices]
    questions = [*ordinary_questions, *selected_refusals]
    questions.sort(
        key=lambda row: (row["point_id"], row["type_id"], row["question_id"])
    )
    balance = build_answer_balance_report(questions)
    if balance["status"] != "pass":
        raise MP3DMultiviewFinalizeError(
            "final multi-view question answer balance failed"
        )
    if len(point_records) != len(specs):
        raise MP3DMultiviewFinalizeError("point finalization records are incomplete")

    output = Path(output_directory).resolve()
    if output.exists() or output.is_symlink():
        raise MP3DMultiviewFinalizeError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = WorkspacePathPolicy.from_roots((output.parent,))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        spec_root = staging / "point_specs"
        spec_root.mkdir()
        for point_id, spec in sorted(specs.items()):
            point_dir = spec_root / point_id
            point_dir.mkdir()
            write_json(point_dir / "spec.json", spec)
        document = {
            "schema": "avengine_mp3d_multiview_questions_v1",
            "status": "pass_research_multiview_batch",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "counts": {
                "points": len(specs),
                "base_questions": len(base_questions),
                "offscreen_candidates": len(offscreen_candidates),
                "selected_offscreen": len(selected_offscreen),
                "refusal_candidates": len(refusal_candidates),
                "selected_refusals": selected_count,
                "questions": len(questions),
                "regions": len({spec["room_id"] for spec in specs.values()}),
                "viewpoints": len({spec["viewpoint_id"] for spec in specs.values()}),
                "by_type": dict(Counter(row["type_id"] for row in questions)),
            },
            "offscreen_balance": offscreen_report,
            "refusal_quota": {
                "status": "pass",
                "minimum_fraction": 0.10,
                "maximum_fraction": 0.15,
                "selected_count": selected_count,
                "candidate_count": len(refusal_candidates),
                "selected_candidate_indices": list(selected_indices),
                "final_fraction": refusal_fraction,
                "selection_policy": "deterministic_evenly_spaced_candidates",
            },
            "answer_balance": balance,
            "questions": questions,
            "claim_boundary": (
                "Current Habitat-native research multi-view batch with fresh RLR, "
                "same-renderer target-only facts, T5, balanced off-screen, and "
                "audio-only refusal. Exact48 selection and formal admission are "
                "separate downstream actions."
            ),
        }
        write_json(staging / "questions.json", document)
        write_json(
            staging / "batch_report.json",
            {
                "status": document["status"],
                "checks": {
                    "all_planned_points_finalized": True,
                    "fresh_audio_bound_to_target_visual": True,
                    "same_renderer_target_only_bound": True,
                    "offscreen_answer_balance": True,
                    "refusal_quota": 0.10 <= refusal_fraction <= 0.15,
                    "question_answer_balance": balance["status"] == "pass",
                },
                "source_design": str(design),
                "source_target_only": str(target_root),
                "source_audio": str(audio),
                "point_records": point_records,
                "offscreen_candidates": offscreen_candidates,
                "unselected_offscreen_question_ids": sorted(
                    {row["question_id"] for row in offscreen_candidates}
                    - {row["question_id"] for row in selected_offscreen}
                ),
            },
        )
        write_json(
            staging / "manifest.json",
            {
                "status": document["status"],
                "point_specs": "point_specs",
                "questions": "questions.json",
                "batch_report": "batch_report.json",
                "exact48_status": "pending_selector_run",
            },
        )
        return atomic_publish_directory(policy, staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-root", required=True)
    parser.add_argument("--target-only-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = finalize_multiview_batch(
            design_root=args.design_root,
            target_only_root=args.target_only_root,
            audio_root=args.audio_root,
            output_directory=args.output,
        )
    except (MP3DMultiviewFinalizeError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "pass", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
