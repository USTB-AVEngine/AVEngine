#!/usr/bin/env python3
"""Add a real audio-only refusal quota to grounded MP3D mini-pilot rows."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
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


class MP3DRefusalMiniPilotError(RuntimeError):
    """Current native visual/audio evidence cannot ground refusal rows."""


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DRefusalMiniPilotError(f"{owner} must be an object")
    return value


def _load_base_questions(
    paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not paths:
        raise MP3DRefusalMiniPilotError(
            "at least one base question document is required"
        )
    questions: list[dict[str, Any]] = []
    resolved_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        document = load_json(path)
        rows = document.get("questions")
        if not (
            document.get("research_only") is True
            and document.get("episode_counted") is False
            and isinstance(rows, list)
            and rows
            and all(isinstance(row, Mapping) for row in rows)
        ):
            raise MP3DRefusalMiniPilotError(
                f"base question document is not a research mini-pilot: {path}"
            )
        for row in rows:
            copied = deepcopy(dict(row))
            identity = copied.get("question_id")
            if not isinstance(identity, str) or not identity:
                point_id = copied.get("point_id")
                type_id = copied.get("type_id")
                if not all(
                    isinstance(value, str) and value for value in (point_id, type_id)
                ):
                    raise MP3DRefusalMiniPilotError(
                        f"base question lacks a stable identity: {path}"
                    )
                identity = f"{point_id}::{type_id}"
            if identity in seen:
                raise MP3DRefusalMiniPilotError(
                    f"base question identity is repeated: {identity}"
                )
            seen.add(identity)
            questions.append(copied)
        resolved_paths.append(str(path))
    return questions, resolved_paths


def _load_audio_receipt(
    path: str | Path,
    *,
    expected_visual_capture: Path,
) -> tuple[Mapping[str, Any], Path]:
    receipt_path = Path(path).resolve(strict=True)
    receipt = load_json(receipt_path)
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
        and isinstance(outputs, Mapping)
        and outputs
        and view_identity.get("visual_capture_dir") == str(expected_visual_capture)
    ):
        raise MP3DRefusalMiniPilotError(
            "audio receipt is not one fresh RLR render for the supplied visual capture"
        )
    for relative in outputs:
        if not (receipt_path.parent / relative).is_file():
            raise MP3DRefusalMiniPilotError(
                f"audio receipt output is missing: {relative}"
            )
    return receipt, receipt_path


def build_refusal_minipilot(
    *,
    base_question_documents: Sequence[str | Path],
    visual_capture_dir: str | Path,
    audio_receipt_path: str | Path,
    output_directory: str | Path,
) -> Path:
    base_questions, base_paths = _load_base_questions(base_question_documents)
    capture = Path(visual_capture_dir).resolve(strict=True)
    visual_receipt = load_json(capture / "research_receipt.json")
    visibility = _mapping(
        visual_receipt.get("visibility"), owner="visual receipt visibility"
    )
    artifacts = _mapping(
        visual_receipt.get("artifacts"), owner="visual receipt artifacts"
    )
    truth_path = capture / str(artifacts.get("pixel_visibility_truth"))
    pixel_truth = load_json(truth_path)
    if not (
        visual_receipt.get("status") == "research_only"
        and visual_receipt.get("research_only") is True
        and visual_receipt.get("episode_counted") is False
        and visual_receipt.get("capture", {}).get("frame_count") == 75
        and visual_receipt.get("capture", {}).get("actor_pair_collision_free") is True
        and visibility.get("candidate_filtering") == "disabled"
        and visibility.get("formal_modal_visibility") == pixel_truth
    ):
        raise MP3DRefusalMiniPilotError(
            "visual capture lacks bound same-renderer target-only truth"
        )
    audio_receipt, audio_path = _load_audio_receipt(
        audio_receipt_path,
        expected_visual_capture=capture,
    )
    actors = _mapping(visual_receipt.get("actors"), owner="visual receipt actors")
    actor_ids = actors.get("actor_ids")
    source_ids = actors.get("source_ids")
    semantic_ids = actors.get("semantic_ids")
    if not (
        isinstance(actor_ids, list)
        and isinstance(source_ids, list)
        and isinstance(semantic_ids, list)
        and len(actor_ids) == len(source_ids) == len(semantic_ids)
        and len(actor_ids) >= 2
    ):
        raise MP3DRefusalMiniPilotError("visual actor axes are invalid")

    candidates: list[dict[str, Any]] = []
    for actor_id, source_id, semantic_id in zip(
        actor_ids, source_ids, semantic_ids, strict=True
    ):
        entry = evaluate_offscreen_entry_truth(
            pixel_truth,
            target_instance_id=actor_id,
        )
        if entry["status"] != "pass":
            continue
        frame_index = entry["evidence"]["entry_frame_indices"][0]
        frame = pixel_truth["per_instance"][actor_id]["frames"][frame_index]
        visual_state = frame["state"]
        if visual_state not in VISUAL_STATES:
            raise MP3DRefusalMiniPilotError(f"{actor_id} entry visual state is invalid")
        row = {
            "question_id": (
                f"{visual_receipt['case']['case_id']}::audio_only_refusal::{actor_id}"
            ),
            "type_id": "QV2-TR-REFUSAL",
            "question_type": "audio_only_visual_state_refusal",
            "answer_arity": "nonbinary",
            "question_en": (
                f"Using only the provided audio, what was the camera-image "
                f"visibility state of controlled actor {actor_id} at video frame "
                f"{frame_index}?"
            ),
            "question_zh": (
                f"只使用给定音频，受控个体 {actor_id} 在视频第 {frame_index} 帧的"
                "画面可见状态是什么？"
            ),
            "options": [*VISUAL_STATES, REFUSAL_ANSWER],
            "answer": REFUSAL_ANSWER,
            "semantic_answer_source_slot": None,
            "provided_modalities": ["audio"],
            "withheld_modalities": ["video"],
            "modality_expectation": "must_refuse_audio_only_visual_attribute",
            "refusal_gold": True,
            "refusal_reason": (
                "the requested same-renderer camera visibility state belongs to "
                "the withheld video/pixel-visibility modality"
            ),
            "target_actor_id": actor_id,
            "target_source_id": source_id,
            "target_semantic_id": semantic_id,
            "frame_index": frame_index,
            "withheld_visual_gold_state": visual_state,
            "withheld_visual_raw_counts": {
                "visible_pixels": frame["visible_pixels"],
                "target_pixels": frame["target_pixels"],
            },
            "visual_gold_authority": pixel_truth["authority"],
            "native_visual_capture": str(capture),
            "audio_receipt": str(audio_path),
            "audio_program_id": audio_receipt["audio_program"]["program_id"],
            "answerability_certificate": {
                "required_fact": "camera_image_modal_and_target_only_pixels",
                "required_modality": "video",
                "provided_modalities": ["audio"],
                "required_modality_withheld": True,
                "structural_not_model_performance_claim": True,
            },
        }
        row["modality_mask"] = resolve_modality_mask(
            row,
            spec={"input_modalities": ["audio", "video"]},
            point_id=str(row["question_id"]),
            audio_available=True,
        )
        candidates.append(row)
    if not candidates:
        raise MP3DRefusalMiniPilotError(
            "no current actor has grounded visual entry evidence for refusal"
        )

    selected_count, selected_indices, refusal_fraction = choose_refusal_quota(
        len(base_questions),
        len(candidates),
    )
    selected = [candidates[index] for index in selected_indices]
    questions = [*base_questions, *selected]
    if questions[: len(base_questions)] != base_questions:
        raise MP3DRefusalMiniPilotError("base questions changed during refusal append")
    balance = build_answer_balance_report(questions)
    if balance["status"] != "pass":
        raise MP3DRefusalMiniPilotError("combined mini-pilot answer balance is invalid")

    output = Path(output_directory).resolve()
    if output.exists() or output.is_symlink():
        raise MP3DRefusalMiniPilotError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = WorkspacePathPolicy.from_roots((output.parent,))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        document = {
            "schema": "avengine_mp3d_refusal_minipilot_questions_v1",
            "status": "pass_research_refusal_minipilot",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "base_question_documents": base_paths,
            "visual_capture": str(capture),
            "audio_receipt": str(audio_path),
            "counts": {
                "base_questions": len(base_questions),
                "refusal_candidates": len(candidates),
                "selected_refusals": selected_count,
                "final_questions": len(questions),
                "by_type": dict(Counter(row["type_id"] for row in questions)),
            },
            "refusal_quota": {
                "status": "pass",
                "minimum_fraction": 0.10,
                "maximum_fraction": 0.15,
                "selected_count": selected_count,
                "candidate_count": len(candidates),
                "selected_candidate_indices": list(selected_indices),
                "final_fraction": refusal_fraction,
                "selection_policy": "deterministic_evenly_spaced_candidates",
            },
            "answer_balance": balance,
            "questions": questions,
            "claim_boundary": (
                "Research mini-pilot refusal rows over one real fresh RLR input. "
                "Answerability is a structural modality fact, not a model score; "
                "accepted multi-view batch and dataset admission remain pending."
            ),
        }
        gate = {
            "status": document["status"],
            "checks": {
                "base_rows_preserved": True,
                "fresh_audio_bound_to_visual": True,
                "same_renderer_visual_gold": True,
                "audio_only_mask": all(
                    row["modality_mask"] == {"audio": True, "video": False}
                    for row in selected
                ),
                "quota_10_to_15_percent": 0.10 <= refusal_fraction <= 0.15,
                "combined_answer_balance": balance["status"] == "pass",
            },
            "pending": {
                "accepted_multi_view_batch": True,
                "trainer_batch_consumption": True,
                "exact48_selection": True,
                "formal_dataset_admission": True,
            },
        }
        write_json(staging / "questions.json", document)
        write_json(staging / "gate_report.json", gate)
        write_json(
            staging / "manifest.json",
            {
                "status": document["status"],
                "visual_case_id": visual_receipt["case"]["case_id"],
                "audio_program_id": audio_receipt["audio_program"]["program_id"],
                "selected_refusal_question_ids": [
                    row["question_id"] for row in selected
                ],
            },
        )
        return atomic_publish_directory(policy, staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-questions", action="append", required=True)
    parser.add_argument("--visual-capture-dir", required=True)
    parser.add_argument("--audio-receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = build_refusal_minipilot(
            base_question_documents=args.base_questions,
            visual_capture_dir=args.visual_capture_dir,
            audio_receipt_path=args.audio_receipt,
            output_directory=args.output,
        )
    except (MP3DRefusalMiniPilotError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "pass", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
