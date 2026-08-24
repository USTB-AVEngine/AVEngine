#!/usr/bin/env python3
"""Build grounded MP3D off-screen entry questions from native target-only truth."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import load_json, write_json
from avengine.qa.answer_balance import build_answer_balance_report
from avengine.qa.question_spec import evaluate_offscreen_entry_truth
from avengine.security import WorkspacePathPolicy, atomic_publish_directory


class MP3DOffscreenMiniPilotError(RuntimeError):
    """Native target-only evidence cannot ground an off-screen question."""


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DOffscreenMiniPilotError(f"{owner} must be an object")
    return value


def build_offscreen_minipilot(
    *,
    visual_capture_dir: str | Path,
    output_directory: str | Path,
) -> Path:
    capture = Path(visual_capture_dir).resolve(strict=True)
    receipt = load_json(capture / "research_receipt.json")
    visibility = _mapping(receipt.get("visibility"), owner="capture visibility")
    artifacts = _mapping(receipt.get("artifacts"), owner="capture artifacts")
    truth_path = capture / str(artifacts.get("pixel_visibility_truth"))
    truth = load_json(truth_path)
    if not (
        receipt.get("status") == "research_only"
        and receipt.get("research_only") is True
        and receipt.get("episode_counted") is False
        and receipt.get("capture", {}).get("frame_count") == 75
        and receipt.get("capture", {}).get("actor_pair_collision_free") is True
        and visibility.get("candidate_filtering") == "disabled"
        and visibility.get("formal_modal_visibility") == truth
        and visibility.get("authority") == truth.get("authority")
        and truth.get("authority")
        == "same_renderer_same_camera_modal_target_only_semantic_masks_v1"
    ):
        raise MP3DOffscreenMiniPilotError(
            "capture does not carry bound same-renderer target-only authority"
        )
    actors = _mapping(receipt.get("actors"), owner="capture actors")
    actor_ids = actors.get("actor_ids")
    source_ids = actors.get("source_ids")
    semantic_ids = actors.get("semantic_ids")
    if not (
        isinstance(actor_ids, list)
        and isinstance(source_ids, list)
        and isinstance(semantic_ids, list)
        and len(actor_ids) == len(source_ids) == len(semantic_ids)
        and len(actor_ids) >= 2
        and len(set(actor_ids)) == len(actor_ids)
        and len(set(source_ids)) == len(source_ids)
    ):
        raise MP3DOffscreenMiniPilotError("capture actor axes are invalid")

    output = Path(output_directory).resolve()
    if output.exists() or output.is_symlink():
        raise MP3DOffscreenMiniPilotError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = WorkspacePathPolicy.from_roots((output.parent,))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        evaluations = []
        questions = []
        for actor_id, source_id, semantic_id in zip(
            actor_ids, source_ids, semantic_ids, strict=True
        ):
            evaluation = evaluate_offscreen_entry_truth(
                truth, target_instance_id=actor_id
            )
            evaluations.append(evaluation)
            if evaluation["status"] != "pass":
                continue
            answer = evaluation["answer"]["value"]
            questions.append(
                {
                    "question_id": (
                        f"{receipt['case']['case_id']}::offscreen_entry::{actor_id}"
                    ),
                    "type_id": "QV2-TO-OFFSCREEN",
                    "question_type": "offscreen_to_onscreen",
                    "answer_arity": "binary",
                    "question_en": (
                        f"Did controlled actor {actor_id} enter the camera view "
                        "from the left or the right?"
                    ),
                    "question_zh": f"受控个体 {actor_id} 从画面左侧还是右侧进入？",
                    "options": ["left", "right"],
                    "answer": answer,
                    "semantic_answer_source_slot": source_id,
                    "target_actor_id": actor_id,
                    "target_source_id": source_id,
                    "target_semantic_id": semantic_id,
                    "entry_evidence": evaluation["evidence"],
                    "gold_authority": truth["authority"],
                    "modality_expectation": "video_pixel_visibility_required",
                    "modality_mask": {"audio": False, "video": True},
                    "native_visual_capture": str(capture),
                }
            )
        if not questions:
            raise MP3DOffscreenMiniPilotError(
                "no actor has one uniquely sided out-of-view to visible transition"
            )
        balance = build_answer_balance_report(questions)
        if balance["status"] != "pass":
            raise MP3DOffscreenMiniPilotError(
                "off-screen binary answer balance is invalid"
            )
        answer_counts = Counter(question["answer"] for question in questions)
        document = {
            "schema": "avengine_mp3d_offscreen_minipilot_questions_v1",
            "status": "pass_research_offscreen_minipilot",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "native_visual_capture": str(capture),
            "native_visual_receipt": str(capture / "research_receipt.json"),
            "pixel_visibility_truth": str(truth_path),
            "counts": {
                "evaluated_actors": len(evaluations),
                "questions": len(questions),
                "answers": dict(sorted(answer_counts.items())),
            },
            "answer_balance": balance,
            "evaluations": evaluations,
            "questions": questions,
            "claim_boundary": (
                "Research mini-pilot questions grounded by same-Habitat, "
                "same-camera normal and target-only semantic passes. This is "
                "not dataset admission or a completed multi-view partition."
            ),
        }
        gate_report = {
            "status": "pass_research_offscreen_minipilot",
            "checks": {
                "same_renderer_target_only_authority": True,
                "normal_target_camera_readback_match": True,
                "normal_target_actor_readback_match": True,
                "visibility_not_candidate_gate": True,
                "unique_entry_side": True,
                "answer_balance": balance["status"] == "pass",
            },
            "pending": {
                "accepted_multi_view_batch": True,
                "refusal_quota": True,
                "exact48_selection": True,
                "formal_dataset_admission": True,
            },
        }
        write_json(staging / "questions.json", document)
        write_json(staging / "gate_report.json", gate_report)
        write_json(
            staging / "manifest.json",
            {
                "status": document["status"],
                "visual_capture": str(capture),
                "case_id": receipt["case"]["case_id"],
                "region_instance_id": receipt["case"]["region_instance_id"],
                "motion_case": receipt["case"]["motion_case"],
                "question_ids": [item["question_id"] for item in questions],
            },
        )
        return atomic_publish_directory(policy, staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-capture-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = build_offscreen_minipilot(
            visual_capture_dir=args.visual_capture_dir,
            output_directory=args.output,
        )
    except (MP3DOffscreenMiniPilotError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "pass", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
