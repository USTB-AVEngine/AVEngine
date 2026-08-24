#!/usr/bin/env python3
"""Author Habitat-native QA/T5 design for one four-motion MP3D viewpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.qa.answer_balance import build_answer_balance_report
from avengine.qa.extended_questions import (
    T5_FULL_CREDIT_DEG,
    T5_PARTIAL_CREDIT_DEG,
    event_frame_index,
)
from avengine.security import WorkspacePathPolicy, atomic_publish_directory
from tools.qa.design_qa_batch import materialize_point_audio_program


MOTION_ANSWERS = {
    "static_static": ("staying still", "staying still"),
    "s1_moving_s2_static": ("moving", "staying still"),
    "s1_static_s2_moving": ("staying still", "moving"),
    "both_moving": ("moving", "moving"),
}
ENDPOINT_AXIS = {
    "beagle_0_muzzle": (0, "source1"),
    "beagle_1_muzzle": (1, "source2"),
}


class MP3DRegionMiniPilotError(RuntimeError):
    """The native four-point evidence cannot ground the QA design."""


def _load(path: Path, *, owner: str) -> Any:
    try:
        resolved = path.resolve(strict=True)
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MP3DRegionMiniPilotError(
            f"{owner} cannot be loaded: {path}: {exc}"
        ) from exc


def listener_local_azimuth_deg(
    listener_pose_txyz_qxyzw: Sequence[float],
    source_position_m: Sequence[float],
) -> float:
    listener = np.asarray(listener_pose_txyz_qxyzw, dtype=np.float64)
    source = np.asarray(source_position_m, dtype=np.float64)
    if (
        listener.shape != (7,)
        or source.shape != (3,)
        or not (np.all(np.isfinite(listener)) and np.all(np.isfinite(source)))
    ):
        raise MP3DRegionMiniPilotError(
            "listener/source pose must be finite 7/3-vectors"
        )
    rotation = np.asarray(quaternion_xyzw_to_matrix(listener[3:]), dtype=np.float64)
    local = rotation.T @ (source - listener[:3])
    return float(math.degrees(math.atan2(local[0], -local[2])))


def _motion_question(
    *,
    base: Mapping[str, Any],
    type_id: str,
    source_slot: str,
    english_ordinal: str,
    chinese_ordinal: str,
    answer: str,
) -> dict[str, Any]:
    return {
        **dict(base),
        "type_id": type_id,
        "answer_arity": "binary",
        "question_en": (
            f"At the {english_ordinal} bark, was the dog making it moving or "
            "staying still?"
        ),
        "question_zh": f"第{chinese_ordinal}声犬叫响起时，发声的狗在走动还是静止？",
        "options": ["moving", "staying still"],
        "answer": answer,
        "semantic_answer_source_slot": source_slot,
        "modality_expectation": "dual_required_motion_state",
        "modality_mask": {"audio": True, "video": True},
    }


def _t5_question(
    *, base: Mapping[str, Any], source_slot: str, event_frame: int, azimuth: float
) -> dict[str, Any]:
    answer = round(azimuth, 1)
    return {
        **dict(base),
        "type_id": "QV2-T5-DOA",
        "answer_arity": "numeric_banded",
        "question_en": (
            "At the second bark, what was the source azimuth relative to the "
            "listener in degrees? Positive is right and negative is left."
        ),
        "question_zh": "第二声犬叫响起时，声源相对听者的方位角是多少度？右正左负。",
        "options": None,
        "answer": answer,
        "answer_numeric": answer,
        "answer_unit": "degree",
        "format": "numeric_banded",
        "unit": "degree",
        "scoring_bands": [T5_FULL_CREDIT_DEG, T5_PARTIAL_CREDIT_DEG],
        "semantic_answer_source_slot": source_slot,
        "modality_expectation": "audio_spatial_numeric",
        "modality_mask": {"audio": True, "video": True},
        "event_frame": event_frame,
        "azimuth_convention": "listener_local_positive_right_degrees",
        "scoring": {
            "metric": "circular_absolute_error_degrees",
            "full_credit_within_deg": T5_FULL_CREDIT_DEG,
            "partial_credit_within_deg": T5_PARTIAL_CREDIT_DEG,
            "domain_deg": [-180.0, 180.0],
        },
        "gold_authority": "habitat_native_listener_and_emitter_link_readback",
    }


def build_minipilot(
    *,
    native_batch_root: str | Path,
    region_index: int,
    camera_slot_id: str,
    base_audio_program_path: str | Path,
    output_directory: str | Path,
    schedule_seed_base: int = 0,
    existing_fixed_audio_root: str | Path | None = None,
    randomized_audio_root: str | Path | None = None,
) -> Path:
    batch_root = Path(native_batch_root).resolve(strict=True)
    plan = _load(batch_root / "plan.json", owner="native batch plan")
    selected = [
        item
        for item in plan
        if item.get("region_index") == region_index
        and item.get("camera_slot_id") == camera_slot_id
    ]
    selected.sort(key=lambda item: int(item["point_index"]))
    if {item.get("motion_case") for item in selected} != set(MOTION_ANSWERS):
        raise MP3DRegionMiniPilotError(
            "selected viewpoint must contain the four declared motion cases"
        )
    base_program = _load(Path(base_audio_program_path), owner="base AudioProgram")
    output = Path(output_directory).resolve()
    if output.exists() or output.is_symlink():
        raise MP3DRegionMiniPilotError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = WorkspacePathPolicy.from_roots((output.parent,))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    questions: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    try:
        for ordinal, item in enumerate(selected):
            point_index = int(item["point_index"])
            motion_case = str(item["motion_case"])
            point_id = (
                f"mp3d_r{region_index:03d}_{camera_slot_id}_"
                f"p{point_index:03d}_{motion_case}"
            )
            capture = (
                batch_root
                / f"region_{region_index:03d}"
                / f"point_{point_index:03d}_{motion_case}"
            )
            receipt = _load(
                capture / "research_receipt.json", owner=f"{point_id} visual receipt"
            )
            if (
                receipt.get("status") != "research_only"
                or receipt.get("episode_counted") is not False
                or receipt.get("capture", {}).get("actor_pair_collision_free")
                is not True
            ):
                raise MP3DRegionMiniPilotError(
                    f"{point_id} native visual is not usable"
                )
            program, schedule = materialize_point_audio_program(
                base_program,
                point_id=point_id,
                schedule_seed=schedule_seed_base + ordinal,
            )
            listeners = np.load(
                capture / receipt["artifacts"]["actual_logical_listener_poses"],
                allow_pickle=False,
            )
            emitters = np.load(
                capture / receipt["artifacts"]["source_positions_m"],
                allow_pickle=False,
            )
            object_id = np.load(capture / "arrays/object_id.npy", allow_pickle=False)
            event_evidence = []
            for event in program["events"][:2]:
                try:
                    axis_index, source_slot = ENDPOINT_AXIS[event["source_endpoint_id"]]
                except KeyError as exc:
                    raise MP3DRegionMiniPilotError(
                        f"{point_id} event endpoint has no visual axis"
                    ) from exc
                frame = event_frame_index(event["start_sample"])
                if not 0 <= frame < 75:
                    raise MP3DRegionMiniPilotError(
                        f"{point_id} event frame is outside capture"
                    )
                azimuth = listener_local_azimuth_deg(
                    listeners[frame], emitters[frame, axis_index]
                )
                runtime_id = receipt["actors"]["runtime_object_ids"][axis_index]
                event_evidence.append(
                    {
                        "event_id": event["event_id"],
                        "audio_endpoint_id": event["source_endpoint_id"],
                        "source_slot": source_slot,
                        "axis_index": axis_index,
                        "start_sample": event["start_sample"],
                        "frame_index": frame,
                        "azimuth_deg": azimuth,
                        "object_id_pixel_count": int(
                            np.count_nonzero(object_id[frame] == runtime_id)
                        ),
                    }
                )
            fixed_audio_receipt = None
            if existing_fixed_audio_root is not None:
                fixed_path = (
                    Path(existing_fixed_audio_root)
                    / f"point_{point_index:03d}_{motion_case}"
                    / "research_receipt.json"
                )
                fixed = _load(fixed_path, owner=f"{point_id} fixed audio receipt")
                if fixed.get("status") != "pass":
                    raise MP3DRegionMiniPilotError(
                        f"{point_id} fixed audio did not pass"
                    )
                fixed_audio_receipt = str(fixed_path.resolve())
            randomized_audio_receipt = None
            randomized_audio_status = "pending"
            if randomized_audio_root is not None:
                randomized_path = (
                    Path(randomized_audio_root) / point_id / "research_receipt.json"
                )
                randomized = _load(
                    randomized_path, owner=f"{point_id} randomized audio receipt"
                )
                if (
                    randomized.get("status") != "pass"
                    or randomized.get("audio_program", {}).get("program_id")
                    != program["program_id"]
                    or randomized.get("listener", {})
                    .get("view_identity", {})
                    .get("visual_capture_dir")
                    != str(capture.resolve())
                ):
                    raise MP3DRegionMiniPilotError(
                        f"{point_id} randomized audio identity did not pass"
                    )
                randomized_audio_receipt = str(randomized_path.resolve())
                randomized_audio_status = "pass"
            first_motion, second_motion = MOTION_ANSWERS[motion_case]
            base_question = {
                "point_id": point_id,
                "pair_kind": "two_dog_same_beagle_asset",
                "motion_case": motion_case,
                "viewpoint_id": camera_slot_id,
                "camera_m1_request": item["m1_request_path"],
                "native_visual_capture": str(capture),
                "fixed_audio_receipt": fixed_audio_receipt,
                "randomized_audio_program_id": program["program_id"],
                "randomized_audio_program_path": f"{point_id}/audio_program.json",
                "randomized_audio_receipt": randomized_audio_receipt,
                "randomized_audio_render_status": randomized_audio_status,
                "first_event_start_sample": program["events"][0]["start_sample"],
                "second_event_start_sample": program["events"][1]["start_sample"],
                "first_second_onset_interval_samples": (
                    program["events"][1]["start_sample"]
                    - program["events"][0]["start_sample"]
                ),
            }
            questions.extend(
                [
                    _motion_question(
                        base=base_question,
                        type_id="QV2-TA-MOTION",
                        source_slot=event_evidence[0]["source_slot"],
                        english_ordinal="first",
                        chinese_ordinal="一",
                        answer=first_motion,
                    ),
                    _motion_question(
                        base=base_question,
                        type_id="QV2-T7-DURING",
                        source_slot=event_evidence[1]["source_slot"],
                        english_ordinal="second",
                        chinese_ordinal="二",
                        answer=second_motion,
                    ),
                    _t5_question(
                        base=base_question,
                        source_slot=event_evidence[1]["source_slot"],
                        event_frame=event_evidence[1]["frame_index"],
                        azimuth=event_evidence[1]["azimuth_deg"],
                    ),
                ]
            )
            point_dir = staging / point_id
            point_dir.mkdir()
            (point_dir / "audio_program.json").write_text(
                json.dumps(program, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            evidence = {
                "point_id": point_id,
                "motion_case": motion_case,
                "native_visual_capture": str(capture),
                "native_visual_receipt": str(capture / "research_receipt.json"),
                "event_evidence": event_evidence,
                "audio_schedule": schedule,
                "existing_fixed_audio_receipt": fixed_audio_receipt,
                "randomized_audio_receipt": randomized_audio_receipt,
                "randomized_audio_render_status": randomized_audio_status,
            }
            (point_dir / "evidence.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            point_records.append(evidence)

        balance = build_answer_balance_report(questions)
        first_frames = Counter(
            item["audio_schedule"]["first_event_frame"] for item in point_records
        )
        second_frames = Counter(
            item["audio_schedule"]["second_event_frame"] for item in point_records
        )
        intervals = Counter(
            item["audio_schedule"]["first_second_onset_interval_samples"]
            for item in point_records
        )
        t5_rows = [item for item in questions if item["type_id"] == "QV2-T5-DOA"]
        design_checks = {
            "native_visual_and_collision": True,
            "fixed_audio_receipts": (
                existing_fixed_audio_root is None
                or all(item["existing_fixed_audio_receipt"] for item in point_records)
            ),
            "randomized_audio_receipts": (
                randomized_audio_root is None
                or all(item["randomized_audio_receipt"] for item in point_records)
            ),
            "motion_answer_balance": balance["status"] == "pass",
            "first_onset_diversity": len(first_frames) >= 2,
            "second_onset_diversity": len(second_frames) >= 2,
            "onset_interval_diversity": len(intervals) >= 2,
            "t5_native_readback": len(t5_rows) == len(point_records)
            and all(math.isfinite(item["answer_numeric"]) for item in t5_rows),
        }
        if not all(design_checks.values()):
            status = "fail"
        elif randomized_audio_root is None:
            status = "pending_randomized_audio_render"
        else:
            status = "pass_research_minipilot_core"
        gate_report = {
            "status": status,
            "gate_B": {
                "native_visual": "pass",
                "collision": "pass",
                "existing_fixed_audio": (
                    "pass" if existing_fixed_audio_root is not None else "not_checked"
                ),
                "randomized_audio": (
                    "pass" if randomized_audio_root is not None else "pending"
                ),
            },
            "gate_C": {
                "design_checks": design_checks,
                "answer_balance": balance,
                "first_event_frame_counts": dict(sorted(first_frames.items())),
                "second_event_frame_counts": dict(sorted(second_frames.items())),
                "interval_sample_counts": dict(sorted(intervals.items())),
                "side_questions": "not_generated_same_side_weak_current_evidence",
                "formal_offscreen": "pending_same_renderer_authority",
                "refusal_quota": "pending_larger_batch",
            },
        }
        question_document = {
            "schema": "avengine_mp3d_region_minipilot_questions_v1",
            "status": status,
            "research_only": True,
            "episode_counted": False,
            "counts": {
                "points": len(point_records),
                "questions": len(questions),
                "by_type": dict(Counter(item["type_id"] for item in questions)),
            },
            "answer_balance": balance,
            "questions": questions,
            "claim_boundary": (
                "Habitat-native readback QA design. Randomized AudioPrograms are "
                "authored but not rendered; refusal/off-screen and admission pending."
            ),
        }
        for name, value in (
            ("questions.json", question_document),
            ("gate_report.json", gate_report),
            (
                "manifest.json",
                {
                    "status": status,
                    "native_batch_root": str(batch_root),
                    "region_index": region_index,
                    "camera_slot_id": camera_slot_id,
                    "base_audio_program": str(Path(base_audio_program_path).resolve()),
                    "schedule_seed_base": schedule_seed_base,
                    "points": [item["point_id"] for item in point_records],
                },
            ),
        ):
            (staging / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return atomic_publish_directory(policy, staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-batch-root", required=True)
    parser.add_argument("--region-index", required=True, type=int)
    parser.add_argument("--camera-slot-id", required=True)
    parser.add_argument("--base-audio-program", required=True)
    parser.add_argument("--schedule-seed-base", type=int, default=0)
    parser.add_argument("--existing-fixed-audio-root")
    parser.add_argument("--randomized-audio-root")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = build_minipilot(
            native_batch_root=args.native_batch_root,
            region_index=args.region_index,
            camera_slot_id=args.camera_slot_id,
            base_audio_program_path=args.base_audio_program,
            output_directory=args.output,
            schedule_seed_base=args.schedule_seed_base,
            existing_fixed_audio_root=args.existing_fixed_audio_root,
            randomized_audio_root=args.randomized_audio_root,
        )
    except (MP3DRegionMiniPilotError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "pass", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
