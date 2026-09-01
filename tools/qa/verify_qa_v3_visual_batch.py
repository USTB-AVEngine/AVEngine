#!/usr/bin/env python3
"""Verify a materialized QA-v3 visual batch against its runtime readbacks.

The verifier reads only the final selection manifest and capture artifacts.  It
checks point coverage, frame indices, expected versus observed camera/actor
poses, and animation readback errors, then writes a fresh no-clobber summary.
It does not launch Unreal and does not grant question or dataset admission.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _angle_error(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 360.0
    return min(delta, 360.0 - delta)


def verify_point(point_id: str, point_dir: Path, expected_frames: int) -> dict:
    records_path = point_dir / "frame_records.json"
    receipt_path = point_dir / "research_receipt.json"
    if not records_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"{point_id}: missing frame_records or research_receipt")
    records = _read_json(records_path)
    receipt = _read_json(receipt_path)
    frames = records.get("frames") or []
    indices = [int(frame["frame_index"]) for frame in frames]
    if indices != list(range(expected_frames)):
        raise ValueError(
            f"{point_id}: frame indices are not 0..{expected_frames - 1}")
    capture = receipt.get("capture") or {}
    if int(capture.get("completed_frame_count", -1)) != expected_frames:
        raise ValueError(
            f"{point_id}: receipt completed_frame_count is not {expected_frames}")
    if int(capture.get("frame_count", -1)) != expected_frames:
        raise ValueError(
            f"{point_id}: receipt frame_count is not {expected_frames}")

    maximum_actor_position_error_cm = 0.0
    maximum_actor_yaw_error_deg = 0.0
    maximum_camera_position_error_cm = 0.0
    maximum_camera_yaw_error_deg = 0.0
    maximum_animation_error_seconds = 0.0
    slots = set()
    for frame in frames:
        observed = frame.get("observed") or {}
        actor_poses = (observed.get("actor_anchor_poses")
                       or frame.get("actor_anchor_poses") or {})
        by_slot = {state["source_slot_id"]: state
                   for state in frame.get("actor_states", [])}
        if set(actor_poses) != set(by_slot):
            raise ValueError(
                f"{point_id} frame {frame['frame_index']}: actor slots differ")
        slots.update(by_slot)
        for slot, expected in by_slot.items():
            actual = actor_poses[slot]
            maximum_actor_position_error_cm = max(
                maximum_actor_position_error_cm,
                _distance(expected["translation_ue_cm"], actual["location_cm"]))
            maximum_actor_yaw_error_deg = max(
                maximum_actor_yaw_error_deg,
                _angle_error(expected["yaw_ue_deg"], actual["rotation_deg"][2]))

        expected_camera = frame["camera"]
        actual_camera = observed.get("camera_pose") or frame.get("camera_pose")
        if not actual_camera:
            raise ValueError(
                f"{point_id} frame {frame['frame_index']}: no camera readback")
        maximum_camera_position_error_cm = max(
            maximum_camera_position_error_cm,
            _distance(expected_camera["translation_ue_cm"],
                      actual_camera["location_cm"]))
        maximum_camera_yaw_error_deg = max(
            maximum_camera_yaw_error_deg,
            _angle_error(expected_camera["yaw_ue_deg"],
                         actual_camera["rotation_deg"][2]))

        readbacks = (observed.get("animation_readbacks")
                     or frame.get("animation_readbacks") or [])
        if {item["source_slot_id"] for item in readbacks} != set(by_slot):
            raise ValueError(
                f"{point_id} frame {frame['frame_index']}: animation slots differ")
        for item in readbacks:
            recomputed = abs(float(item["observed_position_seconds"])
                             - float(item["requested_position_seconds"]))
            declared = float(item["absolute_error_seconds"])
            if not math.isclose(recomputed, declared, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{point_id} frame {frame['frame_index']}: animation "
                    f"error field {declared} != recomputed {recomputed}")
            maximum_animation_error_seconds = max(
                maximum_animation_error_seconds, recomputed)

    root_summary = capture.get("root_readback_summary") or {}
    for slot in slots:
        key = f"{slot}_actor"
        if key not in root_summary:
            raise ValueError(f"{point_id}: receipt lacks {key} root summary")
    animation_summary = capture.get("animation_readback_summary") or {}
    if animation_summary.get("status") != "pass":
        raise ValueError(f"{point_id}: animation readback summary did not pass")
    return {
        "point_id": point_id,
        "frame_count": len(frames),
        "maximum_actor_position_error_cm": maximum_actor_position_error_cm,
        "maximum_actor_yaw_error_deg": maximum_actor_yaw_error_deg,
        "maximum_camera_position_error_cm": maximum_camera_position_error_cm,
        "maximum_camera_yaw_error_deg": maximum_camera_yaw_error_deg,
        "maximum_animation_error_seconds": maximum_animation_error_seconds,
        "status": "pass",
    }


def verify_batch(selection_manifest: Path, visual_root: Path,
                 expected_frames: int) -> dict:
    selection = _read_json(selection_manifest)
    point_ids = [item["point_id"] for item in selection["selected"]]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("selection contains duplicate point ids")
    actual_dirs = {path.name for path in visual_root.iterdir()
                   if path.is_dir() and (path / "frame_records.json").is_file()}
    if actual_dirs != set(point_ids):
        raise ValueError(
            f"visual point coverage differs: missing={sorted(set(point_ids) - actual_dirs)}, "
            f"extra={sorted(actual_dirs - set(point_ids))}")
    results = [
        verify_point(point_id, visual_root / point_id, expected_frames)
        for point_id in point_ids
    ]

    def maximum(field):
        return max(float(result[field]) for result in results) if results else None

    return {
        "schema": "qa_v3_visual_batch_verification_v1",
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": (
            "runtime visual/readback engineering verification only; no question "
            "admission or single-modality certification"),
        "inputs": {
            "selection_manifest": str(selection_manifest.resolve()),
            "visual_root": str(visual_root.resolve()),
            "expected_frames_per_point": expected_frames,
        },
        "counts": {
            "selected_points": len(point_ids),
            "verified_points": len(results),
            "verified_frames": sum(result["frame_count"] for result in results),
            "failures": 0,
        },
        "maxima": {
            "actor_position_error_cm": maximum(
                "maximum_actor_position_error_cm"),
            "actor_yaw_error_deg": maximum("maximum_actor_yaw_error_deg"),
            "camera_position_error_cm": maximum(
                "maximum_camera_position_error_cm"),
            "camera_yaw_error_deg": maximum("maximum_camera_yaw_error_deg"),
            "animation_error_seconds": maximum(
                "maximum_animation_error_seconds"),
        },
        "points": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--expected-frames", type=int, default=75)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    summary = verify_batch(
        args.selection_manifest, args.visual_root, args.expected_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "status": summary["status"],
        "counts": summary["counts"],
        "maxima": summary["maxima"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
