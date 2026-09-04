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
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from avengine.optional_backends.spear_apartment import (
    ANIMATION_TOLERANCE_SECONDS, POSITION_TOLERANCE_CM, ROTATION_TOLERANCE_DEGREES,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _distance(a, b) -> float:
    if len(a) != 3 or len(b) != 3 or not all(math.isfinite(float(v)) for v in (*a, *b)):
        raise ValueError("pose locations must contain three finite values")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _angle_error(a: float, b: float) -> float:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        raise ValueError("pose angles must be finite")
    delta = abs(float(a) - float(b)) % 360.0
    return min(delta, 360.0 - delta)


def _render_intent(frame):
    keys = ("source_slot_id", "actor_id", "translation_ue_cm", "yaw_ue_deg",
            "action_id", "action_phase")
    return {"camera": frame["camera"], "pts_ticks": frame.get("pts_ticks"),
            "actors": sorted(({key: state.get(key) for key in keys}
                              for state in frame["actor_states"]),
                             key=lambda state: state["source_slot_id"])}


def verify_point(point_id: str, point_dir: Path, expected_frames: int | None = None,
                 *, timeline_path: Path | None = None) -> dict:
    records_path = point_dir / "frame_records.json"
    receipt_path = point_dir / "research_receipt.json"
    if not records_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"{point_id}: missing frame_records or research_receipt")
    records, receipt = _read_json(records_path), _read_json(receipt_path)
    if receipt.get("status") not in {"research_only", "pass"}:
        raise ValueError(f"{point_id}: capture receipt did not succeed")
    capture = receipt.get("capture") or {}
    declared_count = capture.get("frame_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count < 1:
        raise ValueError(f"{point_id}: capture frame_count must be a positive integer")
    expected_frames = declared_count if expected_frames is None else expected_frames
    if expected_frames != declared_count or capture.get("completed_frame_count") != expected_frames:
        raise ValueError(f"{point_id}: receipt frame counts differ from the requested clock")
    frames = records.get("frames") or []
    if [frame.get("frame_index") for frame in frames] != list(range(expected_frames)):
        raise ValueError(f"{point_id}: frame indices are not 0..{expected_frames - 1}")
    ticks = capture.get("ticks_per_frame")
    fps = capture.get("frame_rate_hz")
    if (isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1
            or isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or fps <= 0):
        raise ValueError(f"{point_id}: invalid captured render clock")
    if any(frame.get("pts_ticks") != index * ticks for index, frame in enumerate(frames)):
        raise ValueError(f"{point_id}: captured PTS do not follow the declared clock")
    timeline = _read_json(timeline_path) if timeline_path is not None else None
    if timeline is not None:
        wanted = timeline["frames"]
        render = timeline["render"]
        if (len(wanted) != expected_frames or render["frame_rate_hz"] != fps
                or render["ticks_per_frame"] != ticks):
            raise ValueError(f"{point_id}: captured clock differs from the input timeline")
        if any(_render_intent(a) != _render_intent(b) for a, b in zip(wanted, frames)):
            raise ValueError(f"{point_id}: captured intent differs from the input timeline")
    rgb_ref = (receipt.get("artifacts") or {}).get("rgb")
    if not isinstance(rgb_ref, str) or not rgb_ref:
        raise ValueError(f"{point_id}: capture has no RGB array artifact")
    rgb_path = (point_dir / rgb_ref).resolve()
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if (rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != expected_frames
            or rgb.shape[-1] != 3 or min(rgb.shape[1:3]) < 1):
        raise ValueError(f"{point_id}: RGB array shape/type differs from the capture")
    resolution = list(rgb.shape[1:3])
    if timeline is not None and resolution != timeline["render"]["resolution_hw"]:
        raise ValueError(f"{point_id}: RGB resolution differs from the input timeline")
    del rgb

    maxima = {"actor_position_error_cm": 0.0, "actor_yaw_error_deg": 0.0,
              "camera_position_error_cm": 0.0, "camera_yaw_error_deg": 0.0,
              "animation_error_seconds": 0.0}
    actor_ids, animated_slots = set(), set()
    for frame in frames:
        observed = frame.get("observed") or {}
        actor_poses = observed.get("actor_anchor_poses") or frame.get("actor_anchor_poses") or {}
        by_slot = {state["source_slot_id"]: state for state in frame.get("actor_states", [])}
        if not by_slot or len(by_slot) != len(frame["actor_states"]) or set(actor_poses) != set(by_slot):
            raise ValueError(f"{point_id} frame {frame['frame_index']}: actor slots differ")
        expected_animated = {slot for slot, state in by_slot.items() if state.get("action_id") is not None}
        animated_slots.update(expected_animated)
        for slot, expected in by_slot.items():
            actual = actor_poses[slot]
            actor_ids.add(expected.get("actor_id") or f"{slot}_actor")
            maxima["actor_position_error_cm"] = max(maxima["actor_position_error_cm"],
                _distance(expected["translation_ue_cm"], actual["location_cm"]))
            maxima["actor_yaw_error_deg"] = max(maxima["actor_yaw_error_deg"],
                _angle_error(expected["yaw_ue_deg"], actual["rotation_deg"][2]))
        expected_camera = frame["camera"]
        actual_camera = observed.get("camera_pose") or frame.get("camera_pose")
        if not actual_camera:
            raise ValueError(f"{point_id}: no camera readback")
        maxima["camera_position_error_cm"] = max(maxima["camera_position_error_cm"],
            _distance(expected_camera["translation_ue_cm"], actual_camera["location_cm"]))
        maxima["camera_yaw_error_deg"] = max(maxima["camera_yaw_error_deg"],
            _angle_error(expected_camera["yaw_ue_deg"], actual_camera["rotation_deg"][2]))
        readbacks = observed.get("animation_readbacks") or frame.get("animation_readbacks") or []
        if isinstance(readbacks, dict):
            readbacks = list(readbacks.values())
        if len(readbacks) != len(expected_animated) or {item["source_slot_id"] for item in readbacks} != expected_animated:
            raise ValueError(f"{point_id}: animation slots differ from animated actors")
        for item in readbacks:
            recomputed = abs(float(item["observed_position_seconds"]) - float(item["requested_position_seconds"]))
            declared = float(item["absolute_error_seconds"])
            if not math.isfinite(recomputed) or not math.isclose(recomputed, declared, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{point_id}: inconsistent animation readback error")
            maxima["animation_error_seconds"] = max(maxima["animation_error_seconds"], recomputed)
    limits = {"actor_position_error_cm": POSITION_TOLERANCE_CM,
              "camera_position_error_cm": POSITION_TOLERANCE_CM,
              "actor_yaw_error_deg": ROTATION_TOLERANCE_DEGREES,
              "camera_yaw_error_deg": ROTATION_TOLERANCE_DEGREES,
              "animation_error_seconds": ANIMATION_TOLERANCE_SECONDS}
    for field, value in maxima.items():
        if value > limits[field]:
            raise ValueError(f"{point_id}: {field}={value} exceeds renderer tolerance {limits[field]}")
    root_summary = capture.get("root_readback_summary") or {}
    if any(root_summary.get(owner, {}).get("status") != "pass" for owner in (*actor_ids, "camera")):
        raise ValueError(f"{point_id}: root readback summaries did not pass")
    animation_status = (capture.get("animation_readback_summary") or {}).get("status")
    if animation_status != ("pass" if animated_slots else "not_applicable"):
        raise ValueError(f"{point_id}: animation summary disagrees with actor types")
    return {"point_id": point_id, "frame_count": len(frames), "frame_rate_hz": fps,
            "resolution_hw": resolution, "animation_status": animation_status,
            **{f"maximum_{key}": value for key, value in maxima.items()}, "status": "pass"}


def verify_batch(selection_manifest: Path, visual_root: Path,
                 expected_frames: int | None = None) -> dict:
    selection = _read_json(selection_manifest)
    items = selection.get("selected", selection.get("records"))
    if not isinstance(items, list):
        raise ValueError("selection needs selected or records entries")
    point_ids = [item["point_id"] for item in items]
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
    parser.add_argument("--expected-frames", type=int, help="Optional fixed frame count; otherwise read each capture clock")
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
