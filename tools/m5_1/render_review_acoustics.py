#!/usr/bin/env python3
"""Render and retain variable-duration M5.1 binaural RIR evidence."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.m4.runtime import M4SimulationConfig
from avengine.m5_1.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_rir_sequence,
    research_review_trajectory_record,
)


SOURCE_IDS = (
    "source0",
    "source1",
)
ANCHOR_INDICES = (1, 2)


def _absolute_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one persistent-context dynamic binaural RIR sequence from "
            "an M5.1 mixed-capture anchor array."
        )
    )
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--acoustic-package-manifest", required=True, type=Path)
    parser.add_argument("--m4-request", required=True, type=Path)
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    parser.add_argument("--listener-position-m", nargs=3, type=float, required=True)
    parser.add_argument("--listener-yaw-deg", type=float, required=True)
    return parser.parse_args(argv)


def _save_npy(path: Path, value: np.ndarray, *, root: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        readback.shape != array.shape
        or readback.dtype != array.dtype
        or not np.array_equal(readback, array)
    ):
        raise RuntimeError(f"retained array differs on readback: {path}")
    return {
        **file_record(path, relative_to=root),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "readback_verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture = args.capture_dir.resolve()
    acoustic_manifest = args.acoustic_package_manifest.resolve()
    m4_request_path = args.m4_request.resolve()
    hrtf = args.hrtf.resolve()
    destination = args.output_dir.resolve()
    staging = destination.with_name(f".{destination.name}.staging")
    if os.path.lexists(destination) or os.path.lexists(staging):
        raise RuntimeError(
            f"refusing to replace existing output or staging path: {destination}"
        )
    required = (
        capture / "arrays" / "anchor_positions_m.npy",
        capture / "evidence.json",
        acoustic_manifest,
        m4_request_path,
        hrtf,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing M5.1 acoustic input(s): {missing}")

    anchors = np.load(required[0], allow_pickle=False)
    if (
        anchors.ndim != 3
        or anchors.shape[1:] != (3, 3)
        or anchors.shape[0] < 1
        or not np.all(np.isfinite(anchors))
    ):
        raise RuntimeError("capture anchors must be finite [frame,3,3]")
    if args.fps <= 0 or args.rir_stride_frames <= 0:
        raise RuntimeError("fps and RIR stride must be positive")
    listener = np.asarray(args.listener_position_m, dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise RuntimeError("listener position must be a finite length-3 vector")
    if not math.isfinite(args.listener_yaw_deg):
        raise RuntimeError("listener yaw must be finite")
    half_yaw = math.radians(args.listener_yaw_deg) / 2.0
    orientation_wxyz = (
        math.cos(half_yaw),
        0.0,
        math.sin(half_yaw),
        0.0,
    )
    trajectories = {
        source_id: np.ascontiguousarray(anchors[:, anchor_index, :])
        for source_id, anchor_index in zip(
            SOURCE_IDS, ANCHOR_INDICES, strict=True
        )
    }
    grid = build_strided_review_keyframes(
        trajectories,
        visual_frame_rate_hz=args.fps,
        rir_stride_frames=args.rir_stride_frames,
        listener_position_m=listener,
        listener_orientation_wxyz=orientation_wxyz,
    )
    scene = load_compiled_acoustic_scene(
        acoustic_manifest,
        allow_nonpassing_research_qa=True,
    )
    request = load_json(m4_request_path)
    simulation = M4SimulationConfig.from_mapping(request["simulation"])

    staging.mkdir(parents=True)
    try:
        sequence = render_research_review_binaural_rir_sequence(
            scene,
            simulation,
            grid=grid,
            hrtf_file_path=str(hrtf),
        )
        trajectory = research_review_trajectory_record(grid)
        trajectory["trajectory_content_sha256"] = canonical_json_sha256(
            trajectory
        )
        trajectory_path = staging / "trajectory.json"
        metadata_path = staging / "rir" / "metadata.json"
        write_json(trajectory_path, trajectory)
        write_json(metadata_path, dict(sequence.metadata))
        artifacts = {
            "rir_samples": _save_npy(
                staging / "rir" / "samples.npy", sequence.samples, root=staging
            ),
            "rir_lengths": _save_npy(
                staging / "rir" / "lengths.npy", sequence.lengths, root=staging
            ),
            "trajectory": file_record(trajectory_path, relative_to=staging),
            "rir_metadata": file_record(metadata_path, relative_to=staging),
        }
        evidence: dict[str, Any] = {
            "schema": "avengine_m5_1_dynamic_binaural_rir_evidence_v1",
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": (
                "Legacy-Apartment research-review binaural RIR sequence; no "
                "room, material, asset, episode, or dataset admission claim"
            ),
            "capture_evidence": _absolute_file_record(capture / "evidence.json"),
            "acoustic_package_manifest": _absolute_file_record(acoustic_manifest),
            "acoustic_package_gate": {
                "load_policy": "explicit_nonpassing_research_qa_review_only",
                "package_mode": scene.manifest.get("package_mode"),
                "material_semantics": scene.manifest.get("materials", {}).get(
                    "material_semantics"
                ),
                "qualification_claim": scene.manifest.get("materials", {}).get(
                    "qualification_claim"
                ),
                "qa_status_by_report": {
                    name: report.get("status")
                    for name, report in sorted(scene.qa_reports.items())
                },
            },
            "m4_request": _absolute_file_record(m4_request_path),
            "hrtf": _absolute_file_record(hrtf),
            "listener": {
                "position_m": listener.tolist(),
                "yaw_deg": args.listener_yaw_deg,
                "orientation_wxyz": list(orientation_wxyz),
                "motion": "fixed",
            },
            "source_ids": list(sequence.source_ids),
            "capture_anchor_indices": dict(zip(SOURCE_IDS, ANCHOR_INDICES, strict=True)),
            "visual_frame_count": int(anchors.shape[0]),
            "visual_frame_rate_hz": args.fps,
            "rir_stride_frames": args.rir_stride_frames,
            "rir_keyframe_count": len(grid.keyframes),
            "sample_rate_hz": sequence.sample_rate_hz,
            "episode_sample_count": grid.episode_sample_count,
            "trajectory_sha256": sequence.trajectory_sha256,
            "layout": {
                "type": sequence.layout_type,
                "layout_id": sequence.layout_id,
                "channel_labels": list(sequence.channel_labels),
            },
            "artifacts": artifacts,
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        write_json(staging / "evidence.json", evidence)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(destination / "evidence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
