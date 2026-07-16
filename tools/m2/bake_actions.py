#!/usr/bin/env python3
"""Bake the strict M2 Idle/Walking action artifact and hash-bound report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from avengine.contracts.json_io import sha256_file
from avengine.m2.actions import (
    baked_actions_content_sha256,
    bake_required_actions,
    read_baked_actions_npz,
    write_baked_actions_npz,
)
from avengine.m2.glb import load_glb


REPORT_SCHEMA = "avengine_m2_action_bake_report_v1"


def _output(path: Path, label: str) -> Path:
    result = path.resolve()
    if result.exists() or result.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.input_glb.resolve()
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size <= 0
        or source.suffix.lower() != ".glb"
    ):
        raise SystemExit(f"missing or unsafe input GLB: {source}")
    output = _output(args.output_npz, "baked action artifact")
    report = _output(args.report, "action bake report")
    if output == report:
        raise SystemExit("baked action and report paths must differ")

    actions = bake_required_actions(load_glb(source))
    artifact_sha256 = write_baked_actions_npz(actions, output)
    readback = read_baked_actions_npz(output)
    readback_equal = readback == actions
    if not readback_equal:
        raise SystemExit("canonical baked action readback differs")
    if artifact_sha256 != baked_actions_content_sha256(actions):
        raise SystemExit("baked action content hash differs from file hash")

    action_records = []
    for clip in actions.actions:
        array = np.asarray(clip.rotations_xyzw, dtype=np.float64)
        norms = np.linalg.norm(array, axis=-1)
        action_records.append(
            {
                "semantic_action_id": clip.semantic_action_id,
                "source_action_name": clip.source_action_name,
                "clip_start_seconds": clip.clip_start_seconds,
                "clip_end_seconds": clip.clip_end_seconds,
                "sample_count": clip.sample_count,
                "first_sample_tick": clip.sample_ticks[0],
                "last_sample_tick": clip.sample_ticks[-1],
                "loop_duration_ticks": clip.loop_duration_ticks,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.all(np.isfinite(array))),
                "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
            }
        )
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_glb": {
            "path": str(source),
            "sha256": sha256_file(source),
            "byte_size": source.stat().st_size,
        },
        "artifact": {
            "path": str(output),
            "sha256": artifact_sha256,
            "byte_size": output.stat().st_size,
            "canonical_content_sha256": baked_actions_content_sha256(actions),
            "readback_equal": readback_equal,
        },
        "clock": {
            "sample_rate_hz": actions.sample_rate_hz,
            "time_base_hz": actions.time_base_hz,
        },
        "runtime_joint_order": list(actions.runtime_joint_order),
        "actions": action_records,
    }
    with report.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact": str(output),
                "artifact_sha256": artifact_sha256,
                "report": str(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
