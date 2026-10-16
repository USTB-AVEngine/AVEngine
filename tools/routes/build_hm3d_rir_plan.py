#!/usr/bin/env python3
"""Turn an HM3D floor bank and an accepted listener pose into a QA plan-dir.

The certified QA chain aggregates from a plan directory holding
trajectory_bank.json and rir_job_plan.json. The HM3D route banks already ARE
trajectory banks - the navmesh tool builds them with the same
TrajectoryBankBuilder the kujiale planner uses, so the schema matches file
for file - and the listener pose chooser already picks and auditions the
listener. What was missing is only the join: dedup the bank's source states
against that listener into the reusable RIR work plan, with the chain's own
validator as the acceptance gate.

The listener orientation is derived from the pose's accepted aim the same
way the video camera derives it: yaw about +Y from the aim vector via
atan2(-x, -z). That equivalence is load-bearing - this repository once had
two tools each deriving "yaw" differently and the two orientations were
sixty degrees apart - so the quaternion here is checked by rotating -Z and
comparing against the recorded aim before anything is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools/acoustics"))

from build_asset_bound_rir_plan import _load_bank  # noqa: E402
from avengine.contracts.json_io import write_json  # noqa: E402
from avengine.routes.room_feasibility import build_rir_job_plan  # noqa: E402
from avengine.acoustics.rir_cache import validate_rir_job_plan  # noqa: E402


def yaw_quaternion_wxyz(aim_world: list[float]) -> tuple[float, ...]:
    x, _y, z = (float(v) for v in aim_world)
    norm = math.hypot(x, z)
    if norm < 1.0e-9:
        raise SystemExit("listener aim is vertical; no yaw is derivable")
    yaw = math.atan2(-x / norm, -z / norm)
    half = yaw / 2.0
    quaternion = (math.cos(half), 0.0, math.sin(half), 0.0)

    # verify before trusting: rotating -Z by this quaternion must reproduce
    # the recorded aim direction
    w, _qx, qy, _qz = quaternion
    forward_x = -(2.0 * w * qy)
    forward_z = -(1.0 - 2.0 * qy * qy)
    dot = forward_x * (x / norm) + forward_z * (z / norm)
    if dot < 0.999:
        raise SystemExit(
            f"quaternion self-check failed: rotated forward disagrees with the "
            f"recorded aim (dot {dot:.4f})"
        )
    return quaternion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path,
                        help="one floor's *.bank.json from the HM3D route bank")
    parser.add_argument("--listener-pose", required=True, type=Path,
                        help="pose file from choose_listener_pose (accepted entry used)")
    parser.add_argument("--stride-frames", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path,
                        help="plan directory to create (fresh)")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists (fresh/no-clobber): {output}")

    bank = _load_bank(args.bank)
    pose = json.loads(args.listener_pose.read_text(encoding="utf-8"))
    accepted_index = pose.get("accepted_index")
    if accepted_index is None:
        raise SystemExit(
            "the pose file has no accepted_index: run the ambisonic audition "
            "first so the plan uses the same listener the audio accepted"
        )
    candidate = pose["candidates"][int(accepted_index)]
    position = [float(v) for v in candidate["position_m"]]
    orientation = yaw_quaternion_wxyz(candidate["aim_world"])

    plan = build_rir_job_plan(
        bank,
        listener_position_m=position,
        listener_orientation_wxyz=orientation,
        stride_frames=args.stride_frames,
    )
    jobs = validate_rir_job_plan(plan)

    output.mkdir(parents=True)
    write_json(output / "trajectory_bank.json",
               json.loads(args.bank.read_text(encoding="utf-8")))
    write_json(output / "rir_job_plan.json", plan)
    print(json.dumps({
        "plan_dir": str(output),
        "episodes": bank.frame_count and len(bank.episodes),
        "rir_jobs_after_dedup": len(jobs),
        "listener_position_m": position,
        "listener_from": f"accepted candidate {accepted_index} of {args.listener_pose}",
        "validated_by": "avengine.acoustics.rir_cache.validate_rir_job_plan",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
