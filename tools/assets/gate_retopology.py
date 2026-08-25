"""Reject a mesh preparation that damaged the animal, before it is rigged.

Only two readings here are route-independent enough to decide on, and both
describe damage the reduction itself did:

  head-end survival      how much of the front third's triangle share survived,
                         normalised so 1.0 is "lost the same share as the mesh
                         overall". A face collapsed into flat facets reads 0.53
                         where three good assets read 1.18, 1.23 and 1.39. It
                         only works because the reviewed forward direction says
                         which third is the head - an axis-blind version scored
                         0.51 on the ruined mesh and 0.507 on one that had
                         merely thinned its tail.
  face target reached    a mesh whose non-manifold edges block collapse never
                         gets there, and that is worth knowing before rigging.

Everything else is reported and not judged, because it turned out to depend on
which route produced the mesh rather than on whether the result is good:

  faceting               punishes a low face count. The coarsest asset in this
                         batch scored worst on it (0.657) and looked best; a
                         denser one scored 0.383 and looked speckled. Face count
                         and dihedral angle are not independent.
  boundary, non-manifold a remesh drives both to zero, so there is nothing to
                         check; a plain weld-and-collapse inherits whatever the
                         reconstruction had, which is a property of the source
                         and not a defect this step introduced. Two assets that
                         look right carry 1,052 and 1,986 non-manifold edges.
  fidelity               spans fivefold across assets that all look right.

Whether the result actually looks right is decided after rigging, from how the
surface deforms - see gate_rigged_asset.py. That is what lets a pipeline try the
cheap reduction first and stop when it is already good enough.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--min-head-survival", type=float, default=0.7)
    parser.add_argument("--face-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    final = report["stages"]["decimated"]
    bands = report.get("band_survival")
    target = report["target_faces"]

    failures = []
    if bands is None:
        failures.append(
            "no band survival in the report: rerun the retopology with "
            "--front-yaw-deg so the head can be told from the tail")
    elif bands["front"] < args.min_head_survival:
        failures.append(
            f"head-end survival {bands['front']} < {args.min_head_survival}: the "
            "front third paid for the rest of the body")
    if abs(final["faces"] - target) > target * args.face_tolerance:
        failures.append(
            f"faces {final['faces']} missed target {target}: the reduction was "
            "blocked, most likely by non-manifold edges collapse cannot touch")

    verdict = {
        "report": args.report,
        "faces": final["faces"],
        "band_survival": bands,
        "advisory": {
            "faceting_share_over_30deg": report.get("faceting", {}).get(
                "share_over_30deg"),
            "boundary": final["boundary"],
            "nonmanifold": final["nonmanifold"],
            "fidelity_p99": report["fidelity_over_diagonal"]["p99"],
            "source_debris_share": report.get("source_debris_share"),
            "relief_ratio": report.get("relief_ratio"),
            "relief_smooth_iterations": report.get("relief_smooth_iterations"),
        },
        "failures": failures,
    }
    if failures:
        print("RETOPOLOGY_GATE_REJECT " + json.dumps(verdict, ensure_ascii=False))
        return 1
    print("RETOPOLOGY_GATE_OK " + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
