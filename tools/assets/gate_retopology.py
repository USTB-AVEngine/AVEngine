"""Accept or reject a retopology report before spending a rigging slot on it.

The three readings that decide whether a reduction kept the animal:

  boundary edges         a remeshed surface has effectively none, and a hole is
                         where the skin opens during a walk; the allowance is
                         for the stray edge that welding leaves behind, not for
                         a real hole, which arrives in the hundreds
  octant survival span   how unevenly the reduction fell across the body; a
                         starved octant is the head paying for the rest
  fidelity p99           how far the original surface had to move

Thresholds are arguments rather than constants because they were measured on
four animals, which is enough to set a default and not enough to freeze one.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument(
        "--max-boundary", type=int, default=8,
        help="not zero: welding the measurement copy leaves an occasional stray "
             "edge, while a genuine hole shows up in the hundreds")
    parser.add_argument("--max-nonmanifold", type=int, default=16)
    parser.add_argument("--min-octant-survival", type=float, default=0.7)
    parser.add_argument("--max-octant-survival", type=float, default=1.5)
    parser.add_argument("--max-fidelity-p99", type=float, default=0.0025)
    parser.add_argument("--face-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    final = report["stages"]["decimated"]
    low, high = report["octant_survival_span"]
    p99 = report["fidelity_over_diagonal"]["p99"]
    target = report["target_faces"]

    failures = []
    if final["boundary"] > args.max_boundary:
        failures.append(f"boundary edges {final['boundary']} > {args.max_boundary}")
    if final["nonmanifold"] > args.max_nonmanifold:
        failures.append(
            f"non-manifold edges {final['nonmanifold']} > {args.max_nonmanifold}")
    if low < args.min_octant_survival:
        failures.append(
            f"octant survival {low} < {args.min_octant_survival}: one part of the "
            "body paid for the rest")
    if high > args.max_octant_survival:
        failures.append(f"octant survival {high} > {args.max_octant_survival}")
    if p99 > args.max_fidelity_p99:
        failures.append(f"fidelity p99 {p99} > {args.max_fidelity_p99}")
    if abs(final["faces"] - target) > target * args.face_tolerance:
        failures.append(f"faces {final['faces']} missed target {target}")

    verdict = {
        "report": args.report,
        "faces": final["faces"],
        "boundary": final["boundary"],
        "nonmanifold": final["nonmanifold"],
        "octant_survival_span": [low, high],
        "fidelity_p99": p99,
        "relief_ratio": report.get("relief_ratio"),
        "relief_smooth_iterations": report.get("relief_smooth_iterations"),
        "failures": failures,
    }
    if failures:
        print("RETOPOLOGY_GATE_REJECT " + json.dumps(verdict, ensure_ascii=False))
        return 1
    print("RETOPOLOGY_GATE_OK " + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
