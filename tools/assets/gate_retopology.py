"""Accept or reject a retopology report before spending a rigging slot on it.

The three readings that decide whether a reduction kept the animal:

  boundary edges         a remeshed surface has effectively none, and a hole is
                         where the skin opens during a walk; the allowance is
                         for the stray edge that welding leaves behind, not for
                         a real hole, which arrives in the hundreds
  head-end survival      how much of the front third's triangle share survived.
                         This is the reading that separates a ruined asset from
                         a fine one, and it only works because the reviewed
                         forward direction says which third is the head: a
                         face collapsed into flat facets scores 0.51, and so
                         does a tail that merely thinned, which is why the
                         axis-blind octant span is reported and not judged
  faceting               the share of edges bending past thirty degrees. This is
                         the stipple a reviewer sees on a pale coat, and it
                         ranks assets the way the eye does: 4 percent on the
                         cleanest of these four, 23 on the roughest that still
                         reads well, 48 on the one that reads as speckled
  fidelity p99           how far the original surface had to move. Reported, and
                         gated only against catastrophe: two assets that both
                         look right differ fivefold here (0.0019 and 0.0104),
                         so it cannot carry an accept decision

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
    parser.add_argument("--min-head-survival", type=float, default=0.7)
    parser.add_argument("--max-faceting-share", type=float, default=0.30)
    parser.add_argument("--max-fidelity-p99", type=float, default=0.02)
    parser.add_argument("--face-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    final = report["stages"]["decimated"]
    span = report["octant_survival_span"]
    bands = report.get("band_survival")
    p99 = report["fidelity_over_diagonal"]["p99"]
    facets = report.get("faceting", {}).get("share_over_30deg")
    target = report["target_faces"]

    failures = []
    if final["boundary"] > args.max_boundary:
        failures.append(f"boundary edges {final['boundary']} > {args.max_boundary}")
    if final["nonmanifold"] > args.max_nonmanifold:
        failures.append(
            f"non-manifold edges {final['nonmanifold']} > {args.max_nonmanifold}")
    if bands is None:
        failures.append(
            "no band survival in the report: rerun the retopology with "
            "--front-yaw-deg so the head can be told from the tail")
    elif bands["front"] < args.min_head_survival:
        failures.append(
            f"head-end survival {bands['front']} < {args.min_head_survival}: the "
            "front third paid for the rest of the body")
    if facets is not None and facets > args.max_faceting_share:
        failures.append(
            f"faceting {facets} > {args.max_faceting_share}: the surface will "
            "read as speckle, which more relief smoothing removes")
    if p99 > args.max_fidelity_p99:
        failures.append(f"fidelity p99 {p99} > {args.max_fidelity_p99}")
    if abs(final["faces"] - target) > target * args.face_tolerance:
        failures.append(f"faces {final['faces']} missed target {target}")

    verdict = {
        "report": args.report,
        "faces": final["faces"],
        "boundary": final["boundary"],
        "nonmanifold": final["nonmanifold"],
        "octant_survival_span": span,
        "band_survival": bands,
        "fidelity_p99": p99,
        "faceting_share_over_30deg": facets,
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
