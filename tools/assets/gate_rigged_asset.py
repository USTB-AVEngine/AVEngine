"""Accept or reject a rigged animal from how its surface deforms in the walk.

The pre-rig gate can tell that a reduction destroyed the head, but it cannot tell
whether the result will look right, and one of its readings actively misleads:
the share of edges bending past a fixed angle punishes a low face count, so the
coarsest asset of a batch scored worst on it while looking best. Face count and
faceting are not independent, and the eye does not read them separately.

Deformation does separate them. Measured on seven assets covering both failure
modes - a collapsed head and a speckled coat - the share of surface area whose
faces grow past ten times during the walk reads 0.011 to 0.024 percent on every
asset that looked right and 0.070 to 0.123 on both that did not. The default
threshold sits in that gap.

This is what lets a pipeline stop early. Run the cheap reduction first, rig it,
measure here, and only escalate to a full retopology if this rejects - which is
cheaper than escalating always, and avoids spending detail on an asset that was
already good enough.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stretch_report")
    parser.add_argument(
        "--max-share-over-10x", type=float, default=0.0003,
        help="share of area whose faces grow past ten times during the walk")
    parser.add_argument(
        "--max-share-over-4x", type=float, default=0.006,
        help="advisory companion; reported and gated loosely, because it does "
             "not separate the assets that looked right from the ones that did "
             "not as cleanly as the ten-times share")
    parser.add_argument("--heading-manifest", default=None)
    args = parser.parse_args()

    with open(args.stretch_report, encoding="utf-8") as handle:
        report = json.load(handle)
    over10 = report["share_area_stretched_over_10x"]
    over4 = report["share_area_stretched_over_4x"]

    failures = []
    if over10 > args.max_share_over_10x:
        failures.append(
            f"area stretched past 10x is {over10} > {args.max_share_over_10x}: "
            "the surface will fold visibly during the walk")
    if over4 > args.max_share_over_4x:
        failures.append(f"area stretched past 4x is {over4} > {args.max_share_over_4x}")

    verdict = {
        "report": args.stretch_report,
        "faces": report.get("faces"),
        "share_over_2x": report.get("share_area_stretched_over_2x"),
        "share_over_4x": over4,
        "share_over_10x": over10,
        # Worst-face growth is deliberately not gated: it is one outlier triangle
        # and it ranks assets wrongly, scoring 162 on an asset that looks better
        # than one scoring 24.
        "max_growth_reported_only": report.get("max_growth"),
        "failures": failures,
    }
    if args.heading_manifest:
        with open(args.heading_manifest, encoding="utf-8") as handle:
            verdict["reviewed_front_yaw_deg"] = json.load(handle)["heading"][
                "reviewed_source_front_yaw_deg"]
    if failures:
        print("RIGGED_GATE_REJECT " + json.dumps(verdict, ensure_ascii=False))
        return 1
    print("RIGGED_GATE_OK " + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
