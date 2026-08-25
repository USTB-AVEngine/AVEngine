"""Accept or reject a rigged animal from how its surface tears during the walk.

This is the reading that decides whether an asset looks right, and it is separate
from the pre-rig gate on purpose: the two failure modes do not show up in the
same number. A standard Burmese whose face collapsed into flat facets tears less
than a Jack Russell that looks fine (1.07 against 1.27 percent of posed area in
shards), so no tearing threshold will ever catch it - that one is caught before
rigging, by head-third survival. This gate is only responsible for tearing and
speckle.

Two earlier versions of this measurement were wrong in ways worth recording.
One sampled a single pose at 35 percent through the action, which understated the
worst frame by ten to thirteen times. The other weighted everything by area,
while the artifact that dominates what a viewer sees is a shard - a triangle
stretched into a long thin sliver that fans open at an armpit or a hip and
carries almost no area. Shards are now found by how far a face's longest edge
grew, over every sampled frame of the cycle, and the worst frame decides.

Both thresholds are calibrated against owner judgement on fifteen rigged versions
at ordinary viewing distance, which is the scale these assets are used at.
Magnified four times every version in the batch shows some tearing, so a
threshold argued from magnified stills would reject everything and mean nothing.
Recalibrate if the delivery scale changes.

The three versions called acceptable top out at 1.52 percent of posed area in
shards and the three called unacceptable start at 2.85, so the shard threshold
sits at 2.0 - about 1.3x clear on either side. The ten-times area share draws the
same line independently, 0.52 against 0.86, which is why both are gated: two
unrelated readings agreeing on all fifteen versions is a stronger gate than
either alone.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walk_report")
    parser.add_argument(
        "--max-shard-share", type=float, default=0.020,
        help="share of posed area in shards at the worst frame; acceptable "
             "versions reach 0.0152 and unacceptable ones start at 0.0285")
    parser.add_argument(
        "--max-share-over-10x", type=float, default=0.007,
        help="second, independent reading of the same judgement: acceptable "
             "versions reach 0.0052 and unacceptable ones start at 0.0086")
    args = parser.parse_args()

    with open(args.walk_report, encoding="utf-8") as handle:
        report = json.load(handle)
    shards = report["worst_share_area_shards"]
    over10 = report["worst_share_area_over_10x"]

    failures = []
    if shards > args.max_shard_share:
        failures.append(
            f"shards cover {shards} of posed area at frame "
            f"{report['worst_frame_by_shards']} > {args.max_shard_share}: the "
            "surface tears open visibly during the walk")
    if over10 > args.max_share_over_10x:
        failures.append(f"area stretched past 10x is {over10} > {args.max_share_over_10x}")

    verdict = {
        "report": args.walk_report,
        "faces": report.get("faces"),
        "worst_share_area_shards": shards,
        "worst_share_area_over_10x": over10,
        "worst_share_area_over_4x": report.get("worst_share_area_over_4x"),
        "worst_frame": report.get("worst_frame_by_shards"),
        "frames_sampled": len(report.get("frames_sampled", [])),
        # Max growth is reported and never gated: it is one outlier triangle and
        # it ranks assets wrongly.
        "worst_max_edge_growth_reported_only": report.get("worst_max_edge_growth"),
        "failures": failures,
    }
    if failures:
        print("RIGGED_GATE_REJECT " + json.dumps(verdict, ensure_ascii=False))
        return 1
    print("RIGGED_GATE_OK " + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
