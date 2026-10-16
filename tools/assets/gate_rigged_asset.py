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

The versions called acceptable top out at 1.92 percent of posed area in shards
and the one called unacceptable sits at 3.71, so any threshold between them fits
the labels. Rigging the same prepared mesh four times moves this reading by about
ten percent, so the threshold needs that much clearance on the accepted side:
2.5 leaves 30 percent above the highest accepted version and 33 percent below the
rejected one.

Splitting shards by where they sit was tried and did not earn its place. Owner
judgement is explicit that tearing under the belly matters less than tearing on a
flank, so the measurement reports downward-facing low shards separately - but on
these labels the visible-only reading separates *worse* than the total, 1.56x
between accepted and rejected against 1.94x. It stays a diagnostic.

Only the shard share is gated. The ten-times area share drew the same line across
those fifteen versions and looked like an independent confirmation, but rigging
the *same* prepared mesh four times moves it from 0.54 to 0.97 percent, a 1.8x
spread on identical input, while the shard share stays inside 2.16 to 2.63. So it
was measuring the draw, not the asset, and it had already produced one false
reject on a route the owner accepted. Worst edge growth is worse still, 11.5 to
25.6 on the same input. Both are reported and neither decides.

The rigger being stochastic at all is worth remembering when a verdict looks
marginal: a reading within ten percent of the threshold is inside the noise, and
the ladder's other rungs are the cheap way to get a second opinion.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walk_report")
    parser.add_argument(
        "--max-shard-share", type=float, default=0.025,
        help="share of posed area in shards at the worst frame; acceptable "
             "versions reach 0.0192 and the rejected one sits at 0.0371, and rig "
             "variance is about ten percent")
    parser.add_argument(
        "--max-share-over-10x", type=float, default=None,
        help="not gated by default: it varies 1.8x between rig attempts on "
             "identical input, so it measures the draw rather than the asset. "
             "Pass a value to gate it anyway")
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
    if args.max_share_over_10x is not None and over10 > args.max_share_over_10x:
        failures.append(f"area stretched past 10x is {over10} > {args.max_share_over_10x}")

    verdict = {
        "report": args.walk_report,
        "faces": report.get("faces"),
        "worst_share_area_shards": shards,
        "worst_share_area_over_10x_reported_only": over10,
        "worst_share_area_shards_visible_reported_only":
            report.get("worst_share_area_shards_visible"),
        "worst_share_area_shards_underside_reported_only":
            report.get("worst_share_area_shards_underside"),
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
