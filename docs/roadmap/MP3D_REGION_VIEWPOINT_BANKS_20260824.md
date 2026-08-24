# MP3D per-region CPU viewpoint banks (2026-08-24)

## Scope

This slice consumes the complete 22,752-placement target-agnostic camera
catalog and its independently verified static `.house` membership sidecar. It
does not regenerate candidates and does not change input legality. Only
`unique` membership is eligible for a named region's four-view quota;
`boundary`, `unknown`, and `ambiguous` records remain present in the global
input statistics and are never described as illegal.

Every region bank must contain four distinct floor positions. A second yaw or
height at an already selected floor point cannot fill that quota. If a region
has fewer than four uniquely-bound floor positions, publication fails with the
actual count instead of copying a position. This is a bank-composition rule,
not a camera-legality rule.

Within each region, all 12 yaw values and three height variants remain
available at every eligible floor point. Greedy ranking uses continuous
boundary-distance and central-height quality preferences plus continuous
horizontal floor-position, circular-yaw, and height novelty. It has no numeric
threshold. The 5 cm near-boundary flag is retained as a descriptor and is not a
gate; even a region whose every point is near the boundary remains selectable.

Region category/code is copied only after selection. Unknown category codes
follow the same algorithm. No semantic target, current two-Beagle route,
source/actor identity, source coverage, analytic/native visibility, 60 cm
separation, side margin, crossing, or exchange fact is a ranking input.

## Output and evidence boundary

The output is a fresh/no-clobber atomically published CPU bundle containing a
single bank record and one independently validated M1 request per selected
placement. Every request keeps the camera and listener co-located and binds
Topdown height to that selected floor point's Y coordinate. The base M1 source
declarations are preserved only so each request remains structurally valid;
they are not used to rank or claim a source-feasible room route.

The implementation commit is
`c13551f101de2f1297b64f4e9f015777bb4d3cd0`. Its first clean-code run correctly
produced 40 distinct floor positions and diverse yaw, but the continuous
central-height preference outweighed height novelty and chose `base` for all 40
requests. That valid first artifact is retained without overwrite at
`/data/jzy/tmp/mp3d_region_viewpoint_banks_c13551f_r1`; it is not the selected
evidence run.

Follow-up commit `81eadfda0ff8102463a571b7312def9695659e4f` changed only
the two continuous quality weights from boundary/central-height `0.90/0.10` to
`0.95/0.05`. It added no threshold or eligibility rule. A read-only replay over
all 22,752 placements predicted height diversity before the follow-up was
committed. From the clean follow-up commit, Ruff, Ruff format checking,
`git diff --check`, and 87 related CPU tests passed. Those tests cover this
slice together with the static region sidecar, camera catalog, existing probe
selector, camera-pose helper, and M1 contracts. The fresh log is:

`/data/jzy/tmp/mp3d_region_viewpoints_81eadfd_unit_r1.log`

## Clean-code current result

The authoritative fresh/no-clobber CPU bundle is:

`/data/jzy/tmp/mp3d_region_viewpoint_banks_81eadfd_r1`

It contains one 106,214-byte bank record, one README, and 40 independent M1
request files. All 10 static region instances have exactly four selections at
four distinct floor positions:

| Region | Category | Eligible floors | Selected yaw degrees | Selected heights |
|---:|---|---:|---|---|
| 0 | bedroom | 108 | 0, 180, 120, 300 | base, low, high, high |
| 1 | bathroom | 65 | 0, 180, 90, 270 | base, low, high, high |
| 2 | toilet | 7 | 0, 180, 120, 270 | base, low, high, high |
| 3 | bathroom | 26 | 0, 180, 150, 150 | base, low, high, base |
| 4 | tv | 73 | 0, 180, 150, 120 | base, low, high, high |
| 5 | dining room | 61 | 0, 180, 180, 60 | base, low, high, high |
| 6 | entryway/foyer/lobby | 85 | 0, 180, 180, 180 | base, low, high, base |
| 7 | living room | 133 | 0, 180, 150, 180 | base, low, high, base |
| 8 | familyroom/lounge | 47 | 0, 180, 180, 90 | base, low, high, base |
| 9 | kitchen | 10 | 0, 180, 180, 180 | base, low, high, high |

Across all 40 requests, height distribution is `high=16`, `base=14`, and
`low=10`. Yaw distribution spans eight values: `0=10`, `60=1`, `90=2`,
`120=3`, `150=4`, `180=17`, `270=2`, and `300=1`. The artifact retains exact
floor indices/coordinates and every per-round score component. None of the 40
selected records is within the 5 cm boundary descriptor, but the algorithm and
tests also prove that an all-near-boundary region remains selectable.

The global input accounting remains unchanged: 632 floor points are retained
as 615 unique and 17 unknown; all 22,752 placements are retained as 22,140
unique and 612 unknown. No unknown was converted to illegal or assigned to a
nearest region.

An independent verifier replayed all 22,140 uniquely-bound candidates and
recomputed each greedy round from position, circular yaw, numeric height, and
boundary distance alone. It matched all 40 winners and every score component,
checked 40 distinct selected floor positions, then independently validated all
40 M1 contracts, unique request IDs, camera translations/yaw quaternions,
co-located listeners, Topdown floor Y, contained request paths, and unchanged
base source declarations. Its fresh log is:

`/data/jzy/tmp/mp3d_region_viewpoint_banks_81eadfd_r1_verification.log`

This completes the current scene's target-agnostic CPU region probe bank. It
does not provide native viewpoint certification, Habitat pixels/readback,
region-specific actor paths, visibility, RLR/SoundSpaces audio, four-point
mini-pilots, a counted Episode, or formal release; all remain explicitly
pending.

This slice adds no hash, frozen contract, baseline, permanent gate, or Studio
placement workflow.
