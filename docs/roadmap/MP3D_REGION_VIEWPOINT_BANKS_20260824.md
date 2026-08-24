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

The implementation commit, server tests, clean-code 10-region result, selected
pose distributions, fresh artifact and independent per-record verifier will be
recorded here after they exist. Until then, no native viewpoint certification,
Habitat pixels/readback, actor path, visibility, RLR/SoundSpaces audio,
four-point mini-pilot, counted Episode, or formal release is claimed.

This slice adds no hash, frozen contract, baseline, permanent gate, or Studio
placement workflow.
