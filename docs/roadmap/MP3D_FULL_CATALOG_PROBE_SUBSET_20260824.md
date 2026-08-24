# MP3D full-catalog native-probe subset (2026-08-24)

## Owner boundary

This slice consumes a complete target-agnostic legal camera-placement catalog
after enumeration. It never changes placement legality. For every camera it
joins the retained 75-frame, two-source Habitat emitter-link readback and emits
columnar per-frame listener-local azimuth, elevation, distance, camera depth,
normalized pinhole centre coordinates and centre-frustum flags. Native
occlusion and semantic pixels remain explicitly pending.

The post-hoc join ignores semantic room names and targets. It has no visibility,
60 cm separation, side-margin, crossing, exchange, off-screen or diversity
gate. Zero-coverage placements remain in the full descriptor JSONL. Source
identity is bound through the existing native capture receipt because the NPY
array shape alone cannot distinguish a silent source-axis swap.

## Probe-only selection

The requested twelve native probes are selected by a deterministic greedy
continuous score. Analytic quality prefers the lower of the two source
centre-frustum coverage fractions, their mean, and a small continuous distance
preference. Novelty is a weighted continuous combination of horizontal camera
distance, circular yaw difference, coverage-profile Jaccard distance and camera
height difference. No term is a pass/fail threshold, and unselected cameras are
not rejected or removed.

Each selected placement receives an independent validated M1 request. This
does not reuse or certify RIR identity; any later listener position must produce
its own audio/RIR evidence.

## Full native floor authority correction

The older catalog at
`/data/jzy/tmp/mp3d_camera_placement_catalog_c9c2274_r1` used the floor exporter
default limit of 128 samples and therefore was not the complete discrete legal
domain. The current integration reran the native floor export without that
limit: 632 candidates passed and 228 failed. The expanded unfiltered catalog is:

`/data/jzy/tmp/mp3d_camera_placement_catalog_all_8c4e7df_r1`

It contains 22,752 placements: 632 legal floor points x low/base/high camera
height x twelve 30-degree yaw values. The clean-code real descriptor/subset run
must consume this expanded catalog. The older 4,608-placement artifact remains
historical evidence only.

## Clean-code CPU verification and real artifact

The implementation commit is
`488f9dbddc0373ca70f0eef32079af9fd1bf9041`. The post-commit related CPU log is:

`/data/jzy/tmp/mp3d_probe_subset_488f9db_unit_r1.log`

It records `119 passed` across the new slice, camera enumeration, camera-pose
probe, retained MP3D viewpoint selector, QA fact-table geometry, M5 metrics and
M1 contracts. Ruff, Ruff formatting and `git diff --check` also pass.

The fresh/no-clobber expanded-catalog output is:

`/data/jzy/tmp/mp3d_probe_subset_488f9db_r1`

It contains one 445,375,034-byte JSONL with exactly 22,752 parsed lines and
unique placement IDs, one source-track record numerically identical after
float64 parsing
to the retained `[75,2,3]` emitter-link array, twelve independent validated M1
requests and the compact selection record. Every descriptor preserves its
original placement record and reports legality effect `none`. Analytic joint
centre-frustum coverage across the full retained catalog is 3,639 placements
with both sources inside for all 75 frames, 1,061 with partial joint coverage
and 18,052 with zero joint coverage; none are removed.

The selected twelve (placement-ID suffixes below) use twelve unique floor
positions, eight yaw values and all
three height roles. Both source centres are analytically inside for `75/75`
frames in every selected candidate:

- `11523`: `[-3.669753,1.272447,-2.005304]`, yaw `60`, low;
- `00141`: `[-10.461485,1.872447,-4.513983]`, yaw `240`, high;
- `06084`: `[-7.191391,1.883496,2.510318]`, yaw `330`, high;
- `22097`: `[3.373525,1.872447,-3.510512]`, yaw `120`, high;
- `22370`: `[3.373525,1.572447,1.757714]`, yaw `30`, base;
- `06598`: `[-6.939846,1.272447,-1.754436]`, yaw `270`, low;
- `15328`: `[-0.148114,1.872447,0.001639]`, yaw `90`, high;
- `14165`: `[-1.154296,1.572447,-4.764852]`, yaw `120`, base;
- `00994`: `[-10.209939,1.572447,1.005111]`, yaw `270`, base;
- `09732`: `[-4.675935,1.272447,0.503375]`, yaw `330`, low;
- `06477`: `[-6.939846,1.872447,-4.263115]`, yaw `240`, high;
- `00839`: `[-10.209939,1.272447,-2.005304]`, yaw `300`, low.

Each M1 request retains the co-located camera/listener rig and binds its Topdown
height to the selected catalog floor Y. The streaming verifier parsed every
descriptor, matched it exactly to the expanded input catalog, checked all
per-source arrays are 75 frames, confirmed native pixel/occlusion fields remain
pending/null, revalidated every request and matched the selected IDs:

`/data/jzy/tmp/mp3d_probe_subset_488f9db_r1_verification.log`

## Verification boundary

This is complete CPU descriptor and request-materialization evidence. No
Habitat/GPU process, native pixel capture, occlusion, audio/RIR, mini-pilot,
viewpoint certification or formal Episode ran in this slice. Analytic
centre-frustum inclusion is not a claim of native visibility.

No Studio placement, hash, frozen contract, baseline or permanent gate is
introduced. The artifact stays `research_only=true`,
`episode_counted=false`, and formal release remains `not_run`.
