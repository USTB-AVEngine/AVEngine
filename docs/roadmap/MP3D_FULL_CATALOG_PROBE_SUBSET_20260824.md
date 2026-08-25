# MP3D full-catalog native-probe subset (2026-08-24)

## Owner boundary

This slice consumes a complete target-agnostic legal camera-placement catalog
after enumeration. It never changes placement legality. For every camera it
joins the retained 75-frame, two-source Habitat emitter-link readback and emits
columnar per-frame listener-local azimuth, elevation, distance, camera depth,
normalized pinhole centre coordinates and centre-frustum flags. Native
occlusion and semantic pixels remain explicitly pending.

The implementation loads the input catalog as one JSON object and keeps the
compact candidate summaries resident while ranking. Only descriptor JSONL
publication and the independent verifier process descriptor records one line at
a time. This slice makes no low-memory or true streaming-catalog claim.

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
height to the selected catalog floor Y. The line-by-line verifier parsed every
descriptor, matched it exactly to the expanded input catalog, checked all
per-source arrays are 75 frames, confirmed native pixel/occlusion fields remain
pending/null, revalidated every request and matched the selected IDs:

`/data/jzy/tmp/mp3d_probe_subset_488f9db_r1_verification.log`

## Verification boundary

This is complete CPU descriptor and request-materialization evidence. No
Habitat/GPU process, native pixel capture, occlusion, audio/RIR, mini-pilot,
viewpoint certification or formal Episode ran in this slice. Analytic
centre-frustum inclusion is not a claim of native visibility.

## Provenance and input-authority follow-up

The first artifact above predated strict cross-file provenance and catalog
authority checks. Its literal same-renderer provenance label must not be treated
as verified by the old code alone, and this follow-up does not retroactively
upgrade it or rerun the 445 MB output.

The follow-up requires a current `research_only` native-Habitat receipt for the
same room, with two ordered actor/source/emitter-link bindings and 75 native
frames. Receipt-declared source-position and frame-record artifacts must resolve
as ordinary files contained by the receipt root. Every frame record must match
the supplied `[75,2,3]` NPY positions exactly. The receipt-declared M1 request
must be valid and equal to the supplied base M1 in every camera-independent
field; only `request_id` and `world_from_rig` may differ. The receipt-declared M2
path must resolve exactly to the new explicit `--source-m2-request` input.

Only after those checks pass may a future artifact write verified same-renderer
emitter-link provenance and the detailed verification facts. Similarly, input
placement preservation is written only after the catalog reports the current
target-agnostic artifact status, live native PathFinder floor authority and the
required legal placement status on every record. These are ordinary cross-file
validations against a concrete stale/pending-input failure; they add no hash,
frozen contract, baseline or data/difficulty gate.

The focused and related CPU log is:

`/data/jzy/tmp/mp3d_probe_subset_followup_f554eb4_unit_r1.log`

It records `134 passed`, including negative cases for missing native receipt
fields, source-axis reversal, artifact escape, frame/NPY mismatch,
camera-independent M1 drift, explicit M2 mismatch, catalog authority drift and
pending/invalid placement status. Ruff, Ruff formatting and `git diff --check`
also pass.

The actual retained receipt, frame records, NPY, M1, M2 and expanded
22,752-placement catalog pass the read-only authority check in:

`/data/jzy/tmp/mp3d_probe_subset_followup_f554eb4_authority_r1.log`

This validates real inputs without republishing the old descriptor bundle.

## Integration clean-code republication

The full implementation and provenance follow-up were merged into the clean
server integration worktree as `37317f4`. The exact related suite was rerun
there and records `134 passed` at:

`/data/jzy/tmp/mp3d_probe_subset_integration_37317f4_unit_r1.log`

A new fresh/no-clobber artifact was then published from that merge commit:

`/data/jzy/tmp/mp3d_probe_subset_37317f4_r1`

The build completed with exit zero in 2 minutes 14 seconds and used 188,656 KiB
maximum RSS. Its descriptor JSONL is 447,082,406 bytes. An independent
line-by-line verifier parsed all 22,752 unique placement records, matched the
complete input catalog, revalidated all twelve independent M1 requests and
confirmed the source-track provenance and catalog authority facts. It reports
`status=pass` at:

`/data/jzy/tmp/mp3d_probe_subset_37317f4_r1_verification.log`

This republication still performs no native probe, pixel capture, audio/RIR or
viewpoint certification. The selected twelve remain bounded pipeline probes,
not a per-room 4--8-view bank.

No Studio placement, hash, frozen contract, baseline or permanent gate is
introduced. The artifact stays `research_only=true`,
`episode_counted=false`, and formal release remains `not_run`.
