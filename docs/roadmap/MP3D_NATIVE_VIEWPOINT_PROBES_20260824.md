# MP3D Habitat-native viewpoint probes (2026-08-24)

## Scope

This slice turns the six CPU-authored MP3D camera requests into bounded native
Habitat observations. `tools/m1/probe_camera_pose_native.py` now requires an
explicit installed runtime, Magnum Python site, RLR SDK root, MP3D root and
expected PathFinder island. It derives the intended floor from the request's
single Topdown view instead of snapping the elevated camera eye directly.

For every request the probe:

- loads the validated room and request through the existing M1 validators;
- prepares the explicitly selected installed Habitat runtime without accepting
  MP3D state from ambient environment variables;
- opens a real Simulator and loads the room's declared NavMesh;
- reads back the intended floor snap, island and camera eye height;
- captures RGB, metric depth and semantic observations through the shared M1
  observation writer;
- captures the declared native PathFinder Topdown view; and
- exits with status 2 when the written receipt reports a failure.

The output remains `research_only` in use, is not release evidence, does not
request audio and does not count an Episode. The probe does not add a new hash,
frozen contract, baseline or permanent admission gate.

## Fresh native run

After a process/device preflight, six requests were run sequentially on the
otherwise idle GPU 0. The fresh root is:

`/data/jzy/tmp/mp3d_viewpoint_native_probes_9f501b6_r1`

All six `receipt.json` files report `status=pass`. Measured facts across the six
views are:

- intended floor island: `1` for every view;
- horizontal floor snap error: exactly `0.0 m` for every view;
- maximum floor-height error: `1.934051516161439e-9 m`;
- camera eye heights: five at `1.5 m`, one at `1.2 m`;
- RGB shape: `240x320x4`, with color standard deviation from `33.462` to
  `65.050` (no constant frame);
- positive finite depth fraction: `0.9954036458` to `1.0`;
- semantic unique-ID counts: `12, 27, 30, 19, 35, 45`; and
- Topdown navigable pixels: `21,643` for every view.

Each view contains its own RGB image, depth array and preview, semantic array
and preview, Topdown PNG and receipt. The combined inspection artifacts are:

- `contact_sheet_rgb_depth_semantic.png`;
- `topdown_camera_route_overlay.png`.

## Agent visual inspection

I inspected all six RGB/depth/semantic columns in the contact sheet and the
camera/route Topdown overlay. The observations are deliberately retained as
raw composition clues rather than camera-legality verdicts:

- view 01 is a legal camera pose whose RGB is dominated by a close blank wall;
- view 02 exposes a bedroom/doorway region with foreground clutter;
- view 03 is a corridor/doorway view with a large wall fraction;
- view 04 has a close television foreground with bed and doorway behind it;
- views 05 and 06 expose related bathroom/bedroom directions and are visually
  similar, while still being distinct legal camera poses.

The Topdown overlay confirms that the six camera floor points and both retained
75-frame actor routes share the intended native island. These static probes do
not render either actor. Consequently they do not certify per-frame actor
projection, in-frame state, occlusion or visible pixels. Those fields must stay
pending until the route-track descriptor and same-renderer dynamic evidence are
attached.

Under the owner's current rule, blank-wall fraction, clutter, view similarity,
side margin, route exchange and any fixed pair-distance threshold are not
camera-legality filters. They are preserved for later diversity and difficulty
analysis. A later 4--8-view certified bank may select a diverse subset for a
mini-pilot without deleting these legal raw candidates.

## Committed-code native rerun

The intended-floor probe implementation and its tests were committed as
`839cfdf03994f3690e5181104d9ed983e2fc0c65`. After rechecking GPU/process state,
the same six requests were rerun sequentially from that clean commit on the
otherwise idle GPU 0 into the fresh root:

`/data/jzy/tmp/mp3d_viewpoint_native_probes_839cfdf_r1`

All six committed-code receipts report `status=pass`. The existing observation
array hashes, navigation objects and Topdown artifact hashes are exactly equal
between the earlier run and this committed-code rerun for every view. The
fresh comparison log is:

`/data/jzy/tmp/mp3d_probe_839cfdf_r1_vs_prior_r2.log`

It records `all_pass True`, `all_nav_equal True`,
`all_observations_equal True` and `all_topdown_equal True`. An initial inline
comparison command produced only a Python syntax error and stopped; the fixed
single-line comparison used the fresh `_r2` log above.

## Verification

The authoritative focused log is:

`/data/jzy/tmp/mp3d_native_probe_fix_9f501b6_tests_r2.log`

It records `23 passed in 8.83s` for the native probe mocks, MP3D floor-sample
tests and viewpoint-prefilter tests. Ruff, Python compilation and `git diff
--check` also passed. An earlier `r1` command named two nonexistent test files,
ran zero tests and stopped; the actual test names were listed before the fresh
`r2` run.

This checkpoint is not a four-point mini-pilot and does not certify any view for
production. Still required are committed per-frame raw route descriptors,
same-renderer actor visibility evidence, audio/RIR identity, question authoring,
answer reports and agent review of the eventual dynamic frames.
