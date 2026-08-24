# MP3D explicit viewpoint-track descriptors (2026-08-24)

## Scope

`tools/m5/describe_mp3d_viewpoint_tracks.py` is a CPU-only consumer for a
retained MP3D viewpoint bank, its exact two-actor route explanation and one
passing native static-probe receipt per view. It does not launch Habitat and it
does not select or reject cameras.

For every viewpoint the tool:

- loads both caller-supplied `actor_skin_root_paths_m` arrays and requires
  exactly 75 finite frames without endpoint interpolation;
- preserves the two actor IDs through every frame, including crossings or
  endpoint exchanges;
- derives the listener position and forward/right/up basis from the exact M1
  `world_from_rig` quaternion;
- calls the shared explicit-track descriptor with Habitat Y-up mapping
  `(horizontal X/Z, vertical Y)`;
- records per-frame positions, horizontal/XYZ distance, closest frames,
  observed coincidence/order changes and listener-local azimuth/elevation;
- binds the request to its passing intended-floor native receipt and retains
  the RGB/depth/semantic/Topdown artifact records.

The route authority is explicitly `visual_actor_skin_root_not_audio_emitter`.
Audio endpoint/emitter tracks remain separate work. The static native probes do
not contain either actor, so projection, in-frame state, occlusion and visible
pixels remain `pending/null`; the tool never infers them from endpoint geometry.

## Legality and difficulty boundary

The complete legal raw viewpoint catalog is retained. Side margin, pair
distance, crossing/exchange, composition, blank-wall fraction, projection,
occlusion and view similarity do not filter this output. Native collision and
runtime actor-pose readback are also not claimed by this CPU descriptor.

The output is an ordinary research artifact with `research_only=true`,
`episode_counted=false`, formal release not run, candidate filtering `none` and
fresh/no-clobber atomic publication. No new hash, frozen contract, baseline or
permanent gate is introduced.

## Verification before real-data execution

Focused server verification is retained at:

`/data/jzy/tmp/mp3d_track_descriptor_unit_r1.log`

It records `28 passed` across the new tool, the shared explicit-track helper and
the MP3D viewpoint selector. Tests prove exact hold/motion frame preservation,
identity binding, Habitat axis mapping, explicit pending visibility, no-clobber,
nonpassing-receipt failure, exact 75-frame requirement and bank-path containment.
Ruff, formatting, Python compilation and `git diff --check` also pass.

The first clean-code real-artifact attempt stopped before publication because
the valid `839cfdf` receipts predate the later convenience fields
`topdown.required` and `topdown.passed`; they retain the authoritative Topdown
`reports/checks` instead. Room, request, intended-floor navigation,
camera/listener transform and all observation modalities matched. No output was
published.

The consumer now accepts both receipt generations and explicitly derives the
two convenience booleans only when one retained Topdown report and passing
checks prove them. The normalized record says whether this derivation occurred.
A fresh real run remains required from the follow-up commit.

## Clean-code real-data run

The compatibility fix was committed as
`098eef85ec88bdfc0f8403cbdf7c6fae0a042d83`. Its fresh real-data output is:

`/data/jzy/tmp/mp3d_viewpoint_raw_tracks_098eef8_r1`

The ordinary JSON artifact contains all six legal viewpoint candidates and is
1,200,614 bytes. Each viewpoint retains exactly 75 frames for `actor0` and
`actor1`; the stored positions compare equal frame-for-frame to the retained
route explanation. Source identity binding is preserved, candidate filtering
is `none`, and all six passing `839cfdf` native static probes are attached.

For this retained two-Beagle route, the horizontal pair-distance minimum is
`1.422146 m` at frame 17 for every viewpoint. This equality is expected because
the pair track is independent of camera pose. The route does not exchange
endpoints and does not reverse pairwise listener-right order in these six
cameras. Those are recorded observations, not reasons to select or reject any
view.

Every projection/in-frame/occlusion/visible-pixel field remains explicitly
pending/null. The old Topdown receipts are normalized with
`summary_fields_derived_from_reports_checks=true`; no native value is invented.

The fresh verifier is:

`/data/jzy/tmp/mp3d_viewpoint_raw_tracks_098eef8_r1_verification.log`

It reports `status=pass` for six views, no filtering, 75 frames per view, exact
positions, identity binding, passing static probes and pending visibility. The
first `51ea758` attempt published no output; the clean successful artifact above
comes from the committed compatibility fix.

This is still not dynamic actor-pixel evidence, an audio-emitter trajectory,
an RIR/audio run, question authoring or a four-point mini-pilot.

## Runtime emitter-track binding

The consumer now accepts one repeated `--dynamic-capture-dir` per viewpoint.
When supplied, the directories must form an exact M1-request bijection with the
bank. The tool validates the research receipt, room, 75-frame native run,
source/actor identities, no-filter visibility policy, contained artifact paths,
finite `[75,2,3]` emitter positions and nonnegative `[75,2]` semantic pixel
array. Receipt per-actor counts must equal that array.

The same explicit-track helper then describes the actual emitter positions,
while native visible-pixel counts are attached per frame. Projection, in-frame
and occlusion remain pending rather than inferred from zero pixels. Static-only
operation remains supported. A fresh real six-view binding is required after
the implementation commit.
