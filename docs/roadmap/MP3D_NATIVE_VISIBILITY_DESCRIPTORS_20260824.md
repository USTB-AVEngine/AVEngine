# MP3D native visibility descriptors (2026-08-24)

## Owner correction

The previous current-MP3D visual capture raised an error whenever either actor
had zero semantic pixels in any frame. That behavior conflated same-renderer
visibility with camera/path legality and deleted the off-screen or fully
occluded evidence needed by later QA difficulty analysis.

The capture now retains all native semantic pixel counts, including zero. It
still fails closed for invalid room/runtime inputs, navmesh/scene drift,
pre-existing actor semantic-ID collisions, camera/actor pose readback errors,
render mutation and time advancement. Only the visibility-as-legality rule is
removed.

## Evidence emitted

The existing `semantic_visibility_pixels.npy` remains a `[75,2]` exact native
array. The research receipt now adds, for each stable actor ID:

- all 75 per-frame pixel counts;
- visible-frame count;
- zero-visible frame indices;
- minimum and maximum pixels.

The receipt states `all_frames_visibility_required=false`, candidate filtering
`disabled`, and `candidate_legality_effect=descriptor_only`. These values do
not certify projection, amodal visibility or physical collision; they preserve
same-renderer modal semantic observations for later off-screen, occlusion and
difficulty strata.

## Verification boundary

The focused mock capture now makes actor1 disappear only on frame 74 and proves
that the full 75-frame capture succeeds, `[1,0]` is retained for that frame,
actor0 has 75 visible frames, actor1 has 74, and the zero frame is reported
without filtering the candidate. Existing no-actor semantic-ID collision tests
still reject a scene that uses a reserved actor ID before actor instantiation.

The authoritative focused log is:

`/data/jzy/tmp/mp3d_visibility_descriptor_unit_r3.log`

It records `30 passed` across current visual capture/review, MP3D dynamic audio
and explicit viewpoint tracks. The earlier `r1` and `r2` runs reached the newly
successful receipt path but exposed missing fields in the old failure-oriented
test fixture (`habitat_sim.__file__`, then runtime/input provenance fields).
After inspecting the actual receipt writer, those fixture fields were supplied;
no production check was weakened. Ruff, formatting and `git diff --check` pass.

This checkpoint is CPU/unit evidence until a fresh real Habitat dynamic capture
runs. It does not claim a certified viewpoint, audio/RIR, a question, a
four-point mini-pilot or a formal Episode. No new hash, frozen contract,
baseline or permanent gate is added.

## Clean-code six-view native execution

The implementation was committed as
`509838e81abef0d044ea60c550584a73846acb17`. After a GPU/process preflight,
all six M1 viewpoint requests were captured sequentially on otherwise idle GPU
0 into fresh roots named:

`/data/jzy/tmp/mp3d_dynamic_visibility_509838e_viewXX_r1`

All six research receipts report 75 frames, `research_only=true` and
`episode_counted=false`. Each output contains co-registered RGB/depth/semantic,
actor matrices, true emitter-link source positions, per-frame visibility and
runtime readback records. Visibility observations are:

- view 01: actor0 `0/75`, actor1 `0/75` visible frames;
- view 02: actor0 `0/75`, actor1 `0/75`;
- view 03: actor0 `25/75`, actor1 `34/75`, maxima 16 and 23 pixels;
- view 04: actor0 `0/75`, actor1 `0/75`;
- view 05: actor0 `0/75`, actor1 `0/75`;
- view 06: actor0 `0/75`, actor1 `75/75`, maxima 0 and 16 pixels.

The summary log is:

`/data/jzy/tmp/mp3d_dynamic_visibility_509838e_summary_r1.log`

The true emitter arrays are finite `[75,2,3]` and exactly equal across all six
camera-only variants, as expected. Start positions are
`[-4.655152,0.481077,-2.680603]` and
`[-4.405152,0.481077,-4.080603]`; end positions are
`[-5.255169,0.474952,-1.937720]` and
`[-5.005169,0.474952,-3.337720]`.

I inspected RGB frames 0, 37 and 74 for every view in:

`/data/jzy/tmp/mp3d_dynamic_visibility_509838e_contact_sheet_r1.png`

The room composition matches the earlier static probes. Views 01, 02, 04 and
05 contain no visible dog in those frames; view 03 has only tiny intermittent
semantic footprints, and view 06 retains only a tiny actor1 footprint. These
are valid off-screen/occlusion observations, not camera failures. They also
show that the present six-view bank is insufficient by itself for a balanced
mix of visually grounded and off-screen questions; the generic camera/yaw
iterator must add candidates before 4--8-view certification.

This native run proves the visibility descriptor and supplies actual emitter
tracks. It still does not provide audio/RIR, question authoring or a four-point
mini-pilot.
