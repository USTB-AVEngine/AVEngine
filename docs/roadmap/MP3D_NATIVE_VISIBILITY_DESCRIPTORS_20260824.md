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
