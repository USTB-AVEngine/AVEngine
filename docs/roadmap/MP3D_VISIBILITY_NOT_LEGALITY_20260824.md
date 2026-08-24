# MP3D visibility is a descriptor, not a capture gate (2026-08-24)

## Concrete failure

The first fresh selected-twelve OBJECT_ID attempt used clean integration
`7487fe0`, the fresh `732f264` Habitat prefix, and an isolated GPU0:

`/data/jzy/tmp/mp3d_selected12_object_id_7487fe0_r1`

View `11523` completed with 75 frames. The wrapper then stopped at the first
failure, view `00141`, with:

`frame 0 lost an actor semantic ID: [0, 0]`

The retained wrapper log is:

`/data/jzy/tmp/mp3d_selected12_object_id_7487fe0_r1.log`

This was an obsolete legality gate in
`src/avengine/m5/visual.py`. It rejected a valid capture whenever either
registered semantic ID had zero pixels. That directly contradicted the owner
decision that in-frame, off-screen and occlusion observations are raw clues,
not candidate or capture legality.

## Implementation

Commit `ad6174b` removes only that semantic-pixel gate. The producer still
counts both semantic IDs for every frame and persists the exact counts. Zero
is retained as an observation.

All existing fail-fast checks remain in force, including:

- finite and unique actor/source/semantic identities;
- exact 75-frame M2/M1 clock alignment;
- camera, sensor, actor and skin-root post-render readback;
- fixed-state render immutability and unchanged Habitat time; and
- OBJECT_ID runtime-object identity and artifact consistency.

The change adds no hash, frozen contract, baseline, candidate filter or
replacement gate. Semantic-ID coverage remains incomplete and diagnostic.

## Fresh native canary

The same `00141` request was rerun from the independent server worktree on an
empty GPU0 with fresh/no-clobber output:

`/data/jzy/tmp/mp3d_semantic_zero_gatefix_7487fe0_00141_r1`

It completed with 75 frames, `research_only=true`,
`episode_counted=false`, and wrapper exit zero. The offline reviewer also
completed:

`/data/jzy/tmp/mp3d_semantic_zero_gatefix_7487fe0_00141_review_r1`

The independent receipt assertion is:

`/data/jzy/tmp/mp3d_semantic_zero_gatefix_7487fe0_00141_r1_verification.log`

It proves both actors have zero OBJECT_ID pixels in all 75 frames while the
receipt retains:

- `zero_pixels_affect_legality=false`;
- `candidate_legality_effect=descriptor_only`;
- 75 frames;
- `research_only=true`; and
- `episode_counted=false`.

This canary proves the concrete gate is removed. It does not by itself certify
P1, amodal visibility, occlusion truth, or a formal off-screen QA item.

## Agent visual review of the isolated A/B

The accepted three-row contact sheet is:

`/data/jzy/tmp/mp3d_object_id_native_732f264_isolated_rerun_r3_contact_sheet.png`

The owner agent inspected it at original resolution. In `06598` and
`11523`, both dogs and their per-actor OBJECT_ID boxes align at frames
0, 37 and 74. In `00677`, no dog or OBJECT_ID box appears at any of the
three inspected frames. This supports the tested modal-raster route and its
negative control only. It is not amodal or occlusion authority.

## CPU evidence

The post-commit adjacent suite is recorded at:

`/data/jzy/tmp/mp3d_semantic_zero_gate_ad6174b_unit_r1.log`

It reports `61 passed` across the visual producer, current visual publisher,
offline reviewer and MP3D raw-clue consumer. Ruff check, Python byte
compilation, `git diff --check`, and the clean branch state are recorded at:

`/data/jzy/tmp/mp3d_semantic_zero_gate_ad6174b_quality_r1.log`

## Remaining boundary

The failed selected-twelve root remains immutable evidence and is not resumed
or counted as a complete batch. After this slice reaches integration, all
twelve views must be rerun into a new fresh root with the per-case GPU0 guard,
reviewed, and then joined to raw clues. Visibility must remain descriptor-only
and must never alter candidate retention.
