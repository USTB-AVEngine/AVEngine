# M5.1 Status — Mixed Real-Room Research Review

Status: bounded research-review `pass` for the completed route, source/event,
legacy-Apartment delivery, and MP3D visual, acoustic, and listening-video
delivery described here. This is not a dataset-admission gate. The retained
source, capture, acoustic, and delivery records explicitly keep
`research_only: true` and/or
`qualification_claim: false` as defined by their schemas.

Formal M5 remains independently `pass`: its fixed two-Beagle canary passed 9/9
declared checks and 12/12 independently recomputed verification groups. M5.1
does not modify Timeline v2 or broaden that formal M5 result.

## What M5.1 adds

M5.1 replaces the controlled two-Beagle review with one animated Rocketbox
human (`source0`) and one animated Beagle (`source1`) in two research-review
rooms. It adds:

- a hash-bound 18-second, 270-frame, 15 fps migration of the old Apartment
  route and camera;
- explicit center-only obstacle/navigation gates;
- a versioned source, taxonomy, provenance, event, relationship, and flag
  manifest;
- retained animated head/mouth/muzzle link trajectories;
- variable-position binaural RIRs and exact dry/stem/mixture WAVs for both
  room reviews;
- an annotated Habitat main-view + right-side Topdown listening video; and
- a direct old UE | new Habitat | new Topdown comparison video.

The Topdown panel is QA-only. It is not a second dataset view, and it does not
change the single-view sensor contract.

## Frozen legacy Apartment result

The route authority is
`examples/m5_1/legacy_apartment/route_manifest.json`. It preserves the old
18-second duration, 270 frames, camera position `[-0.7, 1.471, 0.65]`, 55-degree
Habitat yaw, and 105-degree horizontal FOV. The imported human trajectory hash
is `9da8a3be24b0dcece1c6728982a0fc0f7e61ef8722c22acecdee970137edf1a0`;
the separately derived safe dog route hash is
`312d7f3adcb3fff6ff6555551c9605863cc136c88b7ea4bbd65ba07cacc427c9`.

The zero-radius horizontal center-point AABB gate passed on all 270 positions
for both actors:

| Check | Retained result |
| --- | --- |
| Human center collisions | 0; minimum clearance `0.2 m` |
| Beagle center collisions | 0; minimum clearance `0.0995654373 m` |
| Minimum human/Beagle center separation | `0.3590786749 m` against a `0.3 m` gate |

This proves only that the declared actor roots do not enter the migrated
horizontal obstacle AABBs. It does not prove swept articulated-mesh clearance,
limb clearance, or collision-free skin geometry.

The mixed fixed-state capture is retained at
`tmp/m5_1/mixed_legacy_20260718_01/evidence.json`. It uses an actually
articulated Rocketbox walking action and the validated 45-state Beagle walk
block rather than static or sliding substitutes. Root, spherical/prismatic
joint, skin-FK, semantic-ID, and mouth/muzzle anchor readback passed. Under the
frozen legacy camera, the human semantic ID is visible in 206/270 frames and
the Beagle in 185/270 frames; M5.1 therefore does not claim that either actor
remains visible for the whole legacy clip.

The retained dynamic Apartment acoustics are
`tmp/m5_1/legacy_apartment_acoustics_20260718_05/evidence.json`. A fixed,
camera-co-located listener receives 90 two-source binaural RIR keyframes at
5 Hz, rendered at 16 kHz through the hash-bound MIT KEMAR HRTF. The exact
sequence is stored as `[90, 2 sources, 2 channels, 20516]` float32 samples plus
per-pair valid lengths.

This acoustic run deliberately loaded a non-passing package only through the
explicit research-review escape hatch. Its material semantics are
`research_placeholder`, its qualification is
`unqualified_research_placeholder`, and the retained QA states are:

| Acoustic report | Retained status |
| --- | --- |
| compiler source-to-package parity | `fail` |
| geometry | `fail` |
| material coverage | `pass` |
| ray leakage | `not_run` |

Those states are not rewritten as passing merely because a listening artifact
was generated. No physical room-material, RT60/EDT, renderer-parity, or room
qualification claim follows.

The final Apartment delivery is
`tmp/m5_1/legacy_apartment_delivery_20260718_01/evidence.json`. Its exact
18-second audio authority is the retained 288,000-sample WAV set. The ordinary
MP4 contains a two-channel binaural AAC listening copy:

- `tmp/m5_1/legacy_apartment_delivery_20260718_01/videos/legacy_apartment_habitat_annotated_binaural.mp4`
  — 1280x480, 270-frame annotated Habitat + Topdown review;
- `tmp/m5_1/legacy_apartment_delivery_20260718_01/videos/legacy_apartment_ue_vs_habitat.mp4`
  — 1920x480 old UE | new Habitat | new Topdown comparison; and
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/clips/rocketbox_adults_male_adult_01_original_ue_v1/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4`
  — the hash-bound old reference consumed by the comparison builder.

## Source, event, and flag contract

The versioned schema is
`schemas/m5_1_source_manifest_v1.schema.json`; the executable human+Beagle
example is `examples/m5_1/legacy_apartment/source_manifest.json`.

`source0` is the Rocketbox adult human with the `Bip01 MJaw` emitter and
`voice_en_read_speech` taxonomy. `source1` is the adult Beagle with the
`beagle Xtra Mouth` emitter and `call_canine_bark` taxonomy. Every visual and
audio asset has a URI, SHA-256, origin, and rights state. The human speech
event occupies frames `[75, 171)`. Three Beagle bark events occupy
`[90, 95)`, `[120, 125)`, and `[150, 155)`; each interval overlaps the human
event, so simultaneous emission is explicit rather than inferred.

The manifest retains per-source, pairwise, and clip-level assessments using
`present`, `absent`, and `not_evaluated` states with reasons and evidence.
Unknown FOV/occlusion facts remain `not_evaluated`; they are never coerced to
false. The declared emitter paths are pre-execution root-plus-offset paths.
The delivery separately retains the actual animated link trajectories and
their hashes.

The Beagle dry-audio item remains
`unresolved_item_level_review`. Consequently, source-manifest validation is
not redistribution permission or dataset admission.

## Real MP3D visual result

The real scanned-room capture is retained at
`tmp/m5_1/mp3d_mixed_20260718_02/`. Its authoritative gate record is
`mp3d_gate_evidence.json`; all 14/14 declared visual/navigation gates passed.
The declared MP3D navmesh was loaded and fingerprinted. Both 270-point actor
routes are navigable on one shared island; all 269 no-sliding segment endpoint
checks pass per actor; each actor travels `1.1 m`; and their center separation
remains `0.9 m`.

Both semantic IDs are visible in all 270 frames. The independently recomputed
minimum masks are 2,384 pixels for the human and 605 pixels for the Beagle,
above the 256-pixel gate. Border-touch is retained as a diagnostic: visibility
does not imply that the whole articulated body is uncropped.

This is a real `habitat_sim.PathFinder` actor-root-center test, not a full
articulated mesh, swept-volume, or limb-collision test. It proves neither
full-body clearance nor physical room admission.

The original compiled MP3D acoustic package could not be uploaded to RLR
because `node58_mesh58_primitive0` consisted entirely of 458 zero-area
triangles with repeated indices. M5.1 does not edit the MP3D visual asset or
the source acoustic package. The generic research-only filter in
`src/avengine/m3/research_cleanup.py` instead derives
`tmp/m3/root_mp3d_package_rlr_clean_20260718_01/manifest.json`. It removes only
those 458 faces and the empty object's 416-vertex partition, leaving 215,299
triangles. The next retained triangle has area
`3.4570714767e-07 m2`. The removal indices, source/derived array hashes, object
identity and implementation hash are retained. Production and qualified
material packages are rejected by this filter.

The derived package passes its internal file/object/material contract so RLR
can load it, but it does not disguise the scan's remaining QA. Compiler
source-to-package parity and geometry are `fail`, material coverage is `pass`,
and ray leakage is `not_run`. Its materials remain
`research_placeholder`/`unqualified_research_placeholder`.

The final MP3D dynamic RIR evidence is
`tmp/m5_1/mp3d_acoustics_20260718_02/evidence.json`. It is bound to the final
`mp3d_mixed_20260718_02` anchors and room ID, and retains 90 two-source
binaural RIR keyframes at 5 Hz. The sample array is
`[90, 2 sources, 2 channels, 12996]` float32; per-pair lengths range from
11,942 to 12,996 samples.

The final listening delivery is
`tmp/m5_1/mp3d_delivery_20260718_01/evidence.json`. Its principal video is
`tmp/m5_1/mp3d_delivery_20260718_01/videos/mp3d_human_beagle_annotated_binaural.mp4`:
1280x480, 15 fps, 270 frames/18 seconds, H.264 plus two-channel 16 kHz AAC.
Separate 288,000-sample float32 dry buses, binaural source stems and mixture
are retained; the unnormalized/unlimited mixture peak is `0.0574638871`.

The right panel comes from the authenticated Habitat Pathfinder navmesh. The
human and Beagle share island 1, their minimum root-center separation is
`0.9 m`, and their minimum distances to a navmesh edge are
`0.0093427803 m` and `0.2597377002 m`, respectively. These are center-only
navigation diagnostics, not articulated-mesh clearances. The MP3D delivery's
`source_program_reuse.json` imports only taxonomy, event timing and audio
programs from the common source contract; legacy observer, trajectory,
spatial flags, migration and visual-asset provenance are explicitly excluded.

## Claim boundary

M5.1 is a bounded research review, not M6 admission. It does not admit the
human, Beagle, dry audio, HRTF, Apartment, MP3D room, acoustic materials,
episode, or dataset sample. `dataset_admission` remains false/not requested.
The Apartment and MP3D material proposals remain research placeholders and
unqualified.

Four-channel FOA remains an independent WAV authority when produced; it is not
placed in MP4. Review MP4s carry only two-channel binaural audio because
generic players do not reliably preserve Ambisonics order and normalization
metadata.
