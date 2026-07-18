# M5.1 Status — Mixed Real-Room Research Review

Status: bounded research-review `pass` for the completed route, source/event,
legacy-Apartment delivery, and MP3D visual, acoustic, and listening-video
delivery described here. This is not a dataset-admission gate. The retained
source, capture, acoustic, and delivery records explicitly keep
`research_only: true` and/or
`qualification_claim: false` as defined by their schemas.
The separately retained same-room UE/SPEAR MP3D result below is a passed
supplementary visual canary; it does not broaden the Habitat gate or its
dataset-admission claim.

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
- a direct old Apartment UE | new Habitat | new Topdown legacy-route
  comparison video; and
- a separate same-source MP3D UE main | Habitat main | Habitat Topdown visual
  triptych.

The Topdown panel is QA-only. It is not a second dataset view, and it does not
change the single-view sensor contract. Both Topdown implementations now use
one shared Habitat `world_from_local` basis: local `-Z` is camera/listener
forward, local `+X` points toward the right ear, and local `+Y` is up. The
panel draws the visual HFOV wedge and explicit `F`/`L`/`R` axes. The wedge is
not an audio gate: M5.1 applies no microphone-FOV or microphone-distance
cutoff, so event scheduling and RLR propagation do not depend on whether a
source is inside the camera image.

## Retained corrected authority

The heading/lighting correction pass supersedes the earlier
`mixed_legacy_20260718_01` and `mp3d_mixed_20260718_02` captures and their
downstream acoustic/delivery outputs. Those earlier reviews exposed the human
walking backward and the Beagle moving sideways because they assumed one
common actor-local forward axis; they are retained only as debugging history,
not as M5.1 authority.

| Room/stage | Retained evidence | Content SHA-256 |
| --- | --- | --- |
| Legacy capture | `tmp/m5_1/mixed_legacy_heading_lighting_20260718_01/evidence.json` | `cdfc547e9c39d50b350fa2c6ccb85022c27bc1e18223ca58d4ad375ae7e55e0d` |
| Legacy acoustics | `tmp/m5_1/legacy_apartment_acoustics_heading_lighting_20260718_01/evidence.json` | `25cede89292cc9139142e49298220d8238b9bba066aac242ea63fdd3bb78ea95` |
| Legacy delivery | `tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/evidence.json` | `51c0d9c5b24ccbdfc559f6dafacf82798146f9f64047c2a4d36445ec8b69aca9` |
| MP3D capture | `tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/evidence.json` | `fca4dd6919e0b3708071601725b98b519575a5a8e62e1bc56f18b27727ce6ac6` |
| MP3D visual/navmesh gate | `tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/mp3d_gate_evidence.json` | `c19b295d9b43cd5189b5468dc5ad0ac3b332a040058856ec6a4fde471f4db4c9` |
| MP3D acoustics | `tmp/m5_1/mp3d_acoustics_heading_lighting_20260718_01/evidence.json` | `e1912ce3769f1b7f97c4a642cb8b5a3942615bfb01f5f50fe561fe2887d88f71` |
| MP3D delivery | `tmp/m5_1/mp3d_delivery_heading_lighting_20260718_02/evidence.json` | `14264969e1dec7056914a83652aab976ff105d0c81525a6c5d14c4e9cfd70ab7` |
| MP3D UE/Habitat supplementary visual comparison | `/data/jzy/code/AVEngine/external/SPEAR/tmp/mp3d_ue_comparison_20260718_01/render/evidence.json` | `dd2f2efc1feed9aaa86c271ab2b380b6bbbfef4960d59b438f3fb147e112d75e` |

The absolute SPEAR path above identifies a local auditable `tmp` result. It is
not a versioned repository asset or a dataset artifact.

The Rocketbox human declares anatomical forward as local `+Z`; the Beagle
uses the M2 QA declaration local `+X`. Each actor's world-space anatomical
forward matches its nearest non-zero route tangent on 270/270 frames in each
room, within a `1e-6`-degree gate. The evidence records the axis-authority file
and hash instead of inferring facing from the mesh at delivery time.

Both actors request and read back PBR and are created against the same
`avengine_m5_1_room_lighting` setup copied from the loaded room. HBAO is
enabled and read back in both captures. The Legacy room has three current and
three registered lights; MP3D has zero current and zero registered lights.
HBAO is screen-space ambient occlusion, not dynamic shadow-map evidence, and
MP3D's apparent room illumination remains baked into its scan textures. M5.1
therefore does not claim UE-quality dynamic shadows for the MP3D actors.

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
`tmp/m5_1/mixed_legacy_heading_lighting_20260718_01/evidence.json`. It uses an actually
articulated Rocketbox walking action and the validated 45-state Beagle walk
block rather than static or sliding substitutes. Root, spherical/prismatic
joint, skin-FK, semantic-ID, and mouth/muzzle anchor readback passed. Under the
frozen legacy camera, the human semantic ID is visible in 206/270 frames and
the Beagle in 192/270 frames; M5.1 therefore does not claim that either actor
remains visible for the whole legacy clip. The retained RGB array is
`[270, 240, 320, 3]` uint8 with SHA-256
`e6e722004705742736a5e24ad5688d7f8ff8ad60c4d45ac53db17e9e601b45f4`;
the `[270, 240, 320]` uint32 semantic array SHA-256 is
`1bbeefc9f81c4514e12d9af922c88af461e50987241d3630e5395f7422b62807`.

The retained dynamic Apartment acoustics are
`tmp/m5_1/legacy_apartment_acoustics_heading_lighting_20260718_01/evidence.json`. A fixed,
camera-co-located listener receives 90 two-source binaural RIR keyframes at
5 Hz, rendered at 16 kHz through the hash-bound MIT KEMAR HRTF. The exact
sequence is stored as `[90, 2 sources, 2 channels, 20199]` float32 samples;
per-pair valid lengths range from 19,377 to 20,199 samples. The sample-array
SHA-256 is
`523cb9948a67f58010df6c921b22ce69aebd534ff2572d15de016fd1395c2a85`.

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
`tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/evidence.json`. Its exact
18-second audio authority is the retained 288,000-sample WAV set. The ordinary
MP4 contains a two-channel binaural AAC listening copy. The unnormalized,
unlimited mixture peak is `0.0470916646`, and its WAV SHA-256 is
`0f73d1f0f5d13c114d4af66e00db65e5c85d661f82879435fd3de60d9ab26c30`:

- `tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/videos/legacy_apartment_habitat_annotated_binaural.mp4`
  — 1280x480, 270-frame annotated Habitat + Topdown review, SHA-256
  `33a50d21e7713245d70e9192b993c188f8f131ebf496243261c5f3591e2c6da5`;
- `tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/videos/legacy_apartment_ue_vs_habitat.mp4`
  — 1920x480 old UE | new Habitat | new Topdown comparison, SHA-256
  `b3ac687e7142a50087cd200bb232b6c08521b8604c636bf23911662323dde6a3`; and
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/clips/rocketbox_adults_male_adult_01_original_ue_v1/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4`
  — the hash-bound old reference consumed by the comparison builder.

## Source, event, and flag contract

The versioned schema is
`schemas/m5_1_source_manifest_v1.schema.json`; the executable human+Beagle
example is `examples/m5_1/legacy_apartment/source_manifest.json`.
Its corrected file SHA-256 is
`324859a2c7038c2385f0ee8992d9bfedad5090478b285bf058747d8eb609aef0`.

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
their hashes. Camera-local lateral signs now come from the same Habitat basis
as Topdown and RLR: human (`source0`) spans
`[-1.8807635032, 4.5020047092] m`, and Beagle (`source1`) spans
`[-2.0815152560, 4.3012529565] m`. Both therefore retain
`crosses_azimuth_zero: present`; the numeric signs and manifest hash supersede
the mirrored pre-fix calculation.

The Beagle dry-audio item remains
`unresolved_item_level_review`. Consequently, source-manifest validation is
not redistribution permission or dataset admission.

## Real MP3D visual result

The real scanned-room capture is retained at
`tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/`. Its authoritative gate record is
`mp3d_gate_evidence.json`; all 14/14 declared visual/navigation gates passed.
The declared MP3D navmesh was loaded and fingerprinted. Both 270-point actor
routes are navigable on one shared island; all 269 no-sliding segment endpoint
checks pass per actor; each actor travels `1.1 m`; and their center separation
remains `0.9 m`.

Both semantic IDs are visible in all 270 frames. The independently recomputed
minimum masks are 2,262 pixels for the human and 361 pixels for the Beagle,
above the 256-pixel gate. Border-touch is retained as a diagnostic: visibility
does not imply that the whole articulated body is uncropped. The retained RGB
array is `[270, 240, 320, 3]` uint8 with SHA-256
`c03614d175e4be2010d528397131dc85fd3b2a34d560217994482fc424e283e2`;
the `[270, 240, 320]` uint32 semantic array SHA-256 is
`d17f1446352cc6a334554c5fe49553e651f03c477968fdb8f3101168fbd31db9`.

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
`tmp/m5_1/mp3d_acoustics_heading_lighting_20260718_01/evidence.json`. It is bound to the final
`mp3d_mixed_heading_lighting_20260718_01` anchors and room ID, and retains 90 two-source
binaural RIR keyframes at 5 Hz. The sample array is
`[90, 2 sources, 2 channels, 13760]` float32; per-pair lengths range from
11,747 to 13,760 samples. Its sample-array SHA-256 is
`fe7436bdb7e03c05f9c7a3933406e38eaf3143fe54c863e1c02f905cce364fb3`,
and its corrected animated-emitter trajectory SHA-256 is
`004069d4e639ede2508970bc1e4742f02e952a70f6c8db90b3261feefdeb87f6`.

The final listening delivery is
`tmp/m5_1/mp3d_delivery_heading_lighting_20260718_02/evidence.json`. Its principal video is
`tmp/m5_1/mp3d_delivery_heading_lighting_20260718_02/videos/mp3d_human_beagle_annotated_binaural.mp4`:
1280x480, 15 fps, 270 frames/18 seconds, H.264 plus two-channel 16 kHz AAC.
Separate 288,000-sample float32 dry buses, binaural source stems and mixture
are retained; the unnormalized/unlimited mixture peak is `0.0505205464`.
The mixture WAV SHA-256 is
`f51097fb29ffe71b29aa45b24ffc177b7f906fef3be8566769e3acd3cec09fe5`,
and the MP4 SHA-256 is
`c011a6063dcfe578ff2d15f8bec8d91cc22f87bff356d1e53216ac9fd4b5e359`.

The right panel comes from the authenticated Habitat Pathfinder navmesh. The
human and Beagle share island 1, their minimum root-center separation is
`0.9 m`, and their minimum distances to a navmesh edge are
`0.0093427803 m` and `0.2597377002 m`, respectively. These are center-only
navigation diagnostics, not articulated-mesh clearances. The MP3D delivery's
`source_program_reuse.json` imports only taxonomy, event timing and audio
programs from the common source contract; legacy observer, trajectory,
spatial flags, migration and visual-asset provenance are explicitly excluded.

## Same-room MP3D UE/SPEAR visual comparison

The passed supplementary comparison evidence is retained locally at
`/data/jzy/code/AVEngine/external/SPEAR/tmp/mp3d_ue_comparison_20260718_01/render/evidence.json`,
with SHA-256
`dd2f2efc1feed9aaa86c271ab2b380b6bbbfef4960d59b438f3fb147e112d75e`.
It binds scene `17DRP5sb8fy`, route
`m5_1_mp3d_human_beagle_parallel_18s_v1`, the same raw MP3D source with
SHA-256 `334456925e056c83a9a7a5c768b3d37cdd23425d8ca20743bfce015be3f56b04`,
the exact Habitat camera, and both Pathfinder-qualified 270-point actor-center
routes. The imported scene contains 71 meshes; camera, human-root, dog-root and
explicit per-frame animation-phase readbacks passed.

The formal local review video is
`/data/jzy/code/AVEngine/external/SPEAR/tmp/mp3d_ue_comparison_20260718_01/render/mp3d_17DRP5sb8fy_ue_vs_habitat_triptych_binaural.mp4`,
with SHA-256
`54e17225c74341de837dd1415a8639d0ed0fdc5ea94d5ad738238be6ddfb2a23`.
It is 1920x480, 15 fps and 270 frames/18 seconds: a 2x upscale of the 320x240
UE main view, then the unchanged Habitat main view and unchanged Habitat
Topdown QA panel. Its two-channel 16 kHz binaural AAC packets are copied
unchanged from the Habitat review; both inputs and the triptych read back the
same audio-packet SHA-256
`be768ea6c279bf3aa1deb82165786806da28177316f8a57bff41e523387ea975`.
This is therefore a visual-engine comparison, not a UE audio or acoustic
comparison.

The imported UE scene uses `NoCollision`; Habitat Pathfinder remains the sole
navigation authority and proves only the two actor root centers. Because the
MP3D scan has baked apparent illumination and no runtime room lights, this UE
review adds a 10-lux movable shadow-casting directional light and a bounded
`0.35` skylight. Those lights make dynamic-shadow behavior reviewable but do
not recover the unknown Matterport capture lights or establish photometric
parity. The local `tmp` evidence and video are auditable run products, not
versioned assets.

## Claim boundary

M5.1 is a bounded research review, not M6 admission. It does not admit the
human, Beagle, dry audio, HRTF, Apartment, MP3D room, acoustic materials,
episode, or dataset sample. `dataset_admission` remains false/not requested.
The Apartment and MP3D material proposals remain research placeholders and
unqualified. The completed same-room UE/SPEAR result is retained as a separate
supplementary visual-engine canary. It does not enter the Habitat gate or
qualify full-body clearance, room materials, lighting equivalence, acoustics,
or dataset admission.

Four-channel FOA remains an independent WAV authority when produced; it is not
placed in MP4. Review MP4s carry only two-channel binaural audio because
generic players do not reliably preserve Ambisonics order and normalization
metadata.
