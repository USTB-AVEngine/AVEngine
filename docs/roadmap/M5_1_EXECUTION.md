# M5.1 Execution

Run from the AVEngine Habitat-native repository with the pinned Habitat Python
environment. Output paths must not already exist unless a command explicitly
offers `--overwrite`.

```bash
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$PWD:$PWD/src"
export RUNTIME=/data/jzy/code/habitat-sim-AVEngine
export LEGACY=/data/jzy/code/AVEngine

# Replace this once per replay. Every output below must be a new path.
REPLAY_TAG=replace_with_unique_tag
LEGACY_ROUTE_REPLAY="tmp/m5_1/route_manifest_replay_${REPLAY_TAG}.json"
LEGACY_CAPTURE_REPLAY="tmp/m5_1/mixed_legacy_replay_${REPLAY_TAG}"
LEGACY_ACOUSTICS_REPLAY="tmp/m5_1/legacy_apartment_acoustics_replay_${REPLAY_TAG}"
LEGACY_DELIVERY_REPLAY="tmp/m5_1/legacy_apartment_delivery_replay_${REPLAY_TAG}"
MP3D_CLEAN_REPLAY="tmp/m3/root_mp3d_package_rlr_clean_replay_${REPLAY_TAG}"
MP3D_CAPTURE_REPLAY="tmp/m5_1/mp3d_mixed_replay_${REPLAY_TAG}"
MP3D_ACOUSTICS_REPLAY="tmp/m5_1/mp3d_acoustics_replay_${REPLAY_TAG}"
MP3D_DELIVERY_REPLAY="tmp/m5_1/mp3d_delivery_replay_${REPLAY_TAG}"
REPLICACAD_CAPTURE_REPLAY="tmp/m5_1/replicacad_mixed_replay_${REPLAY_TAG}"
REPLICACAD_ACOUSTICS_REPLAY="tmp/m5_1/replicacad_acoustics_replay_${REPLAY_TAG}"
REPLICACAD_DELIVERY_REPLAY="tmp/m5_1/replicacad_delivery_replay_${REPLAY_TAG}"
export AVENGINE_REPLICACAD_ROOT=/path/to/replica_cad
```

The commands below reference local research inputs and retained M0-M4/M2
artifacts. They are reproducibility records for this workspace, not a promise
that third-party assets may be redistributed.

## 1. Rebuild and verify the legacy route contract

To rebuild into a disposable path without replacing the checked-in authority:

```bash
python tools/m5_1/import_legacy_apartment_route.py \
  --legacy-root "$LEGACY" \
  --output "$LEGACY_ROUTE_REPLAY"
```

Compare the replay's route hashes and gates with
`examples/m5_1/legacy_apartment/route_manifest.json`. The command reads the
old 270-frame Apartment specification, furniture/shell maps, and furniture
categories, then recomputes the zero-radius horizontal actor-center AABB gate.
It does not perform full-mesh collision testing.

## 2. Validate the source/event/flag authority

The schema and checked-in example are:

- `schemas/m5_1_source_manifest_v1.schema.json`
- `examples/m5_1/legacy_apartment/source_manifest.json`

Run both JSON Schema and semantic/hash validation through the strict loader:

```bash
python - <<'PY'
from avengine.m5_1.source_contracts import load_source_manifest

manifest = load_source_manifest(
    "examples/m5_1/legacy_apartment/source_manifest.json"
)
print(manifest["schema"], manifest["manifest_id"])
PY
```

The loader verifies clip/frame/sample arithmetic, source/taxonomy/event
references, half-open event windows, reconstructed frame-current state,
simultaneous overlap windows, hashes, provenance, and tri-state flag
aggregation. A `not_evaluated` visibility flag is expected when its required
geometric evidence is absent. The retained corrected manifest SHA-256 is
`324859a2c7038c2385f0ee8992d9bfedad5090478b285bf058747d8eb609aef0`.
Its camera-local lateral values must be human
`[-1.8807635032, 4.5020047092] m` and Beagle
`[-2.0815152560, 4.3012529565] m`; both cross zero under the shared Habitat
`world_from_local` basis.

## 3. Capture the animated legacy Apartment route

```bash
python tools/m5_1/capture_human_beagle_legacy.py \
  --route-manifest examples/m5_1/legacy_apartment/route_manifest.json \
  --room-manifest tmp/m1/legacy_apartment_package/room_manifest.json \
  --m1-request examples/m5_1/legacy_apartment/m1_capture_request.json \
  --human-runtime-glb "$LEGACY/external/SPEAR/tmp/rocketbox_native_runtime_ue_v3/rocketbox_male_adult_01_original_ue_v3/runtime.glb" \
  --beagle-manifest tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --beagle-m2-request tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json \
  --runtime-root "$RUNTIME" \
  --output "$LEGACY_CAPTURE_REPLAY"
```

The retained reference output is
`tmp/m5_1/mixed_legacy_heading_lighting_20260718_01/evidence.json`. The runner consumes both
270-point routes verbatim, applies one fixed articulated state per frame, and
retains RGB, semantic masks, actor/skin transforms, animated anchors, and
readback receipts. It must also report human local anatomical forward `+Z`,
Beagle local anatomical forward `+X`, and 270/270 heading-alignment frames per
actor. Both actors must read back PBR, and HBAO must read back enabled. Both
actor creation calls must record and use the same
`avengine_m5_1_room_lighting` key; the registered setup must read back equal
to the current room setup. The pinned Habitat binding does not expose a native
per-actor light-key getter, so this is creation-parameter evidence rather than
an actor-key readback claim. The Legacy room-light setup has three
current/registered lights.

## 4. Render the Apartment dynamic binaural RIR sequence

```bash
python tools/m5_1/render_review_acoustics.py \
  --capture-dir "$LEGACY_CAPTURE_REPLAY" \
  --acoustic-package-manifest tmp/m3/root_ue_package_current_20260718_02/manifest.json \
  --m4-request examples/m4/blender_custom/multi_source_canary_request.json \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --listener-position-m -0.7 1.471 0.65 \
  --listener-yaw-deg 55 \
  --fps 15 \
  --rir-stride-frames 3 \
  --output-dir "$LEGACY_ACOUSTICS_REPLAY"
```

The retained reference is
`tmp/m5_1/legacy_apartment_acoustics_heading_lighting_20260718_01/evidence.json`. This command
uses the explicit research-review load policy because the legacy package has
non-passing QA. The evidence must continue to report those `fail`/`not_run`
states; successful RIR generation does not qualify the package or materials.
The retained RIR array is float32 `[90, 2, 2, 20199]`, with valid lengths from
19,377 to 20,199 samples.

## 5. Build the annotated Apartment delivery

```bash
python tools/m5_1/build_legacy_delivery.py \
  --capture-dir "$LEGACY_CAPTURE_REPLAY" \
  --acoustics-dir "$LEGACY_ACOUSTICS_REPLAY" \
  --source-manifest examples/m5_1/legacy_apartment/source_manifest.json \
  --route-manifest examples/m5_1/legacy_apartment/route_manifest.json \
  --old-review-video "$LEGACY/external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/clips/rocketbox_adults_male_adult_01_original_ue_v1/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4" \
  --human-gain 0.18 \
  --beagle-gain 0.18 \
  --fade-samples 80 \
  --output-dir "$LEGACY_DELIVERY_REPLAY"
```

The retained reference is
`tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/evidence.json`. Verify that both
videos read back as 270 frames at 15 fps and 18 seconds, and that the audio
stream is two-channel 16 kHz AAC. Exact audio authority is the separate
288,000-sample WAV set under `audio/`; AAC packet/sample padding is not an
authoritative timeline count.

The two primary outputs are:

```text
tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/videos/legacy_apartment_habitat_annotated_binaural.mp4
tmp/m5_1/legacy_apartment_delivery_heading_lighting_20260718_02/videos/legacy_apartment_ue_vs_habitat.mp4
```

The right-hand Topdown is derived QA media only. The comparison's three panels
are old UE main, new Habitat main, and new Habitat Topdown QA. The Topdown
camera/listener uses Habitat local `-Z/+X/+Y` as forward/right-ear/up, draws a
visual-HFOV wedge plus `F`/`L`/`R` axes, and does not impose an audio FOV or
distance cutoff.

## 6. Execute the real MP3D visual gate

```bash
python tools/m5_1/capture_human_beagle_mp3d.py \
  --route-manifest examples/m5_1/mp3d_articulated_review/route_manifest.json \
  --room-manifest examples/m2/rooms/habitat_mp3d_articulated_review/room_manifest.json \
  --m1-request examples/m2/rooms/habitat_mp3d_articulated_review/capture_request.json \
  --human-runtime-glb "$LEGACY/external/SPEAR/tmp/rocketbox_native_runtime_ue_v3/rocketbox_male_adult_01_original_ue_v3/runtime.glb" \
  --beagle-manifest tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --beagle-m2-request tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json \
  --runtime-root "$RUNTIME" \
  --output "$MP3D_CAPTURE_REPLAY"
```

The retained reference is
`tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/mp3d_gate_evidence.json`. It must report
14/14 gates passing, exact declared-navmesh loading/fingerprinting, 270/270
navigable center positions for both actors, 269/269 no-sliding endpoint checks
per route, exact actor-root readback, and both semantic IDs visible in all 270
frames. Independently recomputed minimum masks are 2,262 human pixels and 361
Beagle pixels. Each actor must also pass 270/270 anatomical-forward alignment
frames. HBAO and PBR readback must pass against the common room-light key; the
MP3D room has zero current/registered runtime lights because its apparent
illumination is baked into the scan texture. HBAO is not dynamic-shadow proof.

The test is explicitly `actor_root_center_only`. Do not describe it as
articulated-mesh clearance or full-body framing; the evidence retains
border-touch diagnostics.

## 7. Build the final MP3D listening review

The source MP3D acoustic package contains one wholly degenerate scan primitive
that RLR rejects. Derive a separate research-only package; do not edit or
replace the original visual/acoustic assets:

```bash
python tools/m3/derive_research_rlr_package.py \
  --source-manifest tmp/m3/root_mp3d_package_current_20260718_01/manifest.json \
  --output-dir "$MP3D_CLEAN_REPLAY"
```

The retained reference is
`tmp/m3/root_mp3d_package_rlr_clean_20260718_01/manifest.json`. The report must
show exactly 458 removed zero-area triangles and 416 removed vertices from
`node58_mesh58_primitive0`, with 215,299 triangles retained. It must keep
package mode `research_candidate`, material semantics `research_placeholder`,
qualification `unqualified_research_placeholder`, and QA statuses
`fail`/`fail`/`pass`/`not_run` for source parity, geometry, material coverage
and ray leakage.

Render the room-bound dynamic binaural sequence from the final MP3D capture:

```bash
python tools/m5_1/render_review_acoustics.py \
  --capture-dir "$MP3D_CAPTURE_REPLAY" \
  --acoustic-package-manifest "$MP3D_CLEAN_REPLAY/manifest.json" \
  --m4-request examples/m4/blender_custom/multi_source_canary_request.json \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --listener-position-m -4.1499128342 1.572447 -1.2454376221 \
  --listener-yaw-deg 0 \
  --fps 15 \
  --rir-stride-frames 3 \
  --output-dir "$MP3D_ACOUSTICS_REPLAY"
```

The retained reference is
`tmp/m5_1/mp3d_acoustics_heading_lighting_20260718_01/evidence.json`: 90 keyframes at 5 Hz,
two stable source IDs, binaural left/right, and the final capture trajectory
hash `004069d4e639ede2508970bc1e4742f02e952a70f6c8db90b3261feefdeb87f6`.
The retained float32 RIR array is `[90, 2, 2, 13760]`, with valid lengths from
11,747 to 13,760 samples.

Build the annotated listening video and exact WAV buses:

```bash
python tools/m5_1/build_mp3d_delivery.py \
  --capture-dir "$MP3D_CAPTURE_REPLAY" \
  --acoustics-dir "$MP3D_ACOUSTICS_REPLAY" \
  --source-manifest examples/m5_1/legacy_apartment/source_manifest.json \
  --route-manifest examples/m5_1/mp3d_articulated_review/route_manifest.json \
  --m1-request examples/m2/rooms/habitat_mp3d_articulated_review/capture_request.json \
  --human-gain 0.18 \
  --beagle-gain 0.18 \
  --fade-samples 80 \
  --output-dir "$MP3D_DELIVERY_REPLAY"
```

The retained evidence and video are:

```text
tmp/m5_1/mp3d_delivery_heading_lighting_20260718_02/evidence.json
tmp/m5_1/mp3d_delivery_heading_lighting_20260718_02/videos/mp3d_human_beagle_annotated_binaural.mp4
```

The video must read back as H.264 1280x480, 15 fps, 270 frames/18 seconds,
with two-channel 16 kHz AAC. The authoritative WAVs remain 288,000 samples.
The Topdown panel is a derived real-Pathfinder QA view, not a second sensor.
The source-program reuse record must explicitly exclude every legacy spatial
flag, observer, trajectory, migration and visual-provenance field.

## 8. Execute the ReplicaCAD v2 furnished root-center review

This is a bounded research review, not a room qualification.  It loads the
official `apt_0` scene instance and declared navmesh, then validates the two
270-frame actor-root routes against PathFinder and every loaded furnished rigid
object collision OBB.  It does not claim full articulated-body collision.

```bash
python tools/m5_1/capture_human_beagle_replicacad.py \
  --route-manifest examples/m5_1/replicacad_articulated_review/route_manifest.json \
  --room-manifest examples/m5_1/replicacad_articulated_review/room_manifest.json \
  --m1-request examples/m5_1/replicacad_articulated_review/capture_request.json \
  --human-runtime-glb "$LEGACY/external/SPEAR/tmp/rocketbox_native_runtime_ue_v3/rocketbox_male_adult_01_original_ue_v3/runtime.glb" \
  --beagle-manifest tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --beagle-m2-request tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json \
  --replicacad-root "$AVENGINE_REPLICACAD_ROOT" \
  --runtime-root "$RUNTIME" \
  --output "$REPLICACAD_CAPTURE_REPLAY"
```

The gate must report 19/19: exact selected closure, declared navmesh, both
routes on one island without sliding, navmesh clearance, camera/listener floor
placement, actor LOS, semantic visibility, articulated readback and furnished
rigid-object root-center clearance.  The complete package is built in one
sibling staging directory, reopens every array/file/JSON locator and is only
then published atomically.

Render the room-bound dynamic RIR sequence.  The command verifies that the
acoustic package `source_room`, capture, route, request, source manifest and
explicit source-to-actor/emitter binding all describe the same room state.

```bash
python tools/m5_1/render_review_acoustics.py \
  --capture-dir "$REPLICACAD_CAPTURE_REPLAY" \
  --source-manifest examples/m5_1/legacy_apartment/source_manifest.json \
  --acoustic-package-manifest \
    tmp/m3/replicacad_apt_0_package_rlr_clean_20260719_01/manifest.json \
  --m4-request examples/m4/blender_custom/multi_source_canary_request.json \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --runtime-root "$RUNTIME" \
  --listener-position-m 2.6 1.47 3.4 \
  --listener-yaw-deg 180 \
  --fps 15 \
  --rir-stride-frames 3 \
  --output-dir "$REPLICACAD_ACOUSTICS_REPLAY"
```

Build the independent dry buses, binaural stems, exact mixture and annotated
Habitat+Topdown review:

```bash
python tools/m5_1/build_mp3d_delivery.py \
  --capture-dir "$REPLICACAD_CAPTURE_REPLAY" \
  --acoustics-dir "$REPLICACAD_ACOUSTICS_REPLAY" \
  --source-manifest examples/m5_1/legacy_apartment/source_manifest.json \
  --route-manifest examples/m5_1/replicacad_articulated_review/route_manifest.json \
  --m1-request examples/m5_1/replicacad_articulated_review/capture_request.json \
  --room-family replicacad \
  --replicacad-root "$AVENGINE_REPLICACAD_ROOT" \
  --human-gain 0.18 \
  --beagle-gain 0.18 \
  --fade-samples 80 \
  --output-dir "$REPLICACAD_DELIVERY_REPLAY"
```

Both gains must be finite and positive.  Every declared event window must have
nonzero energy in its source dry bus and binaural stem.  The standalone video
permanently labels `RESEARCH ONLY`, `UNQUALIFIED ACOUSTICS`, `ROOT-CENTER
CLEARANCE ONLY`, `ACOUSTIC GEOMETRY: STAGE SURFACE ONLY`, and the unresolved
Beagle dry-audio rights item.  A successful render does not qualify the room,
materials, topology or ray leakage.

## 9. Run focused tests

```bash
pytest -q \
  tests/unit/test_m5_1_legacy_route.py \
  tests/unit/test_m5_1_source_contracts.py \
  tests/unit/test_m5_1_dry_audio.py \
  tests/unit/test_m5_1_lighting.py \
  tests/unit/test_m5_1_mixed_capture.py \
  tests/unit/test_m5_1_mp3d_capture.py \
  tests/unit/test_m5_1_replicacad_capture.py \
  tests/unit/test_m5_1_acoustics.py \
  tests/unit/test_m5_1_orientation.py \
  tests/unit/test_m5_1_topdown.py \
  tests/unit/test_m5_1_review.py \
  tests/unit/test_m5_1_delivery.py \
  tests/unit/test_m5_1_mp3d_delivery.py \
  tests/unit/test_m5_1_replicacad_delivery.py \
  tests/unit/test_m3_research_cleanup.py
```

Four-channel FOA, when produced by M4/M5 dataset-audio paths, remains an
independent WAV with explicit order/normalization metadata. M5.1 review MP4s
carry only the two-channel binaural listening copy.
