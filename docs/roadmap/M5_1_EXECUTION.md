# M5.1 Execution

Run from the AVEngine Habitat-native repository with the pinned Habitat Python
environment. Output paths must not already exist unless a command explicitly
offers `--overwrite`.

```bash
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$PWD:$PWD/src"
export RUNTIME=/data/jzy/code/habitat-sim-AVEngine
export LEGACY=/data/jzy/code/AVEngine
```

The commands below reference local research inputs and retained M0-M4/M2
artifacts. They are reproducibility records for this workspace, not a promise
that third-party assets may be redistributed.

## 1. Rebuild and verify the legacy route contract

To rebuild into a disposable path without replacing the checked-in authority:

```bash
python tools/m5_1/import_legacy_apartment_route.py \
  --legacy-root "$LEGACY" \
  --output tmp/m5_1/route_manifest_replay.json
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
geometric evidence is absent.

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
  --output tmp/m5_1/mixed_legacy_replay
```

The retained reference output is
`tmp/m5_1/mixed_legacy_20260718_01/evidence.json`. The runner consumes both
270-point routes verbatim, applies one fixed articulated state per frame, and
retains RGB, semantic masks, actor/skin transforms, animated anchors, and
readback receipts.

## 4. Render the Apartment dynamic binaural RIR sequence

```bash
python tools/m5_1/render_review_acoustics.py \
  --capture-dir tmp/m5_1/mixed_legacy_20260718_01 \
  --acoustic-package-manifest tmp/m3/root_ue_package_current_20260718_02/manifest.json \
  --m4-request examples/m4/blender_custom/multi_source_canary_request.json \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --listener-position-m -0.7 1.471 0.65 \
  --listener-yaw-deg 55 \
  --fps 15 \
  --rir-stride-frames 3 \
  --output-dir tmp/m5_1/legacy_apartment_acoustics_replay
```

The retained reference is
`tmp/m5_1/legacy_apartment_acoustics_20260718_05/evidence.json`. This command
uses the explicit research-review load policy because the legacy package has
non-passing QA. The evidence must continue to report those `fail`/`not_run`
states; successful RIR generation does not qualify the package or materials.

## 5. Build the annotated Apartment delivery

```bash
python tools/m5_1/build_legacy_delivery.py \
  --capture-dir tmp/m5_1/mixed_legacy_20260718_01 \
  --acoustics-dir tmp/m5_1/legacy_apartment_acoustics_20260718_05 \
  --source-manifest examples/m5_1/legacy_apartment/source_manifest.json \
  --route-manifest examples/m5_1/legacy_apartment/route_manifest.json \
  --old-review-video "$LEGACY/external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/clips/rocketbox_adults_male_adult_01_original_ue_v1/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4" \
  --human-gain 0.18 \
  --beagle-gain 0.18 \
  --fade-samples 80 \
  --output-dir tmp/m5_1/legacy_apartment_delivery_replay
```

The retained reference is
`tmp/m5_1/legacy_apartment_delivery_20260718_01/evidence.json`. Verify that both
videos read back as 270 frames at 15 fps and 18 seconds, and that the audio
stream is two-channel 16 kHz AAC. Exact audio authority is the separate
288,000-sample WAV set under `audio/`; AAC packet/sample padding is not an
authoritative timeline count.

The two primary outputs are:

```text
tmp/m5_1/legacy_apartment_delivery_20260718_01/videos/legacy_apartment_habitat_annotated_binaural.mp4
tmp/m5_1/legacy_apartment_delivery_20260718_01/videos/legacy_apartment_ue_vs_habitat.mp4
```

The right-hand Topdown is derived QA media only. The comparison's three panels
are old UE main, new Habitat main, and new Habitat Topdown QA.

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
  --output tmp/m5_1/mp3d_mixed_replay
```

The retained reference is
`tmp/m5_1/mp3d_mixed_20260718_02/mp3d_gate_evidence.json`. It must report
14/14 gates passing, exact declared-navmesh loading/fingerprinting, 270/270
navigable center positions for both actors, 269/269 no-sliding endpoint checks
per route, exact actor-root readback, and both semantic IDs visible in all 270
frames.

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
  --output-dir tmp/m3/root_mp3d_package_rlr_clean_replay
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
  --capture-dir tmp/m5_1/mp3d_mixed_20260718_02 \
  --acoustic-package-manifest tmp/m3/root_mp3d_package_rlr_clean_20260718_01/manifest.json \
  --m4-request examples/m4/blender_custom/multi_source_canary_request.json \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --listener-position-m -4.1499128342 1.572447 -1.2454376221 \
  --listener-yaw-deg 0 \
  --fps 15 \
  --rir-stride-frames 3 \
  --output-dir tmp/m5_1/mp3d_acoustics_replay
```

The retained reference is
`tmp/m5_1/mp3d_acoustics_20260718_02/evidence.json`: 90 keyframes at 5 Hz,
two stable source IDs, binaural left/right, and the final capture trajectory
hash `6ba59a6544d214455bba06d441e94afc4b2e6adad4f1faadd9195358e813190d`.

Build the annotated listening video and exact WAV buses:

```bash
python tools/m5_1/build_mp3d_delivery.py \
  --capture-dir tmp/m5_1/mp3d_mixed_20260718_02 \
  --acoustics-dir tmp/m5_1/mp3d_acoustics_20260718_02 \
  --source-manifest examples/m5_1/legacy_apartment/source_manifest.json \
  --route-manifest examples/m5_1/mp3d_articulated_review/route_manifest.json \
  --m1-request examples/m2/rooms/habitat_mp3d_articulated_review/capture_request.json \
  --human-gain 0.18 \
  --beagle-gain 0.18 \
  --fade-samples 80 \
  --output-dir tmp/m5_1/mp3d_delivery_replay
```

The retained evidence and video are:

```text
tmp/m5_1/mp3d_delivery_20260718_01/evidence.json
tmp/m5_1/mp3d_delivery_20260718_01/videos/mp3d_human_beagle_annotated_binaural.mp4
```

The video must read back as H.264 1280x480, 15 fps, 270 frames/18 seconds,
with two-channel 16 kHz AAC. The authoritative WAVs remain 288,000 samples.
The Topdown panel is a derived real-Pathfinder QA view, not a second sensor.
The source-program reuse record must explicitly exclude every legacy spatial
flag, observer, trajectory, migration and visual-provenance field.

## 8. Run focused tests

```bash
pytest -q \
  tests/unit/test_m5_1_legacy_route.py \
  tests/unit/test_m5_1_source_contracts.py \
  tests/unit/test_m5_1_dry_audio.py \
  tests/unit/test_m5_1_mixed_capture.py \
  tests/unit/test_m5_1_mp3d_capture.py \
  tests/unit/test_m5_1_acoustics.py \
  tests/unit/test_m5_1_topdown.py \
  tests/unit/test_m5_1_review.py \
  tests/unit/test_m5_1_delivery.py \
  tests/unit/test_m5_1_mp3d_delivery.py \
  tests/unit/test_m3_research_cleanup.py
```

Four-channel FOA, when produced by M4/M5 dataset-audio paths, remains an
independent WAV with explicit order/normalization metadata. M5.1 review MP4s
carry only the two-channel binaural listening copy.
