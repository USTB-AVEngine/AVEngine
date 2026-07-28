# Runtime room and source-asset selection

AVEngine keeps three independent choices:

1. a room identity and its geometry/navigation/acoustic resources;
2. a visible source asset with geometry, articulation and emitter placement;
3. a dry sound asset and event program.

Changing a cat Mesh must not change the room. Changing a bark recording must
not change the dog Mesh. None of these choices is inferred from `source1` or
`source2`; those names are only stable episode slots.

## Source asset runtime registry

The default registry is
[`examples/runtime/source_asset_runtime_profiles.json`](../../examples/runtime/source_asset_runtime_profiles.json).
Its schema is
[`source_asset_runtime_registry_v1.schema.json`](../../schemas/source_asset_runtime_registry_v1.schema.json).

One articulated animal record contains:

- exact asset ID and runtime revision;
- species, breed and realized `size`, `body_build`, `life_stage` and
  breed-scoped coat;
- the logical Pixel3D/other source Mesh URI and rig authority;
- Timeline template/body-plan, anatomical forward axis and Idle/Walking IDs;
- one or more measured emitter anchors;
- backend-specific UE Blueprint, explicit SkeletalMesh binding policy and
  optional path, animation paths, forward-yaw calibration, component/floor
  correction and optional anonymous-bone role mapping. Existing prepared
  Blueprints use `skeletal_mesh_binding: blueprint_component`; a backend that
  assigns the Mesh directly uses `explicit_path` and must provide the path.

The UE fields describe how an already prepared asset is executed. They do not
authorize substituting a template Mesh for a generated breed. A new breed
still follows the complete generated-asset workflow before it receives a
runtime record.

New generated assets use the optional exact `asset_bound_lineage` closure
before formal admission or exact-bound loading. That closure binds the
SPEAR `source_asset_v2` record and registry, raw Pixel3D GLB, unchanged or
bounded-same-Pixel3D repair, TokenRig/animation closure, UE asset-bound import
and runtime readback, emitter measurement and admission evidence by path,
size and SHA-256. Its geometry enum intentionally has no template-replacement
mode: Rocketbox or Quaternius may donate compatible motion, never geometry.

An exact binding also declares a positive `actor_scale`, a complete
right-handed emitter-local basis (`forward x up = right`), an explicit
SkeletalMesh object path and an action-ID-to-UE-animation map that exactly
matches Timeline Idle/Walking IDs. With
`skeletal_mesh_binding: blueprint_component`, the Mesh path is the expected
runtime readback identity rather than an instruction to overwrite the
Blueprint component. Historical research records may omit this closure and
continue to load through the legacy helpers; `formal` cannot.

Pair selection can now name only assets:

```json
{
  "asset_selection": {
    "source1": "rocketbox_human_male_adult_01_m5_1_candidate",
    "source2": "generated_abyssinian_ruddy_medium_standard_adult_research_v1"
  }
}
```

The selector resolves the emitter height, forward axis and exact revision from
the registry. The generated source manifest also carries the selected
species/breed and realized appearance attributes, while the Topdown label
shows the chosen asset rather than treating the slot name as a species. See
[`apartment_generated_asset_pair_templates.json`](../../examples/m7/apartment_generated_asset_pair_templates.json).
An optional object form may select an exact `revision` or a non-default
`anchor_id`. Unknown assets, anchors and revisions fail instead of falling
back to a generic animal.

Use `--source-asset-registry` on both the trajectory selector, Apartment UE
input builder and SPEAR/UE runner when selecting a non-default registry.

## Camera/listener pose selection

The formal view and acoustic listener are one co-located, co-oriented rig.
Its position and yaw therefore belong upstream in the M1 capture request,
not in a UE-only camera override.  That single request is consumed by Habitat
visual capture, Topdown/listener metadata, RIR generation and the optional UE
coordinate adapter.

Create a request at any chosen room position and horizontal orientation with:

```bash
PYTHONPATH=src python tools/m1/build_camera_pose_request.py \
  --base-request examples/m6x/fixed_apartment/m1_capture_request_review_720p.json \
  --room-manifest tmp/m1/legacy_apartment_package/room_manifest.json \
  --request-id apartment_view_test \
  --position-m -2.081 1.60 -1.345 \
  --yaw-deg -65 \
  --output tmp/camera_pose_test/request.json
```

The tool changes `primary_camera_rig.world_from_rig` and preserves the rigid
identity transform from that rig to `listener0`.  It never reuses RIR or
binaural audio made for another listener pose.  Before scheduling a full
dataset run, use the lightweight native probe to confirm that a NavMesh floor
exists beneath the position, the eye height is plausible, and all three
Habitat sensors render:

```bash
PYTHONPATH=src python tools/m1/probe_camera_pose_native.py \
  --room tmp/m1/legacy_apartment_package/room_manifest.json \
  --request tmp/camera_pose_test/request.json \
  --runtime-root ../habitat-sim-AVEngine \
  --output tmp/camera_pose_test/native_probe
```

This probe is interactive native evidence, not a substitute for the formal M1
release capture.  The SPEAR/UE adapter converts the same accepted Habitat pose
to UE coordinates and gates the runtime camera position/yaw readback.

The retained Apartment interface probe executes two requests at
`(-2.081, 1.60, -1.345) m / -65 degrees` and
`(4.279, 1.60, 2.749) m / 145 degrees`. Both pass live floor snapping and
RGB/depth/semantic readback. Their receipts are under
`tmp/runtime_interface_probe_20260724_01/habitat_native_camera_*`; changing
the listener pose for a production episode still requires matching RIR/audio.

## Room runtime profile registry

Stable room identity and resource lineage remain in the M6
[`room_registry.json`](../../examples/m6/rooms/room_registry.json). Concrete
backend execution data is separate in
[`room_runtime_profiles.json`](../../examples/runtime/room_runtime_profiles.json),
validated by
[`room_runtime_profile_registry_v1.schema.json`](../../schemas/room_runtime_profile_registry_v1.schema.json).

A room runtime profile references one exact M6 room revision and records:

- backend and adapter;
- native scene/map identity;
- layout and exterior-view policies;
- render transport and warmup settings;
- supported input layouts and default lighting profile.

Select it with:

```bash
PYTHONPATH=src python tools/m6y/run_spear_apartment_canary.py \
  --input-layout asset-bound-batch \
  --bundle-root PATH_TO_UE_INPUT_BUNDLE \
  --source-asset-registry examples/runtime/source_asset_runtime_profiles.json \
  --room-runtime-profiles examples/runtime/room_runtime_profiles.json \
  --room-profile spear_apartment_0000 \
  --output-dir PATH_TO_NEW_OUTPUT \
  --dry-run
```

`spear_apartment_v1` currently fixes the accepted five-second, 75-frame,
1280x720 transport. Another room may reuse it only when its RoomCapsule and UE
scene satisfy that adapter. A room requiring a different coordinate or render
adapter must declare a new adapter rather than hiding the difference inside
the profile. The registry schema accepts additional backend and adapter IDs;
the selected runner is responsible for rejecting profiles it cannot execute.

## Adding a source asset

1. Complete and review the source-specific Mesh, rig, actions, heading,
   support-plane, emitter and UE runtime gates.
2. Add one source runtime record with its exact asset-bound lineage. Do not
   copy another breed's skeleton roles, scale, floor correction or muzzle
   offset.
3. Add the desired ordered pairing using `asset_selection`.
4. Run registry/unit tests and an exact UE dry run.
5. Run native UE readback before using the asset in a generated dataset.

The dry sound library remains in the sound registry and dataset/audio
configuration. It is intentionally absent from the source visual runtime
registry.

The retained Labrador cross-check follows this procedure with
`generated_labrador_yellow_medium_standard_adult_research_v1`. Its own
generated Mesh and actions pass standalone import/cook, concrete-emitter
trajectory selection, native RIR/binaural assembly and real Apartment runtime
readback. Evidence and listening videos are under
`tmp/runtime_interface_probe_20260724_01/labrador_ue_native_retry1/`.
