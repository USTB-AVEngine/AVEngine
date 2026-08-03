# M1 Execution Runbook

M1 proves the visual room, sensor-pose, navigation, and evidence contracts. It
does not prove audio propagation.

## Fixed M1 contract

- There is one logical rig, `camera_rig_0`, and exactly one formal dataset view,
  `view0`.
- RGB, depth, and semantic are three modalities at the same position,
  orientation, resolution, projection, and capture state. They are not three
  viewpoints.
- `listener0` is attached to and co-located with `camera_rig_0`.
- At least two sources have unique stable names and pairwise-distinct world
  transforms. M1 checks their identity and transform round-trip only.
- M1 does **not** instantiate an AudioSensor, run RLR, render audio, or claim an
  RIR. Multi-source acoustic propagation is the M4 gate.
- The top-down navigation QA map is diagnostic-only and must never appear in
  `formal_view_ids`.

## Prerequisites

Run with the Habitat runtime at the commit in
[`locks/m1_runtime_v1.yaml`](../../locks/m1_runtime_v1.yaml), selected through
the root `runtime.lock.yaml` index, and with both worktrees clean; worktree
cleanliness is a required evidence check. Resolve every machine-local checkout,
tool and output location through environment variables:

```bash
export REPO="${AVENGINE_REPOSITORY_ROOT:-$(git rev-parse --show-toplevel)}"
export AVENGINE_REPOSITORY_ROOT="$REPO"
export AVENGINE_HABITAT_RUNTIME_ROOT="${AVENGINE_HABITAT_RUNTIME_ROOT:-$REPO/../habitat-sim-AVEngine}"
export AVENGINE_UNREAL_ENGINE_ROOT="${AVENGINE_UNREAL_ENGINE_ROOT:-/path/to/UnrealEngine-5.5}"
export AVENGINE_SPEAR_ROOT="${AVENGINE_SPEAR_ROOT:-$REPO/external/SPEAR}"
export AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT="${AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT:-$REPO/tmp/m1/legacy_apartment_export}"
export AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT="${AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT:-$REPO/tmp/m1/legacy_apartment_package}"

export HABPY="${HABPY:-$REPO/.venv/bin/python}"
export SPEARPY="${SPEARPY:-python3}"
export BLENDER="${BLENDER:-blender}"
export RUNTIME="$AVENGINE_HABITAT_RUNTIME_ROOT"
export UE="$AVENGINE_UNREAL_ENGINE_ROOT"
export SPEAR="$AVENGINE_SPEAR_ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
```

SPEAR is an independent Git repository, not content bundled into AVEngine.
Clone it separately and detach it at the revision declared in
`manifest.yaml`:

```bash
git clone https://github.com/Eastforward/spear.git "$AVENGINE_SPEAR_ROOT"
git -C "$AVENGINE_SPEAR_ROOT" checkout --detach 7fbf3632fdb63cc2eceea564811c9597cabfb199
```

On the shared `48g-jump` server, UE 5.5 and an Apartment provenance
checkout that consumers must treat as read-only are available at:

```bash
export AVENGINE_UNREAL_ENGINE_ROOT=/data/UE_5.5
export AVENGINE_SPEAR_ROOT=/data/datasets/avengine_workspaces/shared/SPEAR-7fbf3632
git config --global --add safe.directory /data/datasets/avengine_workspaces/shared/SPEAR-7fbf3632
```

The `safe.directory` command is a one-time, per-user trust opt-in because the
shared checkout has a different owner. Do not apply it system-wide. That
checkout is a sparse, pinned provenance source. Its group permissions may allow
writes, but consumers must not modify it: use it to package, capture and verify
the tracked Apartment map, and do not launch UE or write project files there.
Re-exporting with UE requires a contributor-owned full clone:

```bash
export AVENGINE_SPEAR_ROOT="/data/datasets/avengine_workspaces/users/$USER/SPEAR"
git clone https://github.com/Eastforward/spear.git "$AVENGINE_SPEAR_ROOT"
git -C "$AVENGINE_SPEAR_ROOT" checkout --detach 7fbf3632fdb63cc2eceea564811c9597cabfb199
```

The shared UE path is only the engine root; it is not a SPEAR checkout. Public
Apartment export and package artifacts may remain under
`/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m1/`, or use
any other shared readable locations through the two Apartment root variables.

The runtime environment must contain the pinned audio-enabled Habitat build,
`numpy-quaternion`, NumPy, Pillow, and a working headless GPU/EGL context. The
capture adapter imports `quaternion` before `habitat_sim` to preserve the known
runtime import workaround.

Install the official Habitat MP3D example package locally if it is absent:

```bash
"$HABPY" "$RUNTIME/src_python/habitat_sim/utils/datasets_download.py" \
  --uids mp3d_example_scene \
  --data-path "$RUNTIME/data" \
  --no-replace
```

Before capture, the following official local-only assets must exist beneath
`$RUNTIME/data/scene_datasets/mp3d_example`:

```text
mp3d.scene_dataset_config.json
17DRP5sb8fy/17DRP5sb8fy.glb
17DRP5sb8fy/17DRP5sb8fy_semantic.ply
17DRP5sb8fy/17DRP5sb8fy.house
17DRP5sb8fy/17DRP5sb8fy.navmesh
```

These MP3D files remain subject to the Matterport3D terms and are not committed
to this repository. UE export additionally requires UE 5.5.x, its GLTFExporter
plugin, the SPEAR `SpearSim` project, and the apartment map at
`/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`.

## 1. Build the Blender custom room and navmesh

```bash
"$BLENDER" --background --factory-startup --python-exit-code 2 \
  --python "$REPO/tools/blender/build_m1_custom_room.py" -- \
  --output-dir "$REPO/examples/m1/rooms/blender_custom/visual"

"$HABPY" -m avengine.cli m1 validate-room \
  "$REPO/examples/m1/rooms/blender_custom/room_manifest.json"
"$HABPY" -m avengine.cli m1 validate-request \
  "$REPO/examples/m1/requests/blender_custom.json" \
  --room "$REPO/examples/m1/rooms/blender_custom/room_manifest.json"

"$HABPY" -m avengine.cli m1 build-navmesh \
  --room "$REPO/examples/m1/rooms/blender_custom/room_manifest.json" \
  --request "$REPO/examples/m1/requests/blender_custom.json" \
  --runtime-root "$RUNTIME" \
  --output "$REPO/examples/m1/rooms/blender_custom/visual/navmeshes/m1_custom_room.navmesh"
```

Commit intentional changes to the small tracked custom-room assets before the
formal capture so both worktrees are clean.

`build-navmesh` materializes a candidate file; it is not the capture-time
admission shortcut. The official, custom and legacy room manifests used for
formal M1 evidence must all declare `navmesh_policy: load_declared` and name an
existing navmesh.

## 2. Export and audit the legacy UE apartment

Export the loaded apartment world as UE StaticMesh **render LOD0**. Actor AABBs
are measurements only and must never replace surface geometry.

```bash
mkdir -p "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT"

"$SPEARPY" "$SPEAR/tools/run_editor_script.py" \
  --unreal-engine-dir "$UE" \
  --unreal-project-dir "$SPEAR/cpp/unreal_projects/SpearSim" \
  --map /Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000 \
  --launch-mode full \
  --render-offscreen \
  --script "$REPO/tools/ue/export_apartment_gltf.py" \
  --output "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/scene.glb" \
  --manifest "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/ue_export_manifest.json" \
  --spear-root "$SPEAR" \
  --texture-size 512

"$BLENDER" --background --factory-startup --python-exit-code 2 \
  --python "$REPO/tools/mesh/audit_real_surface_mesh.py" -- \
  --input "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/scene.glb" \
  --output "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/mesh_audit.json" \
  --minimum-triangles 1000
```

The exporter requires UE 5.5.x, proves that UE is running the `SpearSim`
project inside `$SPEAR`, rejects unsaved map/content packages, and reloads the
exact apartment map from disk. It deletes any stale manifest at startup and
deletes any stale target GLB immediately before export, so an older output
cannot be admitted as a new export. It records the clean SPEAR commit and
source `.umap` hash before invoking UE's GLTF exporter, then requires an
identical post-export snapshot and another clean-package check.

Every selected `/Game` StaticMesh or material reference must resolve to a
unique `.uasset` beneath the SPEAR project, be tracked by Git, and match the
path and SHA-256 recorded in the export manifest. `/Engine` references are
enumerated separately because they are not SPEAR repository files. The
admitted apartment export records 144 selected `/Game` package files and one
separate `/Engine` reference; these are measured values for this scene, not
fixed counts for future exports. The later packager and evidence verifier
replay the full package closure. They cannot attach provenance to an
unrelated, older GLB.

The audit must report `real_surface_gate.status == "pass"`, at least 1,000
triangles, a triangle count other than the known 252-triangle bounds signature,
and `all_mesh_nodes_are_simple_boxes == false`.

The packaging step also requires the SPEAR tracked worktree to be clean and
records both its Git commit and the source `apartment_0000.umap` SHA-256.

Producer absolute locators recorded in an older `ue_export_manifest.json` are
diagnostic only; they are not portable identity. Admission binds the exact
SPEAR commit, each repository-relative package path,
and current file size/SHA-256, plus the exported GLB and full selected-package
closure hashes. A consumer may therefore relocate the same clean checkout by
setting `AVENGINE_SPEAR_ROOT`; matching a producer's private absolute root is
neither required nor sufficient.

A render-only consumer outside this formal route can use an already-produced
GLB or render package without installing UE or cloning SPEAR. The formal M1
package, capture and evidence route deliberately requires a clean, pinned
SPEAR provenance checkout so it can replay the map and selected-package
identity. Do not weaken that contract by removing the
`legacy_source_map_package` role or making the source map optional.

Package the audited GLB for Habitat and build its navmesh:

```bash
"$HABPY" "$REPO/tools/m1/prepare_legacy_apartment.py" \
  --scene-glb "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/scene.glb" \
  --ue-manifest "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/ue_export_manifest.json" \
  --mesh-audit "$AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT/mesh_audit.json" \
  --spear-root "$SPEAR" \
  --output-dir "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT"

"$HABPY" -m avengine.cli m1 validate-room \
  "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/room_manifest.json"
"$HABPY" -m avengine.cli m1 validate-request \
  "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/capture_request.json" \
  --room "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/room_manifest.json"

"$HABPY" -m avengine.cli m1 build-navmesh \
  --room "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/room_manifest.json" \
  --request "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/capture_request.json" \
  --runtime-root "$RUNTIME" \
  --output "$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/visual/navmeshes/legacy_apartment_0000.navmesh"
```

During capture, AVEngine unconditionally and explicitly loads the room's
declared navmesh into the active Habitat Pathfinder. It also creates an
independent Pathfinder and loads the same file there. Their full fingerprints
must match: all 17 `NavMeshSettings` values plus the shapes and SHA-256 hashes
of canonical little-endian float32 vertices and int32 indices. The requested
agent height, radius and `include_static_objects` setting must also match the
settings stored in the declared navmesh.

M1 separately closes the actual Habitat loading graph. For a path scene, the
active dataset, scene/stage, render/collision mesh, navmesh, semantic mesh and
semantic descriptor must resolve exactly to declared assets. For a handle
scene, dataset search paths and selected scene instance, stage, object
templates/meshes, live object semantic IDs and poses, and lighting template and
current setup must be exact and unambiguous. The evidence verifier replays the
recorded loaded-graph snapshot against the manifests and current files.

## 3. Capture each room in two independent processes

Define the three room/request pairs and output roots:

```bash
export NATIVE_ROOM="$REPO/examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
export NATIVE_REQ="$REPO/examples/m1/requests/habitat_mp3d_example.json"
export CUSTOM_ROOM="$REPO/examples/m1/rooms/blender_custom/room_manifest.json"
export CUSTOM_REQ="$REPO/examples/m1/requests/blender_custom.json"
export LEGACY_ROOM="$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/room_manifest.json"
export LEGACY_REQ="$AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT/capture_request.json"
export FORMAL="$REPO/tmp/m1/formal"
mkdir -p "$FORMAL"
```

Run each command separately. Each invocation is a fresh OS process;
`--repeat 3` also proves same-process immutability. Every evidence file records
a module-lifetime UUID plus process identity metadata. The second run is
accepted only when its UUID differs from the first run, so two captures in one
Python process cannot impersonate independent-process evidence. A first run
has no cross-process reference, so its expected overall status is `not_run`
and its expected exit code is 3. It is valid as a reference only if the sole
required non-pass check is `independent_process_repeatability=not_run`. Any
`fail` or `blocked` check must be resolved before the second run.

Habitat-native room:

```bash
"$HABPY" -m avengine.cli m1 capture \
  --room "$NATIVE_ROOM" --request "$NATIVE_REQ" \
  --output "$FORMAL/habitat_native/run1" \
  --runtime-root "$RUNTIME" --repeat 3

"$HABPY" -m avengine.cli m1 capture \
  --room "$NATIVE_ROOM" --request "$NATIVE_REQ" \
  --output "$FORMAL/habitat_native/run2" \
  --runtime-root "$RUNTIME" --repeat 3 \
  --reference-evidence "$FORMAL/habitat_native/run1/evidence.json"
```

Blender custom room:

```bash
"$HABPY" -m avengine.cli m1 capture \
  --room "$CUSTOM_ROOM" --request "$CUSTOM_REQ" \
  --output "$FORMAL/blender_custom/run1" \
  --runtime-root "$RUNTIME" --repeat 3

"$HABPY" -m avengine.cli m1 capture \
  --room "$CUSTOM_ROOM" --request "$CUSTOM_REQ" \
  --output "$FORMAL/blender_custom/run2" \
  --runtime-root "$RUNTIME" --repeat 3 \
  --reference-evidence "$FORMAL/blender_custom/run1/evidence.json"
```

Legacy UE real-surface room:

```bash
"$HABPY" -m avengine.cli m1 capture \
  --room "$LEGACY_ROOM" --request "$LEGACY_REQ" \
  --output "$FORMAL/legacy_ue/run1" \
  --runtime-root "$RUNTIME" --repeat 3

"$HABPY" -m avengine.cli m1 capture \
  --room "$LEGACY_ROOM" --request "$LEGACY_REQ" \
  --output "$FORMAL/legacy_ue/run2" \
  --runtime-root "$RUNTIME" --repeat 3 \
  --reference-evidence "$FORMAL/legacy_ue/run1/evidence.json"
```

Only the three `run2/evidence.json` files are completion candidates.
On a valid second run, the CLI copies the first evidence JSON plus its
observation and QA artifacts into `run2/independent_reference/`. This makes the
cross-process proof self-contained; deleting or changing that copy invalidates
`run2` verification.

## 4. Verify and aggregate

```bash
"$HABPY" -m avengine.cli m1 verify \
  "$FORMAL/habitat_native/run2/evidence.json"
"$HABPY" -m avengine.cli m1 verify \
  "$FORMAL/blender_custom/run2/evidence.json"
"$HABPY" -m avengine.cli m1 verify \
  "$FORMAL/legacy_ue/run2/evidence.json"

"$HABPY" -m avengine.cli m1 aggregate \
  "$FORMAL/habitat_native/run2/evidence.json" \
  "$FORMAL/blender_custom/run2/evidence.json" \
  "$FORMAL/legacy_ue/run2/evidence.json" \
  --output "$FORMAL/aggregate.json"
```

M1 passes only when all three `verify` commands and the exact three-room
aggregate return `status: pass`.

All tracked code, schemas, fixtures and configuration must be committed before
these formal captures. Do not change a tracked file after capture: Git binds
the exact clean AVEngine HEAD, while the evidence separately binds external
result-changing inputs and the selected historical runtime profile.

## Status and artifact rules

Required checks aggregate with precedence `fail > blocked > not_run > pass`:

- `pass`: the command executed and every required measured threshold passed.
- `fail`: execution completed but a contract, deterministic comparison,
  geometry, sensor, connectivity, provenance, or artifact-integrity threshold
  was violated. Invalid input contracts are also failures.
- `blocked`: execution could not reach the measurement because a required
  runtime, asset, GPU/UE/Blender capability, or external dependency was
  unavailable. Preserve the concrete blocker; do not relabel it `not_run`.
- `not_run`: the measurement was deliberately not attempted. The first capture
  process uses this only for the missing independent-process comparison. It is
  not evidence of success.

`pass` exits 0; ordinary evidence failure exits 1; contract validation errors
exit 2; `blocked` and `not_run` exit 3. Always use the JSON status and checks,
not the exit code alone.

The UE apartment `scene.glb` is currently about 63 MiB, and the complete RGB,
depth, semantic, top-down navigation QA map, navmesh, and evidence rerun
outputs can also be large. They belong under the ignored `tmp/` tree and must
not enter ordinary Git history. Commit only code, schemas, documentation,
small controlled custom room fixtures, and compact status/provenance records.
Large immutable evidence requires a separately approved artifact store or Git
LFS policy.
