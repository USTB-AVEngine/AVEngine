# Mixamo + ReplicaCAD Expansion Design

> Date: 2026-07-09
>
> Scope: extend the current AVEngine/SPEAR animal audio-visual dataset pipeline
> from approved Hunyuan3D animals in `apartment_v1` to a minimal, reviewable
> vertical slice that can use new Mixamo animation assets and one ReplicaCAD
> room scene.

## 1. Goal

Build the next dataset-expansion slice without losing the hard-won review
guarantees from Plan 1, Plan 1.5, and Plan 2:

- Existing work is committed and pushed before risky data-download/import work.
- Mixamo animation import is probed through the existing SPEAR example path.
- ReplicaCAD is treated as the next room source, not `shoebox_v2`.
- Every generated clip still produces side-by-side review video, topdown, audio
  metadata, and visible actor markers.
- Hunyuan3D mesh direction is automatically gated as far as possible; low
  confidence cases are rejected for review instead of silently accepted.

The first deliverable is one reviewable clip, not a large dataset batch.

## 2. Non-Goals

- Do not make a broad dataset run before one Mixamo + ReplicaCAD smoke clip is
  visually and numerically inspectable.
- Do not commit downloaded ReplicaCAD scenes, Mixamo FBX files, generated UE
  cooked content, or generated review videos.
- Do not remove the human review UI; the new automatic direction gate is a
  prefilter and confidence report, not a claim that every Hunyuan mesh can be
  semantically understood from geometry alone.
- Do not depend on `shoebox_v2` for the new room path except as a regression
  test and API reference.

## 3. External Data Layout

Use `/data/datasets` for large external assets:

- ReplicaCAD root: `/data/datasets/replicacad`
- Mixamo root: `/data/datasets/mixamo`

The repo should only store scripts, manifests, tiny metadata, and docs. If a
dataset is not present, the downloader/prober must say exactly which path is
missing and which command or manual action is required.

ReplicaCAD is expected to need official Habitat/ReplicaCAD assets. If the
download is gated or license-protected, the script must stop with a clear
status file instead of spinning without progress.

Mixamo is expected to need user-downloaded FBX files from the Mixamo website.
The pipeline should accept an existing directory of FBX files and import a
small allowlist first.

## 4. Architecture

### 4.1 Existing Pipeline To Preserve

Current approved flow:

1. Hunyuan3D animal asset is generated, painted, oriented, and approved.
2. `species_rig_map.py` maps each tag to an approved mesh, rig, animation, and
   diffuse texture.
3. Runtime proxy + rig swap is validated by `tools/gate_check_animal.sh`.
4. Scene generator writes `spec.json`, trajectories, flags, audio paths.
5. UE render writes `videos/apartment_v1_view0.mp4` and
   `videos/actor_visual_metadata.json`.
6. RLR audio writes per-source and mixed audio plus `apartment_v1_metadata.json`.
7. Review builder writes topdown and `side_by_side_review_annotated.mp4`.

New work must add sources behind the same review surface instead of creating a
parallel unreviewed path.

### 4.2 Mixamo Animation Adapter

Add a small probe layer around `external/SPEAR/examples/import_mixamo_dataset`:

- Discover FBX files under `/data/datasets/mixamo`.
- Import a named sample into `/Game/Mixamo`.
- Record skeleton name, animation sequence names, frame rate, root motion
  availability, bounding box, and forward-axis behavior.
- Produce an import report JSON under `external/SPEAR/tmp/mixamo_probe`.

For the first slice, Mixamo is a visual animation source for UE review. It is
not required to become the only quadruped animation source. If Mixamo assets
are human animations, the adapter must not force them onto quadruped rigs; it
should classify them as human-only unless a compatible animal skeleton is
detected.

### 4.3 ReplicaCAD Room Adapter

Add a room adapter that can turn one ReplicaCAD scene into:

- A UE visual import target or loadable map.
- An RLR scene mesh and acoustic material sidecar.
- A walkable/collision footprint for trajectory generation.
- A topdown extent and wall/obstacle overlay for review.

The first slice can use one scene and one camera/mic. It must prove:

- UE visual render and RLR mesh use the same coordinate frame.
- Source trajectory stays inside walkable space and does not pass through walls.
- Marker projection uses `actor_visual_metadata.json` when UE render provides
  it.
- Audio listener yaw follows the camera/mic yaw.

### 4.4 Automatic Direction Gate

The gate should report confidence, not pretend to infer impossible semantics.

Inputs:

- Approved `mesh_oriented.glb`
- Optional `direction.json`
- Optional generated reference image
- Runtime/rigged actor preview frames
- A simple motion trajectory whose direction is known

Checks:

1. Geometry axis sanity: bounding box dimensions, up-axis, foot contact plane.
2. Visual forward sanity: render front/back/left/right thumbnails and compute a
   deterministic image-hash report.
3. Motion sanity: render a short clip moving along +X with body yaw 0 and verify
   actor visual center advances in the expected direction.
4. Review artifact: write a contact sheet and JSON report.

Pass condition:

- High-confidence assets are allowed into automatic batch.
- Low-confidence assets are blocked with a report path and need human review.
- The gate never silently flips an approved human rotation unless it can explain
  the evidence in JSON.

This is not a proof of semantic head direction for all Hunyuan meshes. It is a
guard against silent 180 degree mistakes.

## 5. Git And Push Policy

Before risky data work:

1. Run the focused tests that cover review video, RLR yaw, Hunyuan paint paths,
   approved asset requirements, and UV preservation.
2. Commit AVEngine root docs and AGENTS updates separately from SPEAR code.
3. Commit SPEAR implementation changes on the current branch or a new feature
   branch.
4. Push AVEngine to `origin`.
5. Push SPEAR to `eastforward`.

Do not commit:

- `/data/datasets`
- `external/SPEAR/tmp`
- generated MP4/WAV/PNG outputs
- downloaded FBX/GLB datasets
- UE `Saved`, `Intermediate`, or cooked outputs

## 6. Test Strategy

Lightweight tests:

- `tests/tools/spike_rlr/test_review_videos.py`
- `tests/tools/spike_rlr/test_room_conventions.py`
- `tests/tools/spike_rlr/test_run_audio_pass_cli.py`
- `tests/tools/test_hy3d_bake_diffuse_paths.py`
- `tests/tools/test_species_rig_map_approved_assets.py`
- `tests/tools/spike_rlr/test_auto_orient_ingest.py`

New tests to add:

- Direction gate report schema and pass/block behavior.
- ReplicaCAD path resolver with missing-data error.
- Mixamo path resolver with missing-data/manual-download status.
- Room adapter coordinate transform on a tiny synthetic fixture.

End-to-end smoke:

- One clip with a current approved animal in one ReplicaCAD room.
- If Mixamo data is present, one Mixamo import probe report.
- Review output contains UE video, topdown, actor visual metadata, RLR metadata,
  and annotated side-by-side video.

## 7. Completion Report

At the end of the run, write:

`docs/superpowers/status/2026-07-09-mixamo-replicacad-expansion-status.md`

The status file must say:

- What was completed.
- What was not completed.
- Why unfinished items stopped.
- Exact paths to generated review artifacts.
- Exact tests that passed or failed.
- Whether any download is still running. The intended final state is no running
  download or render process.
