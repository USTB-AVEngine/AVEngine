# Rocketbox Human Review And Retarget Design

> Date: 2026-07-10
>
> Scope: establish one reviewable Rocketbox human baseline before testing a
> Hunyuan-to-Rocketbox skinning spike or batching the remaining locomotion
> clips.

## 1. Goal

Build a human asset path with an explicit human review gate at both sides of
animation binding:

1. Restore one male and one female Rocketbox avatar with their official
   textures.
2. Render their untouched bind poses from multiple views and obtain human
   approval for identity, appearance, up axis, and forward axis.
3. Review the source walk skeleton and root-motion direction independently.
4. Retarget one neutral walk with rest-pose correction rather than assigning
   the source Blender `Action` directly to a different bind skeleton.
5. Render a multi-view walking review and obtain a second human approval.
6. Only after both approvals, consider batching the remaining 68 Rocketbox
   walk/run clips.

The first accepted baseline is Rocketbox-only. Hunyuan human binding is a
separate technical spike that starts only after the Rocketbox result passes
the second review.

## 2. Confirmed Constraints

### 2.1 The Existing Videos Are Not A Valid Retarget Result

The current page renderer imports these official Rocketbox avatars:

- `/data/datasets/rocketbox/sample/Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx`
- `/data/datasets/rocketbox/sample/Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01.fbx`

The avatar FBXs contain 80-bone `Bip01` skeletons. The animation FBXs contain
121-bone `Bip01` skeletons with extra nub bones. Common bone names do not imply
identical rest transforms. Measured rest-frame differences include about
44-47 degrees on upper arms, 20-25 degrees on thighs, and 12-15 degrees on the
head.

The current renderer performs direct action assignment:

```python
armature.animation_data.action = action
```

This bypasses retargeting and applies source pose channels in the target's
different rest basis. The old 18 locomotion pages are therefore inventory and
failure-diagnosis artifacts, not approved animation evidence.

The avatar's measured world forward axis and the neutral walk root-motion
direction are both `-Y`. A global 180-degree rotation is not the root fix.

### 2.2 Original Materials Are Currently Hidden Or Missing

The probe scripts clear every original material and replace the whole person
with one flat preview color. Each sampled avatar actually has one skinned mesh
with three material slots: body, head, and opacity. Clothing, skin, and shoes
are not separate mesh objects or separate shirt/pants material slots.

The lightweight sample does not contain a complete texture set. The male body
color exists, the male head color is partial, and the female textures are
missing. Textured appearance review must fetch and hash the official TGA files
before it can pass.

### 2.3 A UV Atlas Is Not A Normal Photograph

The body texture may contain disconnected UV islands, seams, mirrored regions,
and skin plus several garments in one atlas. Sending the raw atlas to an image
editing model with a natural-language instruction is not a reliable way to
select a shirt or trousers.

For plain color changes, deterministic masked material edits are the primary
path. Image editing is reserved for patterns, logos, fabric detail, or other
appearance changes that benefit from generation.

## 3. Staged Architecture

### 3.1 Stage A: Restore And Inspect The Official Avatar

For each sampled avatar:

1. Download the exact official body, head, opacity, normal, and specular TGA
   files listed in the Rocketbox Git tree.
2. Verify every file against the official Git blob SHA.
3. Import the FBX without deleting its material slots.
4. Reconnect external TGA paths explicitly instead of relying on stale FBX
   Windows paths.
5. Record mesh, skeleton, material, texture, and source hashes in an inspection
   JSON file.

Missing required color or opacity textures block appearance approval. A gray
fallback may be rendered for geometry debugging, but it cannot be marked as an
appearance-approved asset.

### 3.2 Stage B: Pre-Bind Human Review

Render the untouched avatar with no locomotion action attached. Produce:

- one front view;
- one back view;
- left and right side views;
- one top view;
- one turntable video;
- a contact sheet with explicit `UP +Z` and `FRONT -Y` arrows;
- a close face, shoulder, wrist, and foot sheet.

The review record is stored under:

```text
external/SPEAR/tmp/rocketbox_human_review/<asset_id>/source_review.json
```

Required review fields:

```json
{
  "schema_version": "rocketbox_human_source_review_v1",
  "asset_id": "rocketbox_male_adult_01",
  "source_fbx": "/data/datasets/rocketbox/sample/Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx",
  "source_sha256": "8d1edb51b4dc3427ae2456f4407fc105532c145dd019e53cd42bab31cc948a29",
  "up_axis": "+Z",
  "forward_axis": "-Y",
  "geometry_status": "approved",
  "appearance_status": "approved",
  "direction_status": "approved",
  "approved_by": "jzy",
  "approved_at": "ISO-8601 timestamp",
  "notes": null
}
```

Downstream retarget commands must reject records missing any of the three
approval statuses. Existing Hunyuan static-mesh review code must not be reused
to bake Rocketbox FBXs through `trimesh`, because that path strips the armature
and animation data.

### 3.3 Stage C: Source Motion Review

Review `m_walk_neutral.max.fbx` and `f_walk_neutral.max.fbx` independently of
the target mesh. This is an internal engineering diagnostic, not a user-facing
approval gate. Produce a source-skeleton video with:

- front, side, and top views;
- a visible root trajectory;
- a forward arrow derived from the source skeleton;
- frame range and displacement labels;
- a bind/rest-pose comparison against the selected target avatar.

The report records source and target bone counts, mapped core bones, per-bone
rest angular differences, source forward axis, root displacement, and the dot
product between facing and travel direction.

The pipeline may stop here when the source report is malformed or contradicts
the already approved `FRONT -Y` target convention. A reviewer is never asked to
approve an FBX, a JSON file, or an unbound skeleton. Source-skeleton media may be
shown beside the final avatar only as optional diagnostic context.

### 3.4 Stage D: Rest-Corrected Retarget

Retarget only one neutral walk first. The retargeter must:

1. Map the shared body bones explicitly and ignore source-only nub bones.
2. Compute animation deltas in each source bone's parent-local rest basis.
3. Apply those deltas in the corresponding target parent-local rest basis.
4. Preserve the target avatar's bind mesh, UVs, material slots, skin weights,
   and facial bones.
5. Rotate and scale root translation from the reviewed source frame into the
   reviewed target frame exactly once.
6. Bake the result into a new target action without modifying the source FBX.

Finger animation is optional for the first baseline. The hand may follow the
hand bone as one rigid palm if that is needed for a stable first smoke, but
shoulder, elbow, wrist, hip, knee, and ankle motion are required.

The direct-action implementation remains available only as a negative
regression fixture. A 180-degree container rotation is not an accepted
retarget fix.

### 3.5 Stage E: Post-Retarget Review

For the retargeted neutral walk, produce:

- synchronized source skeleton and target avatar views;
- target front and side videos;
- a top view with root trajectory and facing arrow;
- a close shoulder, elbow, wrist, hip, knee, ankle, and foot-contact video;
- a contact sheet sampled across one complete gait cycle;
- `retarget_metrics.json` and an exported test GLB.

Automatic checks block approval when:

- required core bones are unmapped;
- target travel direction opposes the reviewed forward axis;
- the root trajectory contains a discontinuity at the loop boundary;
- a foot penetrates the floor beyond the configured tolerance;
- the mesh or material slot count changes unexpectedly;
- the target action cannot be re-imported from the exported GLB.

Human review remains authoritative for unnatural joint bending, visible foot
sliding, shoulder collapse, wrist separation, facial deformation, and overall
motion style.

Review is performed in a dedicated browser UI that serves the rendered media,
not the FBX. It presents the approved Rocketbox male and female avatars as two
independent review items. Each item provides looped videos for the target front,
side, top/root-trajectory, joint close-up, and foot-contact views; synchronized
source-skeleton context is optional and never replaces the skinned target view.

The reviewer selects `approve` or `reject` and may enter a note. The service
writes `motion_review.json` atomically without moving, rewriting, or deleting
the source FBX, target FBX, exported GLB, or rendered evidence. The UI displays
the recorded decision and supports replacing a previous decision so a rerender
can be reviewed without manual JSON editing.

Male and female decisions are independent, but the batch gate requires both
records to be approved and to match the hashes of the currently served retarget
manifest and review videos. Only then may the remaining 68 actions be queued.

## 4. Prompt-Controlled Appearance

### 4.1 Semantic Control Layer

Natural language is converted into a structured appearance request before any
texture operation:

```json
{
  "avatar_id": "rocketbox_male_adult_01",
  "shirt": {"color": "green", "pattern": "plain"},
  "trousers": {"color": "black"},
  "headwear": {"asset_id": "baseball_cap_0001", "color": "red"}
}
```

The renderer consumes this schema, not raw prose. Every generated variant
records the original prompt, parsed request, base asset hash, masks, editing
model and version, seed, and output texture hash.

### 4.2 One-Time UV Semantic Masks

Because Rocketbox uses stable topology, semantic masks are authored once per
avatar or compatible atlas family:

1. Select garment triangles on the 3D mesh for shirt, trousers, shoes, skin,
   hair, and other relevant regions.
2. Rasterize triangle labels into UV space.
3. Detect overlaps where two semantic classes share the same UV texels.
4. Block independent editing of any ambiguous overlap until the mesh is split
   or re-unwrapped.
5. Dilate mask borders and run seam checks to avoid color gaps.

Plain color changes use mask-aware HSV/LAB adjustment or shader parameters.
They do not call a diffusion model.

### 4.3 Image-Editing Use

For generated patterns or fabric detail, do not edit the raw UV atlas blindly.
Use a multi-view render-and-bake path:

1. Render front, back, and side views with RGB, normal, depth, UV-position, and
   semantic garment masks.
2. Run the approved editor only inside the garment mask; the current local
   scope permits FLUX.2 Klein and no other image model.
3. Project edited pixels back to the unchanged UV layout using the UV-position
   pass.
4. Blend overlapping views, optimize seams, and inpaint only unseen texels.
5. Re-render all review views and compare identity, pose, exposed skin, and
   non-target garments against the base avatar.

Headwear and garment silhouette changes are geometry changes. They use
license-approved accessory or clothing meshes attached to reviewed bones.
Prompt text selects an accessory and its parameters; it does not regenerate the
runtime human mesh. Existing Rocketbox construction, pilot, fire, military,
police, and security avatars may provide initial hat or helmet variants, but
their mesh and texture structure must be audited individually.

## 5. Hunyuan-On-Rocketbox Follow-Up Spike

This spike begins only after one Rocketbox neutral walk passes both reviews.
It is technical evidence, not a formal dataset asset, because Hunyuan license
constraints still require separate resolution for training and evaluation use.

### 5.1 Attempt One: Part-Aware Skin Transfer

Use one strict soft T-pose Hunyuan human whose arms, forearms, wrists, palms,
legs, and background gaps are visibly separated. Before binding:

1. remove background and floor-card components;
2. reject disconnected shoulder, wrist, hip, or ankle geometry;
3. align reviewed body landmarks to the approved Rocketbox bind avatar;
4. classify Hunyuan vertices into torso, upper arm, forearm, palm, thigh, calf,
   foot, neck, and head regions;
5. transfer weights only from the matching Rocketbox body region;
6. smooth and normalize weights within each joint neighborhood;
7. collapse finger weights into the palm for the first smoke;
8. attach the result to the already validated Rocketbox target skeleton and
   neutral-walk action.

This avoids the previous global nearest-surface failure where sleeve, torso,
thumb, and finger regions contaminated one another.

### 5.2 Attempt Two: Stable-Template Projection

If the direct Hunyuan mesh still tears, stop after that documented attempt.
The fallback keeps Rocketbox topology and skin weights, then fits or deforms
the stable Rocketbox surface toward the Hunyuan silhouette and projects
Hunyuan appearance onto the stable UV/material representation.

The fallback is the preferred production architecture because animation never
depends on arbitrary generated topology. It is not another hand/forearm proxy
overlay.

### 5.3 Spike Success Criteria

The Hunyuan result must pass the same post-retarget review as Rocketbox, plus:

- no detached or duplicated hands;
- no shoulder or sleeve overlay seam;
- no background card or leg-gap fan;
- no torso-to-arm weight contamination;
- connected, solid palms throughout one gait cycle;
- deterministic reproduction from saved alignment and transfer metadata.

Failure remains a valid result and does not block the Rocketbox baseline.

## 6. Model Weight Storage

The machine's existing canonical model directory is lowercase:

```text
/data/models
```

`/data/Models` does not exist. The existing Hugging Face cache-style root is:

```text
/data/models/hub
```

All image and 3D model repositories downloaded for this work will ultimately
be physically stored as complete Hugging Face cache directories under
`/data/models/hub/models--<org>--<repo>`.

### 6.1 Migration Safety

Do not move a model while any downloader, checker, or probe holds files in its
cache directory. For each repository:

1. let active downloads finish, or intentionally stop a confirmed-stalled
   process;
2. verify target files and record all remaining `.incomplete` files;
3. stop the model's watcher;
4. move the entire `models--<org>--<repo>` directory on the same `/data`
   filesystem so relative `snapshots/* -> ../../blobs/*` links remain valid;
5. set `HF_HUB_CACHE=/data/models/hub` in AVEngine download and probe wrappers;
6. temporarily leave a compatibility symlink at the old per-repository path
   only when an external process still requires it;
7. run snapshot verification and one `local_files_only=True` load from the new
   root;
8. remove the compatibility symlink after all AVEngine helpers use the
   canonical root.

The migration covers the model repositories selected by this human-asset
work. It must not move or rewrite unrelated global caches owned by other
projects.

### 6.2 Probe Ordering

After migration and verification:

1. use only the verified FLUX.2 Klein snapshot for the next image-generation
   or low-drift image-edit probe;
2. keep Qwen, LongCat, FireRed, HiDream, and other image-model downloads and
   watchers stopped unless the user explicitly reopens their evaluation;
3. defer every image-model probe until the Rocketbox male and female neutral
   walks have passed the browser motion-review gate;
4. keep every 3D generator as a separate technical spike rather than treating
   it as a production human source.

An unrelated stale `.incomplete` file must not block a snapshot when every
selected target file resolves to a complete cached blob. The checker should
validate selected files and their referenced blobs rather than rejecting every
residual file in the repository cache.

## 7. Implementation Boundaries

The first implementation plan covers only:

- official texture completion for the two sampled Rocketbox avatars;
- source inspection and pre-bind review artifacts;
- source motion review artifacts;
- one rest-corrected neutral-walk retarget;
- post-retarget review artifacts and gates;
- safe model-cache path configuration and migration of repositories that are
  complete or intentionally stopped.

The following need separate implementation plans after the Rocketbox result is
approved:

- batching all 68 actions;
- semantic UV mask authoring and prompt-controlled appearance generation;
- the Hunyuan part-aware skin-transfer spike;
- the stable-template projection fallback;
- production registry promotion or formal dataset use.

## 8. Test And Verification Strategy

Implementation follows test-first development for reusable code.

Required automated tests:

- shared bone-map coverage and source-only nub handling;
- source/target rest-delta calculation on synthetic skeleton fixtures;
- root translation frame conversion and forward/travel dot-product checks;
- review-gate rejection when required approvals or textures are missing;
- motion-review media allowlisting, safe path handling, atomic decisions, and
  rejection of stale media or retarget-manifest hashes;
- the batch gate requiring current approvals for both male and female items;
- preservation of target mesh, UV, material-slot, and vertex-group counts;
- model-root resolution preferring `/data/models/hub`;
- stale unrelated `.incomplete` files not blocking complete selected files;
- incomplete referenced blobs still blocking load and probes.

Required Blender smokes:

- import untouched male and female FBXs and render bind-pose views;
- retarget one male and one female neutral walk;
- export and re-import each test GLB;
- verify animation frame ranges, mapped bones, materials, and skinning survive;
- render front, side, top, and joint-close review videos.

Required human decisions:

1. approve or reject each untouched source avatar;
2. approve or reject each bound, retargeted neutral walk in the browser;
3. separately approve any later appearance or Hunyuan variant.

## 9. Completion Criteria

This first phase is complete only when:

- both sampled Rocketbox avatars have complete, verified official textures;
- pre-bind review artifacts exist and have explicit human decisions;
- source neutral-walk direction reports exist;
- direct action assignment is no longer used as the accepted retarget path;
- one male and one female rest-corrected neutral walk can be exported,
  re-imported, and reviewed;
- post-retarget review records carry explicit human decisions;
- model weights used by completed probes reside under `/data/models/hub` and
  load locally from that root;
- no Hunyuan result is promoted into the formal source registry by this phase.
