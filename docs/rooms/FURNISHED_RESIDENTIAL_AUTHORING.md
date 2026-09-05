# Furnished residential room authoring

This runbook is for a new static residential room that will be rendered by an
already imported private SPEAR/UE stage. AVEngine owns the room plan, seat
placement, actor/camera selection, clock and readback; UE owns the native
pixels. The room asset, UE project, editor and generated outputs stay outside
Git.

## Room package

Provide one ordinary room handoff JSON. Paths below are placeholders supplied
by the caller:

```json
{
  "kind": "avengine_polished_room_handoff",
  "status": "research_candidate",
  "room_id": "my_room_v1",
  "envelope": {"bounds_xy_m": [-6.0, -5.0, 6.0, 5.0]},
  "artifacts": {
    "editable_blend": "<ROOM_ROOT>/my_room.blend",
    "visual_glb": "<ROOM_ROOT>/visual/my_room.glb",
    "collision_glb": "<ROOM_ROOT>/visual/my_room_collision.glb",
    "objects": "<ROOM_ROOT>/object_semantics.json",
    "seated_affordances": "<ROOM_ROOT>/seated_affordances.json",
    "usd": "<ROOM_ROOT>/usd/my_room.usda"
  },
  "native_execution": {
    "backend": "spear_unreal",
    "map_or_stage": null,
    "status": "pending_root_execution"
  },
  "qualification_claim": false
}
```

`objects` must describe static furniture and the actual architectural mesh
must contain floors, walls, doorframes and other blocking structure. A room
AABB is useful for grid bounds and clearance; it is not a wall mesh and cannot
prove visibility. Each seat point needs an affordance ID, furniture ID,
position in room-local Blender coordinates, support height and chair-to-table
facing. Four seats intended for one dining episode must share the same table
object/group in the metadata.

A polished visual GLB may be accompanied by `fallback_artifacts` for older
object/seat sidecars while the polished exporter is being repaired. The
fallback is a metadata bridge; it does not make missing polished wall or chair
geometry native-validated.

Audit the visual surface with Blender before planning:

```bash
BLENDER=<BLENDER_BIN>
ROOM_GLB=<ROOM_ROOT>/visual/my_room.glb
AUDIT_OUT=<ROOM_ROOT>/qa/visual_mesh_audit.json
"$BLENDER" --background --factory-startup --python-exit-code 2 \
  --python "$AVENGINE_ROOT/tools/rooms/audit_real_surface_mesh.py" -- \
  --input "$ROOM_GLB" --output "$AUDIT_OUT"
```

A successful audit still describes the source mesh only. It does not prove a
SPEAR stage import or a camera/actor readback.

## Pose and seat frames

Use the real UE seated import manifest plus its request when available:

- `blueprint` or `blueprint_class` identifies the imported asset; a Blueprint
  object path is normalized to its generated `_C` class path by the planner.
- `skeletal_mesh` and the full animation object path are copied into the actor
  binding.
- `emitter_offset_avengine_m` is converted to `emitter_local_ue_cm` using the
  AVEngine asset basis.
- A request `root_offset_from_seat_anchor_blender_m` may already be rotated by
  `reference_chair_yaw_degrees`. The planner first removes that reference
  rotation, then applies the room seat yaw once. `reference_actor_yaw_degrees`
  is reference calibration and never becomes a new room heading.
- The pose root is derived from the seat reference and pose `seat_top_m`; the
  seat surface itself is never used as the actor root.

An explicit chair-forward field from the room exporter takes precedence. A
facing value inferred from a table center is only a candidate when the room
has no verified chair-forward metadata. It must remain marked as a candidate
until the chair mesh or native pixels confirm the direction.

## Plan a four-person room

Set variables to your private paths and the actual `/Game` map produced by the
UE import. Do not invent a map path in the planner:

```bash
export AVENGINE_ROOT=<AVENGINE_CHECKOUT>
export PYTHON=<AVENGINE_PYTHON>
export ROOM_HANDOFF=<ROOM_ROOT>/room_handoff.json
export ASSET_ROOT=<ROOM_ROOT>
export POSE_MANIFEST=<POSE_ROOT>/seated_human_ue_import_manifest.json
export POSE_REQUEST=<POSE_ROOT>/seated_human_ue_import_request.json
export MAP_PATH=/Game/AVEngine/MultiHome/my_room_polished_v2
export PLAN_ROOT=<EXTERNAL_OUTPUT_ROOT>/my_room_plan_150

cd "$AVENGINE_ROOT"
PYTHONPATH=src "$PYTHON" tools/rooms/plan_furnished_residential_episode.py \
  --room "$ROOM_HANDOFF" \
  --asset-root "$ASSET_ROOT" \
  --pose-bindings "$POSE_MANIFEST" \
  --pose-request "$POSE_REQUEST" \
  --map-path "$MAP_PATH" \
  --seat-count 4 \
  --actor-count 4 \
  --frame-count 150 \
  --frame-rate-hz 15 \
  --sample-rate-hz 16000 \
  --output "$PLAN_ROOT"
```

The plan contains `scene.map_path`, `scene.scene_id`, `clock`,
`visual_lighting`, four actor declarations and per-frame actor/camera states.
Camera candidates are generated from geometry-grid points and the complete
orientation cross. After actor seats are selected, AVEngine scores target
AABBs with FOV coverage, room/furniture clearance and available static-GLB
rays. A score is a planning observation; `target_los_status` and
`native_validation_status` remain explicit until UE readback.

## Overview without people

For a room overview, omit pose inputs and request the explicit overview mode:

```bash
PYTHONPATH=src "$PYTHON" tools/rooms/plan_furnished_residential_episode.py \
  --room "$ROOM_HANDOFF" \
  --asset-root "$ASSET_ROOT" \
  --map-path "$MAP_PATH" \
  --frame-count 150 --frame-rate-hz 15 --sample-rate-hz 16000 \
  --overview-only --output "$EXTERNAL_OUTPUT_ROOT/my_room_overview_150"
```

This emits zero actors and zero per-frame actor states. The existing SPEAR
runner accepts this only when `--visual-only-research` is supplied. It uses the
same camera renderer and readback path; it does not create a second renderer
or an actor/action engine.

## Native SPEAR render

After the private UE stage has been imported and its `/Game` map path has been
read back, use the existing runner. The exact extension, project and editor
paths are private runtime inputs:

```bash
SPEAR_EXT=<SPEAR_EXTENSION_DIR>
UPROJECT=<PRIVATE_UPROJECT>
UNREAL_EDITOR=<PRIVATE_UNREAL_EDITOR>
CAPTURE_OUT=<EXTERNAL_OUTPUT_ROOT>/my_room_native_visual

PYTHONPATH=src "$PYTHON" tools/rooms/run_spear_residential_episode.py \
  --episode-root "$PLAN_ROOT" \
  --spear-ext-dir "$SPEAR_EXT" \
  --uproject "$UPROJECT" \
  --unreal-editor "$UNREAL_EDITOR" \
  --output "$CAPTURE_OUT" \
  --graphics-adapter <GPU_INDEX> \
  --rpc-port <FREE_RPC_PORT> \
  --visual-only-research
```

Use `--native-multimodal` only when native depth/object-ID readback is needed.
For a four-person plan, the runner spawns the exact Blueprint classes, checks
per-frame actor closure, reads camera/actor/emitter transforms, and writes
frame readbacks. For an overview, the runner skips actor spawn, keeps camera
readback and writes normal metric depth/object-ID artifacts when multimodal
capture is enabled.

A visual-only receipt is research evidence. It does not claim formal room
admission, actor-forward correctness, complete wall visibility or a counted
Episode. Review the rendered frames and readback coordinates before proceeding.

## Audio and QA follow-up

Audio is a separate AVEngine authority. Build or select the room-local
acoustic scene and dry sound inputs with the existing residential source
planner, then render/validate RIR work through the declared acoustic backend:

```bash
PYTHONPATH=src "$PYTHON" tools/rooms/build_residential_source_episode.py \
  --scene-metadata "$SCENE_METADATA" \
  --profile "$ROOM_PROFILE" \
  --dry-root "$DRY_ROOT" \
  --output "$AUDIO_PLAN_ROOT"
```

The audio plan must use the same room identity and coordinate convention as the
native visual plan. Do not infer acoustic materials from a visual GLB and do
not use review lighting as acoustic truth. Run the room/asset QA and media
readback tools appropriate to the selected backend; keep `not_run`, `blocked`
and `research_only` explicit when a native layer or rights-qualified input is
not available. Only after native scene, camera, actor, emitter and media
readbacks pass may an owner decide whether the room is eligible for a later
formal episode route.

## Native camera review

A single geometry score is only a candidate ranking. To compare real
occlusion, body overlap and exposure, build a short review plan from the
four-person plan:

```bash
REVIEW_PLAN_ROOT=<EXTERNAL_OUTPUT_ROOT>/my_room_camera_review_150
PYTHONPATH=src "$PYTHON" tools/rooms/plan_furnished_camera_review.py \
  --episode-root "$PLAN_ROOT" \
  --output "$REVIEW_PLAN_ROOT" \
  --candidate-count 8 \
  --hold-frames 6 \
  --grid-step-m 0.75
```

Each selected candidate occupies six consecutive frames while actor states
remain fixed. The review plan is intended for the existing runner with
`--native-multimodal --visual-only-research`; compare the per-target depth and
object-ID readback for those frame ranges before choosing a production camera.
A review plan is research evidence and does not promote a camera or claim that
all chairs, walls or doorframes have passed native visibility review.
