# Multi-home furniture semantics and seated poses

User authorized implementation on 2026-09-05. All final room pixels for this task must use AVEngine SPEAR/UE. Author editable realistic rooms following the referenced Astra Blender-to-Unreal workflow; AVEngine owns production cameras/listeners, actor placement/routes, audio, facts and QA.

Scope is static furniture semantics and reliable seating/standing positions, reusable seated poses and existing walking/standing behavior. No grasping, opening drawers, dynamic furniture or interaction state machine. The user explicitly authorizes the new authored/reconstructed room family in this task; this does not re-enable the historical excluded Blender-custom room ID or change existing MP3D production routing.

Implementation remains entirely on server 48g-jump. No changes to active Studio or existing worktrees/stages. No push or main merge is authorized. Use ordinary configuration, Git identity and focused tests; no new hash locks, frozen contracts or gates.

## Workspaces

Base verified against actual GitHub main: 2ee3ffd23357c0d9d54a3d7594d399a2e5766a19.

- Root: /data/jzy/tmp/wt-multi-home-activity-integration, codex/multi-home-activity-integration.
- Room A: /data/jzy/tmp/wt-multi-home-room-a, codex/multi-home-room-a.
- Room B: /data/jzy/tmp/wt-multi-home-room-b, codex/multi-home-room-b.
- Poses: /data/jzy/tmp/wt-multi-home-poses, codex/multi-home-poses.
- Fresh shared task asset parent: /data/avengine_external/workspaces/multi_home_activity_20260905. Every worker owns its own subdirectory.
- Each repository tmp symlink resolves to a distinct directory below /data/datasets/avengine_workspaces/multi_home_activity_20260905.

Initial device check: GPU2 free; GPUs0/1 and GPU3 contain other users/tasks and are not allocated here. Root schedules GPU2/UE jobs and rechecks occupancy before each job. CPU/Blender workers must keep bounded thread counts and independent outputs.

## New findings to reuse

The retained /data/jzy/blender_projects/avengine_life_rooms_v1 contains semantic room builders, textures/furniture, seated GLBs, and prior Habitat and UE outputs. It is a read-only audit/adaptation source. Product entrypoints must live in the new AVEngine branches; they must not keep calling external project scripts. Inspect real files/pixels instead of accepting its completion documents. Old room specs contain authored cameras/routes and cannot supply the production plan for this task.

Root will extend the existing SPEAR editor residential runner and add room-semantic placement/planning. It will not replace final UE pixels with Habitat or Blender renders. Room workers supply scene geometry and seat metadata, without production cameras.

## Status

As of 2026-09-05, root branch includes room A/B/C builders, the shared furniture layout planner, four seated asset preparation/import tools, generalized N-actor/explicit-clock SPEAR residential capture, isolated UE stage assembly, polished room export, and per-room acoustic packaging. Latest integrated implementation commit: a35bed1. Focused source tests: 86 passed.

### Completed evidence

- Three independent editable rooms with GLB/USD and furniture/seat metadata. Current second visual revision: /data/avengine_external/workspaces/multi_home_activity_20260905/polished_v2/{room_a,room_b,room_c}.
- Four Rocketbox seated assets imported into the private UE stage; actual mesh/animation/material binding readback passed. Manifest: /data/avengine_external/workspaces/multi_home_activity_20260905/poses/import_runs/b492d6b_retry2/seated_human_ue_import_manifest.json.
- Private stage: /data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim/SpearSim.uproject. Original room A USD map import passed (tmp/room_a_ue_import_v1/result.json).
- General camera/placement planner has explicit Blender -> Habitat (X,Z,-Y) -> UE (100X,-100Y,100Z) conversion; camera UE orientation derives from the forward vector. Blueprint asset paths normalize to generated _C classes.
- Original A/B/C acoustic packages compiled. One real native RLR impulse response passed for A. A/C geometry QA is not passing, so these remain explicit research candidates, not formal acoustic admission. Polished-room packages are being prepared separately.

### Current work

- Sequential second-revision UE map imports: tmp/polished_v2_map_import, coordinator PID 1465813. Verify result.json AND process_result.json; do not infer completion from launch.
- First corrected seated capture input: /data/datasets/avengine_workspaces/multi_home_activity_20260905/planning/room_a_polished_ue_plan_150_v1/episode_plan.json; map /Game/AVEngine/MultiHome/room_a_polished_v2. Four dining seats, 150 frames at 15 Hz. No successful seated-in-room native pixels yet.
- Blender previews inspected: second revision improves A but upholstery remains simplified; B author preview is too close to sofa; C shows a dark horizontal wall band. Worker is preparing a fresh third revision and complete sidecars. Author preview cameras are not production camera plans.
- Remaining delivery: verify actual seating/camera/lighting in SPEAR/UE, produce all three final room outputs, connect actual emitter/listener paths to audio and example questions, run appropriate regression, and document limitations.

### Preserved failed attempts

- tmp/native_editor_transport_smoke_v1: retained map absent from fresh private project; not a successful capture.
- tmp/transport_map_import_v1: null-RHI actor creation crashed without a viewport. Subsequent map creation uses RenderOffscreen and GPU 2.
- tmp/room_a_seated_ue_check_v1: Blueprint asset path lacked generated-class _C and native actor spawn failed before frames. Fixed in integrated ba2408f. Old planning room_a_ue_plan_150_v1/v2/v3 are superseded; only the polished plan above includes the current class-path correction.

All output attempts are preserved. Active Studio, old worktrees, original assets and other GPU jobs remain untouched. No push, main merge, formal admission or service cutover has been performed.
