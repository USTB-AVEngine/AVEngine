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

Workspace preparation complete. Room authoring, seated native binding, generalized planning, source integration and actual multi-room SPEAR/UE delivery remain in progress.
