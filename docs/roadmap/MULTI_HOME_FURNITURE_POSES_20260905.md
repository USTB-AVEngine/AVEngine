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

As of 2026-09-05, root branch includes room A/B/C builders, the shared furniture layout planner, four seated asset preparation/import tools, generalized N-actor/explicit-clock SPEAR residential capture, isolated UE stage assembly, polished room export, and per-room acoustic packaging. Latest integrated implementation commit: 6589d37. The shared runner now also supports explicit actor-free room overviews, native RGB exposure compensation/readback, and camera-buffer teardown.

### Completed evidence

- Three independent editable rooms with GLB/USD and furniture/seat metadata. Current second visual revision: /data/avengine_external/workspaces/multi_home_activity_20260905/polished_v2/{room_a,room_b,room_c}.
- Four Rocketbox seated assets imported into the private UE stage; actual mesh/animation/material binding readback passed. Manifest: /data/avengine_external/workspaces/multi_home_activity_20260905/poses/import_runs/b492d6b_retry2/seated_human_ue_import_manifest.json.
- Private stage: /data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim/SpearSim.uproject. Original room A USD map import passed (tmp/room_a_ue_import_v1/result.json).
- General camera/placement planner has explicit Blender -> Habitat (X,Z,-Y) -> UE (100X,-100Y,100Z) conversion; camera UE orientation derives from the forward vector. Blueprint asset paths normalize to generated _C classes.
- Original A/B/C acoustic packages compiled. One real native RLR impulse response passed for A. A/C geometry QA is not passing, so these remain explicit research candidates, not formal acoustic admission. Polished-room packages are being prepared separately.

### Current native results and active fixes

- Second-revision maps for A/B/C all imported successfully: tmp/polished_v2_map_import/completed.json and each room's result.json/process_result.json.
- Actual 1280x720, 150-frame/15-Hz (10-second) SPEAR captures completed for all three rooms: tmp/room_a_polished_seated_ue_check_v2, tmp/room_b_polished_seated_ue_check_v1, tmp/room_c_polished_seated_ue_check_v1. Actor/animation/camera readbacks and encoded media passed, but these are debugging research outputs, not quality-approved rooms.
- Native A overview also completed: tmp/room_a_overview_ue_exposure_v1. Explicit -3 EV compensation readback passed and corrected the overexposure. A's original seated capture had no compensation. B/C used -3 EV.
- Actual image review: A's old camera is occluded by a wall/doorframe; B clips the fourth person; C contains wrong seat-facing metadata and a person facing away from the table. Old overview camera favored a corridor. These plans are superseded by ongoing geometry-aware planning work; do not promote their passing transport receipts into visual-quality claims.
- Root found source sidecars lacked real wall geometry. Worker is producing complete physical wall/doorframe/furniture bounds and measured seat surfaces in a fresh polished_v3 directory. Planner is using actual GLB triangles for static ray checks, preserving target-independent candidate generation.
- Root also identified a pose-offset frame bug: blue/pink request offsets [0,+.18,-.01] were already rotated by their reference chair yaw 180, while green/white [0,-.18,-.01] used reference yaw 0. Treating all as actor-local double-rotates the first pair. Planner is correcting this calibration interpretation; reference actor yaw variations are not new room placement authority.
- Worker tasks: finish third-revision assets and complete self-contained semantics; fix static mesh LOS, grouped seats, seat facing and offset calibration; prepare matching acoustic packages plus four-speaker audio from real SPEAR emitter/listener readbacks.
- The C run emitted a nonfatal shared-memory BufferError at interpreter teardown after valid media output. Root fixed explicit SceneCapture component termination before closing the instance in 6589d37; focused tests passed, next native run must confirm clean teardown.
- Full unit run: 3402 passed, 3 failed, 118 skipped, 52 subtests passed. All three failures were missing soundfile in the native Python 3.12 environment, not mismatched export behavior. Added the ordinary soundfile dependency to the native extra and installed soundfile/cffi/pycparser ONLY into tmp/native_python_addons_v1. All three failed tests then passed. Use PYTHONPATH=src:tmp/native_python_addons_v1 for the final full run. Shared Conda environments were not changed.

Remaining delivery is visual/placement quality acceptance in all three SPEAR rooms, matching spatial audio and question examples, final relevant regression and concise reproducible commands. No final room-quality or formal dataset admission claim is made.

### Preserved failed attempts

- tmp/native_editor_transport_smoke_v1: retained map absent from fresh private project; not a successful capture.
- tmp/transport_map_import_v1: null-RHI actor creation crashed without a viewport. Subsequent map creation uses RenderOffscreen and GPU 2.
- tmp/room_a_seated_ue_check_v1: Blueprint asset path lacked generated-class _C and native actor spawn failed before frames. Fixed in integrated ba2408f. Old planning room_a_ue_plan_150_v1/v2/v3 are superseded; only the polished plan above includes the current class-path correction.

All output attempts are preserved. Active Studio, old worktrees, original assets and other GPU jobs remain untouched. No push, main merge, formal admission or service cutover has been performed.
