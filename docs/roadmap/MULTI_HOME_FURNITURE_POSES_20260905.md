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

Latest integrated room authoring code: 24c828a (2026-09-05). Four calibrated seated people, room semantics, AVEngine camera selection, native SPEAR/UE capture, RLR binaural audio and three existing QA types have completed the v3 three-room research chain. v6 asset detail is now under native UE review; its passing export/import does not yet mean final visual acceptance.

### Completed evidence

- Reliable v3 assets: /data/avengine_external/workspaces/multi_home_activity_20260905/polished_v3_final/{room_a,room_b,room_c}. Source semantics include physical wall/furniture bounds, chair fronts, and measured dining seat tops (A/C .53m, B .52m).
- Correct-pose v3 native output: root tmp/room_{a,b,c}_v3_posefix_native_v1. Every room has actual 240 frames at 15 Hz / 16 seconds, 1280x720 RGB, metric depth, object IDs, target-only depth, per-frame actor/camera/emitter readbacks and a passing native receipt. Plans ending posefix_camera_native_v2 additionally correct nested quaternion metadata; actual UE poses/cameras are unchanged from the v1 captures because the renderer used explicit UE yaw.
- Matching v3 audio and QA: /data/datasets/avengine_workspaces/multi_home_activity_20260905/integration_sol/room_{a,b,c}_v3_posefix_audio_v1, corresponding research_qa_v1 and review_v1 directories. Actual mixtures and all four wet stems decode to 16-kHz stereo / 256000 samples. Dry speech intervals do not overlap; wet reverberation tails may overlap naturally. Review muxes contain 240 frames / 16 seconds plus stereo sound.
- Each room's research adapter materializes QS-003 who_spoke_first, QS-012 appearance_to_spoken_content and QS-002 sound_to_appearance, without deferred cases. These are registry-bound QA with native pixel/PCM evidence, not model performance or formal admission. Current three-room questions target the same blue-clothed actor with the same utterance/order, so room transfer is verified but content/identity diversity is not.
- Four seated human UE assets were actually imported and mesh/animation/material readbacks passed: /data/avengine_external/workspaces/multi_home_activity_20260905/poses/import_runs/b492d6b_retry2/seated_human_ue_import_manifest.json.
- Pose calibration is fixed in integrated e49b64e: convert the reference-world offset back to canonical frame, then rotate using seat yaw plus anatomical forward. The previously omitted 90-degree compensation caused 25.5cm lateral closure error. Independent four-direction closure now reaches zero. Nested Habitat quaternion agrees with actor yaw after 505f32d.
- Listener basis uses actual UE forward/up, with correct UE Z-up to M3 Y-up conversion. Wet convolution consumes the actual gain/fade-applied M6 bus segment after 758394d. Event offsets and source/listener coordinates come from readbacks, not idealized zero positions.
- General camera candidates use real mesh triangles and physical floor clearance. Native pixel selection preserves AVEngine composition by default; only a question may require a particular target to be visible. Authoring closeup cameras are strictly visual review tools.
- Full regression on the server: 3427 passed, 118 skipped, 52 subtests passed (Sol, 248.76s), using PYTHONPATH=src:/data/jzy/tmp/wt-multi-home-activity-integration/tmp/native_python_addons_v1. soundfile/cffi/pycparser are installed only in that task tmp directory, not shared Conda. New Blender authoring scripts additionally pass py_compile and git diff --check.

### Current v6 assets and visual work

- A and C: /data/avengine_external/workspaces/multi_home_activity_20260905/detailed_v6/{room_a,room_c}; 311 and 405 meshes; all 8/13 seats preserved. Rounded upholstery, wooden supports, thin curved dining chair backs, hollow tableware, real tabletop contacts, wall art mounting and PBR material provenance are included.
- B: /data/avengine_external/workspaces/multi_home_activity_20260905/detailed_v6_retry/room_b; 185 meshes, 8 preserved seats. Countertop/cabinet no longer fill the steel sink basin; nine assembled-scene down-rays hit steel/drain first. Smooth tangent-continuous faucet, thin plates, hollow cups and supported sofa cushions were checked by actual re-import and images. Preserve but do not select detailed_v6/room_b.
- Three fresh UE maps /Game/AVEngine/MultiHome/room_{a,b,c}_detailed_v6 have successful actual root-layer readbacks pointing to the selected USDs. Import evidence and new review outputs live under root tmp/multi_home_v6_native_review_20260905.
- A's first actual v6 UE overview preserves cloth/wood texture and upholstery shape at -3 EV. Root viewed this image. The plant and pet cushion are still visibly simplified; an independent CPU-only living_props_v7_candidate is being developed for review, not automatically adopted.
- B's first UE overview loses the authored sage tint because USD export connects the raw base-color image while skipping the Blender multiply node. Astra is preparing an exportable material correction and checking -4 EV against retained -3 EV. Production 240-frame capture waits for this correction. The failed/old asset is retained.
- C's first v6 overview stalled before RPC became ready (zero frames, shader/DDC initialization). The owning Astra worker is stopping only that verified task process and retrying with fresh output. No passing capture is claimed for that attempt.
- Fresh v6 source acoustic packages and 4-person calibrated plans exist in /data/datasets/avengine_workspaces/multi_home_activity_20260905/integration_sol_v6. Plans ending ue_plan_240_v2 are the valid 4-person versions; v1 is a preserved failed 2-person diagnostic. Production rendering and final PCM/QA are pending native visual acceptance and the final selected asset paths.
- v6 acoustic source QA: all per-object nonmanifold counts are zero, but global mesh QA fails (A duplicate 43/boundary 20/nonmanifold 226; B 0/40/15; C 48/40/199). Existing research proxy handling and research_placeholder materials are explicit. Fresh native RLR still has to prove load/render for each selected final geometry. No v3 RIR is relabeled as v6.

### User decisions and agent allocation

- Natural tabletop occlusion and people outside a composition are acceptable. The user's review of the old B frame explicitly rejected treating the fourth person's partial/out-of-frame placement as a global room failure. Chair facing was the actual issue, now corrected. Any older checkpoint language saying “fourth person clipped” was an overstrict diagnosis.
- The user explicitly selected GPT-6 Astra for 3D/modeling and GPT-5.6 Sol for ordinary code in this session, overriding the earlier Luna-only preference.
- b_detail_astra: authored B detail; now owns fresh UE maps, native visual review, the B material correction and GPU2 scheduling. No edits to shared renderer/planner.
- ac_detail_astra: authored A/C detail; now owns a bounded new living-prop detail tool/candidate, CPU only.
- integration_sol: camera reuse/plans, matching acoustic/audio/QA integration and source audit. It coordinates any GPU need with Astra and does not launch competing UE jobs.
- Root: code integration/review, paper/source comparison, current capability report, regression and final artifact inspection. No changes to live Studio, original worktrees, shared assets or GPU0/1 training.

### QA and paper findings

The stable catalog is 12 types, not 17. All 12 have implementation/unit/native sample evidence; only 3 are regenerated in the new rooms so far. The 17-card QA-v3 document mixes reused types, new meanings, controls and future extensions. Its cross-segment memory card is explicitly deferred. Current per-Episode IDs and short event order do not implement persistent identity, voice re-identification, explicit preference updates or long-term spatiotemporal history.

The two supplied papers were read from their original PDFs, including tables/figures. Hear you are motivates spatial disambiguation beyond semantic matching; VoxParadox motivates controlled lexical/acoustic contradiction. Neither establishes our proposed personalization/memory layer. The full 12-type list, all 17 card mappings, concrete examples, limitations and new research suggestions are in docs/roadmap/MULTI_HOME_QA_CAPABILITY_AUDIT_20260905.md.

Remaining room delivery is final selected native visual quality, matching audio/QA for that version, updated reproducible commands and usable preview media. More diverse activities/multi-segment memory data remain separate from the current four-person static seating sample. No formal dataset readiness is claimed.

### Preserved failed attempts

- tmp/native_editor_transport_smoke_v1: retained map absent from fresh private project; not a successful capture.
- tmp/transport_map_import_v1: null-RHI actor creation crashed without a viewport. Subsequent map creation uses RenderOffscreen and GPU 2.
- tmp/room_a_seated_ue_check_v1: Blueprint asset path lacked generated-class _C and native actor spawn failed before frames. Fixed in integrated ba2408f. Old planning room_a_ue_plan_150_v1/v2/v3 are superseded; only the polished plan above includes the current class-path correction.

All output attempts are preserved. Active Studio, old worktrees, original assets and other GPU jobs remain untouched. No push, main merge, formal admission or service cutover has been performed.
